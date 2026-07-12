<p align="center">
  <strong>Sovereign Runtime</strong> — the inference engine of the Lazarus Sovereign Stack
</p>

# sovereign-vllm

The Lazarus AI Research fork of [vLLM](https://github.com/vllm-project/vllm),
purpose-built to serve as **Sovereign Runtime**: the single inference engine
inside [Lazarus Sovereign Stack](https://github.com/Lazarus-AI-Research/sovereign-stack),
a local-first AI appliance for small offices, workgroups, and customer-owned
hardware.

## The Sovereign Stack, and where this fork sits

Sovereign Stack is one coherent product: a private chat and document
workspace, retrieval over your own data, administration, observability, and
backups — all running on hardware you own, with nothing leaving the building
by default.

```text
        Sovereign Workspace          chat & documents (AnythingLLM)
                │
        Sovereign Gateway            routing, keys, budgets (LiteLLM)
                │
   ┌───► Sovereign Runtime ◄───┐     ★ this repository
   │    one container           │
   │    one process tree        │
   │    one port (8000)         │
   │    one OpenAI-compatible   │
   │    API                     │
   │                            │
   generation role         embedding role
   (assistant-large =      (embedding-omni-default =
    google/gemma-4-E2B-it)  LCO-Embedding/LCO-Embedding-Omni-3B-2605)
                │
        CUDA / ROCm / XPU / Metal / DGX Spark / Strix Halo
```

Everything above the runtime speaks to exactly one endpoint:
`http://sovereign-runtime:8000`. Sovereign Control (the admin plane) drives
the runtime exclusively through its health, manifest, and error surfaces.
That boundary — the **runtime contract** — is defined in the monorepo
(`docs/runtime-contract.md` plus JSON Schemas) and enforced by a conformance
harness that every runtime image must pass before release.

## Why upstream vLLM wasn't enough

vLLM is an excellent engine, and this fork changes as little of it as
possible. But an appliance runtime has obligations a serving library does
not:

1. **One process, multiple model roles.** The stack requires generation and
   embeddings served by a single supervised process behind a single port —
   not one `vllm serve` per model behind a reverse proxy. Upstream serves
   one model per server.
2. **An operational contract instead of a CLI.** Appliances are administered
   by software, not operators at a terminal. The runtime must expose an
   explicit state machine (`initializing → downloading → loading →
   smoke_testing → healthy | degraded | configuration_error |
   runtime_error`), keep its control API alive through recoverable failures
   (a bad model config must never crash-loop the container), and publish a
   machine-readable manifest of what is actually loaded — discovered, not
   assumed.
3. **Platform coverage upstream doesn't own.** The product ships on Macs
   (where Docker containers cannot reach Metal), NVIDIA and AMD
   workstations, unified-memory systems, and Intel XPU. Making one contract
   hold across all of them is Lazarus scope, not upstream's.
4. **The omni embedding model.** The stack's default embedding model
   (`LCO-Embedding-Omni-3B-2605`, a Qwen2.5-Omni-based multimodal embedder)
   needs pooling and multimodal-input paths that upstream does not provide
   out of the box.

## What we changed

The fork operates in **overlay mode**: upstream vLLM stays a pinned
dependency (`constraints.txt`), and every Lazarus customization lives in
clearly separated, Lazarus-owned code. The `vllm/` tree gets vendored only
when an in-tree patch becomes unavoidable — the known triggers are
scheduler-level cross-role fairness, engine-internal metrics changes, the
omni multimodal pooling path, and the Metal backend.

### `lazarus/appliance/` — the appliance layer

- **`launcher.py` (`run-sovereign-runtime`)** — the container entrypoint.
  Owns the startup sequence: config → control API up → download/verify
  weights → load roles serially → startup self-test → terminal state →
  manifest. Role failures degrade honestly; process exit on role failure is
  opt-in configuration, never a default.
- **`state.py`** — the contract state machine with deduplicated, structured
  error records (`MODEL_LOAD_FAILED`, `MODEL_REVISION_NOT_FOUND`,
  `HOST_AGENT_UNREACHABLE`, …) served at `/runtime/errors`.
- **`config.py`** — the `runtime.yaml` parser (validated against the
  monorepo's JSON Schema), including per-role priorities, best-effort memory
  weights, throttling policy, and an `engine_args` escape hatch.
- **`api.py`** — one FastAPI server on port 8000: health endpoints with
  liveness/readiness split, manifest and errors, role-routed OpenAI surface
  (`model` alias → role; wrong-role requests 404), bearer auth, per-role
  admission control with embedding throttling under generation pressure,
  and normalized `sovereign_*` Prometheus metrics labeled by role and
  served model.
- **`manifest.py`** — the `/runtime/manifest` document. Reports observed
  reality: embedding dimensions are probed from the loaded checkpoint
  (LCO-Omni probes at 2048), backends report what actually executes.
- **`healthcheck.py` (`sovereign-runtime-healthcheck`)** — Docker
  healthchecks probe liveness only, so model loads and downloads never
  cause restart loops.

### Multi-role serving (`lazarus/appliance/backends/vllm_engine.py`)

Each role gets vLLM's **own fully-assembled OpenAI application**
(`build_app` + `init_app_state`, the same assembly `vllm serve` uses),
running as a second engine inside the same supervised process. The appliance
dispatches role traffic to the right app over an in-process ASGI transport,
streaming included. Memory weights map to per-engine
`gpu_memory_utilization` with fixed headroom; roles load strictly serially
so memory profiling never races. Weight downloads happen in the appliance's
`downloading` state — engine child processes never touch the network.

### Metal support (`lazarus/agent/` + `agent-dist/`)

Docker on macOS exposes no GPU, so the `metal-arm64` runtime keeps the
container contract while inference runs host-side:

- **`sovereign-runtime-agent`** — a launchd-managed host daemon that
  supervises one llama.cpp server per role (multimodal projector support
  included), fails closed without its bearer token, binds loopback only,
  and exposes a single private port with an `/agent/manifest` and a
  streaming role proxy.
- **The `agent` engine backend** — the container half. Discovers roles from
  the agent manifest, forwards role traffic, and degrades to
  `configuration_error` (alive, diagnosable, no crash loop) when the agent
  is unreachable.
- **`agent-dist/`** — launchd plist template plus install/uninstall
  scripts. Default models are the canonical GGUF releases:
  `google/gemma-4-E2B-it-qat-q4_0-gguf` (+mmproj, reasoning budget 0) and
  `marksverdhei/LCO-Embedding-Omni-3B-2605-GGUF` (+mmproj).

### Backends behind one seam (`lazarus/appliance/backends/`)

`vllm` (in-process engines), `agent` (Metal host agent), and `fake` (a
deterministic engine so the entire appliance is testable on machines that
cannot run vLLM). The engine is swappable; the contract is not.

### Patches (`patches/`)

- `0001-gate-cpu-moe-bindings-on-apple.patch` — upstream's CPU build
  excludes `cpu_fused_moe.cpp` on Apple Silicon while binding it
  unconditionally, leaving `vllm._C` unloadable there. Carried until
  upstreamed.

### Docker build contexts (`docker/`)

Per-profile runtime images (`cuda`, `cpu`, `metal`) that layer the appliance
onto the pinned engine and ship the two contract binaries at their canonical
paths. Production images use immutable version tags.

## Verification

Every runtime image and backend must pass the same gates, wherever it runs:

- the **contract conformance harness** (`sovereign-evals conformance`) —
  health/readiness semantics, schema-valid manifest, chat, streaming,
  embeddings with dimension and normalization checks, role routing, auth;
- the **§25 lifecycle chaos suite** — configuration errors stay alive and
  diagnosable, recovery happens without crash loops;
- unit tests that run without vLLM installed (`pip install -e '.[dev]' && pytest`).

## Development

```bash
pip install -e '.[dev]'            # appliance + agent + tests, no engine
pip install -e '.[dev,engine]'     # with pinned vLLM (Linux)
ruff check lazarus tests && pytest tests
```

## License

Upstream vLLM is Apache-2.0; its license and notices are retained for all
upstream code. Lazarus additions: see `LICENSE`.
