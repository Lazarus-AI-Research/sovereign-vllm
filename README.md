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
   generation role         optional embedding role
   (assistant-large =      (configured by Control only
    google/gemma-4-E2B-it)  when a custom model is needed)
                │
        CUDA / ROCm / XPU / Metal / DGX Spark / Strix Halo
```

Everything above the runtime speaks to exactly one endpoint:
`http://sovereign-runtime:8000`. Sovereign Control (the admin plane) drives
the runtime through its health, manifest, error, and fixed authenticated
generation-control surfaces.
That boundary — the **runtime contract** — is defined in the monorepo
(`docs/runtime-contract.md` plus JSON Schemas) and enforced by a conformance
harness that every runtime image must pass before release.

## Why upstream vLLM wasn't enough

vLLM is an excellent engine, and this fork changes as little of it as
possible. But an appliance runtime has obligations a serving library does
not:

1. **One process, optional multiple model roles.** The normal stack uses this
   runtime for generation and a dedicated EmbeddingGemma service. Customers
   can add a custom embedding role behind the same supervised port instead of
   deploying another `vllm serve` process. Upstream serves one model per server.
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

## What we changed

The fork operates in **overlay mode**: upstream vLLM stays a pinned
dependency (`constraints.txt`), and every Lazarus customization lives in
clearly separated, Lazarus-owned code. The `vllm/` tree gets vendored only
when an in-tree patch becomes unavoidable — the known triggers are
scheduler-level cross-role fairness, engine-internal metrics changes, and the
Metal backend.

### `lazarus/appliance/` — the appliance layer

- **`launcher.py` (`run-sovereign-runtime`)** — the container entrypoint.
  Owns the startup sequence: config → control API up → download/verify
  weights → load roles serially → startup self-test → terminal state →
  manifest. Role failures degrade honestly; process exit on role failure is
  opt-in configuration, never a default.
- **`state.py`** — the contract state machine with deduplicated, structured
  error records (`MODEL_LOAD_FAILED`, `MODEL_REVISION_NOT_FOUND`,
  `HOST_AGENT_UNREACHABLE`, …) served at `/runtime/errors`.
- **`config.py`** — the strict `runtime.yaml` parser (validated against the
  monorepo's JSON Schema), including per-role priorities, best-effort memory
  weights, throttling policy, and Control-derived ordered NVIDIA GPU UUIDs
  with an exact tensor-parallel cardinality. Raw engine arguments are rejected.
- **`api.py`** — one FastAPI server on port 8000: health endpoints with
  liveness/readiness split, manifest and errors, role-routed OpenAI surface
  (`model` alias → role; wrong-role requests 404), bearer auth, per-role
  admission control with embedding throttling under generation pressure,
  and normalized `sovereign_*` Prometheus metrics labeled by role and
  served model.
- **`manifest.py`** — the `/runtime/manifest` document. Reports observed
  reality: embedding dimensions are probed from the loaded checkpoint, and
  backends report what actually executes. A generation-only runtime is fully
  healthy and simply omits the embedding role.
  Managed CUDA generation also reports the exact observed device UUIDs in
  local-rank order, device count, and applied tensor-parallel size; omission or
  disagreement keeps Control from treating a multi-GPU candidate as ready.
- **`healthcheck.py` (`sovereign-runtime-healthcheck`)** — Docker
  healthchecks probe liveness only, so model loads and downloads never
  cause restart loops.

### Generation gate observations and managed controls

Legacy fixed Runtime publishes manifest schema **1.3**, independently of the
compatible **1.2** configuration schema. Managed Runtime instances publish
configuration schema **1.3** and manifest schema **1.4**, pairing distinct
lowercase UUID `runtime_instance_id` and `deployment_id` values in both
documents. A Stack must reject managed identity on an older wire version; the
published runtime release remains unchanged and does not thereby gain this
capability. The required top-level `generation_paused` boolean reports the
actual first-party generation ingress gate for every backend. `true` means
ingress is closed, including while drain or an engine acknowledgement is
pending and after failed or cancelled control. `false` means only that this
ingress gate is open: it is not proof of engine availability, readiness, idle
work, response completion, or model qualification. Historical manifests at
1.0–1.2 omit this observation; that absence remains unknown and must not be
rewritten as `false`.

The existing bearer-authenticated, empty-body controls are
`POST /runtime/admin/generation/quiesce` and
`POST /runtime/admin/generation/resume`. Quiesce closes ingress, drains admitted
generation responses, and requires the engine's idle acknowledgement or
positively verified absence of an owned generation engine before returning
`{"quiesced":true}`. Resume does not reload the model or replace the
backend or configuration: it requires the real engine's unpaused acknowledgement
before reopening ingress and returning `{"resumed":true}`. A known successful
physical quiescence therefore needs its matching acknowledged resume, including
when a fenced gateway disruption leaves the Runtime and loaded model unchanged.
Unsupported legacy engine controls remain unavailable rather than inventing
acknowledgements.

The managed native llama.cpp path closes host ingress and drains complete
responses before observing both `llamacpp:requests_processing` and
`llamacpp:requests_deferred` at zero. The audited `b9960` source is
`a935fbffe1a3d31509c325c116454ab5d56b2eb8`. Each owned generation process receives
a private random API key through its environment. The metrics endpoint must
reject an unauthenticated probe and accept that key; process identity/liveness
must still match. Health is never idle proof. The key's fixed public suffix
prevents upstream last-four-character diagnostics from exposing key entropy.
Custom generation API-key/key-file arguments are rejected; embedding configuration
and credentials remain independent.

Incomplete or ambiguous responses, missing streaming terminal events, transport
failure and cancellation latch execution uncertainty for that exact native
child. Zero metrics or repeated commands cannot clear it; only proven process
replacement establishes fresh ownership. Native switch/cutback cannot stop an
unproven-idle child. An existing agent without these controls remains unavailable
for safe physical disruption rather than receiving an inferred acknowledgement.

A fully received nonstream native `400` can retire a source-proven single task
only when its bounded exact error envelope is `invalid_request_error` or
`exceed_context_size_error` and the same owned child's protected read-only
`GET /lora-adapters` returns its validated FIFO acknowledgement within five
seconds. Canonical `n_cmpl` overrides `n`; batch cardinality is never inferred
from `n` alone. This fences the completed request's release, not global engine
idle, and does not pause unrelated generation or clear earlier uncertainty.
Unknown errors, partial batches, malformed/truncated bodies, error streams and
missing acknowledgements remain fenced. The pinned source ordering and local
protocol fixtures are not physical engine qualification.

Managed Runtime start adopts the freshly observed host pause. Only that explicit
start action may resume after enabled roles are healthy and the authenticated
host returns exact owned-idle/resume acknowledgement. Passive manifest refresh
never resumes ingress. A lost reply keeps Runtime closed even if the host
completed resume. These protocol tests do not qualify a physical native engine.
Readiness, manifest publication and generation forwarding refresh authenticated
host admission facts. Missing, invalid, unavailable or paused host observations
can close Runtime ingress; later passive unpaused observations never reopen it.
Embedding service remains independent. Replacing a surviving native SlimServe
process after restart, repair or cutback requires scheduler quiesce
acknowledgement before shutdown, not an empty host response counter.

The ephemeral SlimServe child credential protects every HTTP path, including
health, inference and metrics. Legitimate probes and both native/CUDA role clients
use the existing private client headers; no health exemption or new caller key is
introduced, and the credential is not written to argv, manifests or logs.

Absence is not inferred from a missing app/client or an error label. SlimServe
retains process ownership until its leader is reaped and the owned process
group is affirmatively absent for the fixed reviewed local launch topology.
Unresolved spawns, remaining descendants, signal failures, and unknown group
checks remain fenced. In-process vLLM shutdown after native construction does
not by itself prove all workers exited; such uncertainty cannot acknowledge
quiescence. Cleanup also retains returned engines when app assembly fails and
collects cancelled off-loop constructor results. Same-instance restoration
preserves a closed gate until real resume; native validate/cleanup before the
first launch does not manufacture an existing pause. These are source and
local-process supervision contracts, not physical engine qualification.

The live manifest and configured on-disk publisher update at gate closure,
acknowledgement, and failure. A failed publication cannot produce a successful
control acknowledgement and keeps ingress closed. `/health` and the manifest's
loaded-model state may remain healthy while the gate is closed;
`/health/ready` and public generation remain unavailable until resume succeeds.
Embedding admission is independent.

This current source requires a Stack reader compatible with legacy manifest
1.3 and managed manifest 1.4. It does not change historical manifest
acceptance, advance the published Runtime release pin, or establish physical
serving or release qualification.

### Multi-role serving (`lazarus/appliance/backends/vllm_engine.py`)

Each role gets vLLM's **own fully-assembled OpenAI application**
(`build_app` + `init_app_state`, the same assembly `vllm serve` uses),
running as a second engine inside the same supervised process. The appliance
dispatches role traffic to the right app over an in-process ASGI transport,
streaming included. Memory weights map to per-engine
`gpu_memory_utilization` with fixed headroom; roles load strictly serially
so memory profiling never races. Hugging Face sources download in the
appliance's `downloading` state. Managed CUDA generation uses `source: local`
and consumes a complete directory prepared by Control, not a checkpoint file
or a repository fallback.

The managed local directory contains one approved primary checkpoint under
its original `*.safetensors` basename, matching `config.json`, `tokenizer.json`
and `tokenizer_config.json`, and the pinned runtime metadata selected by its
immutable engine profile. Gemma4 requires `processor_config.json`; Qwen3.5
also requires `preprocessor_config.json` and `video_preprocessor_config.json`.
Control includes matching generation defaults and chat templates. An optional
`model.safetensors.index.json` must reference only the staged primary filename;
an unsharded bundle does not require an index. The curated Fast bundle omits
its upstream stale index rather than rewriting metadata or aliasing weights.

Runtime checks this local CUDA generation shape before engine construction:
nonempty regular files only, no symlinks, nested directories, extra checkpoints
or unsupported metadata names; at most 16 fixed-name support files, each at
most 64 MiB and at most 256 MiB combined. Missing or ambiguous inputs produce
a bounded `CONFIG_INVALID` error while the control API remains alive. The
consumer check does not establish authenticity: Control owns the immutable
support-file manifest, SHA-256 verification, exact bundle contents and atomic
staging. Legacy Metal GGUF local-file inputs are unchanged.

The directory is passed unchanged as structured `--model`, alongside the pinned
revision, served alias, tensor-parallel size and eager setting. Runtime does
not repair local metadata with a Hub snapshot, switch to the source repository,
or add `hf_config_path`, tokenizer, remote-code or arbitrary engine overrides.
The private runtime manifest's `engine_model` is observed from the engine;
`served_model_name` and the public OpenAI model identity remain the public
alias, never the local directory. This input contract does not constitute
physical CUDA qualification.

### Metal support (`lazarus/agent/` + `agent-dist/`)

Docker on macOS exposes no GPU, so the `metal-arm64` runtime keeps the
container contract while inference runs host-side:

- **`sovereign-runtime-agent`** — a launchd-managed host daemon that
  supervises native generation and embedding processes, fails closed without
  its bearer token, binds loopback only, and exposes a single private port
  with an `/agent/manifest` and a streaming role proxy. Managed instances
  require their exact instance/deployment identity and an EmbeddingGemma
  embedding process; generation uses llama.cpp or the selected SlimServe path.
- **The `agent` engine backend** — the container half. Discovers roles from
  the agent manifest, forwards role traffic, and degrades to
  `configuration_error` (alive, diagnosable, no crash loop) when the agent
  is unreachable.
- **`agent-dist/`** — launchd plist template plus install/uninstall scripts.
  The default is the canonical `google/gemma-4-E2B-it-qat-q4_0-gguf`
  generation model (+mmproj, reasoning budget 0). For legacy fixed agents,
  Control can add or remove a checksum-verified GGUF embedding role through
  the constrained agent API. Managed instances reject these legacy mutations
  with HTTP 409; their composite configuration is owned by deployment lifecycle.

Native llama roles report the actual started child's managed file identity, not
the requested repository or the file's basename. `SOVEREIGN_AGENT_MODEL_ROOT`
sets the existing host model root; otherwise it is `models/` beside the agent
configuration file, or `~/.sovereign/models` when no configuration path is given.
An existing, canonically spelled regular file strictly inside that resolved root
projects to `/models/<root-relative-path>`, preserving every directory component.
For example, a child started with
`<root>/staged/<artifact-id>/<checksum>/artifact` reports
`/models/staged/<artifact-id>/<checksum>/artifact`. Merely naming a file
`artifact` elsewhere does not give it that identity. Relative, missing,
out-of-root, directory, traversal, and symlinked file paths fail closed before
starting a native child. If a previously started file ceases to satisfy the
contract, the role is unhealthy and no model identity or private host path is
published.

Runtime refreshes authenticated native role observations before publishing its
manifest, health/readiness and model list, and before forwarding role requests,
including embedding-only configurations. A missing, unhealthy, malformed or
unexpectedly changed started model, revision or context observation withdraws
that role's cached identity and dimensions. It is not silently replaced or
reaccepted by later passive observations; an explicit Runtime startup must
accept and probe the role again. Withdrawing generation proof closes Runtime
generation admission, but does not close the independent host gate or interrupt
already-admitted streams. A child can retain usable weights in memory after a
file disappears: loaded memory is not current managed-file identity proof.

Overall readiness still requires every enabled role. Request forwarding checks
the requested role independently, so withdrawn or paused generation does not
disable a healthy embedding role, and an embedding-only Runtime ignores host
generation admission. A failed transport observation closes generation admission
without asserting that another role's loaded identity was withdrawn. Passive
observations never reopen generation admission. SlimServe retains its separate
validated generation observation and existing monitor; native ancillary roles
use the same exact-identity withdrawal check.

The projected `/models/<root-relative-path>` identity is limited to **512 UTF-8
bytes including `/models/`** (at most 504 bytes for the relative part), not 512
bytes for the private absolute host path. Each component is at most 255 UTF-8
bytes. Ordinary spaces, Unicode names, and filename punctuation are preserved
exactly, including repeated dots inside a filename. Empty, `.` or `..`
components, repeated/trailing `/` separators, invalid UTF-8, and Unicode control
characters (including NUL) are rejected. Paths use POSIX separators; a backslash
inside a filename is not a separator. Nothing is normalized, truncated, hashed,
or replaced with a basename. This deliberately bounded contract supports
space-bearing parents and nested paths longer than the old Control decoder's
257-character ceiling. Existing host filesystem limits can be lower; an
existing managed regular file is still required.
Agent startup, checksum-verified embedding admission, native observation, and
the Runtime adapter enforce the same native bound. Control applies it only to
native absolute file observations on Metal; served aliases, repository IDs,
command validators, and SlimServe generation's separate staged-path grammar
remain unchanged. Longer native paths must be restaged under a supported exact
managed path before configuring a child.

Revision and context metadata stay associated with the child's startup inputs;
later desired configuration changes cannot relabel that child. The Runtime
adapter copies the canonical identity unchanged and rejects older basename-only
native manifests, so agent and Runtime must be upgraded together, with the
coordinated Control decoder for space-bearing, Unicode, or longer native
identities. The common manifest schema already carries `engine_model` as a
nonempty string; this native-only admission/decoder correction introduces no
new schema version or published artifact dependency. Native
embedding observations use this same contract even when generation uses
SlimServe; SlimServe generation retains its independently verified host/staged
mapping. Installed engine discovery is not loaded-engine version evidence:
native llama `engine_version` remains unknown and the Runtime manifest does not
invent an `engine` object. The file projection is not a checksum measurement or
physical serving qualification.

The supported native llama-server contract is the reviewed **b9960** CLI, not
arbitrary historical versions. The agent makes `model_path` authoritative by
placing the primary-file argument after extra role arguments and clearing the
primary URL, Hugging Face, and Docker selectors with their supported empty
string arguments. This also clears inherited selector environment values and
primary-model presets; it does not replace auxiliary projector/draft inputs.
Operators must stage the intended local file and set `model_path` to it rather
than select another primary model through `args` or environment. Binaries that
do not support this CLI are not covered; this source-reviewed argument policy
does not attest the version of a loaded child. See the
[b9960 string setters](https://github.com/ggml-org/llama.cpp/blob/b9960/common/arg.cpp#L2733-L2768)
and [remote-selection handling](https://github.com/ggml-org/llama.cpp/blob/b9960/common/arg.cpp#L459-L576).

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

### Optional SlimServe source builds (M9)

**These recipes have not been built or GPU-qualified as part of this change.**
They do not establish a published SlimServe image, a working model/profile,
performance, multi-GPU correctness, or release eligibility. The normal
`docker/cuda/Dockerfile` path remains available and unchanged.

SlimServe's actual distribution is named **`vllm`**, so installing it into the
stock engine interpreter would replace that engine. The optional recipes build
SlimServe at `44a7d1a21851c7164c098c93fbc1baa12ab99847` inside the fixed, isolated
`/opt/sovereign-slimserve/bin/python` environment, with **another installation of
the first-party `sovereign-runtime` package, without its `engine` extra**. No
system site-packages are inherited, and the CUDA image does not put this
interpreter on its global `PATH`. Its original Runtime entrypoint and stock
vLLM remain the default. Runtime, not the upstream SlimServe installer or CLI,
owns configuration and launch.

#### Required external inputs

- A **complete, platform-specific wheelhouse** containing `requirements.lock`
  with exact versions and SHA256 hashes for every direct and transitive
  dependency. The installer uses `--require-hashes`, `--only-binary=:all:` and
  no package index. The lock must jointly satisfy the pinned SlimServe
  [`pyproject.toml`](https://github.com/QuixiAI/SlimServe/blob/44a7d1a21851c7164c098c93fbc1baa12ab99847/pyproject.toml),
  [`requirements/build/cuda.txt`](https://github.com/QuixiAI/SlimServe/blob/44a7d1a21851c7164c098c93fbc1baa12ab99847/requirements/build/cuda.txt)
  for CUDA builds, and the target's
  [`requirements/cuda.txt`](https://github.com/QuixiAI/SlimServe/blob/44a7d1a21851c7164c098c93fbc1baa12ab99847/requirements/cuda.txt)
  or [`requirements/metal.txt`](https://github.com/QuixiAI/SlimServe/blob/44a7d1a21851c7164c098c93fbc1baa12ab99847/requirements/metal.txt)
  (including `common.txt`), plus this repository's base dependencies and
  `hatchling==1.31.0`. Include a pinned `pip`; exclude `vllm`, `sovereign-runtime`
  and the Runtime `engine` extra because those two packages are built locally.
  FlashInfer packages are **not allowed**: the pinned source declares no
  reviewed FlashInfer dependency/artifact closure. The builder rejects
  importable `flashinfer`, `flashinfer_cubin` or `flashinfer_jit_cache` modules
  and installed `flashinfer` / `flashinfer-*` distributions, both after
  dependency installation and before emitting installed provenance.
  CUDA requires **torch 2.13.0, torchvision 0.28.0, numba 0.65.0**; Metal
  requires **torch 2.13.0 with MPS**. Both need CMake, Ninja, packaging,
  setuptools `>=77.0.3,<81`, setuptools-scm, wheel and Jinja2. Use a pinned
  CMake `>=3.26.1,<3.30` for the upstream legacy FetchContent calls. No working
  wheelhouse, dependency lock, or availability of these wheels is asserted
  here; supplying and qualifying those real artifacts is a prerequisite.
- Python **3.11–3.14**, Git and network access to the pinned source repositories
  in `docker/slimserve/provenance.json`. Git fetches exact commits without
  submodule initialization. Each required nested CUDA header dependency is
  fetched independently at its recorded gitlink commit. Upstream control-plane
  scripts, installers, ROCm dependencies and floating branches are not used.
- Sufficient build disk/RAM and an empty, writable
  `/opt/sovereign-slimserve`. The installer refuses to overwrite an existing
  installation; a failed build must be removed deliberately before retrying.
  Building these recipes requires native compiler/toolchain resources and is
  not the same operation as selecting an engine profile in Control.

#### CUDA source image

`docker/slimserve/Dockerfile` requires **`CUDA_BUILD_IMAGE`**, an actual immutable
`name@sha256:...` reference supplied by the operator. It must be a
toolchain-equipped derivative of this repository's existing CUDA Runtime
image: stock `vllm==0.25.0`, Runtime's two `/usr/local/bin` contract binaries,
Python with pip/venv, Git, a C++20 compiler (**GCC >=11.3** when using GCC), and
a complete CUDA toolkit with `nvcc` compatible with the supplied torch 2.13.0
wheel and target driver. No speculative CUDA/SlimServe image reference or
unpinned apt installation is embedded in this recipe.

The pinned SlimServe source refers to `tools/build_deepgemm_C.py`, but that
file is absent from its committed tree. Consequently this recipe explicitly
accepts **SM80, SM86 and SM89 only** (`8.0`, `8.6`, `8.9`, separated by spaces
or semicolons). Its SM90+ DeepGEMM build path is blocked on that missing
upstream prerequisite; Hopper/Blackwell builds are not silently advertised.
The restriction is specific to this optional source recipe, not the existing
stock vLLM image. An explicit target architecture is required even on a build
host without a GPU.

After supplying the actual base image and locked wheelhouse:

```bash
: "${CUDA_BUILD_IMAGE:?Set an immutable toolchain-equipped Runtime CUDA image}"
: "${SLIMSERVE_WHEELHOUSE:?Set the absolute directory containing requirements.lock and wheels}"
docker buildx build --load -f docker/slimserve/Dockerfile \
  --build-arg CUDA_BUILD_IMAGE \
  --build-arg TORCH_CUDA_ARCH_LIST=8.0 \
  --build-arg MAX_JOBS=4 \
  --build-context slimserve-dependencies="${SLIMSERVE_WHEELHOUSE}" \
  -t sovereign-runtime-slimserve:local .
```

The tag above names the **local output**, not a published artifact. The image
also updates the first-party Runtime wheel in the stock interpreter using
`--no-deps`; it never installs SlimServe or replaces stock vLLM there.

#### Native Apple Silicon source install

Docker cannot expose Metal. The native path is
`docker/metal/build-slimserve.sh`, run on Apple Silicon with the Metal-specific
locked wheelhouse and Python 3.11–3.14. It additionally requires full Xcode,
`xcrun metal` supporting **`-std=metal4.0`**, `install_name_tool`, the Metal /
Foundation / QuartzCore SDK frameworks, and the matching MPS-enabled torch
wheel. Command Line Tools or an older Metal SDK alone are insufficient. The
script prepends `/opt/homebrew/bin` to `PATH`; set `SLIMSERVE_PYTHON` to the
absolute native interpreter path when necessary. Arrange permission to create
the fixed `/opt` installation before invoking it:

```bash
bash docker/metal/build-slimserve.sh "${SLIMSERVE_METAL_WHEELHOUSE}"
```

This compiles the actual `vllm._quixicore_C` ObjC++ extension and adjacent
`vllm/quixicore_metal.metallib` using SlimServe's pinned CMake targets. It does
not use QuixiCore's standalone JIT package, modify `install-agent.sh`, start a
daemon, or represent a container-to-Metal bridge. Host-agent deployment and
actual native serving still require their own configuration and qualification.

#### Provenance and qualification boundary

`docker/slimserve/provenance.json` is the checked-in **source lock**. After a
successful install, the builder creates a different file at
`/opt/sovereign-slimserve/provenance.json` containing actual installed package
versions, source-file SHA256s, compiled-extension SHA256s (and the Metal shader
hash), dependency-lock hash and retained license-file hashes. Python and kernel
artifact paths are relative to the isolated interpreter's `site-packages`.
No build digest is filled in before the corresponding bytes exist.

QuixiCore-CUDA's base is `08780aaa22cdc2d144b6beacb24df953134b34be`; Metal's is
`71a08cd4cbcdc622ce31b3fc91e1f505e144b516`. SlimServe has changed both vendored
trees. The build retains the exact pinned fork additions and modifications,
and records the combined kernel version as
`<base-commit>+slimserve.44a7d1a21851c7164c098c93fbc1baa12ab99847`, not as an
unmodified upstream kernel release. CUDA CMake compiles the supplied
`csrc/quixicore/{tm_cuda,quant,serving}` inputs; there is no invented
`QUIXICORE_SOURCE_DIR` API or floating QuixiCore fetch. External CUDA CMake
dependencies use pinned source overrides, including the normal copy/import
rewrite path for FlashAttention's Python modules rather than development-only
symlinks to a discarded build tree.

Offline **dependency installation** is not proof of offline **serving**.
The pinned FlashInfer compatibility helpers can otherwise probe NVIDIA's
artifact service independently of Hugging Face's offline flags. Supported
SM80/SM86/Metal profiles use native QuixiCore, FlashAttention or Marlin paths;
their Runtime-owned launch configuration disables TRTLLM attention and rejects
optional FlashInfer importability before loading the engine. This is a narrow
source-audited exclusion, not a general network sandbox or a prepopulated
FlashInfer cache. A profile that requires FlashInfer remains unsupported until
its exact dependency and compiled/JIT artifacts are pinned and reviewed.
Cold-start serving with external network access genuinely denied, including
multimodal and speculative paths, remains a required **unrun** qualification
gate; model closure and `HF_HUB_OFFLINE` alone do not satisfy it.

The generated metadata is local build evidence, not a signed attestation or a
claim that a kernel executed. Runtime must verify installed paths/hashes and
observe the child actually invoking the native kernel before publishing
kernel facts. All existing contract, lifecycle and real-hardware release gates
remain required. Fixed source commits and a hashed wheelhouse make inputs
traceable; bit-for-bit reproducibility has not been demonstrated. See
`THIRD_PARTY_NOTICES.md` for exact source licenses and notice retention.

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

Sovereign Runtime is licensed under Apache-2.0. Upstream vLLM remains
Apache-2.0, and all bundled components retain their own licenses. See
`LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
