# sovereign-vllm

The Lazarus Sovereign Runtime: the inference engine layer of
[Sovereign Stack](https://github.com/Lazarus-AI-Research/sovereign-stack).

## Overlay mode

This repository is currently an **overlay** on upstream
[vLLM](https://github.com/vllm-project/vllm), pinned in `constraints.txt`.
It contains only Lazarus-owned code:

| Path | Contents |
| --- | --- |
| `lazarus/appliance/` | Runtime launcher: config, state machine, health/manifest/errors API, healthcheck. |
| `lazarus/multi_role/` | Single-process multi-role serving: routing, admission control, memory policy. |
| `lazarus/models/` | Out-of-tree model plugins (e.g. LCO-Omni embedding pooling). |
| `lazarus/api/` | Contract endpoint implementations. |
| `lazarus/platforms/` | Platform-specific integration (cuda, metal host-agent client, ...). |
| `docker/` | Runtime image build contexts per platform profile. |
| `tests/` | Runtime unit tests. Contract conformance lives in the monorepo's evals harness. |

It converts to a true fork (vendoring the `vllm/` tree, per design.md §6.3)
only when an in-tree patch is forced. Known triggers: scheduler-level
cross-role fairness, engine-internal metrics changes, LCO-Omni audio pooling
beyond the plugin API, and the Metal backend.

The invariant the images must uphold (design.md §9, §24): one container, one
entrypoint, one supervising API process, **one port (8000)**, one
OpenAI-compatible API, multiple model roles, shared process-tree fate.

## Development

The `sovereign-runtime` Python package installs without vLLM (the engine is
an extra) so it can be developed on machines that can't build vLLM:

```bash
pip install -e '.[dev]'            # appliance code + tests, no engine
pip install -e '.[dev,engine]'     # with pinned vLLM (linux)
```

Runtime images are built from `docker/<profile>/Dockerfile` with this repo as
context and must pass the conformance harness in the monorepo
(`sovereign-stack/evals`) before release.
