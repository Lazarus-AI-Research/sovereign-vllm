# Host inference agent

`sovereign-runtime-agent` (`lazarus.agent`) runs on a Mac beside the appliance
because containers cannot reach Metal. It supervises one server process per
deployment (`llama-server` for language and embedding models, stable-diffusion.cpp's
`sd-server` for image models), proxies inference to each, and exposes a small
bearer-authenticated admin API on `127.0.0.1:9100` that Sovereign Control drives.
It fails closed: no token, no service.

A deployment is what Control creates while the appliance runs: the shipped
assistant, a second generation model, an embedding profile's model, an image
model. Each has its own id, port, process and admission gate, is persisted in
`agent.yaml` so a restarted agent serves it again, and is reached through
`/deployments/{id}/v1/*`. Starting or failing one never touches another. A fresh
agent serves nothing until Control asks.

The fixed `generation` and `embedding` roles of earlier agents, the
`X-Sovereign-Role` proxy under `/v1/*` and the SlimServe generation path behind
them are retired. An `agent.yaml` that still carries `roles` (or the
managed-instance identity that went with them) loads with those keys ignored
and dropped at the next save.

## Deployments API

| Method and path | Purpose |
| --- | --- |
| `PUT /agent/admin/deployments/{id}` | Create or replace a deployment. Waits until it serves; a replacement that cannot serve puts the previous process back. |
| `DELETE /agent/admin/deployments/{id}` | Stop and forget a deployment. Absent is not an error. |
| `GET /agent/deployments` | Every deployment with `status`, `kind`, `model`, `port`, `served_model_name`, `context_length`, `revision`, `engine`. |
| `GET /agent/manifest` | `agent_version`, `backend`, the installed engines, and the same `deployments`. |
| `POST /deployments/{id}/v1/{chat/completions,completions,embeddings,models}` | Inference. The paths a deployment answers follow its kind. |

`{id}` is a short lowercase slug (`^[a-z0-9][a-z0-9-]{0,63}$`). The request body
is constrained; arbitrary `llama.cpp` flags never cross this boundary:

```json
{
  "kind": "generation",
  "artifact": "metal/model.gguf",
  "sha256": "<64 hex>",
  "mmproj": "metal/model-mmproj.gguf",
  "mmproj_sha256": "<64 hex>",
  "revision": "<40 or 64 hex>",
  "served_model_name": "assistant-second",
  "context_length": 8192
}
```

An embedding deployment takes `pooling` (`mean`, `last`, `cls`) and
`normalization` (`l2`, `none`) instead of a projector. An image deployment
(`"kind": "image"`) names its diffusion weights as the artifact and the files
served beside them under `components` (`clip_l`, `t5xxl`, `vae`, each
`{"artifact", "sha256"}`), with the sampling it is pinned to: `steps`,
`cfg_scale`, `sampler` (`euler`, `euler_a`, `heun`, `dpm2`, `dpm++2m`, `lcm`).
It answers `images/generations` and `models`; its server carries no API key of
its own and listens on loopback behind the agent's proxy. Artifacts are relative
to the managed model root, must already exist, must be GGUF or safetensors, and
are checksummed before anything starts: the agent never downloads.

Ports are allocated from `9110`–`9199`; the agent itself listens on `9100`. A
generation deployment's process is started with an ephemeral API key the agent
alone holds, so nothing on the host reaches it except through the agent.

Replacing or removing a deployment closes its admission gate, waits for
requests in flight (bounded), then stops the process. A request that arrives
while the gate is closed is answered `503`.
