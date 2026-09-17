# Host inference agent

`sovereign-runtime-agent` (`lazarus.agent`) runs on a Mac beside the appliance
because containers cannot reach Metal. It supervises one `llama-server` process
per served model, proxies inference to each, and exposes a small
bearer-authenticated admin API on `127.0.0.1:9100` that Sovereign Control drives.
It fails closed: no token, no service.

Two kinds of process:

- **Roles** — the installer's fixed pair, `generation` (the shipped assistant)
  and an optional `embedding`, configured in `agent.yaml` and reached through
  `/v1/*` with the `X-Sovereign-Role` header. The Metal runtime container fronts
  them under the one-port runtime contract.
- **Deployments** — what an operator adds and removes while the appliance runs.
  Each has its own id, port, process and admission gate, is persisted in
  `agent.yaml` so a restarted agent serves it again, and is reached directly
  through `/deployments/{id}/v1/*`. Starting or failing one never touches
  another.

## Deployments API

| Method and path | Purpose |
| --- | --- |
| `PUT /agent/admin/deployments/{id}` | Create or replace a deployment. Waits until it serves; a replacement that cannot serve puts the previous process back. |
| `DELETE /agent/admin/deployments/{id}` | Stop and forget a deployment. Absent is not an error. |
| `GET /agent/deployments` | Every deployment with `status`, `kind`, `model`, `port`, `served_model_name`, `context_length`, `revision`, `engine`. |
| `GET /agent/manifest` | Roles as before, plus `deployments`. |
| `POST /deployments/{id}/v1/{chat/completions,completions,embeddings,models}` | Inference. The paths a deployment answers follow its kind. |

`{id}` is a short lowercase slug (`^[a-z0-9][a-z0-9-]{0,63}$`) and never a role
name. The request body is constrained; arbitrary `llama.cpp` flags never cross
this boundary:

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
`normalization` (`l2`, `none`) instead of a projector. Artifacts are relative to
the managed model root, must already exist, must be GGUF, and are checksummed
before anything starts: the agent never downloads.

Ports are allocated from `9110`–`9199`; roles keep `9101` and `9102`. A
generation deployment's process is started with an ephemeral API key the agent
alone holds, so nothing on the host reaches it except through the agent.

Replacing or removing a deployment closes its admission gate, waits for
requests in flight (bounded), then stops the process. A request that arrives
while the gate is closed is answered `503`.
