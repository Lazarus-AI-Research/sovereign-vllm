"""The server the agent starts for an MLX deployment: mlx-lm's own, which
answers only the agent's key and only for the model it was started with.

mlx-lm's server takes no key and loads whatever model a request names, so
both are closed here before it starts: a request without the key the agent
gave this child is refused, and every request is served by the model on the
command line. The weights are on disk already; nothing is downloaded."""

from __future__ import annotations

import hmac
import os


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm import server

    key = os.environ.pop("LLAMA_API_KEY", "")
    os.environ.pop("VLLM_API_KEY", None)
    if not key:
        raise SystemExit("an MLX server answers only the agent, which gives it a key")
    expected = f"Bearer {key}"

    def guarded(handle):
        def answer(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            handle(self)

        return answer

    server.APIHandler.do_GET = guarded(server.APIHandler.do_GET)
    server.APIHandler.do_POST = guarded(server.APIHandler.do_POST)
    load = server.ModelProvider.load

    def started_with(self, model_path, adapter_path=None, draft_model_path=None):
        return load(self, "default_model", None, None)

    server.ModelProvider.load = started_with
    server.main()


if __name__ == "__main__":
    main()
