"""The server the agent starts for an MLX deployment: mlx-lm's own, which
answers only the agent's key, only for the model it was started with, and
says it is healthy only once that model is loaded.

mlx-lm's server takes no key, loads whatever model a request names, and
answers its health route while it is still loading the model, so all three
are closed here before it starts: a request without the key the agent gave
this child is refused; every request is served by the model on the command
line; and the health route waits for the model, while a model that cannot
load ends the process, so the agent sees the failure at once. A model's
thinking is answered as reasoning_content, as vLLM and llama.cpp answer it,
where mlx-lm names it reasoning, a field the gateway does not carry. The
weights are on disk already; nothing is downloaded."""

from __future__ import annotations

import hmac
import logging
import os


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm import server

    key = os.environ.pop("LLAMA_API_KEY", "")
    os.environ.pop("VLLM_API_KEY", None)
    if not key:
        raise SystemExit("an MLX server answers only the agent, which gives it a key")
    # Headers arrive as Latin-1 text; compared as bytes, any header is a
    # clean refusal rather than an error.
    expected = f"Bearer {key}".encode("latin-1")

    def guarded(handle):
        def answer(self):
            given = self.headers.get("Authorization", "").encode("latin-1", "replace")
            if not hmac.compare_digest(given, expected):
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
    available = server.ResponseGenerator.is_healthy.fget
    server.ResponseGenerator.is_healthy = property(lambda self: available(self) and self.model_provider.model is not None)
    generate = server.ResponseGenerator._generate

    def generate_or_end(self):
        try:
            generate(self)
        except BaseException:
            logging.exception("the MLX server stopped: its model could not be loaded or run")
            os._exit(1)

    server.ResponseGenerator._generate = generate_or_end
    respond = server.APIHandler.generate_response

    def with_reasoning_content(self, *arguments, **options):
        response = respond(self, *arguments, **options)
        for choice in response.get("choices", []):
            for part in (choice.get("message"), choice.get("delta")):
                if isinstance(part, dict) and "reasoning" in part:
                    part["reasoning_content"] = part.pop("reasoning")
        return response

    server.APIHandler.generate_response = with_reasoning_content
    server.main()


if __name__ == "__main__":
    main()
