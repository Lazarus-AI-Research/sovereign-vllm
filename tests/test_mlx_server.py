"""The MLX server the agent starts answers only the agent's key, and only
with the model it was started with, without reaching the network."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lazarus.agent import mlx_server


@pytest.fixture()
def fake_mlx(monkeypatch):
    """mlx-lm's server module, standing in for the real one: its handler, its
    model loader, and its entry point."""
    calls = SimpleNamespace(answered=[], loaded=[], started=False)

    class APIHandler:
        def __init__(self, authorization):
            self.headers = {"Authorization": authorization} if authorization else {}
            self.status = None

        def send_response(self, status):
            self.status = status

        def send_header(self, *args):
            pass

        def end_headers(self):
            pass

        def do_GET(self):
            calls.answered.append("GET")

        def do_POST(self):
            calls.answered.append("POST")

        # mlx-lm 0.32.0's own order; a whole answer's token counts come
        # positionally, and whether it streams is the handler's.
        def generate_response(self, text, finish_reason, prompt_token_count=None, completion_token_count=None,
                              prompt_cache_count=None, token_logprobs=None, top_tokens=None, tokens=None,
                              tool_calls=None, reasoning_text=None):
            key = "delta" if self.stream else "message"
            part = {"role": "assistant", "content": text}
            if reasoning_text:
                part["reasoning"] = reasoning_text
            return {"choices": [{"index": 0, key: part, "finish_reason": finish_reason}]}

    class ModelProvider:
        def __init__(self):
            self.model = None

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            calls.loaded.append((model_path, adapter_path, draft_model_path))

    class ResponseGenerator:
        def __init__(self, provider, fails=False):
            self.model_provider, self.fails = provider, fails

        @property
        def is_healthy(self):
            return True

        def _generate(self):
            if self.fails:
                raise RuntimeError("unsupported model type")

    def main():
        calls.started = True

    server = ModuleType("mlx_lm.server")
    server.APIHandler, server.ModelProvider, server.ResponseGenerator, server.main = APIHandler, ModelProvider, ResponseGenerator, main
    package = ModuleType("mlx_lm")
    package.server = server
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.server", server)
    return SimpleNamespace(server=server, calls=calls)


def test_the_server_answers_only_the_agents_key_and_its_own_model(fake_mlx, monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "child-key")
    monkeypatch.setenv("VLLM_API_KEY", "child-key")
    mlx_server.main()
    import os

    assert fake_mlx.calls.started and os.environ["HF_HUB_OFFLINE"] == "1"
    assert "LLAMA_API_KEY" not in os.environ and "VLLM_API_KEY" not in os.environ
    for authorization, answered in (("Bearer child-key", ["GET", "POST"]), ("Bearer other", []), (None, [])):
        fake_mlx.calls.answered.clear()
        for method in ("do_GET", "do_POST"):
            handler = fake_mlx.server.APIHandler(authorization)
            getattr(handler, method)()
            assert handler.status == (None if answered else 401)
        assert fake_mlx.calls.answered == answered
    fake_mlx.server.ModelProvider().load("someone/else", "adapter", "drafter")
    assert fake_mlx.calls.loaded == [("default_model", None, None)]
    # A header no key can be is a refusal, not an error.
    handler = fake_mlx.server.APIHandler("Bearer caf\u00e9")
    handler.do_GET()
    assert handler.status == 401


def test_the_server_is_healthy_only_once_its_model_is_loaded_and_ends_when_it_cannot_load(fake_mlx, monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "child-key")
    mlx_server.main()
    provider = fake_mlx.server.ModelProvider()
    generator = fake_mlx.server.ResponseGenerator(provider)
    assert not generator.is_healthy
    provider.model = object()
    assert generator.is_healthy
    ended = []
    monkeypatch.setattr(mlx_server.os, "_exit", lambda code: ended.append(code))
    fake_mlx.server.ResponseGenerator(provider, fails=True)._generate()
    assert ended == [1]


def test_the_server_refuses_to_start_without_a_key(fake_mlx, monkeypatch):
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        mlx_server.main()
    assert not fake_mlx.calls.started


def test_thinking_is_answered_as_reasoning_content(fake_mlx, monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "child-key")
    mlx_server.main()
    handler = fake_mlx.server.APIHandler("Bearer child-key")
    handler.stream = False
    part = handler.generate_response("no", "stop", 12, 3, 0, reasoning_text="91 is 7 times 13")["choices"][0]["message"]
    assert part == {"role": "assistant", "content": "no", "reasoning_content": "91 is 7 times 13"}
    assert "reasoning_content" not in handler.generate_response("hi", "stop", 12, 1, 0)["choices"][0]["message"]
    handler.stream = True
    part = handler.generate_response("", None, reasoning_text="91 is")["choices"][0]["delta"]
    assert part == {"role": "assistant", "content": "", "reasoning_content": "91 is"}
