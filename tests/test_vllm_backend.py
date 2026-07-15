import asyncio
import time

from lazarus.appliance.backends.vllm_engine import VllmBackend


def test_engine_construction_does_not_block_api_loop(monkeypatch):
    backend = VllmBackend()

    def slow_constructor(_name, _role):
        time.sleep(0.08)
        return "built"

    monkeypatch.setattr(backend, "_construct_role_engine_sync", slow_constructor)

    async def exercise():
        task = asyncio.create_task(backend._construct_role_engine("generation", None))
        await asyncio.sleep(0.01)
        assert not task.done(), "synchronous model loading blocked the API event loop"
        return await task

    assert asyncio.run(exercise()) == "built"
