"""run-sovereign-runtime: the appliance entrypoint (design.md §24).

Startup sequence: read config → start control API → initializing → load roles
serially → smoke test → terminal state → write manifest. A configuration
error leaves the control API serving with /runtime/errors populated — never
a crash loop (§3.2). Process exit on role failure is opt-in via the
startup.fail_process_on_*_error flags (§12).
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from lazarus.appliance import api
from lazarus.appliance.backends import select_backend
from lazarus.appliance.backends.base import BackendStartError, EngineBackend
from lazarus.appliance.config import ConfigError, RuntimeConfig, load_config
from lazarus.appliance.manifest import ManifestBuilder
from lazarus.appliance.state import StateMachine

logger = logging.getLogger("sovereign.launcher")

ROLE_LOAD_ERROR_MESSAGES = {
    "MODEL_LOAD_FAILED": "the engine failed to load the configured model",
    "OUT_OF_MEMORY": "the model did not fit in available memory",
}


class Appliance:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        backend: EngineBackend | None = None,
    ) -> None:
        self.state = StateMachine()
        api.wire_state_metric(self.state)
        self.config_path = config_path or os.environ.get(
            "SOVEREIGN_RUNTIME_CONFIG", "/runtime-config/runtime.yaml"
        )
        self.config: RuntimeConfig | None = None
        self.config_error: str | None = None
        try:
            self.config = load_config(self.config_path)
        except ConfigError as exc:
            self.config_error = str(exc)

        self.backend = backend if backend is not None else select_backend()
        self.port = self.config.runtime.port if self.config else int(
            os.environ.get("SOVEREIGN_RUNTIME_PORT", "8000")
        )
        self.manifest = ManifestBuilder(
            state=self.state, backend=self.backend, config=self.config, port=self.port
        )
        self.app = api.build_app(
            state=self.state,
            backend=self.backend,
            config=self.config,
            manifest=self.manifest,
            lifecycle=self.run_lifecycle,
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    async def run_lifecycle(self) -> None:
        try:
            await self._lifecycle_inner()
        except Exception as exc:  # never let the lifecycle kill the API silently
            logger.exception("lifecycle failed unexpectedly")
            self.state.record_error("ENGINE_DEAD", f"lifecycle crashed: {exc}", recoverable=False)
            self.state.transition("runtime_error")
            self.manifest.write()

    async def _lifecycle_inner(self) -> None:
        if self.config is None:
            self.state.record_error("CONFIG_INVALID", self.config_error or "unknown", recoverable=True)
            self.state.transition("configuration_error")
            self.manifest.write()
            return

        try:
            await self.backend.start(self.config, on_state=self.state.transition)
        except BackendStartError as exc:
            self.state.record_error(exc.code, str(exc), recoverable=exc.recoverable, role=exc.role)
            self.state.transition("configuration_error" if exc.recoverable else "runtime_error")
            self.manifest.write()
            return

        enabled = self.config.enabled_roles()
        for name in enabled:
            info = self.backend.role_info(name)
            if info.status == "unhealthy":
                code = info.error_code or "MODEL_LOAD_FAILED"
                message = ROLE_LOAD_ERROR_MESSAGES.get(code, "role failed to load")
                self.state.record_error(code, message, recoverable=True, role=name)

        healthy = [n for n in enabled if self.backend.role_info(n).status == "healthy"]

        if healthy and self.config.startup.smoke_test_on_start:
            self.state.transition("smoke_testing")
            await self._smoke_test(healthy)
            healthy = [n for n in enabled if self.backend.role_info(n).status == "healthy"]

        if len(healthy) == len(enabled) and healthy:
            self.state.transition("healthy")
        elif healthy:
            self.state.transition("degraded")
        else:
            self.state.transition("configuration_error")
        self.manifest.write()
        self._maybe_fail_process(enabled)

    async def _smoke_test(self, healthy_roles: list[str]) -> None:
        """Startup self-test (design.md §20): exercise each healthy role once."""
        for name in healthy_roles:
            role = self.config.role(name)
            try:
                if name == "generation":
                    await self.backend.chat_completion(
                        {
                            "model": role.served_model_name,
                            "messages": [{"role": "user", "content": "Say OK."}],
                            "max_tokens": 8,
                        }
                    )
                elif name == "embedding":
                    result = await self.backend.embeddings(
                        {"model": role.served_model_name, "input": "smoke test"}
                    )
                    vector = result["data"][0]["embedding"]
                    info = self.backend.role_info("embedding")
                    if info.dimensions and len(vector) != info.dimensions:
                        raise RuntimeError(
                            f"probe dimension {len(vector)} != discovered {info.dimensions}"
                        )
            except Exception as exc:
                logger.exception("startup smoke test failed for role %s", name)
                self.backend.role_info(name).status = "unhealthy"
                self.backend.role_info(name).error_code = "SMOKE_TEST_FAILED"
                self.state.record_error(
                    "SMOKE_TEST_FAILED", f"{name}: {exc}", recoverable=True, role=name
                )

    def _maybe_fail_process(self, enabled: dict) -> None:
        startup = self.config.startup
        for name, flag in (
            ("generation", startup.fail_process_on_generation_error),
            ("embedding", startup.fail_process_on_embedding_error),
        ):
            if name in enabled and flag and self.backend.role_info(name).status != "healthy":
                logger.error("role %s unhealthy and fail_process_on_%s_error=true — exiting", name, name)
                os.kill(os.getpid(), signal.SIGTERM)
                return


def main() -> int:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    appliance = Appliance()
    if appliance.config_error:
        logger.error("configuration error (staying alive for diagnosis): %s", appliance.config_error)
    host = appliance.config.runtime.listen_address if appliance.config else "0.0.0.0"
    uvicorn.run(appliance.app, host=host, port=appliance.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
