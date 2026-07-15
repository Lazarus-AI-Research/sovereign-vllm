import json

from lazarus.appliance import healthcheck


class Response:
    def __init__(self, body: dict, status: int = 200):
        self.body = json.dumps(body).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_live_mode_is_default(monkeypatch):
    seen = []
    monkeypatch.setattr("sys.argv", ["healthcheck"])
    monkeypatch.setattr(
        healthcheck.urllib.request,
        "urlopen",
        lambda url, timeout: seen.append((url, timeout)) or Response({"status": "alive"}),
    )

    assert healthcheck.main() == 0
    assert seen == [("http://127.0.0.1:8000/health/live", 5)]


def test_ready_mode_requires_true_ready(monkeypatch):
    responses = iter([Response({"ready": False}), Response({"ready": True})])
    seen = []
    monkeypatch.setattr("sys.argv", ["healthcheck", "--ready", "--port", "9000"])
    monkeypatch.setattr(
        healthcheck.urllib.request,
        "urlopen",
        lambda url, timeout: seen.append((url, timeout)) or next(responses),
    )

    assert healthcheck.main() == 1
    assert healthcheck.main() == 0
    assert seen == [
        ("http://127.0.0.1:9000/health/ready", 5),
        ("http://127.0.0.1:9000/health/ready", 5),
    ]


def test_probe_failure_is_unhealthy(monkeypatch):
    monkeypatch.setattr("sys.argv", ["healthcheck", "--live"])

    def fail(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", fail)
    assert healthcheck.main() == 1
