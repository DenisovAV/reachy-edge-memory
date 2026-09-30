"""Tests for demo/embed_service.py's main(): the warm-before-bind ordering
fix. No real model load, no real socket
— ThreadingHTTPServer and Embedders are both replaced with recording fakes.
"""
from __future__ import annotations


class _FakeServer:
    """Stands in for ThreadingHTTPServer: main() must not actually bind a
    socket or block on serve_forever() in a test."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def serve_forever(self) -> None:
        pass


def test_main_warms_models_before_binding_the_socket(monkeypatch):
    """Regression test: ThreadingHTTPServer's constructor itself calls
    socket.bind()+listen(), so building it BEFORE embedders.warm() leaves the
    port accepting connections while the models are still loading — a robot
    connecting to embed_service in that window gets a request queued behind
    a server that never accepts it, instead of a clean, fast connection
    refusal. warm() must complete before the socket exists at all.
    """
    import demo.embed_service as embed_service_mod

    order = []

    class RecordingEmbedders:
        def warm(self):
            order.append("warm")

    def fake_server(*_args, **_kwargs):
        order.append("bind")
        return _FakeServer()

    monkeypatch.setattr(embed_service_mod, "Embedders", RecordingEmbedders)
    monkeypatch.setattr(embed_service_mod, "ThreadingHTTPServer", fake_server)

    embed_service_mod.main([])

    assert order == ["warm", "bind"]


def test_main_no_warmup_skips_warm_but_still_binds(monkeypatch):
    import demo.embed_service as embed_service_mod

    order = []

    class RecordingEmbedders:
        def warm(self):
            order.append("warm")

    def fake_server(*_args, **_kwargs):
        order.append("bind")
        return _FakeServer()

    monkeypatch.setattr(embed_service_mod, "Embedders", RecordingEmbedders)
    monkeypatch.setattr(embed_service_mod, "ThreadingHTTPServer", fake_server)

    embed_service_mod.main(["--no-warmup"])

    assert order == ["bind"]
