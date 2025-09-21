import os
from typing import Any

import pytest

from core.database.arango import memory_client as memory_client_module
from core.database.arango.memory_client import ArangoMemoryClient, resolve_memory_config
from core.database.database_factory import DatabaseFactory


class DummyHttp2Client:
    """Minimal stand-in for ArangoHttp2Client used in tests."""

    instances: list["DummyHttp2Client"] = []

    def __init__(self, config: Any):
        self.config = config
        self.closed = False
        DummyHttp2Client.instances.append(self)

    def close(self) -> None:
        self.closed = True

    # The memory client may call these helpers during tests; keep minimal stubs.
    def query(self, *_, **__):
        return []

    def request(self, *_, **__):
        return {}

    def insert_documents(self, *_args, **_kwargs):
        return {"created": 0}


@pytest.fixture(autouse=True)
def clear_dummy_instances():
    DummyHttp2Client.instances.clear()
    yield
    DummyHttp2Client.instances.clear()


def test_resolve_memory_config_fills_missing_write_socket(monkeypatch):
    monkeypatch.setenv("ARANGO_PASSWORD", "secret")
    monkeypatch.setenv("ARANGO_RO_SOCKET", "/tmp/ro.sock")
    monkeypatch.delenv("ARANGO_RW_SOCKET", raising=False)

    config = resolve_memory_config(use_proxies=True)

    assert config.read_socket == "/tmp/ro.sock"
    # When write socket is missing we expect a safe default, not None.
    assert config.write_socket is not None
    assert config.write_socket.endswith("/arangod.sock")


def test_get_arango_returns_memory_client(monkeypatch):
    monkeypatch.setenv("ARANGO_PASSWORD", "secret")

    monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

    client = DatabaseFactory.get_arango(
        database="unit_tests",
        username="root",
        use_unix=True,
        host="localhost",
        port=8529,
    )

    try:
        assert isinstance(client, ArangoMemoryClient)
        assert len(DummyHttp2Client.instances) in {1, 2}
        # Both sockets should be resolved even in minimal configuration.
        cfg = DummyHttp2Client.instances[0].config
        assert cfg.socket_path is not None
    finally:
        client.close()

    # All dummy clients should be closed after closing the wrapper.
    assert all(instance.closed for instance in DummyHttp2Client.instances)
