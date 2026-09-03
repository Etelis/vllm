# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.v1.kv_offload.tiering.p2p.control.zmq."""

from __future__ import annotations

import socket
import time

import msgspec
import pytest
import zmq

from vllm.v1.kv_offload.tiering.p2p.control.zmq import (
    ZmqConnection,
    ZmqTransport,
    _Sockets,
)


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_transport(host: str = "127.0.0.1", attempts: int = 8):
    """Construct a ZmqTransport on a fresh port, retrying on bind collisions.

    Why: _free_port() releases the probe socket before ZmqTransport binds the
    same port — a parallel test run can steal it in between. Retrying on
    ZMQError/OSError closes that race without a production change.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        port = _free_port()
        try:
            return ZmqTransport(f"{host}:{port}", host, port), port
        except (zmq.ZMQError, OSError) as e:
            last_err = e
    assert last_err is not None
    raise last_err


def _wait_for_inbound(transport: ZmqTransport, deadline: float = 2.0):
    """Poll until at least one new inbound connection is accepted, or fail."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        new = transport.poll()
        if new:
            return new
        time.sleep(0.005)
    raise AssertionError(f"no inbound connection within {deadline}s")


def _wait_for_messages(
    transport: ZmqTransport,
    conn: ZmqConnection,
    n: int,
    deadline: float = 2.0,
) -> list[dict]:
    """Poll until `conn` has received at least `n` messages, then return them."""
    end = time.monotonic() + deadline
    msgs: list[dict] = []
    while time.monotonic() < end:
        transport.poll()
        msgs.extend(conn.recv())
        if len(msgs) >= n:
            return msgs
        time.sleep(0.005)
    raise AssertionError(f"got {len(msgs)}/{n} messages within {deadline}s")


def _make_mock_connection(peer_id: str = "test:1234") -> ZmqConnection:
    """Create a ZmqConnection with mock sockets for unit testing."""
    from unittest.mock import MagicMock

    sockets = _Sockets(
        dealer=MagicMock(), monitor=MagicMock(), monitor_addr="inproc://test"
    )
    return ZmqConnection(peer_id, sockets)


class TestZmqConnection:
    """Tests for ZmqConnection in isolation (no real sockets)."""

    def test_enqueue_and_recv(self):
        """Messages enqueued are returned by recv() in order."""
        conn = _make_mock_connection()

        conn.enqueue({"type": "a"})
        conn.enqueue({"type": "b"})

        msgs = conn.recv()
        assert list(msgs) == [{"type": "a"}, {"type": "b"}]
        # Second recv is empty
        assert not conn.recv()

    def test_recv_returns_empty_initially(self):
        conn = _make_mock_connection()
        assert not conn.recv()

    def test_alive_initially_true(self):
        conn = _make_mock_connection()
        assert conn.alive is True

    def test_mark_dead(self):
        conn = _make_mock_connection()
        conn.mark_dead()
        assert conn.alive is False

    def test_send_raises_when_closed(self):
        conn = _make_mock_connection()
        conn.mark_dead()

        with pytest.raises(RuntimeError, match="closed connection"):
            conn.send({"type": "test"})


class TestZmqTransportConnectivity:
    """Integration tests for ZmqTransport with real ZMQ sockets."""

    def test_connect_and_send_message(self):
        """Two transports can connect and exchange messages."""
        transport_a, port_a = _make_transport()
        transport_b, port_b = _make_transport()

        try:
            peer_a_id = f"127.0.0.1:{port_a}"
            conn_b_to_a = transport_b.connect(peer_a_id)
            conn_b_to_a.send({"type": "hello", "data": 42})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            conn_a_from_b = new_conns[0]
            assert conn_a_from_b.peer_id == f"127.0.0.1:{port_b}"

            msgs = _wait_for_messages(transport_a, conn_a_from_b, 1)
            assert msgs == [{"type": "hello", "data": 42}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_bidirectional_messaging(self):
        """Both sides can send and receive after connection."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "from": "b"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            conn_a.send({"type": "reply", "from": "a"})

            msgs = _wait_for_messages(transport_b, conn_b, 1)
            assert msgs == [{"type": "reply", "from": "a"}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_poll_returns_empty_when_no_connections(self):
        transport, _ = _make_transport()
        try:
            assert not transport.poll()
        finally:
            transport.close()

    def test_multiple_messages(self):
        """Multiple messages are buffered and returned together."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"seq": 1})
            conn_b.send({"seq": 2})
            conn_b.send({"seq": 3})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            msgs = _wait_for_messages(transport_a, conn_a, 3)
            assert [m["seq"] for m in msgs] == [1, 2, 3]
        finally:
            transport_a.close()
            transport_b.close()

    def test_duplicate_connect_asserts(self):
        """Connecting to the same peer twice raises AssertionError."""
        # port_a is never bound — we just need a syntactically-valid peer id.
        port_a = _free_port()
        transport_b, _ = _make_transport()
        try:
            transport_b.connect(f"127.0.0.1:{port_a}")
            with pytest.raises(AssertionError, match="already exists"):
                transport_b.connect(f"127.0.0.1:{port_a}")
        finally:
            transport_b.close()

    def test_dead_connection_removed_on_poll(self):
        """Dead connections are cleaned up during poll."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "hello"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            # Mark the inbound connection dead manually.
            new_conns[0].mark_dead()

            # Pruning is synchronous within poll().
            transport_a.poll()
            assert len(transport_a._connections) == 0
        finally:
            transport_a.close()
            transport_b.close()

    def test_close_is_idempotent(self):
        """Calling close() twice doesn't raise."""
        transport, _ = _make_transport()
        transport.close()
        transport.close()  # should not raise


class TestReconnectSurvival:
    """A peer whose connection died must be able to re-open it.

    Peers reconnect after any 10s silence on the control socket, and their
    DEALER identity is their own fixed host:port — so a reconnect always
    collides with the pipe it replaces, and its first message always races
    the sweep of the connection it replaces. Both had to be survivable
    before a transient network stall stopped being a permanent outage.
    """

    def test_sweep_releases_a_dead_connection_sockets(self):
        """A reaped peer's DEALER must be released, not merely flagged.

        A DEALER left open keeps holding its ZMTP identity against the peer's
        ROUTER, so the peer's next connect is dropped rather than replacing
        it — besides leaking two file descriptors on every reap.
        """
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn = transport_b.connect(f"127.0.0.1:{port_a}")
            dealer, monitor = conn._sockets.dealer, conn._sockets.monitor
            assert not dealer.closed

            conn.mark_dead()
            transport_b.poll()

            assert dealer.closed
            assert monitor.closed
        finally:
            transport_a.close()
            transport_b.close()

    def test_mark_dead_alone_leaves_the_sockets_to_the_transport(self):
        """mark_dead() stops use; the transport still owes the release."""
        conn = _make_mock_connection()

        conn.mark_dead()

        assert conn.alive is False
        conn._sockets.dealer.close.assert_not_called()

    def test_close_after_mark_dead_still_releases(self):
        conn = _make_mock_connection()

        conn.mark_dead()
        conn.close()

        conn._sockets.dealer.close.assert_called_once()
        conn._sockets.monitor.close.assert_called_once()

    def test_a_reused_identity_takes_over_the_router_pipe(self):
        """The replacement pipe must win while the old one is still open.

        Without ROUTER_HANDOVER libzmq silently drops the newcomer, and its
        TCP and ZMTP handshakes both succeed anyway — so the peer logs a
        successful connect and is then mute for the life of the process.
        """
        transport_a, port_a = _make_transport()
        ctx = zmq.Context()
        first = ctx.socket(zmq.DEALER)
        second = ctx.socket(zmq.DEALER)

        try:
            for sock in (first, second):
                sock.identity = b"127.0.0.1:9999"
            first.connect(f"tcp://127.0.0.1:{port_a}")
            first.send(msgspec.msgpack.encode({"seq": 1}))

            conns = _wait_for_inbound(transport_a)
            assert _wait_for_messages(transport_a, conns[0], 1) == [{"seq": 1}]

            # The first pipe is never closed — that is the reconnect race.
            second.connect(f"tcp://127.0.0.1:{port_a}")
            second.send(msgspec.msgpack.encode({"seq": 2}))

            assert _wait_for_messages(transport_a, conns[0], 1) == [{"seq": 2}]
        finally:
            first.close(linger=0)
            second.close(linger=0)
            ctx.destroy(linger=0)
            transport_a.close()

    def test_a_reconnects_message_is_not_swept_with_the_old_connection(self):
        """Queued before the sweep, delivered after it.

        A session announces itself with exactly one ConnectMsg and nothing
        retransmits it. Reading the ROUTER before sweeping would enqueue that
        message onto the connection being dropped and discard it, leaving the
        peer unable to ever complete its handshake.
        """
        transport_a, port_a = _make_transport()
        transport_b, port_b = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect"})
            first = _wait_for_inbound(transport_a)
            assert _wait_for_messages(transport_a, first[0], 1) == [{"type": "connect"}]

            # The peer re-announces itself while we still hold the connection
            # it is replacing; its message reaches our ROUTER queue first.
            conn_b.send({"type": "connect", "epoch": "2"})
            assert transport_a._router.poll(2000) & zmq.POLLIN
            first[0].mark_dead()

            second = _wait_for_inbound(transport_a)
            assert len(second) == 1
            assert second[0] is not first[0]
            assert second[0].peer_id == f"127.0.0.1:{port_b}"
            assert _wait_for_messages(transport_a, second[0], 1) == [
                {"type": "connect", "epoch": "2"}
            ]
        finally:
            transport_a.close()
            transport_b.close()

    def test_reopening_a_peer_gets_a_fresh_monitor_endpoint(self):
        """Each connection needs its own inproc monitor endpoint.

        A peer's address is stable across reconnects, so an endpoint named
        after it alone is still bound by the connection being replaced —
        libzmq releases an inproc binding asynchronously after close — and
        re-opening that peer then raises EADDRINUSE instead of reconnecting.
        Asserting the endpoints differ pins the invariant rather than the
        race, which only sometimes lands on the collision.
        """
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()
        peer = f"127.0.0.1:{port_a}"

        try:
            addrs = []
            for _ in range(3):
                conn = transport_b.connect(peer)
                addrs.append(conn.monitor_addr)
                conn.mark_dead()
                transport_b.poll()

            assert len(set(addrs)) == len(addrs), addrs
            assert peer not in transport_b._connections
        finally:
            transport_a.close()
            transport_b.close()
