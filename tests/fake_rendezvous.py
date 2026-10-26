"""A stand-in for the official rendezvous Space *and* the robot's `mediad` relay, on this machine.

What it copies from the real ones (docs/design/remote-access-design.md §3, read off their app.py):
- `Authorization: Bearer <token>` on every route, 401 otherwise;
- `GET /api/robot-status` lists the duck without opening anything;
- `GET /events` is an SSE stream that starts with a `welcome`, and `POST /send` before it is a 400;
- `startSession` is answered in the POST body; a busy robot answers `sessionRejected`;
- a `peer` envelope's `rpc` key is relayed verbatim, and answers come back on the stream;
- 1200 requests a minute per token (then 429), 400 notifications a minute from the robot.

The robot behind it is a unix socket speaking robotd's JSON-RPC: a fake one in the tests, or the real
simulator, which makes the whole remote path runnable with no Hugging Face account and no GStreamer:

    python tests/fake_rendezvous.py                    # relays to ~/.cache/duck-sim/duck.sock on :8790
    DUCK_TRANSPORT=rendezvous REACHY_CENTRAL_URL=http://127.0.0.1:8790 HF_TOKEN=local ./run.sh
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROBOT_PEER = "duck-peer-1"
REQUESTS_PER_MINUTE = 1200
NOTIFICATIONS_PER_MINUTE = 400


class Window:
    def __init__(self, limit: int):
        self.limit, self.times = limit, collections.deque()

    def take(self) -> bool:
        now = time.monotonic()
        while self.times and now - self.times[0] > 60:
            self.times.popleft()
        if len(self.times) >= self.limit:
            return False
        self.times.append(now)
        return True


class FakeRendezvous:
    def __init__(self, robot_socket: str, token: str = "local", name: str = "sim-duck", port: int = 0):
        self.robot_socket, self.token, self.name = robot_socket, token, name
        self.busy = False
        self.posts = 0
        self.rejected_429 = 0
        self.dropped_notifications = 0
        self._consumer: queue.Queue | None = None  # the open event stream's outbox
        self._session: str | None = None
        self._sessions = itertools.count(1)
        self._limit = Window(REQUESTS_PER_MINUTE)
        self._robot_budget = Window(NOTIFICATIONS_PER_MINUTE)
        self._writer = None
        self._subscriptions: list[socket.socket] = []
        self._lock = threading.Lock()
        handler = type("Handler", (_Handler,), {"hub": self})
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.drop_stream()
        self.server.shutdown()
        self.server.server_close()

    def drop_stream(self) -> None:
        """What a wifi drop looks like from the client: the stream ends, the session is gone."""
        with self._lock:
            outbox, self._consumer, self._session = self._consumer, None, None
            subscriptions, self._subscriptions = self._subscriptions, []
        for conn in subscriptions:
            conn.close()
        if outbox is not None:
            outbox.put(None)

    # ── routes ──

    def robot_status(self) -> dict:
        # Busy while someone holds a session with it, as the real service reports it
        return {"robots": [{"peerId": ROBOT_PEER, "busy": self.busy or self._session is not None,
                            "meta": {"name": self.name, "kind": "microduck", "simulated": True}}]}

    def open_stream(self) -> queue.Queue:
        self.drop_stream()  # a second /events on the same token supersedes the first
        outbox: queue.Queue = queue.Queue()
        with self._lock:
            self._consumer = outbox
        outbox.put({"type": "welcome", "peerId": "consumer-1"})
        return outbox

    def send(self, message: dict) -> tuple[int, dict]:
        self.posts += 1
        if not self._limit.take():
            self.rejected_429 += 1
            return 429, {"detail": "rate limited"}
        if self._consumer is None:
            return 400, {"detail": "no such peer: open /events first"}
        kind = message.get("type")
        if kind == "setPeerStatus":
            return 200, {"status": "ok"}
        if kind == "startSession":
            if self.busy or message.get("peerId") != ROBOT_PEER:
                return 200, {"type": "sessionRejected", "reason": "robot_busy", "activeApp": "the console"}
            self._session = f"session-{next(self._sessions)}"
            return 200, {"type": "sessionStarted", "sessionId": self._session}
        if kind == "endSession":
            self.drop_stream()
            return 200, {"status": "ok"}
        if kind == "peer":
            if message.get("sessionId") != self._session or not isinstance(message.get("rpc"), dict):
                return 200, {"status": "ok"}  # the real service drops a peer message for an unknown session
            threading.Thread(target=self._relay, args=(message["rpc"],), daemon=True).start()
            return 200, {"status": "ok"}
        return 200, {"status": "ok"}

    # ── the robot side: mediad's control lane ──

    def _to_consumer(self, rpc: dict) -> None:
        if rpc.get("id") is None and not self._robot_budget.take():
            self.dropped_notifications += 1
            return
        outbox = self._consumer
        if outbox is not None:
            outbox.put({"type": "peer", "sessionId": self._session, "rpc": rpc})

    def _relay(self, rpc: dict) -> None:
        line = json.dumps(rpc) + "\n"
        try:
            if rpc.get("id") is None:
                with self._lock:
                    if self._writer is None:
                        conn = socket.socket(socket.AF_UNIX)
                        conn.connect(self.robot_socket)
                        self._writer = conn.makefile("w")
                    self._writer.write(line)
                    self._writer.flush()
                return
            conn = socket.socket(socket.AF_UNIX)
            conn.connect(self.robot_socket)
            stream = conn.makefile("rw")
            stream.write(line)
            stream.flush()
            if rpc.get("method") != "robot.subscribe":
                answer = stream.readline()
                conn.close()
                if answer:
                    self._to_consumer(json.loads(answer))
                return
            with self._lock:
                self._subscriptions.append(conn)
            for answer in stream:
                self._to_consumer(json.loads(answer))
        except (OSError, ValueError):
            with self._lock:
                self._writer = None


class _Handler(BaseHTTPRequestHandler):
    hub: FakeRendezvous
    # HTTP/1.1 and a chunked stream, like the real Space: a client reading "as it arrives" relies on chunks
    protocol_version = "HTTP/1.1"

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:
        pass

    def _authorised(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {self.hub.token}":
            return True
        self._json(401, {"detail": "invalid token"})
        return False

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if not self._authorised():
            return
        if self.path == "/api/robot-status":
            self._json(200, self.hub.robot_status())
            return
        if self.path != "/events":
            self._json(404, {"detail": "not found"})
            return
        outbox = self.hub.open_stream()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                try:
                    message = outbox.get(timeout=5)
                except queue.Empty:
                    self._chunk(b"event: ping\ndata: {}\n\n")  # the real one pings every 30 s
                    continue
                if message is None:
                    self.wfile.write(b"0\r\n\r\n")
                    self.close_connection = True
                    return
                self._chunk(f"data: {json.dumps(message)}\r\n\r\n".encode())  # CRLF, as a proxy may
        except OSError:
            return

    def do_POST(self) -> None:
        if not self._authorised():
            return
        if self.path != "/send":
            self._json(404, {"detail": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self._json(*self.hub.send(body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--socket", default=str(Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim")) / "duck.sock"))
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--token", default="local")
    args = parser.parse_args()
    hub = FakeRendezvous(args.socket, args.token, port=args.port)
    print(f"fake rendezvous on {hub.url} → {args.socket} (token {args.token!r})", flush=True)
    try:
        while True:
            time.sleep(10)
            print(f"posts {hub.posts}, 429s {hub.rejected_429}, robot notifications dropped {hub.dropped_notifications}", flush=True)
    except KeyboardInterrupt:
        hub.close()
