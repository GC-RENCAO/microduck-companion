"""How JSON-RPC reaches robotd: a unix socket on this machine, or the official rendezvous from anywhere.

Both speak the same thing — JSON-RPC 2.0 objects from microduck's duck-ipc-proto — and offer the same
three verbs, so nothing above this file knows which one it holds:

- `request(method, params)`: a call with an id, answered once.
- `notify(method, params)`: a continuous intent (robot.move / robot.head / robot.pose), no answer.
- `subscribe(method, params)`: a call whose answer is followed by a stream of notifications.

`UnixTransport` is the socket `scripts/duck-sim` (or a robot, to a program running on it) exposes.

`RendezvousTransport` is the control lane of microduck's remote access (docs/design/remote-access-design.md
§3.8, client half in spaces/shared/wire.py): an HTTP event stream plus `POST /send` to the Hugging Face
Space both the robot and this program sign in to. A JSON-RPC object rides in the `rpc` key of a `peer`
envelope, which the rendezvous relays verbatim and the robot's `mediad` answers from the same routing
table as its WebRTC channel. No WebRTC, so nothing a NAT can refuse — and no video, which the duck screen
does not need. What it does need to respect is the rate limit: 1200 requests a minute per token on our
side (a 429 takes out everything that token does), 400 notifications a minute on the robot's. Hence the
two numbers each transport publishes: how often to resend a continuous intent, and how fast to ask for
robot.state.

Configuration (environment variables):
- DUCK_TRANSPORT       `unix` (default) or `rendezvous`
- DUCK_SOCKET          the unix socket, default the duck-sim one
- DUCK_ROBOT           rendezvous: which duck, by name or peer id (default: the only one listed)
- HF_TOKEN             rendezvous: a Hugging Face token of the duck's account; otherwise the one
                       `hf auth login` stored. Not the robot's own token — peers are keyed by token,
                       so sharing it would take the robot off the listing.
- REACHY_CENTRAL_URL   rendezvous: the Space, default the official one (same name the official clients read)
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import queue
import random
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import requests

DEFAULT_SOCKET = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim")) / "duck.sock"
DEFAULT_CENTRAL_URL = "https://pollen-robotics-reachy-mini-central.hf.space"
DUCK_KIND = "microduck"
# Hugging Face's edge rate-limits clients that do not say who they are (spaces/shared/rendezvous.py).
USER_AGENT = "microduck-companion/0.1 (+https://github.com/GC-RENCAO/microduck-companion)"

CONNECT_TIMEOUT = 20
# The rendezvous pings the event stream every 30 s; much longer silence is a dead stream.
READ_TIMEOUT = 90
# Our side of the rendezvous limit is 1200 requests per 60 s. Intents are dropped before that, so a
# burst of them can never cost a request (or the robot its connection).
REQUEST_BUDGET = 1000
RECONNECT_MIN_S, RECONNECT_MAX_S = 5.0, 60.0


class DuckError(RuntimeError):
    pass


class Transport:
    name = "?"
    resend_s = 0.1  # how often a continuous intent is repeated (robotd drops it 500 ms after the last one)
    state_hz = 10  # how fast to ask for robot.state (robotd wants a whole number)

    def request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        raise NotImplementedError

    def notify(self, method: str, params: dict) -> None:
        raise NotImplementedError

    def subscribe(self, method: str, params: dict) -> Iterator[dict]:
        """Yields the `params` of each notification; ends (or raises DuckError) when the stream does."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"transport": self.name}

    def close(self) -> None:
        pass


def _message(method: str, params: dict | None, id: int | None = None) -> dict:
    message: dict = {"jsonrpc": "2.0", "method": method}
    if id is not None:
        message["id"] = id
    if params is not None:
        message["params"] = params
    return message


def _result(answer: dict) -> dict:
    if "error" in answer:
        raise DuckError(answer["error"].get("message", str(answer["error"])))
    return answer.get("result", {})


# ── Unix socket ──────────────────────────────────────────────────────────────


class UnixTransport(Transport):
    name = "unix"

    def __init__(self, socket_path: Path | str = DEFAULT_SOCKET):
        self.socket_path = str(socket_path)
        self._ids = itertools.count(1)
        self._writer = None  # one long-lived connection for intents
        self._writer_lock = threading.Lock()

    def _connect(self, timeout: float | None) -> socket.socket:
        conn = socket.socket(socket.AF_UNIX)
        conn.settimeout(timeout)
        try:
            conn.connect(self.socket_path)
        except OSError as error:
            conn.close()
            raise DuckError(f"连不上鸭子（{self.socket_path}）：先运行 scripts/duck-sim") from error
        return conn

    def request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        # A connection per call: calls are rare, and a daemon serves one request at a time per
        # connection, so sharing one would queue a call behind a slow one.
        with self._connect(timeout) as conn:
            stream = conn.makefile("rw")
            stream.write(json.dumps(_message(method, params, next(self._ids))) + "\n")
            stream.flush()
            line = stream.readline()
        if not line:
            raise DuckError(f"{method}: the robot closed the connection without answering")
        return _result(json.loads(line))

    def notify(self, method: str, params: dict) -> None:
        line = json.dumps(_message(method, params)) + "\n"
        with self._writer_lock:
            try:
                if self._writer is None:
                    self._writer = self._connect(3).makefile("w")
                self._writer.write(line)
                self._writer.flush()
            except (OSError, DuckError) as error:
                self._writer = None
                raise DuckError(f"{method}: {error}") from error

    def subscribe(self, method: str, params: dict) -> Iterator[dict]:
        conn = self._connect(5)
        try:
            stream = conn.makefile("rw")
            stream.write(json.dumps(_message(method, params, next(self._ids))) + "\n")
            stream.flush()
            _result(json.loads(stream.readline() or "{}"))
            conn.settimeout(5)  # a subscription that goes quiet this long has died
            for line in stream:
                message = json.loads(line)
                if "method" in message:
                    yield message.get("params", {})
        except (OSError, ValueError) as error:
            raise DuckError(f"{method}: {error}") from error
        finally:
            conn.close()


# ── The rendezvous ───────────────────────────────────────────────────────────


def hf_token() -> str | None:
    """HF_TOKEN, else what `hf auth login` stored (huggingface_hub's own lookup order, without the dependency)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", home / "token"))
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def list_ducks(token: str, base: str) -> list[dict]:
    """This account's ducks from `GET /api/robot-status`, which opens no stream and takes no session slot."""
    try:
        answer = requests.get(f"{base}/api/robot-status", timeout=CONNECT_TIMEOUT,
                              headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT})
    except requests.RequestException as error:
        raise DuckError(f"连不上 rendezvous（{base}）：{error}") from None
    if answer.status_code == 401:
        raise DuckError("rendezvous 不认这个 Hugging Face token（HF_TOKEN 或 `hf auth login`）")
    if answer.status_code != 200:
        raise DuckError(f"rendezvous /api/robot-status 返回 HTTP {answer.status_code}")
    ducks = []
    for entry in answer.json().get("robots") or []:
        meta = entry.get("meta") or {}
        peer = entry.get("peerId") or entry.get("id")
        if peer and meta.get("kind") == DUCK_KIND:
            flag = meta.get("simulated")
            ducks.append({"peer_id": peer, "name": meta.get("name") or peer[:8], "busy": bool(entry.get("busy")),
                          "simulated": flag is True or str(flag).lower() == "true"})
    return ducks


class _Budget:
    """At most `limit` sends in any 60 s, so we never earn the rendezvous's 429."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window_s = limit, window_s
        self._sent: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def used(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for at in self._sent if now - at <= self.window_s)

    def take(self, reserve: int = 0) -> bool:
        """Spend one unless fewer than `reserve` would be left (intents leave room for calls)."""
        now = time.monotonic()
        with self._lock:
            while self._sent and now - self._sent[0] > self.window_s:
                self._sent.popleft()
            if len(self._sent) >= self.limit - reserve:
                return False
            self._sent.append(now)
            return True


class RendezvousTransport(Transport):
    name = "rendezvous"
    # Three intent channels at ~3 Hz each is ≤ 600 requests a minute, half our budget. Still inside
    # robotd's 500 ms expiry with a round trip to spare.
    resend_s = 0.3
    # robot.state is a notification on the robot's side, rationed to 400 a minute with everything else.
    state_hz = 3

    def __init__(self, token: str | None = None, robot: str | None = None, base: str | None = None,
                 label: str = "microduck-companion"):
        self.base = (base or os.environ.get("REACHY_CENTRAL_URL") or DEFAULT_CENTRAL_URL).rstrip("/")
        self.token = token or hf_token()
        if not self.token:
            raise DuckError("远程模式需要 Hugging Face token：设 HF_TOKEN，或先运行 `hf auth login`")
        self.wanted = robot if robot is not None else os.environ.get("DUCK_ROBOT") or None
        self.label = label
        self.robot: dict | None = None  # {"peer_id", "name", "simulated", ...} once found
        self.error: str | None = None
        self._ids = itertools.count(1)
        self._budget = _Budget(REQUEST_BUDGET)
        self._pending: dict[int, queue.Queue] = {}
        self._subscribers: dict[str, list[queue.Queue]] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self.rtt_ms: int | None = None  # the last call's round trip, through the Space and back
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._local = threading.local()  # a requests.Session per thread: they are not thread-safe
        self._stream: requests.Response | None = None
        threading.Thread(target=self._run, name="rendezvous", daemon=True).start()

    # ── the three verbs ──

    def request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        id = next(self._ids)
        answers: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[id] = answers
        try:
            started = time.monotonic()
            self._send_rpc(_message(method, params, id), intent=False)
            try:
                answer = answers.get(timeout=max(timeout, 8.0))  # a round trip through a Space
                self.rtt_ms = round((time.monotonic() - started) * 1000)
                return _result(answer)
            except queue.Empty:
                raise DuckError(f"{method}: 机器人没有应答（经 rendezvous）") from None
        finally:
            with self._lock:
                self._pending.pop(id, None)

    def notify(self, method: str, params: dict) -> None:
        self._send_rpc(_message(method, params), intent=True)

    def subscribe(self, method: str, params: dict) -> Iterator[dict]:
        # Notifications on this lane carry no subscription id, so they are routed by their method:
        # robot.subscribe streams robot.state.
        topic = {"robot.subscribe": "robot.state"}.get(method, method)
        frames: queue.Queue = queue.Queue(maxsize=32)
        with self._lock:
            self._subscribers[topic].append(frames)
        try:
            self.request(method, params)
            while True:
                try:
                    item = frames.get(timeout=10)
                except queue.Empty:
                    raise DuckError(f"{method}: 机器人 10 秒没有发来状态") from None
                if item is None:  # the session ended
                    raise DuckError(self.error or "rendezvous 连接断了")
                yield item
        finally:
            with self._lock:
                self._subscribers[topic].remove(frames)

    def describe(self) -> dict:
        robot = self.robot or {}
        return {"transport": self.name, "connected": self._connected.is_set(), "robot": robot.get("name"),
                "simulated": robot.get("simulated"), "error": None if self._connected.is_set() else self.error,
                "rtt_ms": self.rtt_ms, "requests_last_minute": self._budget.used()}

    def close(self) -> None:
        self._closed.set()
        session, self._session_id = self._session_id, None
        if session:
            try:
                # Short: this runs while the program is exiting, and an unreachable service must not hold it
                self._post({"type": "endSession", "sessionId": session}, timeout=3)
            except DuckError:
                pass  # the service frees the duck anyway once it notices the stream is gone
        if self._stream is not None:
            self._stream.close()

    # ── the wire ──

    def _http(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        return session

    def _send_rpc(self, rpc: dict, intent: bool) -> None:
        session = self._session_id
        if session is None:
            raise DuckError(self.error or "还没连上远程鸭子")
        # Intents keep a fifth of the budget free for calls; a dropped intent is resent next tick anyway.
        if not self._budget.take(reserve=REQUEST_BUDGET // 5 if intent else 0):
            raise DuckError("发得太快了：rendezvous 每分钟最多 1200 个请求")
        self._post({"type": "peer", "sessionId": session, "rpc": rpc})

    def _post(self, message: dict, timeout: float = CONNECT_TIMEOUT) -> dict | None:
        """`POST /send`. `startSession` and `list` are answered in the body; everything else on the stream."""
        try:
            answer = self._http().post(f"{self.base}/send", json=message, timeout=timeout,
                                       headers={"Authorization": f"Bearer {self.token}"})
        except requests.RequestException as error:
            raise DuckError(f"POST /send：{error}") from None
        if answer.status_code == 429:
            raise DuckError("rendezvous 限流了（每分钟 1200 个请求）")
        if answer.status_code == 400:
            raise DuckError("rendezvous 说这个连接不存在了（事件流断开），正在重连")
        if answer.status_code != 200:
            raise DuckError(f"POST /send 返回 HTTP {answer.status_code}：{answer.text[:200]}")
        try:
            body = answer.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) and body.get("type") else None

    def _run(self) -> None:
        """Connect, pump the event stream, and reconnect with capped, jittered backoff when it drops."""
        delay = RECONNECT_MIN_S
        while not self._closed.is_set():
            try:
                self._connect_once()
                delay = RECONNECT_MIN_S
                self._pump()
            except DuckError as error:
                self.error = str(error)
            except Exception as error:  # noqa: BLE001 — whatever urllib3 raises when a stream dies
                self.error = f"事件流断开：{type(error).__name__}: {error}"
            self._session_id = None
            self._connected.clear()
            self._abandon()
            if self._closed.is_set():
                return
            # A token the service refuses is not fixed by retrying fast.
            wait = RECONNECT_MAX_S / 2 if self.error and "token" in self.error else delay
            self._closed.wait(wait * random.uniform(0.9, 1.1))
            delay = min(RECONNECT_MAX_S, delay * 2)

    def _connect_once(self) -> None:
        ducks = list_ducks(self.token, self.base)
        if self.wanted:
            found = [d for d in ducks if self.wanted in (d["name"], d["peer_id"])]
        else:
            found = ducks if len(ducks) == 1 else []
        if not found:
            names = "、".join(d["name"] for d in ducks) or "没有"
            hint = f"没有叫 {self.wanted} 的鸭子" if self.wanted else ("有好几只，用 DUCK_ROBOT 选一只" if ducks else "这个账号下没有在线的鸭子")
            raise DuckError(f"{hint}（在线的：{names}）")
        robot = found[0]
        if robot["busy"]:
            raise DuckError(f"{robot['name']} 正被别的程序占用（一次只能有一个连接，机器人自己的网页控制台也算）")

        # The stream first: `POST /send` before `GET /events` is a 400, the stream is what binds the token to a peer.
        try:
            stream = requests.get(f"{self.base}/events", stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                  headers={"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream",
                                           "User-Agent": USER_AGENT})
        except requests.RequestException as error:
            raise DuckError(f"连不上 rendezvous：{error}") from None
        if stream.status_code != 200:
            stream.close()
            raise DuckError(f"rendezvous /events 返回 HTTP {stream.status_code}")
        self._stream = stream
        self._frames = self._sse(stream)
        welcome = next(self._frames, None)
        if not welcome or welcome.get("type") != "welcome":
            raise DuckError("rendezvous 接受了连接但没有打招呼")

        # The consumer's name is what the owner's other devices see as `activeApp`.
        self._post({"type": "setPeerStatus", "roles": ["listener"], "meta": {"name": self.label}})
        answer = self._post({"type": "startSession", "peerId": robot["peer_id"]}) or {}
        if answer.get("type") == "sessionRejected":
            raise DuckError(f"{robot['name']} 拒绝了连接：正被 {answer.get('activeApp') or '别的程序'} 使用")
        if answer.get("type") != "sessionStarted" or not answer.get("sessionId"):
            raise DuckError(f"startSession 的回答不对：{answer!r}")
        self.robot = robot
        self._session_id = answer["sessionId"]
        self.error = None
        self._connected.set()

    @staticmethod
    def _sse(stream: requests.Response) -> Iterator[dict]:
        """Server-sent events → JSON messages. CRLF is normalised: a proxy may rewrite line endings."""
        buffer = ""
        for chunk in stream.iter_content(chunk_size=None):
            buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
            while "\n\n" in buffer:
                frame, _, buffer = buffer.partition("\n\n")
                data = "".join(line[5:].strip() for line in frame.split("\n") if line.startswith("data:"))
                if data:
                    try:
                        yield json.loads(data)
                    except ValueError:
                        continue

    def _pump(self) -> None:
        for message in self._frames:
            if self._closed.is_set():
                return
            kind = message.get("type")
            if kind == "peer" and isinstance(message.get("rpc"), dict):
                self._deliver(message["rpc"])
            elif kind in ("endSession", "sessionRejected"):
                raise DuckError(f"远程会话结束了：{message.get('reason') or '没有说原因'}")
        raise DuckError("rendezvous 关闭了事件流")

    def _deliver(self, rpc: dict) -> None:
        with self._lock:
            if rpc.get("id") is not None:
                answers = self._pending.get(rpc["id"])
                if answers is not None:
                    answers.put(rpc)
                return
            subscribers = list(self._subscribers.get(rpc.get("method", ""), ()))
        for frames in subscribers:
            while True:
                try:
                    frames.put_nowait(rpc.get("params", {}))
                    break
                except queue.Full:  # a slow reader loses its oldest frame, never blocks the stream
                    try:
                        frames.get_nowait()
                    except queue.Empty:
                        pass

    def _abandon(self) -> None:
        with self._lock:
            for answers in self._pending.values():
                answers.put({"error": {"message": self.error or "rendezvous 连接断了"}})
            subscribers = [q for queues in self._subscribers.values() for q in queues]
        for frames in subscribers:
            try:
                frames.put_nowait(None)
            except queue.Full:
                pass


def from_env() -> Transport:
    kind = os.environ.get("DUCK_TRANSPORT", "unix").strip().lower()
    if kind == "unix":
        return UnixTransport(os.environ.get("DUCK_SOCKET") or DEFAULT_SOCKET)
    if kind == "rendezvous":
        return RendezvousTransport()
    raise DuckError(f"DUCK_TRANSPORT 只能是 unix 或 rendezvous，不是 {kind!r}")
