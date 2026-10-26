"""The duck screen: the 3D duck in the browser, and typing to it.

    ./run.sh            then open http://127.0.0.1:8765

GET  /                the page (three.js renders the 3D in the browser)
GET  /api/scene       parts and mesh index, plus the world and known places (JSON, fetched once)
GET  /api/meshes.bin  mesh data (float32 vertices + uint32 triangles, fetched once)
GET  /api/tex/<n>.png scene textures (floors, wall patterns), referenced by number in the scene
GET  /api/pose        pose stream: each frame = uint32 length + [float64 time, 12×float32 per part]
POST /api/say         {"text": "...", "image": optional base64 JPEG} → NDJSON events, one per line:
                      {"type":"act","steps","act_ms"} the body starts → {"type":"delta","text"} reply chunks…
                      → {"type":"done","brain","ok","timings_ms"}
POST /api/track       camera results {"kind":"hand|face|none","x","y","size","gesture"} → {"event","event_id"},
                      about 20 a second from the browser (tracker.py perception + gestures, no LLM)
POST /api/viewer      where the 3D view's camera is {"x","y","z"}: that is “you” in the duck's world (life.py)
GET  /api/calibration finger feature statistics per gesture from the real camera (for tracker.py's thresholds); POST resets
GET  /api/status      online?, policy, fallen?, current action, LLM ready?, mood, world and link

DUCK_BRAIN=llm (default): the LLM first (brain.py), falling back to keyword mode — and saying so in the
reply — when Ollama is unavailable or fails. DUCK_BRAIN=rules: keyword mode only.

DUCK_TRANSPORT=unix (default) / rendezvous: how to reach the duck (transport.py).
DUCK_WORLD=sim / robot: where the duck's position and 3D pose come from — the simulator's ground truth,
or the joints and odometry the robot reports itself (pose.py). Defaults to sim over unix and robot over
the rendezvous. In robot mode there is no house map and the duck does not walk on its own (DUCK_ROAM=1
allows it).
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import brain as brain_module
import commands
from actions import ActionQueue, Actions
from brain import Brain, BrainError
from life import Life
from navmap import NavMap
import transport
from duck_client import DuckClient, DuckError
from pose import PoseSource
from tracker import Tracker

HOST = os.environ.get("DUCK_CHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("DUCK_CHAT_PORT", "8765"))
USE_LLM = os.environ.get("DUCK_BRAIN", "llm") != "rules"
PAGE = Path(__file__).with_name("static") / "index.html"

try:
    link = transport.from_env()
except DuckError as error:
    sys.exit(f"❌ {error}")
WORLD = os.environ.get("DUCK_WORLD") or ("sim" if link.name == "unix" else "robot")
REAL = WORLD == "robot"
duck = DuckClient(link)
runner = ActionQueue(Actions(duck))
tracker = Tracker(runner)
pose = PoseSource(WORLD, duck)
nav = None
if REAL:
    print(f"🦆 真机模式（{link.name}）：姿态来自机器人自己的关节和里程计，没有房子地图，不会自己走动", flush=True)
else:
    try:
        nav = NavMap(pose.model, pose.moving_bodies)
        print(f"🗺️ 地图：{len(nav.rooms)} 个房间、{len(nav.places)} 个地点", flush=True)
    except ValueError as reason:  # an open floor: no map needed, head straight for goals
        print(f"🗺️ 没有地图：{reason}", flush=True)
roam = os.environ.get("DUCK_ROAM")
life = Life(runner, tracker, pose, nav, real=REAL, roam=None if roam is None else roam == "1")
brain = Brain(runner, life.describe, list(nav.places) if nav else None, real=REAL)
# What the page builds once: the parts, plus which world it is and the places the duck knows (for the chips)
SCENE_JSON = json.dumps({**pose.scene, "world": WORLD, "places": list(nav.places) if nav else []}).encode()


STOP_WORDS = ("停", "stop", "别动", "站住", "不要动")


def _rules(text: str, note: str = ""):
    reply = commands.handle(runner, text)
    yield {"type": "act", "steps": [], "act_ms": 0, "action": reply.action}
    yield {"type": "delta", "text": reply.text + (f"\n（规则模式：{note}）" if note else "")}
    yield {"type": "done", "brain": "rules", "ok": reply.ok, "action": reply.action}


def say(text: str, image: str | None):
    """Yields act → delta… → done events."""
    life.on_chat(text)
    # “Stop” does not wait for the LLM: it is about safety and must act at once
    if any(word in text.lower() for word in STOP_WORDS):
        runner.cancel()
        yield {"type": "act", "steps": [{"name": "stop", "args": {}}], "act_ms": 0}
        yield {"type": "delta", "text": "好，停下了。"}
        yield {"type": "done", "brain": "rules", "ok": True, "action": "stop"}
        return
    if not USE_LLM:
        yield from _rules(text)
        return
    acted = False
    try:
        for event in brain.think_stream(text, image):
            if event["type"] == "act":
                acted = True
                yield event
            elif event["type"] == "delta":
                life.speaking(True)  # nod gently while talking
                yield event
            else:
                thought = event["thought"]
                yield {"type": "done", "brain": brain_module.MODEL, "ok": True, "action": "llm",
                       "timings_ms": thought.timings_ms}
    except BrainError as error:
        if not acted:  # failed before the body decided: keyword mode takes the whole sentence
            yield from _rules(text, str(error))
        else:  # already moving, only the words were cut short
            yield {"type": "delta", "text": f"\n（回话出错：{error}）"}
            yield {"type": "done", "brain": brain_module.MODEL, "ok": False, "action": "llm"}
    finally:
        life.speaking(False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # the pose stream is long-lived; no access log
        pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # ── Routes ────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        self.path = self.path.split("?", 1)[0]
        if self.path == "/":
            self._bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/scene":
            self._bytes(SCENE_JSON, "application/json")
        elif self.path == "/api/meshes.bin":
            self._bytes(pose.meshes, "application/octet-stream")
        elif self.path.startswith("/api/tex/") and self.path.endswith(".png"):
            try:
                self._bytes(pose.textures[int(self.path[len("/api/tex/"):-len(".png")])], "image/png")
            except (KeyError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
        elif self.path == "/api/pose":
            self._pose_stream()
        elif self.path == "/api/status":
            self._status()
        elif self.path == "/api/calibration":
            self._json(tracker.calibration())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/track":
            try:
                self._json(tracker.observe(self._body()))
            except (TypeError, ValueError):
                self._json({"event": None}, HTTPStatus.BAD_REQUEST)
        elif self.path == "/api/calibration":
            tracker.reset_calibration()
            self._json({"ok": True})
        elif self.path == "/api/viewer":
            body = self._body()
            try:
                life.set_viewer(float(body["x"]), float(body["y"]), float(body["z"]))
                self._json({"ok": True})
            except (KeyError, TypeError, ValueError):
                self._json({"ok": False}, HTTPStatus.BAD_REQUEST)
        elif self.path == "/api/say":
            body = self._body()
            image = body.get("image") if isinstance(body.get("image"), str) else None
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for event in say(str(body.get("text", ""))[:500], image):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page closed: actions carry on, the words need not
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _status(self) -> None:
        payload = {
            "body": pose.connected, "robot": False, "busy": runner.busy(), "current": runner.current,
            "world": WORLD, "link": link.describe(),
            "brain": brain_module.MODEL if USE_LLM else "rules", "brain_ready": USE_LLM and brain.available(),
            "life": life.status(),
            "done": runner.log[-3:],  # the last few actions' results (e.g. “to the dock: stuck, cannot get through”)
        }
        try:
            state = duck.state(timeout=0)
            payload.update(
                robot=True,
                policy=state.get("policy"),
                fallen=bool(state.get("safety", {}).get("fallen")),
                moving=duck.moving(),
            )
        except (DuckError, OSError, ValueError):
            pass
        self._json(payload)

    def _pose_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                last, frame = pose.wait_frame(last)
                if frame is not None:
                    self.wfile.write(len(frame).to_bytes(4, "little") + frame)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the tab


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"🦆 鸭子屏幕： http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
