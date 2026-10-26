"""鸭子屏幕：浏览器里看 3D 鸭子、打字和它互动。

    ./run.sh            然后打开 http://127.0.0.1:8765

GET  /                页面（three.js 在浏览器里渲染 3D）
GET  /api/scene      零件清单与网格索引（JSON，只取一次）
GET  /api/meshes.bin 网格数据（float32 顶点 + uint32 三角面，只取一次）
GET  /api/tex/<n>.png 场景贴图（地板、墙面花纹），场景清单里按编号引用
GET  /api/pose       姿态流：每帧 = uint32 长度 + [float64 仿真时间, 每零件 12×float32]
POST /api/say        {"text": "...", "image": 可选 base64 JPEG} → NDJSON 事件流，每行一个：
                     {"type":"act","steps","act_ms"} 身体开始动 → {"type":"delta","text"} 回话片段…
                     → {"type":"done","brain","ok","timings_ms"}
POST /api/track      摄像头识别结果 {"kind":"hand|face|none","x","y","size","gesture"} → {"event","event_id"}，
                     浏览器每秒约 20 次（tracker.py 感知 + 手势，不经过大模型）
POST /api/viewer     3D 视角（相机）的位置 {"x","y","z"}：这就是“你”在鸭子世界里的位置（life.py）
GET  /api/status     鸭子是否在线、当前 policy、是否摔倒、正在做的动作、大模型是否可用、心情

DUCK_BRAIN=llm（默认）：先交给大模型（brain.py）；Ollama 不可用或出错时退回关键词模式，
回复里标明“规则模式”。DUCK_BRAIN=rules：只用关键词模式。
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
from duck_client import DuckClient, DuckError
from pose import PoseSource
from tracker import Tracker

HOST = os.environ.get("DUCK_CHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("DUCK_CHAT_PORT", "8765"))
USE_LLM = os.environ.get("DUCK_BRAIN", "llm") != "rules"
PAGE = Path(__file__).with_name("static") / "index.html"

duck = DuckClient()
runner = ActionQueue(Actions(duck))
tracker = Tracker(runner)
pose = PoseSource()
try:
    nav = NavMap(pose.model, pose.moving_bodies)
    print(f"🗺️ 地图：{len(nav.rooms)} 个房间、{len(nav.places)} 个地点", flush=True)
except ValueError as reason:  # 开阔空地：不需要地图，直接朝目标走
    nav = None
    print(f"🗺️ 没有地图：{reason}", flush=True)
life = Life(runner, tracker, pose, nav)
brain = Brain(runner, life.describe, list(nav.places) if nav else None)
SCENE_JSON = json.dumps(pose.scene).encode()


STOP_WORDS = ("停", "stop", "别动", "站住", "不要动")


def _rules(text: str, note: str = ""):
    reply = commands.handle(runner, text)
    yield {"type": "act", "steps": [], "act_ms": 0, "action": reply.action}
    yield {"type": "delta", "text": reply.text + (f"\n（规则模式：{note}）" if note else "")}
    yield {"type": "done", "brain": "rules", "ok": reply.ok, "action": reply.action}


def say(text: str, image: str | None):
    """产出 act → delta… → done 事件。"""
    life.on_chat(text)
    # “停”不等大模型：安全相关，必须立刻生效
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
                life.speaking(True)  # 说话时轻轻点头
                yield event
            else:
                thought = event["thought"]
                yield {"type": "done", "brain": brain_module.MODEL, "ok": True, "action": "llm",
                       "timings_ms": thought.timings_ms}
    except BrainError as error:
        if not acted:  # 身体还没决定就失败：整句交给关键词模式
            yield from _rules(text, str(error))
        else:  # 已经开始动了，只是话没说完
            yield {"type": "delta", "text": f"\n（回话出错：{error}）"}
            yield {"type": "done", "brain": brain_module.MODEL, "ok": False, "action": "llm"}
    finally:
        life.speaking(False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 姿态流是长连接，不打访问日志
        pass

    # ── 工具 ────────────────────────────────────────────────────────────────

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

    # ── 路由 ────────────────────────────────────────────────────────────────

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
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/track":
            try:
                self._json(tracker.observe(self._body()))
            except (TypeError, ValueError):
                self._json({"event": None}, HTTPStatus.BAD_REQUEST)
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
                pass  # 页面关了：动作照常执行，话不用再说
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _status(self) -> None:
        payload = {
            "body": pose.connected, "robot": False, "busy": runner.busy(), "current": runner.current,
            "brain": brain_module.MODEL if USE_LLM else "rules", "brain_ready": USE_LLM and brain.available(),
            "life": life.status(),
            "done": runner.log[-3:],  # 最近几个动作的结果（比如“去窝：卡住了，过不去”）
        }
        try:
            state = duck.state()
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
            pass  # 浏览器关了标签页


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
