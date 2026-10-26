"""robotd 的 JSON-RPC 客户端：通过 Unix socket 给仿真鸭子发意图（intent）。

协议要点（来自 microduck 的 duck-ipc-proto）：
- 每行一个 JSON-RPC 2.0 对象，不需要握手。
- robot.move / robot.head 是“持续意图”：以通知形式（无 id）10 Hz 左右重复发送，
  robotd 在停发 500 ms 后自动归零——所以“走 3 秒”= 连续发 3 秒。
- robot.do / robot.look / robot.sound / robot.enable 是“离散意图”：带 id 的请求，有应答。
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import threading
import time
from pathlib import Path

DEFAULT_SOCKET = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim")) / "duck.sock"

# 持续意图的重发间隔。robotd 的失效时间是 500 ms，10 Hz 留足余量（与 duck-sim drive 一致）。
RESEND_S = 0.1


class DuckError(RuntimeError):
    pass


class DuckClient:
    def __init__(self, socket_path: Path | str = DEFAULT_SOCKET):
        self.socket_path = str(socket_path)
        self._ids = itertools.count(1)
        self._motion_lock = threading.Lock()
        # 当前持续意图，按方法分通道：{"robot.move": (params, 截止时间), "robot.head": (...)}。
        # 走路和转头互不覆盖——摄像头跟踪转头时，走路照常进行。
        self._intents: dict[str, tuple[dict, float]] = {}
        # robot.pose 的 active=true 会让鸭子进入“摆姿势”模式，而且意图过期后仍停在这个模式里，
        # 这时它不响应走路（实测：直接走 3 秒位移 0，发 active=false 后走 3 秒 0.36 米）。
        # 所以记住是否摆过姿势，走路/做技能前先退出。
        self._posing = False
        self._wake = threading.Event()
        threading.Thread(target=self._motion_loop, name="duck-motion", daemon=True).start()

    # ── 离散意图 ────────────────────────────────────────────────────────────

    def request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        """发一个请求并等应答。每次新开连接：请求频率很低，这样最简单也最不容易卡住。"""
        message = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            message["params"] = params
        with socket.socket(socket.AF_UNIX) as conn:
            conn.settimeout(timeout)
            try:
                conn.connect(self.socket_path)
            except OSError as error:
                raise DuckError(f"连不上鸭子（{self.socket_path}）：先运行 scripts/duck-sim") from error
            reader = conn.makefile("rw")
            reader.write(json.dumps(message) + "\n")
            reader.flush()
            answer = json.loads(reader.readline())
        if "error" in answer:
            raise DuckError(answer["error"].get("message", str(answer["error"])))
        return answer.get("result", {})

    def do(self, skill: str) -> dict:
        self.release_pose()
        return self.request("robot.do", {"skill": skill})

    def sound(self, tag: str) -> dict:
        return self.request("robot.sound", {"tag": tag})

    def enable(self) -> dict:
        return self.request("robot.enable", {"on": True})

    def look(self, x: float, y: float, z: float, hold_s: float = 3.0) -> dict:
        """看向躯干坐标系里的一个点，并保持 hold_s 秒。

        robot.look 只发一次的话，头会很快被 idle look-around 接管；应答里的关节角
        通过 robot.head（持续意图）重发，才能“盯住”。
        """
        result = self.request("robot.look", {"x": x, "y": y, "z": z})
        self.hold_head(result["head"], hold_s)
        return result

    # ── 持续意图 ────────────────────────────────────────────────────────────

    def move(self, vx: float, vyaw: float = 0.0, seconds: float = 3.0) -> None:
        """以给定速度走 seconds 秒，立即返回；新的 move/stop 会覆盖它。"""
        if vx or vyaw:
            self.release_pose()
        self._set_intent("robot.move", {"vx": vx, "vy": 0.0, "vyaw": vyaw}, seconds)

    def stop(self) -> None:
        self._set_intent("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}, RESEND_S * 3)

    def hold_head(self, head: dict, seconds: float) -> None:
        """把头保持在给定关节角 seconds 秒（robot.head）。新的目标立即发出，不等下一拍。"""
        self._set_intent("robot.head", head, seconds)

    def hold_pose(self, pitch: float = 0.0, roll: float = 0.0, seconds: float = 1.0) -> None:
        """站立时的身体姿态（robot.pose）：pitch 正 = 前倾，roll 正 = 向右歪。

        训练范围 ±0.26 rad，robotd 不截断，超出会让策略收到没见过的输入，所以这里截断。
        仿真实测：pitch/roll ±0.2 效果明显；z（下蹲）几乎看不出，所以不提供。
        """
        limit = 0.24
        params = {"z": 0.0, "pitch": max(-limit, min(limit, pitch)), "roll": max(-limit, min(limit, roll)), "active": True}
        self._posing = True
        self._set_intent("robot.pose", params, seconds)

    def release_pose(self, force: bool = False) -> None:
        """退出摆姿势模式（身体立刻回到正常姿态），否则鸭子不会走路。
        没摆过姿势就什么都不做；force=True 时总是发（启动时清掉上次运行留下的状态）。"""
        if not self._posing and not force:
            return
        self._posing = False
        self._set_intent("robot.pose", {"z": 0.0, "pitch": 0.0, "roll": 0.0, "active": False}, RESEND_S * 3)

    def moving(self) -> bool:
        with self._motion_lock:
            intent = self._intents.get("robot.move")
            return intent is not None and intent[1] > time.monotonic()

    def _set_intent(self, method: str, params: dict, seconds: float) -> None:
        with self._motion_lock:
            self._intents[method] = (params, time.monotonic() + seconds)
        self._wake.set()

    def _motion_loop(self) -> None:
        writer = None
        while True:
            now = time.monotonic()
            with self._motion_lock:
                self._intents = {m: i for m, i in self._intents.items() if i[1] > now}
                active = [(m, params) for m, (params, _) in self._intents.items()]
            if not active:
                self._wake.wait()
                self._wake.clear()
                continue
            try:
                if writer is None:
                    conn = socket.socket(socket.AF_UNIX)
                    conn.connect(self.socket_path)
                    writer = conn.makefile("w")
                for method, params in active:
                    writer.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
                writer.flush()
            except OSError:
                # 鸭子重启或没开：丢掉这些意图，下次重新连。持续意图本来就允许丢。
                writer = None
                with self._motion_lock:
                    self._intents.clear()
                continue
            # 每 RESEND_S 重发一次；有新目标（比如摄像头跟踪更新了头的方向）就提前醒来立刻发
            self._wake.wait(RESEND_S)
            self._wake.clear()

    # ── 状态 ────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        """读一帧 robot.state：policy（stand/walk/sit/…）、safety（fallen 等）。"""
        with socket.socket(socket.AF_UNIX) as conn:
            conn.settimeout(3)
            conn.connect(self.socket_path)
            reader = conn.makefile("rw")
            reader.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "robot.subscribe", "params": {"hz": 10}}) + "\n")
            reader.flush()
            ack = json.loads(reader.readline())
            if "error" in ack:
                raise DuckError(ack["error"].get("message"))
            return json.loads(reader.readline()).get("params", {})
