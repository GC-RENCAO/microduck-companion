"""A unix socket that answers like robotd: calls get a result, notifications are recorded,
robot.subscribe streams robot.state frames (joints, odometry, IMU) at the asked rate."""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import threading
import time


class FakeRobot:
    def __init__(self):
        self.path = os.path.join(tempfile.mkdtemp(prefix="duck-"), "duck.sock")
        self.calls: list[tuple[str, dict]] = []
        self.intents: list[tuple[str, dict]] = []
        self.x = 0.0  # odometry the frames report; tests move it
        self._server = socket.socket(socket.AF_UNIX)
        self._server.bind(self.path)
        self._server.listen()
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._closed = True
        self._server.close()

    def state(self) -> dict:
        return {"t": time.monotonic(), "policy": "stand", "safety": {"fallen": False, "limp": False},
                "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
                "joints": [0.01 * i for i in range(15)], "odom": {"position": [self.x, 0.5, 0.12], "yaw": math.pi / 2},
                "imu": {"gyro": [0, 0, 0], "quat": [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]}}

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        stream = conn.makefile("rw")
        try:
            for line in stream:
                message = json.loads(line)
                method, params = message.get("method"), message.get("params") or {}
                if "id" not in message:
                    self.intents.append((method, params))
                    continue
                self.calls.append((method, params))
                if method == "robot.nosuch":
                    answer = {"error": {"code": -32601, "message": 'unknown method "robot.nosuch"'}}
                elif method == "robot.subscribe" and not isinstance(params.get("hz", 10), int):
                    answer = {"error": {"code": -32602, "message": "invalid type: floating point, expected u32"}}
                else:
                    answer = {"result": {"accepted": True, "method": method}}
                stream.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], **answer}) + "\n")
                stream.flush()
                if method == "robot.subscribe" and "result" in answer:
                    period = 1.0 / float(params.get("hz", 10))
                    while not self._closed:
                        stream.write(json.dumps({"jsonrpc": "2.0", "method": "robot.state", "params": self.state()}) + "\n")
                        stream.flush()
                        time.sleep(period)
        except (OSError, ValueError):
            pass
        finally:
            conn.close()
