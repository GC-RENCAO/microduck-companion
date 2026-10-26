"""The duck's body as the rest of duck-chat sees it: intents out, robot.state in, over any transport.

Protocol (microduck's duck-ipc-proto):
- robot.move / robot.head / robot.pose are continuous intents: notifications (no id) repeated while they
  hold, which robotd zeroes 500 ms after the last one — so “walk 3 s” means sending for 3 s.
- robot.do / robot.look / robot.sound / robot.enable are discrete: requests with an id and an answer.
- robot.subscribe streams robot.state: policy, safety, joints, odometry, IMU.

How often an intent is repeated and how fast state is asked for come from the transport (transport.py):
10 Hz on a local socket, ~3 Hz through the rendezvous, whose rate limit is shared by everything.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from transport import DEFAULT_SOCKET, DuckError, Transport, UnixTransport  # noqa: F401 — DuckError is re-exported

STATE_STALE_S = 2.0  # a robot.state older than this is not the robot's state any more


class DuckClient:
    def __init__(self, link: Transport | None = None):
        self.link = link or UnixTransport(DEFAULT_SOCKET)
        self._motion_lock = threading.Lock()
        # The current continuous intents, one channel per method: {"robot.move": (params, until), ...}.
        # Walking and head turns never overwrite each other — the head tracks you while the duck walks.
        self._intents: dict[str, tuple[dict, float]] = {}
        # robot.pose with active=true puts the duck in a posing mode that outlives the intent, and in it
        # the duck will not walk (in the sim: 0 m in 3 s, 0.36 m after active=false). So remember
        # whether we posed, and leave the mode before walking or a skill.
        self._posing = False
        self._wake = threading.Event()
        self._state: dict | None = None
        self._state_at = 0.0
        self._state_cond = threading.Condition()
        self._state_listeners: list[Callable[[dict], None]] = []
        threading.Thread(target=self._motion_loop, name="duck-motion", daemon=True).start()
        threading.Thread(target=self._state_loop, name="duck-state", daemon=True).start()

    # ── Discrete intents ────────────────────────────────────────────────────

    def request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        return self.link.request(method, params, timeout)

    def do(self, skill: str) -> dict:
        self.release_pose()
        return self.request("robot.do", {"skill": skill})

    def sound(self, tag: str) -> dict:
        return self.request("robot.sound", {"tag": tag})

    def enable(self) -> dict:
        return self.request("robot.enable", {"on": True})

    def look(self, x: float, y: float, z: float, hold_s: float = 3.0) -> dict:
        """Look at a point in the trunk frame and hold it for hold_s.

        A single robot.look is soon taken over by the idle look-around; resending the joint angles it
        answers with as robot.head (a continuous intent) is what makes the gaze stick.
        """
        result = self.request("robot.look", {"x": x, "y": y, "z": z})
        self.hold_head(result["head"], hold_s)
        return result

    # ── Continuous intents ──────────────────────────────────────────────────

    def move(self, vx: float, vyaw: float = 0.0, seconds: float = 3.0) -> None:
        """Walk at this velocity for `seconds` and return at once; a newer move/stop replaces it."""
        if vx or vyaw:
            self.release_pose()
        self._set_intent("robot.move", {"vx": vx, "vy": 0.0, "vyaw": vyaw}, seconds)

    def stop(self) -> None:
        self._set_intent("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}, self.link.resend_s * 3)

    def hold_head(self, head: dict, seconds: float) -> None:
        """Hold the head at these joint angles for `seconds` (robot.head). A new target goes out at once."""
        self._set_intent("robot.head", head, seconds)

    def hold_pose(self, pitch: float = 0.0, roll: float = 0.0, seconds: float = 1.0) -> None:
        """Body posture while standing (robot.pose): pitch positive = lean forward, roll positive = tilt right.

        Trained on ±0.26 rad and robotd does not clamp, so clamping here keeps the policy on inputs it has
        seen. In the sim ±0.2 pitch/roll reads clearly; z (crouch) barely shows, so it is not offered.
        """
        if self.moving():
            # Posing mode stops the walk, and an expression that raced a walk command (life decides it
            # was idle a tick before the walk began) must not win. The walk always does.
            return
        limit = 0.24
        params = {"z": 0.0, "pitch": max(-limit, min(limit, pitch)), "roll": max(-limit, min(limit, roll)), "active": True}
        self._posing = True
        self._set_intent("robot.pose", params, seconds)

    def release_pose(self, force: bool = False) -> None:
        """Leave posing mode (the body returns to normal at once), or the duck will not walk.
        Does nothing if we never posed; force=True always sends (clearing a previous run's state at startup)."""
        if not self._posing and not force:
            return
        self._posing = False
        self._set_intent("robot.pose", {"z": 0.0, "pitch": 0.0, "roll": 0.0, "active": False}, self.link.resend_s * 3)

    def moving(self) -> bool:
        with self._motion_lock:
            intent = self._intents.get("robot.move")
            return intent is not None and intent[1] > time.monotonic()

    def _set_intent(self, method: str, params: dict, seconds: float) -> None:
        with self._motion_lock:
            self._intents[method] = (params, time.monotonic() + seconds)
        self._wake.set()

    def _motion_loop(self) -> None:
        sent: dict[str, tuple[float, dict]] = {}  # per channel: when, and what, was last sent
        while True:
            now = time.monotonic()
            with self._motion_lock:
                self._intents = {m: i for m, i in self._intents.items() if i[1] > now}
                active = [(m, params) for m, (params, _) in self._intents.items()]
            if not active:
                self._wake.wait()
                self._wake.clear()
                continue
            # Each channel is resent at the transport's cadence. A *changed* target may go sooner — but
            # not sooner than half of it: the camera retargets the gaze 20 times a second, which a
            # rate-limited lane cannot carry, and a newer target simply replaces the one waiting.
            period = self.link.resend_s
            for method, params in active:
                at, last = sent.get(method, (0.0, None))
                if now - at < (period * 0.5 if params != last else period):
                    continue
                try:
                    self.link.notify(method, params)
                    sent[method] = (now, params)
                except DuckError:
                    # The duck restarted, is off, or the lane is throttled. Intents are allowed to be lost;
                    # the move/head that wanted this will resend or time out on its own.
                    pass
            self._wake.wait(period * 0.5)
            self._wake.clear()

    # ── State ───────────────────────────────────────────────────────────────

    def on_state(self, listener: Callable[[dict], None]) -> None:
        """Call `listener(frame)` for every robot.state frame (pose.py draws the real robot from them)."""
        self._state_listeners.append(listener)

    def state(self, timeout: float = 3.0) -> dict:
        """The latest robot.state: policy (stand/walk/sit/…), safety (fallen, …), joints, odom, imu."""
        with self._state_cond:
            fresh = lambda: self._state is not None and time.monotonic() - self._state_at < STATE_STALE_S
            if not self._state_cond.wait_for(fresh, timeout=timeout):
                raise DuckError("读不到鸭子的状态（它没开，或者连接断了）")
            return self._state

    def _state_loop(self) -> None:
        """One subscription for the life of the program, resubscribing whenever it drops."""
        reported = None
        while True:
            try:
                for frame in self.link.subscribe("robot.subscribe", {"hz": self.link.state_hz}):
                    with self._state_cond:
                        self._state, self._state_at = frame, time.monotonic()
                        self._state_cond.notify_all()
                    for listener in self._state_listeners:
                        listener(frame)
            except DuckError as error:
                if str(error) != reported:  # say why once, not every second
                    reported = str(error)
                    print(f"⚠️ 读不到鸭子的状态：{error}", flush=True)
            time.sleep(1.0)
