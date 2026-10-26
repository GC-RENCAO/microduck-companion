"""The duck's “life”: it knows where you are, has moods, and finds things to do. 10 times a second, no LLM.

Where you are in the 3D world = where the browser's 3D view (camera) is: whichever side you look from,
it knows you are there. Three things decide what the duck is doing at any moment:
- Where you are (/api/viewer) → it watches you; walk round behind it and it turns to face you.
- What the camera sees (tracker.py) → it watches your hand/face; it looks for you when you leave and is
  glad when you are back; it reacts to petting, to being startled, and to you leaning in.
- Its mood (happiness, energy) and how long it has been ignored → it nuzzles up while you are around;
  ignored, it wanders off on its own and eventually sits down for a nap.

Who wins (highest first): your commands (chat/gestures, which interrupt its own behaviour) > getting up
after a fall > reacting to you > its own ideas.
The head: left alone while you have told it where to look or during a skill; otherwise it watches you,
nods gently while talking, and tilts when asking.

The body's measured limits shaped this: pivoting is unreliable (so mostly turns while walking), crouching
barely shows (leaning forward/back/sideways expresses instead), and the model has no mouth joint
(talking is shown by nodding).

With a real robot (`real`), the 3D camera is not a place in the robot's world, so where you are is
unknown, and life starts nothing that walks (it may be on a table; DUCK_ROAM=1 allows it).
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque

from actions import ActionQueue, Step
from duck_client import DuckError
from tracker import Tracker

TICK_S = 0.1
# Steps that move the duck across the floor. With a real robot life starts none of them on its own: it
# may be on a table, and nothing here can see the edge (DUCK_ROAM=1 allows it anyway).
LOCOMOTION = {"walk_forward", "turn", "come_here", "face_me", "go_to", "wander_to"}
LOOK_MIN_S = 1 / 15  # the head target changes at most 15 times a second
TRUNK_Z = 0.12  # trunk origin height above the floor (m)

# How long out of the camera counts as gone, and how often it then looks for you
AWAY_AFTER_S = 8.0
SEARCH_EVERY_S = 18.0
# Being with you: a recent interaction, or you in front of the camera
WITH_YOU_S = 30.0
FACE_YOU_AFTER_S = 1.5  # you behind its shoulder (> 50° off) for this long, and it turns round
FACE_YOU_DEG = 50.0
# It can pivot now (actions._pivot_to), so it turns to you even up close; closer than 0.3 m it only turns its head
FACE_YOU_MIN_M = 0.3
# Ignored
BORED_AFTER_S = 45.0
NAP_AFTER_S = 180.0
WANDER_NEAR_M = (0.6, 1.8)  # with a map: wander to a walkable spot this far away in the same room
NEST = "窝（充电座）"  # sleepy: back to the dock before napping

# Reactions to the camera
PET_SIZE = 0.33  # a hand wider than this fraction of the frame = very close
PET_SPEED = (0.1, 1.5)  # moving slowly = petting
PET_AFTER_S = 0.8
STARTLE_SPEED = 4.0  # a hand swiping past fast
CURIOUS_GROWTH = 1.35  # the face 35% bigger within 1 s = leaning in to look

TALK_BOB_HZ = 1.8  # nodding rate and size while talking
TALK_BOB_RAD = 0.12


class Life:
    def __init__(self, runner: ActionQueue, tracker: Tracker, pose, nav=None, real: bool = False, roam: bool | None = None):
        self.runner, self.tracker, self.pose = runner, tracker, pose
        # A real robot: the 3D view's camera is not a place in its world (its odometry starts wherever it
        # booted), so “where you are” is unknown rather than wrong, and it does not roam unless told to.
        self.real = real
        self.roam = (not real) if roam is None else roam
        self.nav = nav  # navmap.NavMap: the house map; None on an open floor
        self.path: list[tuple[float, float]] | None = None  # the route being walked (drawn in the page)
        self.duck = runner.actions.duck
        runner.actions.world = self
        tracker.on_event = self.on_event

        now = time.monotonic()
        self.viewer: tuple[float, float, float] | None = None  # where you are in the 3D world (m)
        self.happy, self.energy = 0.6, 0.8  # 0–1
        self.last_interaction = now
        self.last_viewer_move = 0.0  # dragging/zooming the view = you moving around the room, which also counts as being around
        self.with_you_since: float | None = None
        self.away_since: float | None = None
        self.napping = False
        self.memory: deque[tuple[float, str]] = deque(maxlen=8)  # recent events, for the LLM's “mouth”

        self._state: dict = {}
        self._state_at = 0.0
        self._cooldown: dict[str, float] = {}
        self._behind_since: float | None = None
        self._pet_since: float | None = None
        self._face_sizes: deque[tuple[float, float]] = deque(maxlen=20)
        self._next_idle = now + random.uniform(15, 30)
        # Head and body expressions
        self._speaking = False
        self._tilt_until, self._tilt_dir = 0.0, 1.0
        self._pose_until, self._pose = 0.0, (0.0, 0.0)
        # After a pose intent expires robotd holds the last pose and stays in posing mode, deaf to walking,
        # so an expression ends by gliding back level and then leaving the mode (duck.release_pose).
        # Leave it once at startup too, clearing whatever a previous run left behind.
        self._glide_until: float | None = None
        self.duck.release_pose(force=True)
        self._head_hold_off_until = 0.0
        self._last_look = 0.0
        threading.Thread(target=self._run, name="life", daemon=True).start()

    # ── Inputs ────────────────────────────────────────────────────────────

    def set_viewer(self, x: float, y: float, z: float) -> None:
        if self.viewer is None or math.dist(self.viewer, (x, y, z)) > 0.05:
            self.last_viewer_move = time.monotonic()
        self.viewer = (x, y, z)

    def _with_you(self, now: float, present: bool) -> bool:
        """You are around: the camera sees you, or you recently interacted / moved the view."""
        return present or now - max(self.last_interaction, self.last_viewer_move) < WITH_YOU_S

    def duck_pose(self) -> tuple[float, float, float]:
        return self.pose.duck_xy[0], self.pose.duck_xy[1], self.pose.duck_yaw

    def viewer_xy(self) -> tuple[float, float]:
        return self.viewer[0], self.viewer[1]

    def show_path(self, path: list[tuple[float, float]] | None) -> None:
        self.path = path

    def where_is_viewer(self) -> tuple[float, float] | None:
        """Your bearing from the duck (rad, positive = on its left) and distance (m); None when unknown."""
        if self.real or self.viewer is None or not self.pose.connected:
            return None
        dx, dy = self.viewer[0] - self.pose.duck_xy[0], self.viewer[1] - self.pose.duck_xy[1]
        bearing = math.atan2(dy, dx) - self.pose.duck_yaw
        return math.atan2(math.sin(bearing), math.cos(bearing)), math.hypot(dx, dy)

    def on_chat(self, text: str) -> None:
        """You said something: an interaction; a question tilts the head; a nap ends."""
        self._interact()
        if _is_question(text):
            self.tilt_head()

    def on_event(self, text: str, kind: str) -> None:
        """Instant events such as gestures."""
        self._interact()
        self._remember(f"主人比了手势 {text}")

    def speaking(self, on: bool) -> None:
        self._speaking = on

    def tilt_head(self, seconds: float = 2.0) -> None:
        self._tilt_until, self._tilt_dir = time.monotonic() + seconds, random.choice((-1.0, 1.0))

    def hold_off_head(self, seconds: float) -> None:
        """You told it where to look: leave the head alone for this long."""
        self._head_hold_off_until = time.monotonic() + seconds

    def describe(self) -> list[str]:
        """Mood and recent events for the LLM's “mouth”."""
        mood = "很开心" if self.happy > 0.75 else "挺开心" if self.happy > 0.55 else "有点闷" if self.happy > 0.35 else "不太开心"
        tired = "精神很好" if self.energy > 0.6 else "有点困" if self.energy > 0.3 else "很困"
        lines = [f"- 心情：{mood}，{tired}" + ("，正在打盹" if self.napping else "")]
        if self.nav is not None and self.pose.connected:
            lines.append(f"- 你在：{self.nav.room_at(*self.pose.duck_xy) or '房子外面'}")
            if self.viewer is not None:
                lines.append(f"- 主人在：{self.nav.room_at(*self.viewer[:2]) or '房子外面'}")
        bearing = self.where_is_viewer()
        if bearing:
            side = "正对着主人" if abs(bearing[0]) < math.radians(30) else ("主人在左边" if bearing[0] > 0 else "主人在右边")
            lines.append(f"- 主人的位置：离你约 {bearing[1]:.1f} 米，{side}")
        if self.tracker.camera_on():
            seen = self.tracker.seen()
            lines.append("- 摄像头：" + {"hand": "看到主人的手", "face": "看到主人的脸"}.get(seen.kind, "现在没看到主人"))
        now = time.monotonic()
        recent = [text for at, text in self.memory if now - at < 180]
        if recent:
            lines.append("- 最近发生：" + "；".join(recent[-4:]))
        return lines

    def status(self) -> dict:
        where = self.where_is_viewer()
        return {"happy": round(self.happy, 2), "energy": round(self.energy, 2), "napping": self.napping,
                "viewer_bearing_deg": round(math.degrees(where[0])) if where else None,
                "viewer_distance_m": round(where[1], 2) if where else None,
                "recent": [text for _, text in list(self.memory)[-3:]],
                "room": self.nav.room_at(*self.pose.duck_xy) if self.nav is not None else None,
                "viewer_room": self.nav.room_at(*self.viewer[:2]) if self.nav is not None and self.viewer else None,
                "path": [[round(x, 2), round(y, 2)] for x, y in self.path] if self.path else None}

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            time.sleep(TICK_S)
            try:
                self._tick()
            except (DuckError, OSError, ValueError, KeyError):
                pass  # the duck is down or restarting: try next tick; chat is unaffected

    def _tick(self) -> None:
        now = time.monotonic()
        state = self._robot_state(now)
        policy = state.get("policy")
        fallen = bool(state.get("safety", {}).get("fallen"))
        seen = self.tracker.seen()
        camera = self.tracker.camera_on()
        present = seen.kind != "none"
        self._update_mood(now, present)

        idle = not self.runner.busy() and not self.duck.moving()
        if fallen:
            self._get_up(now)
        else:
            if camera:
                self._presence(now, present)
                if present:
                    self._react(now, seen)
            if idle:
                self._autonomy(now, present, camera, policy)
        self._head(now, seen, present)
        self._body_pose(now, policy, idle)

    def _robot_state(self, now: float) -> dict:
        if now - self._state_at > 0.5:
            self._state, self._state_at = self.duck.state(), now
        return self._state

    def _update_mood(self, now: float, present: bool) -> None:
        # Happiness drifts back to 0.5; energy recovers with you around and drains when ignored
        self.happy += (0.5 - self.happy) * 0.002
        if self._with_you(now, present):
            self.energy = min(1.0, self.energy + 0.002)
        else:
            self.energy = max(0.0, self.energy - 0.0006)

    # ── Behaviours ────────────────────────────────────────────────────────────────

    def _get_up(self, now: float) -> None:
        if self._ready("get_up", now, 6.0):
            self._do([Step("make_sound", {"kind": "alarm"}), Step("stand_up", {})], "😵 摔倒了，爬起来", now)
            self._remember("刚才摔倒了，自己爬了起来")
            self.napping = False

    def _presence(self, now: float, present: bool) -> None:
        """With the camera on: look for you when you leave, be glad when you are back."""
        if present:
            if self.away_since is not None and now - self.away_since > AWAY_AFTER_S:
                steps = ([Step("stand_up", {})] if self.napping else []) + [Step("make_sound", {"kind": "greet"}), Step("face_me", {})]
                where = self.where_is_viewer()
                if where and where[1] > 1.5:
                    steps.append(Step("come_here", {}))
                self.napping = False
                self._do(steps, "😊 你回来啦", now, preempt=True)
                self.happy = min(1.0, self.happy + 0.3)
                self._remember(f"主人离开了 {now - self.away_since:.0f} 秒，刚刚回来了")
                self._interact()
            self.away_since = None
            return
        if self.away_since is None:
            self.away_since = now
        elif now - self.away_since > AWAY_AFTER_S and not self.napping and not self.runner.busy() \
                and self._ready("search", now, SEARCH_EVERY_S):
            side = random.choice(("left", "right"))
            self._do([Step("make_sound", {"kind": "inquire"}), Step("turn", {"direction": side, "degrees": 90})],
                     "🔍 主人去哪了", now)

    def _react(self, now: float, seen) -> None:
        """React to you in the camera: petting, being startled, you leaning in."""
        # Startled: a hand swiping past fast
        if seen.kind == "hand" and seen.speed > STARTLE_SPEED and seen.size > 0.15 and self._ready("startle", now, 6.0):
            self.duck.sound("alarm")
            self._express(pitch=-0.2, seconds=0.9)
            self.happy = max(0.0, self.happy - 0.05)
            self._event("😳 吓我一跳", now)
            self._remember("主人的手突然挥过来，吓了一跳")
            return
        # Petted: a hand very close, moving slowly
        petting = seen.kind == "hand" and seen.size > PET_SIZE and PET_SPEED[0] < seen.speed < PET_SPEED[1]
        if petting:
            self._pet_since = self._pet_since or now
            if now - self._pet_since > PET_AFTER_S and self._ready("pet", now, 3.0):
                self.duck.sound("coo")
                self._express(pitch=0.12, roll=random.choice((-0.15, 0.15)), seconds=1.8)
                self.happy = min(1.0, self.happy + 0.1)
                self._interact()
                self._event("🥰 好舒服", now)
                self._remember("主人在摸我")
        else:
            self._pet_since = None
        # Curious: your face clearly bigger within 1 s (leaning in to the camera)
        if seen.kind == "face":
            self._face_sizes.append((now, seen.size))
            old = [s for t, s in self._face_sizes if now - t > 0.8]
            if old and seen.size > 0.25 and seen.size > old[-1] * CURIOUS_GROWTH and self._ready("curious", now, 10.0):
                self.duck.sound("inquire")
                self.tilt_head(1.8)
                self._express(pitch=0.15, seconds=1.5)
                self._event("🤔 你在看我吗", now)
                self._remember("主人凑近镜头看我")
        else:
            self._face_sizes.clear()

    def _autonomy(self, now: float, present: bool, camera: bool, policy: str | None) -> None:
        """The duck's own ideas while it is idle."""
        with_you = self._with_you(now, present)
        where = self.where_is_viewer()

        # You are behind its shoulder: turn to face you
        if with_you and where and not self.napping and policy != "sit" and where[1] > FACE_YOU_MIN_M:
            if abs(where[0]) > math.radians(FACE_YOU_DEG):
                self._behind_since = self._behind_since or now
                if now - self._behind_since > FACE_YOU_AFTER_S and self._ready("face", now, 6.0):
                    self._do([Step("face_me", {})], "👀 转过来看你", now)
                    self._behind_since = None
                    return
            else:
                self._behind_since = None

        if now < self._next_idle:
            return
        if with_you:
            self._next_idle = now + random.uniform(15, 30)
            self._companion(now, where)
        elif now - self.last_interaction > NAP_AFTER_S or self.energy < 0.2:
            self._next_idle = now + random.uniform(30, 60)
            self._nap(now, policy)
        elif now - self.last_interaction > BORED_AFTER_S:
            self._next_idle = now + random.uniform(12, 25)
            self._wander(now, where, policy)

    def _companion(self, now: float, where) -> None:
        """With you: nuzzle up now and then, come closer if you are far."""
        if where and where[1] > 1.8 and self.energy > 0.3:
            self._do([Step("come_here", {})], "🐥 凑过来", now)
            return
        choice = random.random()
        if choice < 0.4:
            self.duck.sound("chirp")
            self.tilt_head(1.5)
        elif choice < 0.7:
            self._express(pitch=0.12, seconds=0.6)  # a little nod
        else:
            self.duck.sound("coo")

    def _wander(self, now: float, where, policy: str | None) -> None:
        """Ignored: wander about, but not far."""
        if self.napping:
            return
        if policy == "sit":
            self._do([Step("stand_up", {})], None, now)
            return
        choice = random.random()
        if choice < 0.75:
            spot = self._wander_spot()
            if spot:
                self._do([Step("wander_to", {"x": spot[0], "y": spot[1]})], None, now)
            elif self.nav is None:  # an open floor: stroll and turn at random
                self._do([Step("walk_forward", {"seconds": round(random.uniform(1.5, 3.5), 1)})] if choice < 0.4 else
                         [Step("turn", {"direction": random.choice(("left", "right")), "degrees": random.choice((45, 90, 120))})], None, now)
        else:
            self.duck.sound(random.choice(("chirp", "inquire")))

    def _wander_spot(self) -> tuple[float, float] | None:
        """With a map: any walkable spot in the same room (never into a wall or down the stairwell)."""
        if self.nav is None:
            return None
        x, y, _ = self.duck_pose()
        room = self.nav.room_at(x, y)
        for _ in range(40):
            angle, distance = random.uniform(-math.pi, math.pi), random.uniform(*WANDER_NEAR_M)
            spot = (x + distance * math.cos(angle), y + distance * math.sin(angle))
            if self.nav.is_free(*spot) and self.nav.room_at(*spot) == room:
                return spot
        return None

    def _nap(self, now: float, policy: str | None) -> None:
        if not self.napping:
            self.napping = True
            home = [Step("go_to", {"place": NEST})] if self.nav is not None and NEST in self.nav.places else []
            self._do(home + [Step("make_sound", {"kind": "coo"}), Step("sit_down", {})],
                     "😴 回窝打个盹" if home else "😴 打个盹", now)
            self._remember("没人陪，" + ("回到窝里" if home else "") + "坐下打了个盹")
        elif policy == "sit" and random.random() < 0.5:
            self.duck.sound("coo")  # a coo in its sleep

    # ── Head and body expressions ──────────────────────────────────────────────────────

    def _head(self, now: float, seen, present: bool) -> None:
        current = self.runner.current
        if now < self._head_hold_off_until or current in ("do_trick", "sit_down", "stand_up", "look"):
            return
        with_you = self._with_you(now, present) or self._speaking
        tilt = now < self._tilt_until
        if not (with_you or tilt) or self.napping:
            return  # hand back to robotd's own idle look-around
        if now - self._last_look < LOOK_MIN_S:
            return
        self._last_look = now

        target = self._gaze_target(seen if present else None)
        head = self.duck.request("robot.look", {"x": target[0], "y": target[1], "z": target[2]}, timeout=1.0)["head"]
        if self._speaking:  # nod while talking. The head takes ~220 ms to get halfway, so faster (3.5 Hz) smears out
            head["head_pitch"] += TALK_BOB_RAD * math.sin(2 * math.pi * TALK_BOB_HZ * now)
        if tilt:  # tilt
            head["head_roll"] += 0.28 * self._tilt_dir
        self.duck.hold_head(head, 0.5)

    def _gaze_target(self, seen) -> tuple[float, float, float]:
        """Look at you (your place in the 3D world), fine-tuned by your hand/face in the camera. Trunk frame, m."""
        where = self.where_is_viewer()
        if where is None:
            base = (0.5, 0.0, 0.05)
            distance = 0.5
        else:
            bearing, distance = where
            distance = max(0.4, min(distance, 2.0))
            height = (self.viewer[2] - TRUNK_Z) if self.viewer else 0.1
            height = max(-0.1, min(height * distance / max(where[1], 0.1), 0.8))
            base = (distance * math.cos(bearing), distance * math.sin(bearing), height)
        if seen is None:
            return base
        # You face it: your right is its left. The offset scales with distance, perpendicular to the line of sight
        norm = math.hypot(base[0], base[1]) or 1.0
        left = (-base[1] / norm, base[0] / norm)
        side = seen.x * 0.4 * distance
        return (base[0] + left[0] * side, base[1] + left[1] * side, base[2] + seen.y * 0.3 * distance)

    def _body_pose(self, now: float, policy: str | None, idle: bool) -> None:
        """Leaning forward/back/sideways as expression, only while standing still; level again afterwards."""
        if now < self._pose_until and policy == "stand" and idle:
            pitch, roll = self._pose
            self.duck.hold_pose(pitch=pitch, roll=roll, seconds=0.3)
            self._glide_until = now + 0.6
        elif self._glide_until is not None:
            if now < self._glide_until and idle:
                self.duck.hold_pose(0.0, 0.0, seconds=0.3)  # glide back level first
            else:
                self.duck.release_pose()  # then leave posing mode, or it will not walk
                self._glide_until = None

    def _express(self, pitch: float = 0.0, roll: float = 0.0, seconds: float = 1.0) -> None:
        self._pose, self._pose_until = (pitch, roll), time.monotonic() + seconds

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _interact(self) -> None:
        self.last_interaction = time.monotonic()
        self.energy = min(1.0, self.energy + 0.3)
        if self.napping:
            self.napping = False
            self.runner.submit([Step("stand_up", {}, source="life"), Step("make_sound", {"kind": "greet"}, source="life")])
            self._remember("被主人叫醒了")

    def _do(self, steps: list[Step], label: str | None, now: float, preempt: bool = False) -> None:
        if not self.roam:
            steps = [step for step in steps if step.name not in LOCOMOTION]
            if not steps:
                return
        for step in steps:
            step.source = "life"
        if preempt and self.runner.current_source == "life":
            self.runner.cancel()
        self.runner.submit(steps)
        if label:
            self._event(label, now)

    def _event(self, text: str, now: float) -> None:
        self.tracker.event(text)  # the page shows a toast

    def _remember(self, text: str) -> None:
        self.memory.append((time.monotonic(), text))

    def _ready(self, name: str, now: float, cooldown: float) -> bool:
        if now < self._cooldown.get(name, 0):
            return False
        self._cooldown[name] = now + cooldown
        return True


QUESTION_WORDS = ("?", "？", "吗", "什么", "为什么", "怎么", "哪", "谁")
ENGLISH_QUESTION_STARTS = ("what", "why", "how", "where", "who", "can", "do", "are", "is")


def _is_question(text: str) -> bool:
    words = text.lower().split()
    return any(mark in text for mark in QUESTION_WORDS) or bool(words and words[0] in ENGLISH_QUESTION_STARTS)
