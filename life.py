"""鸭子的“生命”：知道你在哪、有心情、会自己找事做。每秒 10 次，不经过大模型。

你在 3D 世界里的位置 = 浏览器 3D 视角（相机）的位置：你从哪个角度看它，它就知道你在那个方向。
综合三样东西决定鸭子此刻的样子：
- 你在哪（/api/viewer）→ 它盯着你看；你绕到背后，它会转过身来。
- 摄像头看到什么（tracker.py）→ 盯着你的手/脸；你离开会找你，回来会高兴；被摸、被吓到、你凑近都有反应。
- 它的心情（开心、精力）和多久没人理 → 陪着你时偶尔撒娇；没人理就自己溜达，久了坐下打盹。

谁说了算（从高到低）：主人的指令（聊天/手势，会打断自主行为）> 摔倒爬起 > 对你的反应 > 自己找事做。
头：主人明确让它看某处时、做技能时，不去管头；其余时间盯着你，说话时轻轻点头，疑问时歪头。

仿真实测的身体限制决定了这里的写法：不能原地转（只能边走边转）、下蹲几乎看不出（用前倾/后仰/歪身表达）、
模型里没有嘴巴关节（说话用点头表现）。
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
LOOK_MIN_S = 1 / 15  # 头部目标最多每秒更新 15 次
TRUNK_Z = 0.12  # 躯干原点离地高度（米）

# 你多久没出现在摄像头里算离开、离开后多久开始找你
AWAY_AFTER_S = 8.0
SEARCH_EVERY_S = 18.0
# 和你相处：最近有互动或你在镜头前，就算“陪着你”
WITH_YOU_S = 30.0
FACE_YOU_AFTER_S = 1.5  # 你在它侧后方（偏差 > 50°）持续这么久，它就转过来
FACE_YOU_DEG = 50.0
# 现在能原地转身（actions._pivot_to），离得近也能转过来；太近（< 0.3 米）就只用头看
FACE_YOU_MIN_M = 0.3
# 没人理
BORED_AFTER_S = 45.0
NAP_AFTER_S = 180.0
WANDER_NEAR_M = (0.6, 1.8)  # 有地图时：在同一个房间里挑这么远的一个能走的点溜过去
NEST = "窝（充电座）"  # 困了先回窝再打盹

# 摄像头反应
PET_SIZE = 0.33  # 手占画面宽度超过这个比例 = 很近
PET_SPEED = (0.1, 1.5)  # 慢慢地动 = 在摸
PET_AFTER_S = 0.8
STARTLE_SPEED = 4.0  # 手飞快地挥过
CURIOUS_GROWTH = 1.35  # 脸 1 秒内变大 35% = 凑近看它

TALK_BOB_HZ = 1.8  # 说话时点头的频率和幅度
TALK_BOB_RAD = 0.12


class Life:
    def __init__(self, runner: ActionQueue, tracker: Tracker, pose, nav=None):
        self.runner, self.tracker, self.pose = runner, tracker, pose
        self.nav = nav  # navmap.NavMap：房子的地图；开阔空地时为 None
        self.path: list[tuple[float, float]] | None = None  # 正在走的路线（页面上画出来）
        self.duck = runner.actions.duck
        runner.actions.world = self
        tracker.on_event = self.on_event

        now = time.monotonic()
        self.viewer: tuple[float, float, float] | None = None  # 你在 3D 世界里的位置（米）
        self.happy, self.energy = 0.6, 0.8  # 0–1
        self.last_interaction = now
        self.last_viewer_move = 0.0  # 你拖动/缩放视角 = 你在房间里走动，也算“你在身边”
        self.with_you_since: float | None = None
        self.away_since: float | None = None
        self.napping = False
        self.memory: deque[tuple[float, str]] = deque(maxlen=8)  # 最近发生的事，给大模型的“嘴巴”

        self._state: dict = {}
        self._state_at = 0.0
        self._cooldown: dict[str, float] = {}
        self._behind_since: float | None = None
        self._pet_since: float | None = None
        self._face_sizes: deque[tuple[float, float]] = deque(maxlen=20)
        self._next_idle = now + random.uniform(15, 30)
        # 头和身体的表情
        self._speaking = False
        self._tilt_until, self._tilt_dir = 0.0, 1.0
        self._pose_until, self._pose = 0.0, (0.0, 0.0)
        # robotd 在姿态意图过期后会停在最后的姿态，而且留在“摆姿势”模式里不响应走路，
        # 所以表情结束要先平滑摆正、再退出这个模式（duck.release_pose）。
        # 启动时先退出一次，清掉上次运行留下的状态。
        self._glide_until: float | None = None
        self.duck.release_pose(force=True)
        self._head_hold_off_until = 0.0
        self._last_look = 0.0
        threading.Thread(target=self._run, name="life", daemon=True).start()

    # ── 外部输入 ────────────────────────────────────────────────────────────

    def set_viewer(self, x: float, y: float, z: float) -> None:
        if self.viewer is None or math.dist(self.viewer, (x, y, z)) > 0.05:
            self.last_viewer_move = time.monotonic()
        self.viewer = (x, y, z)

    def _with_you(self, now: float, present: bool) -> bool:
        """你在身边：摄像头看到你，或者最近和它互动过 / 动过视角。"""
        return present or now - max(self.last_interaction, self.last_viewer_move) < WITH_YOU_S

    def duck_pose(self) -> tuple[float, float, float]:
        return self.pose.duck_xy[0], self.pose.duck_xy[1], self.pose.duck_yaw

    def viewer_xy(self) -> tuple[float, float]:
        return self.viewer[0], self.viewer[1]

    def show_path(self, path: list[tuple[float, float]] | None) -> None:
        self.path = path

    def where_is_viewer(self) -> tuple[float, float] | None:
        """你相对鸭子的方位（弧度，正 = 在它左边）和距离（米）。"""
        if self.viewer is None or not self.pose.connected:
            return None
        dx, dy = self.viewer[0] - self.pose.duck_xy[0], self.viewer[1] - self.pose.duck_xy[1]
        bearing = math.atan2(dy, dx) - self.pose.duck_yaw
        return math.atan2(math.sin(bearing), math.cos(bearing)), math.hypot(dx, dy)

    def on_chat(self, text: str) -> None:
        """你说了句话：算互动；是疑问句就歪头；在打盹就醒过来。"""
        self._interact()
        if _is_question(text):
            self.tilt_head()

    def on_event(self, text: str, kind: str) -> None:
        """手势等即时事件。"""
        self._interact()
        self._remember(f"主人比了手势 {text}")

    def speaking(self, on: bool) -> None:
        self._speaking = on

    def tilt_head(self, seconds: float = 2.0) -> None:
        self._tilt_until, self._tilt_dir = time.monotonic() + seconds, random.choice((-1.0, 1.0))

    def hold_off_head(self, seconds: float) -> None:
        """主人明确让它看某处：这段时间不去管头。"""
        self._head_hold_off_until = time.monotonic() + seconds

    def describe(self) -> list[str]:
        """给大模型“嘴巴”的心情和最近发生的事。"""
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

    # ── 主循环 ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            time.sleep(TICK_S)
            try:
                self._tick()
            except (DuckError, OSError, ValueError, KeyError):
                pass  # 仿真没开或重启中：下一拍再试，不影响聊天

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
        # 开心慢慢回到 0.5；陪着你时精力恢复，没人理时慢慢变困
        self.happy += (0.5 - self.happy) * 0.002
        if self._with_you(now, present):
            self.energy = min(1.0, self.energy + 0.002)
        else:
            self.energy = max(0.0, self.energy - 0.0006)

    # ── 行为 ────────────────────────────────────────────────────────────────

    def _get_up(self, now: float) -> None:
        if self._ready("get_up", now, 6.0):
            self._do([Step("make_sound", {"kind": "alarm"}), Step("stand_up", {})], "😵 摔倒了，爬起来", now)
            self._remember("刚才摔倒了，自己爬了起来")
            self.napping = False

    def _presence(self, now: float, present: bool) -> None:
        """摄像头开着时：你离开会找你，回来会高兴。"""
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
        """对摄像头里的你做出反应：被摸、被吓到、你凑近看它。"""
        # 吓一跳：手飞快地挥过
        if seen.kind == "hand" and seen.speed > STARTLE_SPEED and seen.size > 0.15 and self._ready("startle", now, 6.0):
            self.duck.sound("alarm")
            self._express(pitch=-0.2, seconds=0.9)
            self.happy = max(0.0, self.happy - 0.05)
            self._event("😳 吓我一跳", now)
            self._remember("主人的手突然挥过来，吓了一跳")
            return
        # 被摸：手很近、慢慢地动
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
        # 好奇：你的脸 1 秒内明显变大（凑近镜头看它）
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
        """鸭子空闲时自己找事做。"""
        with_you = self._with_you(now, present)
        where = self.where_is_viewer()

        # 你在它侧后方：转过来面对你
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
        """陪着你：偶尔撒个娇，你离太远就凑过去。"""
        if where and where[1] > 1.8 and self.energy > 0.3:
            self._do([Step("come_here", {})], "🐥 凑过来", now)
            return
        choice = random.random()
        if choice < 0.4:
            self.duck.sound("chirp")
            self.tilt_head(1.5)
        elif choice < 0.7:
            self._express(pitch=0.12, seconds=0.6)  # 点点头
        else:
            self.duck.sound("coo")

    def _wander(self, now: float, where, policy: str | None) -> None:
        """没人理：自己溜达，但不走太远。"""
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
            elif self.nav is None:  # 开阔空地：随便走走、转转
                self._do([Step("walk_forward", {"seconds": round(random.uniform(1.5, 3.5), 1)})] if choice < 0.4 else
                         [Step("turn", {"direction": random.choice(("left", "right")), "degrees": random.choice((45, 90, 120))})], None, now)
        else:
            self.duck.sound(random.choice(("chirp", "inquire")))

    def _wander_spot(self) -> tuple[float, float] | None:
        """有地图时：同一个房间里随便挑一个能走的点（不会走去撞墙、掉进楼梯井）。"""
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
            self.duck.sound("coo")  # 睡梦里咕一声

    # ── 头和身体的表情 ──────────────────────────────────────────────────────

    def _head(self, now: float, seen, present: bool) -> None:
        current = self.runner.current
        if now < self._head_hold_off_until or current in ("do_trick", "sit_down", "stand_up", "look"):
            return
        with_you = self._with_you(now, present) or self._speaking
        tilt = now < self._tilt_until
        if not (with_you or tilt) or self.napping:
            return  # 交还给 robotd 自己东张西望
        if now - self._last_look < LOOK_MIN_S:
            return
        self._last_look = now

        target = self._gaze_target(seen if present else None)
        head = self.duck.request("robot.look", {"x": target[0], "y": target[1], "z": target[2]}, timeout=1.0)["head"]
        if self._speaking:  # 说话时点头。头约 220 ms 才转过一半，太快（3.5 Hz）会被抹平看不出来
            head["head_pitch"] += TALK_BOB_RAD * math.sin(2 * math.pi * TALK_BOB_HZ * now)
        if tilt:  # 歪头
            head["head_roll"] += 0.28 * self._tilt_dir
        self.duck.hold_head(head, 0.5)

    def _gaze_target(self, seen) -> tuple[float, float, float]:
        """看向你（3D 世界里的位置），再按摄像头里手/脸的位置微调。躯干坐标系，米。"""
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
        # 你面对着它：你的右手边是它的左边。偏移量和距离成比例，垂直于视线
        norm = math.hypot(base[0], base[1]) or 1.0
        left = (-base[1] / norm, base[0] / norm)
        side = seen.x * 0.4 * distance
        return (base[0] + left[0] * side, base[1] + left[1] * side, base[2] + seen.y * 0.3 * distance)

    def _body_pose(self, now: float, policy: str | None, idle: bool) -> None:
        """前倾/后仰/歪身的表情，只在站着不动时做；做完摆正。"""
        if now < self._pose_until and policy == "stand" and idle:
            pitch, roll = self._pose
            self.duck.hold_pose(pitch=pitch, roll=roll, seconds=0.3)
            self._glide_until = now + 0.6
        elif self._glide_until is not None:
            if now < self._glide_until and idle:
                self.duck.hold_pose(0.0, 0.0, seconds=0.3)  # 先慢慢摆正
            else:
                self.duck.release_pose()  # 再退出摆姿势模式，否则不会走路
                self._glide_until = None

    def _express(self, pitch: float = 0.0, roll: float = 0.0, seconds: float = 1.0) -> None:
        self._pose, self._pose_until = (pitch, roll), time.monotonic() + seconds

    # ── 小工具 ──────────────────────────────────────────────────────────────

    def _interact(self) -> None:
        self.last_interaction = time.monotonic()
        self.energy = min(1.0, self.energy + 0.3)
        if self.napping:
            self.napping = False
            self.runner.submit([Step("stand_up", {}, source="life"), Step("make_sound", {"kind": "greet"}, source="life")])
            self._remember("被主人叫醒了")

    def _do(self, steps: list[Step], label: str | None, now: float, preempt: bool = False) -> None:
        for step in steps:
            step.source = "life"
        if preempt and self.runner.current_source == "life":
            self.runner.cancel()
        self.runner.submit(steps)
        if label:
            self._event(label, now)

    def _event(self, text: str, now: float) -> None:
        self.tracker.event(text)  # 页面会弹出提示

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
