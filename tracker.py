"""摄像头感知：浏览器里的 MediaPipe 识别出手/脸，这里整理成“看到了什么”，并处理即时手势。

只负责感知和手势，不决定头往哪看、身体怎么动——那是 life.py 的事（它综合你在 3D 世界里的位置、
摄像头看到的东西和鸭子的心情）。不经过大模型，整条链路目标 0.1–0.2 秒。

坐标约定（浏览器发来的，已按镜像预览换算好）：x ∈ [-1, 1]，正数 = 你在预览里看到的右边（也就是你的右手边）；
y ∈ [-1, 1]，正数 = 上；size = 手或脸的宽度占画面宽度的比例（越大越靠近镜头）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from actions import ActionQueue, Step

PRESENT_S = 0.6  # 这么久没看到手/脸就算“不在”
CAMERA_ON_S = 1.5  # 这么久没收到浏览器的帧就算摄像头关了

GESTURE_COOLDOWN_S = 2.0
MIN_SCORE = 0.75  # MediaPipe 的手势置信度


@dataclass(frozen=True)
class Gesture:
    label: str
    steps: tuple
    hold_s: float  # 要稳定保持多久
    min_size: float = 0.0  # 手至少多大（占画面宽度）：越大越靠近镜头
    max_drift: float = 99.0  # 保持期间手的位置最多漂移多少（画面半宽为 1）：停的手势要举着不动
    min_upright: float = -99.0  # 手指朝上的程度（中指尖比手腕高几个手掌长）：停要竖着的手掌


# “停”要求最严：MediaPipe 把几乎任何张开的手都认成 Open_Palm（用户实测：手一出现就被当成停，
# 打断了刚下的指令）。所以要像交警那样——手掌靠近镜头、举着不动 1 秒才算。
# “不动”看的是这 1 秒里位置漂移的范围，不是瞬时速度：摸它时手来回摆，两端会停顿，平均速度并不高。
# 还要手指朝上（竖着的手掌）：很慢的抚摸在 1 秒里几乎不动，只靠位置分不开，摸的时候手通常是平的或斜的。
GESTURES = {
    "Open_Palm": Gesture("✋ 停", (Step("stop", {}),), hold_s=1.0, min_size=0.28, max_drift=0.1, min_upright=1.0),
    "Thumb_Up": Gesture("👍 开心", (Step("make_sound", {"kind": "coo"}),), hold_s=0.7),
    "Victory": Gesture("✌️ 翻滚", (Step("do_trick", {"name": "roulade"}),), hold_s=0.7),
    "ILoveYou": Gesture("🤟 打招呼", (Step("make_sound", {"kind": "greet"}),), hold_s=0.7),
}

# 🫴 勾勾手 = 过来：整只手基本不动，手指一弯一伸反复（四指一起或只用食指）。
# 伸展度 = 指尖到手腕 / 手掌长：张开约 2，握拳约 1。勾手指时 MediaPipe 的分类会在张开/握拳之间跳，
# 所以不会累积成“停”；挥手时手指一直张开、整只手大幅移动，也不会被当成勾手。
BECKON_WINDOW_S = 2.0
BECKON_MIN_SWING = 0.35  # 伸展度来回摆动的最小幅度
BECKON_HYSTERESIS = 0.1
BECKON_MIN_FLIPS = 4  # 张 → 弯 → 张 → 弯：至少 2 个来回
BECKON_MAX_DRIFT = 0.25  # 这段时间整只手最多漂移多少
BECKON_COOLDOWN_S = 4.0
BECKON = ("🫴 过来", (Step("make_sound", {"kind": "chirp"}), Step("come_here", {})))


def _flips(values: list[float]) -> int:
    """一串伸展度里，张开/弯曲来回切换了几次（带迟滞，抖动不算）。不够幅度直接算 0。"""
    top, bottom = max(values), min(values)
    if top - bottom < BECKON_MIN_SWING:
        return 0
    middle, state, flips = (top + bottom) / 2, None, 0
    for v in values:
        new = "open" if v > middle + BECKON_HYSTERESIS else "curled" if v < middle - BECKON_HYSTERESIS else state
        if state is not None and new != state:
            flips += 1
        state = new
    return flips


@dataclass
class Seen:
    """最近一次看到的东西。"""
    kind: str = "none"  # "hand" / "face" / "none"
    x: float = 0.0
    y: float = 0.0
    size: float = 0.0
    speed: float = 0.0  # 位置变化速度，单位：画面半宽 / 秒（平滑过）
    gesture: str | None = None
    curl: float = 0.0  # 四指平均伸展度（张开约 2，握拳约 1）
    icurl: float = 0.0  # 食指伸展度
    upright: float = 0.0  # 中指尖比手腕高几个手掌长（> 1 = 手指朝上）
    at: float = 0.0  # 最近一次看到手/脸的时间
    frame_at: float = 0.0  # 最近一次收到浏览器帧的时间（不管有没有看到人）


class Tracker:
    def __init__(self, runner: ActionQueue, on_event=None):
        self.runner = runner
        self.on_event = on_event or (lambda text, kind: None)
        self._seen = Seen()
        self._lock = threading.Lock()
        self._gesture, self._gesture_since, self._fired = None, 0.0, False
        self._gesture_path: list[tuple[float, float]] = []  # 保持手势期间手的位置
        self._hand_history: deque[tuple[float, float, float, float, float]] = deque()  # (t, curl, icurl, x, y)
        self._cooldown_until: dict[str, float] = {}
        self.last_event: tuple[float, str] | None = None
        self.event_id = 0

    def observe(self, data: dict) -> dict:
        """收下浏览器的一帧识别结果。"""
        now = time.monotonic()
        kind = str(data.get("kind", "none"))
        with self._lock:
            seen = self._seen
            seen.frame_at = now
            if kind in ("hand", "face"):
                x = max(-1.0, min(1.0, float(data.get("x", 0) or 0)))
                y = max(-1.0, min(1.0, float(data.get("y", 0) or 0)))
                dt = now - seen.at
                if seen.kind == kind and 0 < dt < 0.3:
                    instant = ((x - seen.x) ** 2 + (y - seen.y) ** 2) ** 0.5 / dt
                    seen.speed += 0.5 * (instant - seen.speed)
                else:
                    seen.speed = 0.0
                seen.kind, seen.x, seen.y, seen.at = kind, x, y, now
                seen.size = max(0.0, min(1.0, float(data.get("size", 0) or 0)))
                score = float(data.get("score", 1) or 0)
                seen.gesture = (data.get("gesture") or None) if score >= MIN_SCORE else None
                seen.curl = float(data.get("curl", 0) or 0)
                seen.icurl = float(data.get("icurl", 0) or 0)
                seen.upright = float(data.get("upright", 0) or 0)
            elif now - seen.at > PRESENT_S:
                seen.kind, seen.gesture, seen.speed = "none", None, 0.0
        hand = seen if seen.kind == "hand" else None
        self._gestures(hand, now)
        self._beckon(hand, now)
        event = self.last_event
        recent = event is not None and now - event[0] < 1.5
        # 带编号：页面只提示新事件，不会在 1.5 秒内反复弹同一个
        return {"event": event[1] if recent else None, "event_id": self.event_id}

    def seen(self) -> Seen:
        """当前看到的（拷贝）。超过 PRESENT_S 没更新就当作没看到人。"""
        with self._lock:
            seen = Seen(**vars(self._seen))
        if time.monotonic() - seen.at > PRESENT_S:
            seen.kind, seen.gesture, seen.speed = "none", None, 0.0
        return seen

    def camera_on(self) -> bool:
        with self._lock:
            return time.monotonic() - self._seen.frame_at < CAMERA_ON_S

    def event(self, text: str) -> None:
        """在页面上提示一个事件（life.py 也用它）。"""
        self.last_event = (time.monotonic(), text)
        self.event_id += 1

    def _gestures(self, seen: Seen | None, now: float) -> None:
        """手势要稳定保持够久（且满足大小/静止要求）才触发一次；放下后才能再次触发；每种有冷却时间。"""
        name = seen.gesture if seen is not None and seen.gesture in GESTURES else None
        gesture = GESTURES.get(name)
        if gesture is not None and (seen.size < gesture.min_size or seen.upright < gesture.min_upright):
            name, gesture = None, None  # 姿势对但不够“刻意”（离镜头太远 / 手指没朝上）：当作没比
        if name != self._gesture:
            self._gesture, self._gesture_since, self._fired, self._gesture_path = name, now, False, []
        if gesture is not None:
            self._gesture_path.append((seen.x, seen.y))
            xs, ys = [p[0] for p in self._gesture_path], [p[1] for p in self._gesture_path]
            if max(max(xs) - min(xs), max(ys) - min(ys)) > gesture.max_drift:
                # 手在动：从现在重新计时（摸它、比划时不会累积到触发）
                self._gesture_since, self._gesture_path = now, [(seen.x, seen.y)]
        if gesture is None or self._fired or now - self._gesture_since < gesture.hold_s \
                or now < self._cooldown_until.get(name, 0):
            return
        self._fired = True
        self._cooldown_until[name] = now + GESTURE_COOLDOWN_S
        self.runner.submit(list(gesture.steps))  # stop 会在 submit 里清空队列
        self.event(gesture.label)
        self.on_event(gesture.label, "gesture")

    def _beckon(self, hand: Seen | None, now: float) -> None:
        """勾勾手：最近 BECKON_WINDOW_S 秒里手指来回弯伸 ≥ 2 次、整只手基本不动。"""
        history = self._hand_history
        if hand is None or not hand.curl:
            history.clear()
            return
        history.append((now, hand.curl, hand.icurl, hand.x, hand.y))
        while history and now - history[0][0] > BECKON_WINDOW_S:
            history.popleft()
        if len(history) < 10 or now < self._cooldown_until.get("beckon", 0):
            return
        xs, ys = [h[3] for h in history], [h[4] for h in history]
        if max(max(xs) - min(xs), max(ys) - min(ys)) > BECKON_MAX_DRIFT:
            return
        if max(_flips([h[1] for h in history]), _flips([h[2] for h in history])) < BECKON_MIN_FLIPS:
            return
        self._cooldown_until["beckon"] = now + BECKON_COOLDOWN_S
        history.clear()
        label, steps = BECKON
        self.runner.submit(list(steps))
        self.event(label)
        self.on_event(label, "gesture")
