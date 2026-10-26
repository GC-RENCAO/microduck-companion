"""Webcam perception: MediaPipe in the browser finds hands and faces; this turns them into “what is seen”
and handles instant gestures.

Perception and gestures only — where the head looks and how the body moves is life.py's call (it combines
where you are in the 3D world, what the camera sees and the duck's mood). No LLM; the target for the
whole chain is 0.1–0.2 s.

Coordinates (as the browser sends them, already mirrored like the preview): x ∈ [-1, 1], positive = the
right of the preview (your right hand side); y ∈ [-1, 1], positive = up; size = the hand's or face's width
as a fraction of the frame width (bigger = closer to the camera).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from actions import ActionQueue, Step

PRESENT_S = 0.6  # no hand/face for this long = nobody there
CAMERA_ON_S = 1.5  # no frame from the browser for this long = the camera is off

GESTURE_COOLDOWN_S = 2.0
MIN_SCORE = 0.75  # MediaPipe gesture confidence


@dataclass(frozen=True)
class Gesture:
    label: str
    steps: tuple
    hold_s: float  # how long it has to be held
    min_size: float = 0.0  # minimum hand size (fraction of frame width): bigger = closer to the camera
    max_drift: float = 99.0  # how far the hand may drift while held (frame half-width = 1): stop is held still
    min_upright: float = -99.0  # how upright the fingers are (middle tip above wrist, in palm lengths): stop is a raised palm


# “Stop” is the strictest: MediaPipe calls almost any open hand Open_Palm (in real use a hand merely
# appearing read as stop and cancelled the command just given). So it has to look like a traffic police
# stop — palm close to the camera, held still for 1 s.
# “Still” is the drift over that second, not the instant speed: petting swings back and forth, pausing at
# each end, so its average speed is low.
# And the fingers must point up (a raised palm): very slow petting barely moves in a second and position
# alone cannot tell it apart, but a petting hand is usually flat or tilted.
GESTURES = {
    "Open_Palm": Gesture("✋ 停", (Step("stop", {}),), hold_s=1.0, min_size=0.28, max_drift=0.1, min_upright=1.0),
    "Thumb_Up": Gesture("👍 开心", (Step("make_sound", {"kind": "coo"}),), hold_s=0.7),
    "Victory": Gesture("✌️ 翻滚", (Step("do_trick", {"name": "roulade"}),), hold_s=0.7),
    "ILoveYou": Gesture("🤟 打招呼", (Step("make_sound", {"kind": "greet"}),), hold_s=0.7),
}

# 🫴 Beckon = come here: the hand stays put while the fingers curl and open repeatedly (all four, or the
# index alone). Extension = fingertip-to-wrist / palm length: about 2 open, about 1 in a fist. While
# beckoning, MediaPipe's label flips between open palm and fist, so it never accumulates into “stop”;
# waving keeps the fingers open and moves the whole hand, so it is not a beckon either.
BECKON_WINDOW_S = 2.0
BECKON_MIN_SWING = 0.35  # minimum swing of the extension
BECKON_HYSTERESIS = 0.1
BECKON_MIN_FLIPS = 4  # open → curled → open → curled: at least two round trips
BECKON_MAX_DRIFT = 0.25  # how far the whole hand may drift meanwhile
BECKON_COOLDOWN_S = 4.0
CALIBRATION_SAMPLES = 400  # frames kept per gesture (about 20 s at 20 Hz)

BECKON = ("🫴 过来", (Step("make_sound", {"kind": "chirp"}), Step("come_here", {})))


def _flips(values: list[float]) -> int:
    """How many times a series of extensions switches between open and curled (with hysteresis, so jitter
    does not count). 0 if the swing is too small."""
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
    """What was seen last."""
    kind: str = "none"  # "hand" / "face" / "none"
    x: float = 0.0
    y: float = 0.0
    size: float = 0.0
    speed: float = 0.0  # position change, frame half-widths per second (smoothed)
    gesture: str | None = None
    curl: float = 0.0  # mean extension of the four fingers (about 2 open, about 1 in a fist)
    icurl: float = 0.0  # index finger extension
    upright: float = 0.0  # middle fingertip above the wrist, in palm lengths (> 1 = fingers up)
    at: float = 0.0  # when a hand/face was last seen
    frame_at: float = 0.0  # when the browser last sent a frame (seen anyone or not)


class Tracker:
    def __init__(self, runner: ActionQueue, on_event=None):
        self.runner = runner
        self.on_event = on_event or (lambda text, kind: None)
        self._seen = Seen()
        self._lock = threading.Lock()
        self._gesture, self._gesture_since, self._fired = None, 0.0, False
        self._gesture_path: list[tuple[float, float]] = []  # where the hand was while the gesture was held
        self._hand_history: deque[tuple[float, float, float, float, float]] = deque()  # (t, curl, icurl, x, y)
        # For calibration: real finger features under each MediaPipe gesture (last CALIBRATION_SAMPLES frames)
        self._samples: dict[str, deque[tuple[float, float, float, float]]] = {}
        self._beckon_seen: deque[tuple[float, float, int]] = deque(maxlen=CALIBRATION_SAMPLES)  # (swing, drift, flips)
        self._cooldown_until: dict[str, float] = {}
        self.last_event: tuple[float, str] | None = None
        self.event_id = 0

    def observe(self, data: dict) -> dict:
        """Take one frame of recognition results from the browser."""
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
        if hand is not None and hand.curl:
            label = (data.get("gesture") or "None") if float(data.get("score", 1) or 0) >= MIN_SCORE else "None"
            self._samples.setdefault(label, deque(maxlen=CALIBRATION_SAMPLES)).append(
                (hand.curl, hand.icurl, hand.upright, hand.size))
        self._gestures(hand, now)
        self._beckon(hand, now)
        event = self.last_event
        recent = event is not None and now - event[0] < 1.5
        # Numbered, so the page toasts each event once instead of for 1.5 s
        return {"event": event[1] if recent else None, "event_id": self.event_id}

    def seen(self) -> Seen:
        """What is seen now (a copy); nobody, if nothing for PRESENT_S."""
        with self._lock:
            seen = Seen(**vars(self._seen))
        if time.monotonic() - seen.at > PRESENT_S:
            seen.kind, seen.gesture, seen.speed = "none", None, 0.0
        return seen

    def camera_on(self) -> bool:
        with self._lock:
            return time.monotonic() - self._seen.frame_at < CAMERA_ON_S

    def event(self, text: str) -> None:
        """Show an event in the page (life.py uses this too)."""
        self.last_event = (time.monotonic(), text)
        self.event_id += 1

    def _gestures(self, seen: Seen | None, now: float) -> None:
        """A gesture fires once when held long enough (and close/still/upright enough); it must be dropped
        before it can fire again, and each has a cooldown."""
        name = seen.gesture if seen is not None and seen.gesture in GESTURES else None
        gesture = GESTURES.get(name)
        if gesture is not None and (seen.size < gesture.min_size or seen.upright < gesture.min_upright):
            name, gesture = None, None  # the right shape but not deliberate (too far / fingers not up): ignore
        if name != self._gesture:
            self._gesture, self._gesture_since, self._fired, self._gesture_path = name, now, False, []
        if gesture is not None:
            self._gesture_path.append((seen.x, seen.y))
            xs, ys = [p[0] for p in self._gesture_path], [p[1] for p in self._gesture_path]
            if max(max(xs) - min(xs), max(ys) - min(ys)) > gesture.max_drift:
                # The hand is moving: restart the clock (petting and gesturing never add up to a trigger)
                self._gesture_since, self._gesture_path = now, [(seen.x, seen.y)]
        if gesture is None or self._fired or now - self._gesture_since < gesture.hold_s \
                or now < self._cooldown_until.get(name, 0):
            return
        self._fired = True
        self._cooldown_until[name] = now + GESTURE_COOLDOWN_S
        self.runner.submit(list(gesture.steps))  # a stop clears the queue inside submit
        self.event(gesture.label)
        self.on_event(gesture.label, "gesture")

    def _beckon(self, hand: Seen | None, now: float) -> None:
        """Beckon: over the last BECKON_WINDOW_S the fingers curled and opened ≥ 2 times, the hand staying put."""
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
        drift = max(max(xs) - min(xs), max(ys) - min(ys))
        curls = [h[1] for h in history]
        self._beckon_seen.append((max(curls) - min(curls), drift, _flips(curls)))
        if drift > BECKON_MAX_DRIFT:
            return
        if max(_flips([h[1] for h in history]), _flips([h[2] for h in history])) < BECKON_MIN_FLIPS:
            return
        self._cooldown_until["beckon"] = now + BECKON_COOLDOWN_S
        history.clear()
        label, steps = BECKON
        self.runner.submit(list(steps))
        self.event(label)
        self.on_event(label, "gesture")

    def calibration(self) -> dict:
        """Finger feature statistics from the real camera (the thresholds are at the top of this file).

        Grouped by MediaPipe's gesture label, 5% / 50% / 95% quantiles per feature. `beckon` holds the
        quantiles of the extension swing, hand drift and flips over 2 s windows — beckon at the camera
        and compare them with the BECKON_* thresholds.
        """
        def spread(values):
            values = sorted(values)
            pick = lambda q: round(values[min(len(values) - 1, int(q * len(values)))], 3)
            return {"p5": pick(0.05), "p50": pick(0.5), "p95": pick(0.95)}
        out = {}
        for label, rows in self._samples.items():
            if rows:
                columns = list(zip(*rows))
                out[label] = {"frames": len(rows), **{name: spread(col) for name, col in
                                                      zip(("curl", "icurl", "upright", "size"), columns)}}
        if self._beckon_seen:
            swing, drift, flips = zip(*self._beckon_seen)
            out["beckon"] = {"windows": len(swing), "curl_swing": spread(swing), "drift": spread(drift), "flips": spread(flips),
                             "thresholds": {"min_swing": BECKON_MIN_SWING, "max_drift": BECKON_MAX_DRIFT, "min_flips": BECKON_MIN_FLIPS}}
        out["stop_thresholds"] = {"min_size": GESTURES["Open_Palm"].min_size, "min_upright": GESTURES["Open_Palm"].min_upright,
                                  "max_drift": GESTURES["Open_Palm"].max_drift, "hold_s": GESTURES["Open_Palm"].hold_s}
        return out

    def reset_calibration(self) -> None:
        self._samples.clear()
        self._beckon_seen.clear()
