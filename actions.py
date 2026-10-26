"""What the duck can do, and the queue that runs it in order. Shared by keyword mode and the LLM's tool calls.

Speeds and durations were measured in the sim (alpha policies):
- vx 0.15 steps in place; vx 0.3 walks steadily at about 0.12 m/s; vx 0.5 veers off.
- The gait is small quick steps: about 4.6 steps/s of about 2.6 cm (vx 0.3 steady state, counted at knee peaks).
- Turning in place (vx = 0) is unreliable; turning while walking (vx 0.3, vyaw ±1.0) is reliable but
  asymmetric: including the coast after stopping, about 39°/s left and 50°/s right.
- Backwards (vx < 0) and sideways (vy) do not walk.
- kick_right barely moves the leg in the sim (the policy's own doing; robotd runs it); kick_left is clear.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

from duck_client import DuckClient, DuckError

WALK_VX = 0.3
TURN_VYAW = 1.0
TURN_DEG_PER_S = {"left": 39.0, "right": 50.0}
DEFAULT_TURN_DEG = 90.0
DEFAULT_WALK_S = 3.0
MIN_WALK_S = 1.0  # any shorter and the gait has not started: it barely moves
MAX_WALK_S = 20.0
WALK_SPEED_MPS = 0.12
STEPS_PER_S = 4.6

# Trunk frame (m): x forward, y left, z up; the floor is at z ≈ -0.12.
LOOK_TARGETS = {
    "left": (0.3, 0.5, 0.1),
    "right": (0.3, -0.5, 0.1),
    "up": (0.4, 0.0, 0.5),
    "down": (0.3, 0.0, -0.12),
    "front": (0.5, 0.0, 0.05),
}

TRICKS = ("roulade", "kick_left", "kick_right", "ground_pick")
SOUNDS = ("greet", "chirp", "coo", "inquire", "alarm")

# How long an action takes (s); the queue waits this long before the next. Skills and sit/stand observed in the sim.
TRICK_S = {"roulade": 3.0, "kick_left": 1.5, "kick_right": 1.5, "ground_pick": 6.0}
SIT_STAND_S = 4.5
LOOK_S = 1.5
SOUND_S = 0.5
SETTLE_S = 0.6  # the coast after walking/turning stops


@dataclass
class Step:
    name: str
    args: dict
    source: str = "user"  # "user" = your command (chat/gesture); "life" = the duck's own idea (your commands interrupt it)


def plan_walk(seconds=None, steps=None, meters=None) -> tuple[float, str]:
    """Steps / metres / seconds → the seconds to actually walk (clamped to what is walkable), with an honest description.

    The conversion lives here, not in the LLM: small models are unreliable at arithmetic (“5 steps” once became 50 s).
    """
    if steps is not None:
        wanted, asked = float(steps) / STEPS_PER_S, f"{float(steps):g} 步"
    elif meters is not None:
        wanted, asked = float(meters) / WALK_SPEED_MPS, f"{float(meters):g} 米"
    elif seconds is not None:
        wanted, asked = float(seconds), f"{float(seconds):g} 秒"
    else:
        wanted, asked = DEFAULT_WALK_S, ""
    actual = max(MIN_WALK_S, min(wanted, MAX_WALK_S))
    words = f"向前走 {actual:.1f} 秒（约 {actual * STEPS_PER_S:.0f} 步、{actual * WALK_SPEED_MPS:.2f} 米）"
    if asked:
        words += f"，主人要的是 {asked}"
        if wanted > MAX_WALK_S:
            words += f"，太远了，一次最多走 {MAX_WALK_S:g} 秒"
    return actual, words


def plan_turn(direction="left", degrees=DEFAULT_TURN_DEG) -> tuple[str, float, float, str]:
    """A turn: side, angle (clamped to 10–720°), seconds to walk, an honest description."""
    side = "right" if str(direction).lower().startswith(("r", "右")) else "left"
    degrees = max(10.0, min(float(degrees), 720.0))
    seconds = min(degrees / TURN_DEG_PER_S[side], MAX_WALK_S)
    return side, degrees, seconds, f"边走边{'左' if side == 'left' else '右'}转约 {degrees:g}°"


COME_STOP_M = 0.6  # stop this close to you
STEER_GAIN = 1.8  # heading error (rad) → turn rate

# Following a route (planned by navmap.py), so the duck neither bumps into walls nor gets lost:
LOOKAHEAD_M = 0.25  # steer at the point this far ahead on the route: further cuts corners into door frames (0.3 left 0.11 m clearance)
ARRIVE_M = 0.25  # wandering to a spot: this close counts as there
PLACE_ARRIVE_M = 0.45  # going to a place (dock, sofa…): this close counts; places sit in corners the turning circle cannot reach, and squeezing in bumps
OFF_PATH_M = 0.3  # replan when this far off the route
REPLAN_EVERY_S = 4.0
STUCK_WINDOW_S = 3.0  # barely any progress (< STUCK_MOVE_M) for this long = stuck
STUCK_MOVE_M = 0.06
MAX_STUCK = 3  # give up after this many stucks in a row and say it cannot get through
MAX_FALLS = 3  # falls on the way: get up and replan on the spot; give up after this many
NAV_SPEED_MPS = 0.12  # measured walking speed, for the timeout
VIEWER_REACH_M = 3.0  # called while you stand outside the walls/house: the nearest walkable spot within this distance

# Pivoting (stepping in place while turning). Measured in the sim (3 s standing between runs, 3 runs each):
#   right vx 0.15, vyaw -1.2: -178° / -180° / -176°, drifting only 0.15 m — reliable
#   left  vx 0.15, vyaw +1.2:  162° /  15° /  155° — sometimes barely turns
# So it is closed-loop: turn while watching the heading, and if a left turn stalls for 1.5 s go the long way right.
PIVOT_VX = 0.15
PIVOT_VYAW = 1.2
PIVOT_DONE_DEG = 15.0
PIVOT_STALL_S = 1.5
PIVOT_STALL_DEG = 8.0
PIVOT_TIMEOUT_S = 9.0
PIVOT_BEFORE_WALK_DEG = 70.0  # pivot first when the way to go is this far off (turning while walking pushes into furniture in tight spots)


class Actions:
    """One action each: send the intent at once, return (what happened, seconds to wait). The text goes back to the LLM as the tool result.

    come_here / go_to / face_me are closed-loop: they need where the duck and “you” are and the house map
    (`world`, provided by life.py), and correct course at 10 Hz on the thread running them until done,
    given up, or interrupted by stop (`should_stop` is injected by ActionQueue).
    """

    def __init__(self, duck: DuckClient):
        self.duck = duck
        self.world = None  # life.Life：where_is_viewer()、duck_pose()、viewer_xy()、nav、show_path()
        self.should_stop = lambda: False

    def come_here(self) -> tuple[str, float]:
        """Walk up to you: with a map, around walls and through doors; on an open floor, straight there."""
        if self.world is None or self.world.where_is_viewer() is None:
            return "不知道主人在哪", 0.0
        if self.world.where_is_viewer()[1] < COME_STOP_M:
            return "已经在主人面前了", 0.0
        # You (the camera) often stand outside the walls or the house: go to the walkable spot nearest you
        return self._navigate(self.world.viewer_xy, stop_within_m=COME_STOP_M, label="走到主人面前",
                              goal_reach_m=VIEWER_REACH_M)

    def go_to(self, place: str) -> tuple[str, float]:
        """Go somewhere in the house (a room or beside furniture; names from navmap's places)."""
        nav = self.world.nav if self.world is not None else None
        if nav is None:
            return "这里没有房子的地图，不知道那是哪", 0.0
        spot = nav.places.get(place)
        if spot is None:
            return f"不知道“{place}”在哪，认识的地方有：{'、'.join(nav.places)}", 0.0
        return self._navigate(lambda: (spot.x, spot.y), stop_within_m=PLACE_ARRIVE_M, label=f"去{place}")

    def wander_to(self, x: float, y: float) -> tuple[str, float]:
        """Wander to a spot on its own (for life.py, not offered to the LLM)."""
        return self._navigate(lambda: (x, y), stop_within_m=ARRIVE_M, label="溜达")

    def face_me(self) -> tuple[str, float]:
        """Pivot to face you (no walking away and circling back, so it works up close)."""
        if self.world is None or self.world.where_is_viewer() is None:
            return "不知道主人在哪", 0.0
        def heading():
            x, y, _ = self.world.duck_pose()
            vx, vy = self.world.viewer_xy()
            return math.atan2(vy - y, vx - x)
        if self._pivot_to(heading):
            return "转过来面朝主人了", SETTLE_S
        return "转身：被打断" if self.should_stop() else "转身：没转过来", SETTLE_S

    def _pivot_to(self, heading_fn, timeout_s: float = PIVOT_TIMEOUT_S) -> bool:
        """Pivot to heading_fn() (rad). If turning left stalls, go round to the right. True once there."""
        started = time.monotonic()
        prefer_right = False
        stall_at, stall_yaw = started, self.world.duck_pose()[2]
        while time.monotonic() - started < timeout_s:
            if self.should_stop():
                self.duck.stop()
                return False
            yaw = self.world.duck_pose()[2]
            error = math.atan2(math.sin(heading_fn() - yaw), math.cos(heading_fn() - yaw))
            if abs(error) < math.radians(PIVOT_DONE_DEG):
                self.duck.stop()
                time.sleep(SETTLE_S)
                return True
            left = error > 0 and not prefer_right
            now = time.monotonic()
            if now - stall_at > PIVOT_STALL_S:
                turned = abs(math.atan2(math.sin(yaw - stall_yaw), math.cos(yaw - stall_yaw)))
                if left and turned < math.radians(PIVOT_STALL_DEG):
                    prefer_right = True  # the left turn stalls: go the long way round to the right
                stall_at, stall_yaw = now, yaw
            self.duck.move(PIVOT_VX, PIVOT_VYAW if left else -PIVOT_VYAW, 0.4)
            time.sleep(0.1)
        self.duck.stop()
        return False

    def _navigate(self, goal_fn, stop_within_m: float, label: str, goal_reach_m: float = 1.0) -> tuple[str, float]:
        """Follow a planned route to near goal_fn(), which is asked afresh each time (you may be moving).

        Without a map (an open floor) it heads straight for the goal. Move intents expire on their own
        (robotd drops them 500 ms after the last), so a stuck thread never keeps the duck walking.
        """
        world, nav = self.world, self.world.nav
        started, path, planned_at = time.monotonic(), None, 0.0
        progress_at, progress_pos, stuck = started, world.duck_pose()[:2], 0
        state_at, falls = 0.0, 0
        try:
            while True:
                now = time.monotonic()
                if self.should_stop():
                    self.duck.stop()
                    return f"{label}：被打断", 0.0
                x, y, yaw = world.duck_pose()
                goal = goal_fn()
                if math.dist((x, y), goal) < stop_within_m:
                    self.duck.stop()
                    return f"{label}：到了", SETTLE_S
                # The goal is beyond a wall: the end of the route (the nearest walkable spot) counts as arrived
                if path and math.dist((x, y), path[-1]) < ARRIVE_M and math.dist(path[-1], goal) > stop_within_m:
                    self.duck.stop()
                    return f"{label}：走到离目标最近的地方了（中间有墙）", SETTLE_S

                # Fallen: get up right here (not via life.py — its stand_up would queue behind this action), then replan
                if now - state_at > 0.5:
                    state_at = now
                    state = self.duck.state()
                    if state.get("safety", {}).get("fallen"):
                        falls += 1
                        self.duck.stop()
                        if falls > MAX_FALLS:
                            return f"{label}：路上摔倒了 {MAX_FALLS} 次，先不走了", 0.0
                        self._get_up_from_fall(state)
                        path, progress_at, progress_pos = None, time.monotonic(), world.duck_pose()[:2]
                        continue

                # Plan / replan: no route, off the route, or every so often (you may have moved)
                if nav is not None and (path is None or now - planned_at > REPLAN_EVERY_S
                                        or _distance_to_path((x, y), path) > OFF_PATH_M):
                    path, planned_at = nav.plan((x, y), goal, goal_reach_m), now
                    if path is None:
                        self.duck.stop()
                        return f"{label}：找不到能走过去的路", 0.0
                    world.show_path(path)
                    budget = 3.0 * _path_length(path) / NAV_SPEED_MPS + 20.0
                    if now - started > budget + 60.0:
                        self.duck.stop()
                        return f"{label}：走了太久还没到，先停下", SETTLE_S

                # Stuck: barely any progress → turn towards open space, then replan from here
                if now - progress_at > STUCK_WINDOW_S:
                    if math.dist((x, y), progress_pos) < STUCK_MOVE_M:
                        stuck += 1
                        if stuck >= MAX_STUCK:
                            self.duck.stop()
                            return f"{label}：卡住了，过不去", 0.0
                        self._unstick(x, y, yaw)
                        path = None
                    else:
                        stuck = 0
                    progress_at, progress_pos = time.monotonic(), world.duck_pose()[:2]
                    continue

                target = _lookahead((x, y), path, LOOKAHEAD_M) if path else goal
                bearing = math.atan2(target[1] - y, target[0] - x) - yaw
                bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                if abs(bearing) > math.radians(PIVOT_BEFORE_WALK_DEG):
                    # The way to go is too far off: pivot first (turning while walking pushes into furniture), then carry on
                    self._pivot_to(lambda: math.atan2(target[1] - world.duck_pose()[1], target[0] - world.duck_pose()[0]))
                    progress_at, progress_pos = time.monotonic(), world.duck_pose()[:2]
                    continue
                self.duck.move(WALK_VX, max(-TURN_VYAW, min(TURN_VYAW, STEER_GAIN * bearing)), 0.4)
                time.sleep(0.1)
        finally:
            world.show_path(None)

    def _unstick(self, x: float, y: float, yaw: float) -> None:
        """When stuck (it cannot back up): pivot towards the most open direction and walk half a second that way."""
        nav = self.world.nav
        if nav is None:
            best = yaw + math.pi / 2
        else:
            def openness(heading: float) -> int:
                return sum(nav.is_free(x + d * math.cos(heading), y + d * math.sin(heading)) for d in (0.15, 0.3, 0.45, 0.6))
            best = max((yaw + math.radians(a) for a in range(-180, 180, 30)), key=openness)
        if self._pivot_to(lambda: best, timeout_s=6.0):
            self.duck.move(WALK_VX, 0.0, 0.8)
            time.sleep(0.8 + SETTLE_S)

    def walk_forward(self, seconds=None, steps=None, meters=None) -> tuple[str, float]:
        actual, words = plan_walk(seconds, steps, meters)
        self.duck.move(WALK_VX, 0.0, actual)
        return words, actual + SETTLE_S

    def turn(self, direction: str = "left", degrees: float = DEFAULT_TURN_DEG) -> tuple[str, float]:
        side, _, seconds, words = plan_turn(direction, degrees)
        self.duck.move(WALK_VX, TURN_VYAW if side == "left" else -TURN_VYAW, seconds)
        return words, seconds + SETTLE_S

    def stop(self) -> tuple[str, float]:
        self.duck.stop()
        return "停下了", 0.0

    def sit_down(self) -> tuple[str, float]:
        if self.duck.state().get("policy") == "sit":
            return "已经坐着了", 0.0
        return self._skill("sit_toggle", "坐下了", SIT_STAND_S)

    def stand_up(self) -> tuple[str, float]:
        state = self.duck.state()
        if state.get("safety", {}).get("fallen"):
            return self._get_up_from_fall(state)
        if state.get("policy") == "sit":
            return self._skill("sit_toggle", "站起来了", SIT_STAND_S)
        return "已经站着了", 0.0

    def _get_up_from_fall(self, state: dict) -> tuple[str, float]:
        """Fallen: tuck into sitting first, then rise from sitting (in the sim: trunk 0.063 → 0.121 m).

        robot.enable does nothing for a duck on the ground (it just lies there). robotd has two ways up: init
        from standing, sit_toggle from sitting — getting up off the floor is the latter (vetbook counts it as a rise).
        """
        if state.get("policy") != "sit":
            self.duck.do("sit_toggle")
            if not self._wait_for(lambda s: s.get("policy") == "sit", 6.0):
                return "摔倒了，想先坐起来但没成功", 0.0
        self.duck.do("sit_toggle")
        if self._wait_for(lambda s: not s.get("safety", {}).get("fallen") and s.get("policy") != "sit", 8.0):
            return "摔倒了，自己爬起来了", SETTLE_S
        return "摔倒了，爬了一下没起来", 0.0

    def _wait_for(self, condition, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.should_stop():
                return False
            if condition(self.duck.state()):
                return True
            time.sleep(0.25)
        return False

    def do_trick(self, name: str) -> tuple[str, float]:
        if name not in TRICKS:
            return f"不会 {name}，只会 {', '.join(TRICKS)}", 0.0
        return self._skill(name, f"做了 {name}" + ("（右脚踢得很轻）" if name == "kick_right" else ""), TRICK_S[name])

    def look(self, direction: str = "front") -> tuple[str, float]:
        target = direction if direction in LOOK_TARGETS else "front"
        if self.world is not None:
            self.world.hold_off_head(3.0)  # look where you asked; for these 3 s “watch you” must not take the head back
        self.duck.look(*LOOK_TARGETS[target])
        return f"看向 {target}", LOOK_S

    def make_sound(self, kind: str = "chirp") -> tuple[str, float]:
        tag = kind if kind in SOUNDS else "chirp"
        self.duck.sound(tag)
        return f"叫了一声（{tag}）", SOUND_S

    def get_status(self) -> tuple[str, float]:
        state = self.duck.state()
        policy = state.get("policy", "?")
        mood = {"stand": "站着", "walk": "在走路", "sit": "坐着"}.get(policy, policy)
        fallen = state.get("safety", {}).get("fallen")
        return f"现在{mood}" + ("，摔倒了" if fallen else ""), 0.0

    def _skill(self, skill: str, done: str, wait: float) -> tuple[str, float]:
        result = self.duck.do(skill)
        if result.get("accepted", True):
            return done, wait
        return f"做不了：{result.get('reason', '被拒绝')}", 0.0

    def run(self, step: Step) -> tuple[str, float]:
        method = getattr(self, step.name, None)
        if method is None or step.name.startswith("_") or step.name in ("run",):
            return f"没有 {step.name} 这个动作", 0.0
        try:
            return method(**step.args)
        except TypeError as error:
            return f"{step.name} 参数不对：{error}", 0.0


class ActionQueue:
    """Runs actions in order, each for its duration before the next. stop empties the queue and halts at once."""

    def __init__(self, actions: Actions):
        self.actions = actions
        self._steps: queue.Queue[Step] = queue.Queue()
        self._generation = 0  # bumped by stop, so a waiting action ends early
        self.current: str | None = None
        self.current_source: str | None = None
        self.log: list[str] = []
        threading.Thread(target=self._worker, name="duck-actions", daemon=True).start()

    def submit(self, steps: list[Step]) -> None:
        # Your command interrupts the duck's own behaviour (wandering, coming over…) instead of queueing behind it
        if any(s.source == "user" for s in steps) and self.current_source == "life":
            self.cancel()
        for step in steps:
            if step.name == "stop":
                self.cancel()
            else:
                self._steps.put(step)

    def cancel(self) -> None:
        self._generation += 1
        while not self._steps.empty():
            try:
                self._steps.get_nowait()
            except queue.Empty:
                break
        self.actions.stop()

    def busy(self) -> bool:
        return self.current is not None or not self._steps.empty()

    def _worker(self) -> None:
        while True:
            step = self._steps.get()
            generation = self._generation
            self.current, self.current_source = step.name, step.source
            self.actions.should_stop = lambda: generation != self._generation
            try:
                result, wait = self.actions.run(step)
            except (DuckError, OSError) as error:
                result, wait = f"出错：{error}", 0.0
            self.log = (self.log + [f"{step.name}: {result}"])[-20:]
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and generation == self._generation:
                time.sleep(0.05)
            self.current = self.current_source = None


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def _closest_on_segment(p, a, b) -> tuple[float, tuple[float, float]]:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    q = (ax + t * dx, ay + t * dy)
    return math.dist(p, q), q


def _distance_to_path(p, path) -> float:
    if len(path) < 2:
        return math.dist(p, path[0]) if path else math.inf
    return min(_closest_on_segment(p, path[i], path[i + 1])[0] for i in range(len(path) - 1))


def _lookahead(p, path, ahead_m: float) -> tuple[float, float]:
    """The point ahead_m further along the route than the point nearest p (up to the end)."""
    if len(path) < 2:
        return path[-1]
    best = min(range(len(path) - 1), key=lambda i: _closest_on_segment(p, path[i], path[i + 1])[0])
    _, point = _closest_on_segment(p, path[best], path[best + 1])
    left, i = ahead_m, best
    while i < len(path) - 1:
        segment = math.dist(point, path[i + 1])
        if segment >= left:
            t = left / segment
            return (point[0] + (path[i + 1][0] - point[0]) * t, point[1] + (path[i + 1][1] - point[1]) * t)
        left -= segment
        point, i = path[i + 1], i + 1
    return path[-1]
