"""鸭子会做的动作，以及按顺序执行它们的队列。关键词模式和大模型工具调用共用这一层。

速度和时长都是在仿真里实测出来的（alpha 策略）：
- vx 0.15 原地踏步；vx 0.3 稳定前进，稳态约 0.12 m/s；vx 0.5 会明显跑偏。
- 步态是小碎步：约 4.6 步/秒，每步约 2.6 cm（vx 0.3 稳态，膝关节峰值计数）。
- 原地转（vx=0）方向不稳定；边走边转（vx 0.3, vyaw ±1.0）可靠，但左右不对称：
  含停下后的惯性，左转约 39°/s，右转约 50°/s。
- 后退（vx<0）和侧移（vy）都走不动。
- kick_right 在仿真里几乎不动腿（策略本身如此，robotd 有执行）；kick_left 动作明显。
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
MIN_WALK_S = 1.0  # 再短的话步态还没起来，几乎不动
MAX_WALK_S = 20.0
WALK_SPEED_MPS = 0.12
STEPS_PER_S = 4.6

# 躯干坐标系（米）：x 前，y 左，z 上；地面在 z≈-0.12。
LOOK_TARGETS = {
    "left": (0.3, 0.5, 0.1),
    "right": (0.3, -0.5, 0.1),
    "up": (0.4, 0.0, 0.5),
    "down": (0.3, 0.0, -0.12),
    "front": (0.5, 0.0, 0.05),
}

TRICKS = ("roulade", "kick_left", "kick_right", "ground_pick")
SOUNDS = ("greet", "chirp", "coo", "inquire", "alarm")

# 一个动作做完要多久（秒），队列据此等待再做下一个。技能和坐/站来自仿真观察。
TRICK_S = {"roulade": 3.0, "kick_left": 1.5, "kick_right": 1.5, "ground_pick": 6.0}
SIT_STAND_S = 4.5
LOOK_S = 1.5
SOUND_S = 0.5
SETTLE_S = 0.6  # 走路/转弯停下后的惯性


@dataclass
class Step:
    name: str
    args: dict
    source: str = "user"  # "user" = 主人的指令（聊天/手势）；"life" = 鸭子自己想做的（会被主人的指令打断）


def plan_walk(seconds=None, steps=None, meters=None) -> tuple[float, str]:
    """把“几步 / 几米 / 几秒”换算成实际要走的秒数（截断到能走的范围），并给出如实的描述。

    单位换算在这里做，不交给大模型：小模型做算术不可靠（曾把“5 步”算成 50 秒）。
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
    """转弯：方向、角度（截断到 10–720°）、要走的秒数、如实的描述。"""
    side = "right" if str(direction).lower().startswith(("r", "右")) else "left"
    degrees = max(10.0, min(float(degrees), 720.0))
    seconds = min(degrees / TURN_DEG_PER_S[side], MAX_WALK_S)
    return side, degrees, seconds, f"边走边{'左' if side == 'left' else '右'}转约 {degrees:g}°"


COME_STOP_M = 0.6  # 走到离你这么近就停
STEER_GAIN = 1.8  # 朝向误差（弧度）→ 转向速度

# 沿路线走（navmap.py 规划），目标是“不撞墙、不迷路”：
LOOKAHEAD_M = 0.25  # 盯着路线上前方这么远的点走：太远拐弯会切内角蹭到门框（0.3 时实测离障碍只剩 0.11 米）
ARRIVE_M = 0.25  # 溜达到一个点：离它这么近算到了
PLACE_ARRIVE_M = 0.45  # 去某个地点（窝、沙发……）：这么近就算到了；地点常在角落，转弯半径进不去，硬挤容易撞
OFF_PATH_M = 0.3  # 偏离路线这么远就重新规划
REPLAN_EVERY_S = 4.0
STUCK_WINDOW_S = 3.0  # 这么久几乎没前进（< STUCK_MOVE_M）算卡住
STUCK_MOVE_M = 0.06
MAX_STUCK = 3  # 连续卡住几次就放弃，告诉主人过不去
MAX_FALLS = 3  # 路上摔倒：当场爬起来、重新规划；摔这么多次就放弃
NAV_SPEED_MPS = 0.12  # 实测行走速度，用来估算超时
VIEWER_REACH_M = 3.0  # 叫它过来时，主人在墙外/房子外面：在这么远内找离主人最近的能走的点

# 原地转身（踏步 + 转向）。仿真实测（每组之间先站稳 3 秒，各测 3 次）：
#   右转 vx 0.15, vyaw -1.2：-178° / -180° / -176°，身体只漂 0.15 米 —— 可靠
#   左转 vx 0.15, vyaw +1.2：162° / 15° / 155° —— 有时几乎不转
# 所以闭环：盯着朝向转，左转 1.5 秒没动静就改成往右绕大圈。
PIVOT_VX = 0.15
PIVOT_VYAW = 1.2
PIVOT_DONE_DEG = 15.0
PIVOT_STALL_S = 1.5
PIVOT_STALL_DEG = 8.0
PIVOT_TIMEOUT_S = 9.0
PIVOT_BEFORE_WALK_DEG = 70.0  # 要去的方向偏这么多，先原地转过去再走（窄地方边走边转会顶到家具）


class Actions:
    """单个动作：立即发出意图，返回 (结果说明, 需要等待的秒数)。说明会作为工具结果交给大模型。

    come_here / go_to / face_me 是闭环动作：需要知道鸭子和“你”在哪、房子的地图（world，由 life.py 提供），
    在执行它的线程里以 10 Hz 边看边修正，直到做到、放弃，或被 stop 打断（should_stop 由 ActionQueue 注入）。
    """

    def __init__(self, duck: DuckClient):
        self.duck = duck
        self.world = None  # life.Life：where_is_viewer()、duck_pose()、viewer_xy()、nav、show_path()
        self.should_stop = lambda: False

    def come_here(self) -> tuple[str, float]:
        """走到主人面前：有地图就找路绕过墙、穿过门；没墙的空地就直接走过去。"""
        if self.world is None or self.world.where_is_viewer() is None:
            return "不知道主人在哪", 0.0
        if self.world.where_is_viewer()[1] < COME_STOP_M:
            return "已经在主人面前了", 0.0
        # 主人（相机）常常站在墙外、房子外面：那就走到离主人最近的能走的地方
        return self._navigate(self.world.viewer_xy, stop_within_m=COME_STOP_M, label="走到主人面前",
                              goal_reach_m=VIEWER_REACH_M)

    def go_to(self, place: str) -> tuple[str, float]:
        """去房子里的某个地方（房间或家具旁边，名字见 navmap 的 places）。"""
        nav = self.world.nav if self.world is not None else None
        if nav is None:
            return "这里没有房子的地图，不知道那是哪", 0.0
        spot = nav.places.get(place)
        if spot is None:
            return f"不知道“{place}”在哪，认识的地方有：{'、'.join(nav.places)}", 0.0
        return self._navigate(lambda: (spot.x, spot.y), stop_within_m=PLACE_ARRIVE_M, label=f"去{place}")

    def wander_to(self, x: float, y: float) -> tuple[str, float]:
        """自己溜达到某个点（life.py 用，不给大模型）。"""
        return self._navigate(lambda: (x, y), stop_within_m=ARRIVE_M, label="溜达")

    def face_me(self) -> tuple[str, float]:
        """原地转过来面朝主人（不用先走开再绕回来，离得很近也行）。"""
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
        """原地转到 heading_fn() 的朝向（弧度）。左转转不动就改往右绕。转到了返回 True。"""
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
                    prefer_right = True  # 左转转不动：改往右绕一大圈
                stall_at, stall_yaw = now, yaw
            self.duck.move(PIVOT_VX, PIVOT_VYAW if left else -PIVOT_VYAW, 0.4)
            time.sleep(0.1)
        self.duck.stop()
        return False

    def _navigate(self, goal_fn, stop_within_m: float, label: str, goal_reach_m: float = 1.0) -> tuple[str, float]:
        """沿规划好的路线走到 goal_fn() 附近。goal_fn 每次调用都重新取目标（主人可能在走动）。

        没地图（开阔空地）就直接朝目标走。0.4 秒自动失效的 move 意图保证线程卡住也不会一直走。
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
                # 目标在墙外：走到路线终点（离目标最近的能走的点）也算到了
                if path and math.dist((x, y), path[-1]) < ARRIVE_M and math.dist(path[-1], goal) > stop_within_m:
                    self.duck.stop()
                    return f"{label}：走到离目标最近的地方了（中间有墙）", SETTLE_S

                # 摔倒了：当场爬起来（不能等 life.py——它的“站起来”会排在这个动作后面），再重新规划
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

                # 规划 / 重新规划：没路线、偏离路线、隔一段时间（主人可能走动了）
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

                # 卡住了：几乎没前进 → 朝空的方向转一下，再从当前位置重新规划
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
                    # 要去的方向偏太多：先原地转过去（窄地方边走边转会顶到家具），再接着走
                    self._pivot_to(lambda: math.atan2(target[1] - world.duck_pose()[1], target[0] - world.duck_pose()[0]))
                    progress_at, progress_pos = time.monotonic(), world.duck_pose()[:2]
                    continue
                self.duck.move(WALK_VX, max(-TURN_VYAW, min(TURN_VYAW, STEER_GAIN * bearing)), 0.4)
                time.sleep(0.1)
        finally:
            world.show_path(None)

    def _unstick(self, x: float, y: float, yaw: float) -> None:
        """卡住时（不会后退）：原地转向最空的方向，再往那边走半秒挪开。"""
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
        """摔倒了：先收进坐姿，再从坐姿起身（仿真实测：躯干 0.063 → 0.121 米）。

        robot.enable 对倒地的鸭子没用（实测躺着不动）；robotd 里“站起来”分两种：站着的用 init，
        坐着的用 sit_toggle，从地上爬起来走的是后者（vetbook 把它记作一次 rise）。
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
            self.world.hold_off_head(3.0)  # 主人让看哪就看哪，这 3 秒别被“盯着主人”抢回去
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
    """按顺序执行动作：每个做完（等它的时长）再做下一个。stop 清空队列并立即停下。"""

    def __init__(self, actions: Actions):
        self.actions = actions
        self._steps: queue.Queue[Step] = queue.Queue()
        self._generation = 0  # stop 时递增，正在等待的动作据此提前结束
        self.current: str | None = None
        self.current_source: str | None = None
        self.log: list[str] = []
        threading.Thread(target=self._worker, name="duck-actions", daemon=True).start()

    def submit(self, steps: list[Step]) -> None:
        # 主人的指令打断鸭子自己的行为（溜达、凑过来…），不排在它们后面干等
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
    """路线上离 p 最近的点再往前 ahead_m 的点（到终点为止）。"""
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
