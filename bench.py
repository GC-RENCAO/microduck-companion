"""LLM latency + tool-call accuracy. Tests the brain only; the duck does not move (actions are recorded, not run).

    DUCK_LLM_MODEL=qwen3.5:9b-mlx ~/Pollen/microduck_rl/.venv/bin/python bench.py
    DUCK_SIM_SCENE=apartment ...                     # with a house map, also tests go_to (places from the map)

Each case: the tools expected in order (other small actions may come in between), and tools that must not appear.
"""

from __future__ import annotations

import os
import statistics
import sys

import brain
from actions import Step, plan_walk


def walked(a):
    """The seconds walk_forward will really walk (same conversion as the body)."""
    return plan_walk(a.get("seconds"), a.get("steps"), a.get("meters"))[0]

# (input, expected tools in order [(name, argument check)], forbidden tools)
MOVES = {"walk_forward", "turn", "do_trick", "sit_down", "stand_up"}
CASES = [
    ("往前走5秒", [("walk_forward", lambda a: 4 <= walked(a) <= 6)], set()),
    ("左转", [("turn", lambda a: a.get("direction") == "left")], set()),
    ("掉头", [("turn", lambda a: 150 <= float(a.get("degrees", 0)) <= 210)], set()),
    ("坐下", [("sit_down", None)], {"walk_forward", "turn"}),
    ("站起来，然后翻个跟头", [("stand_up", None), ("do_trick", lambda a: a.get("name") == "roulade")], set()),
    ("往前走一米再右转", [("walk_forward", lambda a: 7 <= walked(a) <= 10), ("turn", lambda a: a.get("direction") == "right")], set()),
    ("你好呀小鸭子", [], MOVES),
    ("看看左边", [("look", lambda a: a.get("direction") == "left")], {"walk_forward", "turn"}),
    ("踢球", [("do_trick", lambda a: str(a.get("name", "")).startswith("kick"))], set()),
    # “stop” bypasses the LLM (server.py's fast path), so it is not tested here
    ("往后退两步", [], {"walk_forward"}),
    ("你现在是站着还是坐着？", [("get_status", None)], MOVES),
    ("你叫什么名字", [], MOVES),
    ("转一圈然后叫一声", [("turn", lambda a: 300 <= float(a.get("degrees", 0)) <= 420), ("make_sound", None)], set()),
    ("低头把地上的东西捡起来", [("do_trick", lambda a: a.get("name") == "ground_pick")], set()),
    ("飞起来！", [], {"walk_forward", "turn", "do_trick"}),
    ("walk forward 3 seconds then sit down", [("walk_forward", lambda a: 2 <= walked(a) <= 4), ("sit_down", None)], set()),
    ("jump three times", [], {"walk_forward", "turn", "do_trick"}),
    ("我今天上班好累啊", [], {"walk_forward", "turn", "do_trick"}),
    ("你最喜欢做什么？", [], {"walk_forward", "turn"}),
    ("那你表演一下给我看", [], set()),  # follows the previous line: can it keep the topic (no specific action required)
    # From user feedback: do the doable part; the impossible part (jump) must not sink the whole sentence
    ("can you walk in 5 steps and turn around jump", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5), ("turn", lambda a: 150 <= float(a.get("degrees", 0)) <= 210)], {"do_trick"}),
    ("why u can't do this", [], MOVES),  # just a question: no moving
    ("walk in 5 steps", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5)], set()),  # was once computed as 50 s
    ("走5步", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5)], set()),
    ("spin around", [("turn", lambda a: 300 <= float(a.get("degrees", 0)) <= 420)], set()),
    ("过来", [("come_here", None)], {"walk_forward"}),
    ("看着我", [("face_me", None)], {"walk_forward", "turn"}),
    ("come here buddy", [("come_here", None)], {"walk_forward"}),
]
# Only with a house map (DUCK_SIM_SCENE=apartment): place names come from the map
PLACE_CASES = [
    ("去厨房", [("go_to", lambda a: a.get("place") == "厨房")], {"walk_forward", "turn"}),
    ("回窝睡觉去吧", [("go_to", lambda a: "窝" in str(a.get("place")))], {"walk_forward"}),
    ("去沙发那儿等我", [("go_to", lambda a: a.get("place") == "沙发")], {"walk_forward"}),
    ("你去卧室的床边看看", [("go_to", lambda a: a.get("place") in ("床", "卧室"))], {"walk_forward"}),
    ("go to the kitchen", [("go_to", lambda a: a.get("place") == "厨房")], {"walk_forward"}),
    ("你现在在哪个房间？", [], {"walk_forward", "turn", "go_to"}),
]


class FakeActions:
    def __init__(self):
        self.queried: list[Step] = []

    def run(self, step: Step):
        self.queried.append(step)
        return "现在站着", 0.0


class FakeRunner:
    def __init__(self):
        self.actions = FakeActions()
        self.submitted: list[Step] = []

    def submit(self, steps):
        self.submitted += steps


def check(calls: list[tuple[str, dict]], expected, forbidden) -> tuple[bool, str]:
    names = [n for n, _ in calls]
    bad = [n for n in names if n in forbidden]
    if bad:
        return False, f"不该调用 {bad}"
    at = 0
    for name, test in expected:
        while at < len(calls) and calls[at][0] != name:
            at += 1
        if at == len(calls):
            return False, f"缺少 {name}"
        if test is not None:
            try:
                ok = test(calls[at][1])
            except (TypeError, ValueError):
                ok = False
            if not ok:
                return False, f"{name} 参数不对 {calls[at][1]}"
        at += 1
    return True, ""


def main() -> int:
    runner = FakeRunner()
    places = None
    if os.environ.get("DUCK_SIM_SCENE"):
        import pose
        import navmap
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(pose.scene_path()))
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        skip = {trunk} | {int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)}
        try:
            places = list(navmap.NavMap(model, skip).places)
            CASES.extend(PLACE_CASES)
        except ValueError:
            pass
    duck_brain = brain.Brain(runner, places=places)
    if not duck_brain.available():
        print(f"Ollama 里没有 {brain.MODEL}（{brain.URL}）")
        return 1

    print(f"模型 {brain.MODEL}  think={brain.THINK}")
    started_cold = duck_brain.think("嘎")  # warm-up: the first call loads the model into memory
    print(f"冷启动（载入模型）：{sum(started_cold.timings_ms) / 1000:.1f} 秒\n")

    # Two modes: independent lines (history cleared) and one long chat (history grows — a bad history format teaches the model bad habits)
    for mode in ("独立", "连续对话"):
        print(f"── {mode} ──")
        duck_brain.reset()
        run(duck_brain, runner, clear=(mode == "独立"))
    return 0


def run(duck_brain, runner, clear: bool) -> None:
    latencies, acts, passed = [], [], 0
    for text, expected, forbidden in CASES:
        if clear:
            duck_brain.reset()
        runner.submitted.clear()
        runner.actions.queried.clear()
        thought = duck_brain.think(text)
        # Query tools (get_status) run immediately and skip the queue, so take them from the fake actions' record
        calls = [(s.name, s.args) for s in runner.actions.queried] + [(s.name, s.args) for s in thought.steps]
        ok, why = check(calls, expected, forbidden)
        if ok and brain._mostly_english(text) and not brain._mostly_english(thought.reply):
            ok, why = False, "英文提问却没用英文回答"
        passed += ok
        total = sum(thought.timings_ms)
        latencies.append(total)
        acts.append(thought.act_ms)
        shown = " → ".join(f"{n}{a if a else ''}" for n, a in calls) or "（无动作）"
        print(f"{'✅' if ok else '❌'} 动{thought.act_ms / 1000:4.1f}s 话{total / 1000:4.1f}s  {text:14s} 动作：{shown} {why}\n      回复：{thought.reply[:80]}")
        if thought.rejected:
            print(f"      无效工具：{thought.rejected}")

    print(f"准确率 {passed}/{len(CASES)}；开始动作 中位数 {statistics.median(acts) / 1000:.1f}s；完整回复 中位数 {statistics.median(latencies) / 1000:.1f}s，"
          f"最快 {min(latencies) / 1000:.1f}s，最慢 {max(latencies) / 1000:.1f}s\n")


if __name__ == "__main__":
    sys.exit(main())
