"""大模型测速 + 工具调用准确率。只测大脑，不让鸭子动（动作被记录、不执行）。

    DUCK_LLM_MODEL=qwen3.5:9b-mlx ~/Pollen/microduck_rl/.venv/bin/python bench.py
    DUCK_SIM_SCENE=apartment ...                     # 有房子地图时，多测 go_to（地点从地图来）

每条用例：期望按顺序出现的工具（可以夹杂别的小动作），以及绝不能出现的工具。
"""

from __future__ import annotations

import os
import statistics
import sys

import brain
from actions import Step, plan_walk


def walked(a):
    """walk_forward 实际会走的秒数（和身体同一套换算）。"""
    return plan_walk(a.get("seconds"), a.get("steps"), a.get("meters"))[0]

# (输入, 期望的工具顺序[(名字, 参数检查)], 禁止的工具)
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
    # “停”不经过大模型（server.py 的快速通道），所以不在这里测
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
    ("那你表演一下给我看", [], set()),  # 接上一句，看它能不能接住话题（不强制具体动作）
    # 用户反馈：能做的部分要做，做不到的（跳）不能拖累整句
    ("can you walk in 5 steps and turn around jump", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5), ("turn", lambda a: 150 <= float(a.get("degrees", 0)) <= 210)], {"do_trick"}),
    ("why u can't do this", [], MOVES),  # 只是提问，不该动
    ("walk in 5 steps", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5)], set()),  # 曾被算成 50 秒
    ("走5步", [("walk_forward", lambda a: 0.9 <= walked(a) <= 1.5)], set()),
    ("spin around", [("turn", lambda a: 300 <= float(a.get("degrees", 0)) <= 420)], set()),
    ("过来", [("come_here", None)], {"walk_forward"}),
    ("看着我", [("face_me", None)], {"walk_forward", "turn"}),
    ("come here buddy", [("come_here", None)], {"walk_forward"}),
]
# 有房子地图时（DUCK_SIM_SCENE=apartment）才测的：地点名来自地图
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
        skip = {trunk} | {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
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
    started_cold = duck_brain.think("嘎")  # 预热：第一次要把模型载入内存
    print(f"冷启动（载入模型）：{sum(started_cold.timings_ms) / 1000:.1f} 秒\n")

    # 两种模式：每句独立（清空历史），和连续对话（历史一直累积——历史格式出错时模型会学坏）
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
        # 查询工具（get_status）当场执行、不进动作队列，所以从假动作层的记录里补回来
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
