"""关键词模式：把一句话（中文或英文）解析成动作。不需要大模型，响应即时。

大模型不可用时的后备，也是 DUCK_BRAIN=rules 时的唯一模式。动作本身在 actions.py。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from actions import DEFAULT_TURN_DEG, DEFAULT_WALK_S, MAX_WALK_S, ActionQueue, Step
from duck_client import DuckError


@dataclass
class Reply:
    text: str
    action: str = ""  # 给页面显示用的动作标签
    ok: bool = True


HELP = """我听得懂这些（中文英文都行）：
• 走路：前进 / 走5秒 / forward
• 转弯：左转 / 右转 / 掉头 / 转圈
• 停下：停 / stop
• 姿势：坐下 / 站起来
• 技能：翻滚 / 踢左脚 / 踢右脚 / 捡东西
• 看：看左边 / 看右边 / 抬头 / 低头 / 看前面
• 叫：叫一声 / 你好 / 摸摸
• 状态：状态 / status"""


def _seconds(text: str, default: float) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(秒|s\b|sec)?", text)
    if not match:
        return default
    return max(0.5, min(float(match.group(1)), MAX_WALK_S))


def _degrees(text: str, default: float) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(度|°|deg)", text)
    return max(10.0, min(float(match.group(1)), 720.0)) if match else default


def _has(text: str, *words: str) -> bool:
    return any(word in text for word in words)


def handle(runner: ActionQueue, raw: str) -> Reply:
    text = raw.strip().lower()
    if not text:
        return Reply("嘎？你想说什么？", ok=False)
    try:
        return _dispatch(runner, text)
    except DuckError as error:
        return Reply(f"出错了：{error}", ok=False)
    except OSError as error:
        return Reply(f"连不上鸭子：{error}。先在 microduck 目录运行 scripts/duck-sim", ok=False)


def _queue(runner: ActionQueue, text: str, action: str, *steps: Step) -> Reply:
    runner.submit(list(steps))
    return Reply(text, action)


def _dispatch(runner: ActionQueue, text: str) -> Reply:
    if _has(text, "帮助", "help", "怎么用", "会什么", "能做什么"):
        return Reply(HELP, "help")

    if text in ("s", "停") or _has(text, "停", "stop", "别动", "站住"):
        runner.cancel()
        return Reply("好，停下了。", "stop")

    if _has(text, "状态", "status", "怎么样"):
        result, _ = runner.actions.get_status()
        return Reply(f"我{result}。", "status")

    if _has(text, "后退", "倒退", "往后", "back"):
        return Reply("我还不会倒着走（策略没学过），可以说「掉头」再「前进」。", "back", ok=False)

    if _has(text, "侧移", "横着", "螃蟹"):
        return Reply("我还不会横着走。", "side", ok=False)

    if _has(text, "踢", "kick"):
        if _has(text, "右", "right"):
            return _queue(runner, "右脚踢一下（我的右脚踢得比较轻）⚽", "kick_right", Step("do_trick", {"name": "kick_right"}))
        return _queue(runner, "看我左脚一踢！⚽", "kick_left", Step("do_trick", {"name": "kick_left"}))

    if _has(text, "翻滚", "打滚", "滚", "roll"):
        return _queue(runner, "翻个跟头！🤸", "roulade", Step("do_trick", {"name": "roulade"}))

    if _has(text, "捡", "pick", "啄"):
        return _queue(runner, "低头捡一下～", "ground_pick", Step("do_trick", {"name": "ground_pick"}))

    if _has(text, "坐", "sit"):
        return _queue(runner, "坐下休息一下。", "sit", Step("sit_down", {}))

    if _has(text, "站", "起来", "stand", "get up"):
        return _queue(runner, "站起来啦！", "stand", Step("stand_up", {}))

    if _has(text, "掉头", "转身", "回头走", "turn around"):
        return _queue(runner, "掉个头～", "turn_around", Step("turn", {"direction": "left", "degrees": 180}))

    if _has(text, "转圈", "转一圈", "circle"):
        return _queue(runner, "转一圈给你看！", "circle", Step("turn", {"direction": "left", "degrees": 360}))

    for side, words, single in (("left", ("左转", "向左", "往左", "turn left"), ("a", "left")),
                                ("right", ("右转", "向右", "往右", "turn right"), ("d", "right"))):
        if text in single or _has(text, *words):
            degrees = _degrees(text, DEFAULT_TURN_DEG)
            label = "左" if side == "left" else "右"
            return _queue(runner, f"往{label}拐 {degrees:g}°。", f"turn_{side}", Step("turn", {"direction": side, "degrees": degrees}))

    look = None
    if _has(text, "看"):
        look = ("left" if _has(text, "左") else "right" if _has(text, "右") else
                "up" if _has(text, "上", "天") else "down" if _has(text, "下", "地") else "front")
    elif _has(text, "抬头", "look up"):
        look = "up"
    elif _has(text, "低头", "look down"):
        look = "down"
    if look:
        words = {"left": "看看左边。", "right": "看看右边。", "up": "抬头看看。", "down": "低头看看。", "front": "看着前面。"}
        return _queue(runner, words[look], f"look_{look}", Step("look", {"direction": look}))

    if text in ("w",) or _has(text, "前进", "往前", "向前", "走", "forward", "walk", "go", "过来"):
        seconds = _seconds(text, DEFAULT_WALK_S)
        return _queue(runner, f"向前走 {seconds:g} 秒。", "walk", Step("walk_forward", {"seconds": seconds}))

    if _has(text, "你好", "hello", "hi", "嗨", "早"):
        return _queue(runner, "嘎嘎！你好呀 👋", "greet", Step("make_sound", {"kind": "greet"}), Step("look", {"direction": "front"}))

    if _has(text, "摸", "乖", "可爱", "pet"):
        return _queue(runner, "咕～好舒服。", "coo", Step("make_sound", {"kind": "coo"}))

    if _has(text, "叫", "quack", "嘎", "唱"):
        return _queue(runner, "嘎！🦆", "chirp", Step("make_sound", {"kind": "chirp"}))

    runner.submit([Step("make_sound", {"kind": "inquire"})])
    return Reply("嘎？我没听懂。输入「帮助」看看我会什么。", "unknown", ok=False)
