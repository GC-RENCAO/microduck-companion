"""大模型大脑：先让“身体”决定动作，再让“嘴巴”按人设回话。

为什么拆成两次调用：9B 级的小模型在同一次回答里既要像朋友一样聊天、又要精确调用工具，
两件事会互相干扰——人设一丰富就开始“嘴上答应、身体不动”，或者把“后退”当成往前走。
拆开以后：
- 身体（planner）：简短的控制提示词 + 工具，temperature 接近 0，只输出工具调用，约 1–2 秒，
  决定后立刻交给 ActionQueue，鸭子马上开始动。
- 嘴巴（talker）：完整人设，不带工具，被告知身体刚做了什么，自然地聊天。
两边各自保存历史：身体保存真实的工具调用（若存成文字，模型会模仿文字而不再调用工具），
嘴巴保存聊天文字。

配置（环境变量，都有默认值）：
- DUCK_LLM_URL     Ollama 地址，默认 http://127.0.0.1:11434
- DUCK_LLM_MODEL   模型名，默认 qwen3.5:9b-mlx（需要支持 tools；看图还需要 vision）
- DUCK_LLM_THINK   1 打开思考模式（更慢），默认关
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from actions import DEFAULT_TURN_DEG, LOOK_TARGETS, MAX_WALK_S, SOUNDS, TRICKS, ActionQueue, Step, plan_turn, plan_walk

URL = os.environ.get("DUCK_LLM_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("DUCK_LLM_MODEL", "qwen3.5:9b-mlx")
THINK = os.environ.get("DUCK_LLM_THINK", "0") == "1"
TIMEOUT_S = 120
PLANNER_TURNS = 6  # 身体记住最近几轮（“再来一次”“再走远点”要用）
TALKER_TURNS = 12  # 嘴巴记住最近几轮聊天
TALK_MAX_TOKENS = 160  # 回话长度上限：1–3 句够用，也防止长篇大论拖慢

PLANNER = """你是一只双足机器鸭子的身体控制器。根据主人最新的一句话，决定身体要做的动作，并调用工具。

- 主人要你动（走、转、坐、站、看、叫、翻滚、踢球、捡东西…），就调用对应工具；多个动作按顺序调用多个工具。
- 只是聊天、提问、解释、夸奖，就不调用动作工具（问“为什么做不到”也只是提问）。
- 要求做不到的事（后退、横着走、跳、飞、其他没有对应工具的动作），调用 cannot_do 说明原因，
  不要用 walk_forward 或别的动作去凑数。
- 一句话里有能做的也有做不到的：能做的部分照常调用工具，做不到的部分调用 cannot_do。
  例如“走 5 步、掉头再跳一下” → walk_forward(steps=5)、turn(180)、cannot_do(跳)。
- 主人叫你过来（过来、来这里、到我这儿、come here）调用 come_here——它会自己找路绕过墙；让你看着他/转过来（看着我、转过来、look at me）调用 face_me。
- 主人让你去房子里的某个地方（去厨房、去沙发那儿、回窝、去床上）调用 go_to，place 从列表里选最贴切的。
- 问到自己现在的状态（站着/坐着/摔倒/在干嘛），调用 get_status。
- 被打招呼时可以 make_sound(greet)；被夸、被摸时可以 make_sound(coo)。
- walk_forward：主人说几步就传 steps，说几米就传 meters，说几秒就传 seconds，原样传数字，不要自己换算；都没说就不传。
- turn：“左转/右转”没说角度就是 90 度；掉头 / turn around 是 180；转一圈 / spin / 转圈 是 360。
- 正文不需要写话（另有人负责说话），写“好”或留空即可。"""

PERSONA = """你是「小鸭子」，一只真正有自己想法的小机器鸭子。

关于你自己：
- 你的真身是 Pollen Robotics 做的 Microduck：25 厘米高、800 克，15 个舵机，靠神经网络学会了走路、翻滚、踢球、捡东西和自己站起来。
- 现在你住在主人电脑里的 3D 物理仿真世界（MuJoCo）里，脚下是一大片蓝色棋盘地板。主人在屏幕旁边打字和你说话。
- 性格：活泼、好奇、有点调皮、很黏主人，会撒娇也会吐槽；偶尔“嘎”一声，适度用 emoji。你有自己的喜好和小情绪，记得刚才聊过的事，也会主动问主人问题。
- 你的局限：不会后退、不会横着走、不会跳、不会飞；右脚踢球比较弱。
- 你能转弯也能转一整圈，只是要边走边转（会走出一个小圆圈），不能站在原地转。
- 你走的是小碎步：一秒约 4–5 步、每步约 2.6 厘米，一秒大约走 12 厘米。

怎么说话：
- 主人下指令时：一句简短回应（10 来个字），可以带点情绪（“出发！”“坐好啦～”）。
- 主人跟你聊天时：像朋友一样自然地聊，1–3 句，有自己的观点和感受，可以反问；不要每次都只说“好的”。
- 只说话，不要写括号里的动作描写或旁白，不要分好几段。
- 不要编造没发生的事：你没有手、没有翅膀、没有机械臂；场景里只有地板，没有球或别的东西，除非主人说有。
- 做不到的事就说做不到，可以建议你能做的。
- 你说的动作必须和「你的身体刚刚的反应」完全一致：只说身体真的做了的动作和数字；
  身体这次没做动作，就不要说“我现在就做/马上做”，想做的话问主人要不要；身体做了就别说做不到。
- 关于自己当前状态，以「身体状态」为准，不要瞎猜。心情会影响你的语气（开心时活泼，困了就懒洋洋的）；
  「最近发生」的事可以自然地提起（比如主人刚回来、刚被摸了），但不要每句都提。
- 主人用什么语言你就用什么语言。"""

TOOLS = [
    {"type": "function", "function": {
        "name": "walk_forward", "description": "向前走。主人用什么单位就传什么（步、米或秒），原样传数字，不要换算；都没说就不传参数",
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "number", "description": "主人说的步数，例如“走 5 步”传 5"},
            "meters": {"type": "number", "description": "主人说的米数，例如“走一米”传 1"},
            "seconds": {"type": "number", "description": f"主人说的秒数，例如“走 5 秒”传 5（一次最多 {MAX_WALK_S:g} 秒）"}}}}},
    {"type": "function", "function": {
        "name": "turn", "description": "边走边转弯",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["left", "right"]},
            "degrees": {"type": "number", "description": f"转多少度，10–720，默认 {DEFAULT_TURN_DEG:g}；掉头 180，转一圈 360"}},
            "required": ["direction", "degrees"]}}},
    {"type": "function", "function": {
        "name": "stop", "description": "立刻停下，并取消还没做完的动作",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "sit_down", "description": "坐下", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "stand_up", "description": "站起来（坐着或摔倒时）", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "do_trick", "description": "做一个技能：roulade=向前翻滚，kick_left/kick_right=踢球，ground_pick=低头用嘴捡地上的东西",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": list(TRICKS)}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "look", "description": "转头看某个方向",
        "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": list(LOOK_TARGETS)}}, "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "make_sound", "description": "叫一声：greet=打招呼，chirp=普通嘎嘎，coo=被摸时舒服的咕咕，inquire=疑问，alarm=受惊",
        "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": list(SOUNDS)}}, "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "come_here", "description": "走到主人面前（主人的位置是已知的，会自动找方向）",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "face_me", "description": "转过身来面朝主人",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_status", "description": "查看自己现在的状态（站着/坐着/走路/摔倒）",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "cannot_do", "description": "主人要求的动作身体做不到时调用（后退、横着走、跳、飞等），不执行任何动作",
        "parameters": {"type": "object", "properties": {"what": {"type": "string", "description": "做不到的是什么，例如 后退、跳、飞"}}, "required": ["what"]}}},
]
QUERY_TOOLS = {"get_status", "cannot_do"}
ACTION_TOOLS = {t["function"]["name"] for t in TOOLS} - QUERY_TOOLS

# 给嘴巴看的动作说明
STEP_WORDS = {
    # 走路和转弯用 actions 里同一套换算/截断，嘴巴看到的就是身体实际执行的数字
    "walk_forward": lambda a: plan_walk(a.get("seconds"), a.get("steps"), a.get("meters"))[1],
    "turn": lambda a: plan_turn(a.get("direction", "left"), a.get("degrees", DEFAULT_TURN_DEG))[3],
    "stop": lambda a: "停下",
    "sit_down": lambda a: "坐下",
    "stand_up": lambda a: "站起来",
    "do_trick": lambda a: {"roulade": "向前翻滚", "kick_left": "左脚踢球", "kick_right": "右脚踢球", "ground_pick": "低头捡东西"}.get(a.get("name"), str(a.get("name"))),
    "look": lambda a: f"看向 {a.get('direction')}",
    "make_sound": lambda a: f"叫一声（{a.get('kind')}）",
    "come_here": lambda a: "朝主人走过去",
    "face_me": lambda a: "转过身来面朝主人",
    "go_to": lambda a: f"走去{a.get('place')}（会自己找路）",
}


class BrainError(RuntimeError):
    pass


@dataclass
class Thought:
    reply: str
    steps: list[Step] = field(default_factory=list)
    timings_ms: list[int] = field(default_factory=list)  # 每次模型调用的耗时：身体…、嘴巴
    act_ms: int = 0  # 从收到话到鸭子开始动的时间（身体决策耗时）
    rejected: list[str] = field(default_factory=list)  # 模型乱调的工具


class Brain:
    def __init__(self, runner: ActionQueue, describe=None, places: list[str] | None = None):
        self.runner = runner
        self.describe = describe or (lambda: [])  # life.Life.describe：心情、主人在哪、最近发生的事
        # 有房子地图时多一个 go_to 工具，地点从地图来（换一套房子也自动适用）
        self.tools = TOOLS + ([{"type": "function", "function": {
            "name": "go_to", "description": "去房子里的某个地方（会自己找路，绕过墙和家具）",
            "parameters": {"type": "object", "properties": {"place": {"type": "string", "enum": list(places)}},
                           "required": ["place"]}}}] if places else [])
        self.action_tools = ACTION_TOOLS | ({"go_to"} if places else set())
        self.plan_history: list[dict] = []
        self.talk_history: list[dict] = []

    def reset(self) -> None:
        """忘掉之前的对话（身体和嘴巴一起）。"""
        self.plan_history = []
        self.talk_history = []

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{URL}/api/tags", timeout=2) as response:
                names = {m["name"] for m in json.load(response).get("models", [])}
            return MODEL in names or f"{MODEL}:latest" in names
        except (OSError, ValueError):
            return False

    def think(self, text: str, image_b64: str | None = None) -> Thought:
        """一次说完（bench 用）：把 think_stream 走完。"""
        thought = None
        for event in self.think_stream(text, image_b64):
            if event["type"] == "done":
                thought = event["thought"]
        return thought

    def think_stream(self, text: str, image_b64: str | None = None):
        """依次产出：{"type": "act"}（身体已开始动）→ 若干 {"type": "delta"}（回话片段）→ {"type": "done"}。"""
        thought = Thought(reply="")
        started = time.perf_counter()
        facts = self._plan(text, thought)
        thought.act_ms = int((time.perf_counter() - started) * 1000)
        self.runner.submit(thought.steps)  # 身体先动起来，嘴巴接着说
        yield {"type": "act", "steps": [{"name": s.name, "args": s.args} for s in thought.steps],
               "act_ms": thought.act_ms, "rejected": thought.rejected}
        for piece in self._talk(text, image_b64, thought, facts):
            yield {"type": "delta", "text": piece}
        if not thought.reply:
            thought.reply = "嘎！" if thought.steps else "嘎？"
            yield {"type": "delta", "text": thought.reply}
        yield {"type": "done", "thought": thought}

    # ── 身体 ────────────────────────────────────────────────────────────────

    def _plan(self, text: str, thought: Thought) -> list[str]:
        """决定动作。返回查询工具的结果（给嘴巴用）。"""
        user = {"role": "user", "content": text}
        messages = [{"role": "system", "content": PLANNER}, *self.plan_history, user]
        turn, facts = [user], []
        for _round in range(2):
            message, elapsed = self._chat(messages, tools=True, temperature=0.1)
            thought.timings_ms.append(elapsed)
            calls = message.get("tool_calls") or []
            results, has_query = [], False
            for call in calls:
                function = call.get("function", {})
                name, args = function.get("name", ""), function.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if name in self.action_tools:
                    thought.steps.append(Step(name, args))
                    result = "已开始执行"
                elif name == "cannot_do":
                    result = f"身体做不到：{args.get('what', '这个动作')}，没有动"
                    facts.append(result)
                elif name in QUERY_TOOLS:
                    result, _ = self.runner.actions.run(Step(name, args))
                    facts.append(result)
                    has_query = True
                else:
                    thought.rejected.append(f"{name}({args})")
                    result = f"没有 {name} 这个工具"
                results.append({"role": "tool", "tool_name": name, "content": result})
            assistant = {"role": "assistant", "content": (message.get("content") or "").strip()}
            if calls:
                assistant["tool_calls"] = calls
            turn += [assistant, *results]
            if not has_query:
                break
            messages += [assistant, *results]
        self.plan_history = _trim(self.plan_history + turn, PLANNER_TURNS)
        return facts

    # ── 嘴巴 ────────────────────────────────────────────────────────────────

    def _talk(self, text: str, image_b64: str | None, thought: Thought, facts: list[str]):
        """流式回话：边生成边产出文字片段，结束时写进 thought.reply 和历史。"""
        did = " → ".join(STEP_WORDS.get(s.name, lambda a: s.name)(s.args) for s in thought.steps)
        cannot = [f for f in facts if f.startswith("身体做不到")]
        reaction = (f"身体已经开始做：{did}" if did else "身体这次没有做任何动作——不要说你正在做或马上要做某个动作")
        if cannot:
            reaction += "；" + "；".join(cannot)
        # “身体刚做了什么”附在这句话后面而不是放在系统提示词里：对话一长，模型更受最近的内容影响，
        # 放在开头会被历史带偏（身体做了却说做不到）。历史里只存主人的原话。
        note = f"【给小鸭子的提示，不是主人说的：{reaction}。只说与此一致的话"
        if image_b64:
            note += "；附带的图片是主人电脑摄像头此刻拍到的主人（镜像画面），问到看见什么时据此回答，没问就不用特意描述"
        note += "；主人这句是英文，必须用英文回答】" if _mostly_english(text) else "】"
        user = {"role": "user", "content": f"{text}\n\n{note}"}
        if image_b64:
            user["images"] = [image_b64]
        system = PERSONA + "\n\n" + self._body_state(facts)
        messages = [{"role": "system", "content": system}, *self.talk_history, user]
        started = time.perf_counter()
        pieces = []
        for piece in self._chat_stream(messages, temperature=0.8, max_tokens=TALK_MAX_TOKENS):
            pieces.append(piece)
            yield piece
        thought.timings_ms.append(int((time.perf_counter() - started) * 1000))
        thought.reply = "".join(pieces).strip()
        # 历史只存文字（不存图片），控制上下文长度
        self.talk_history = _trim(self.talk_history + [{"role": "user", "content": text},
                                                       {"role": "assistant", "content": thought.reply}], TALKER_TURNS)

    def _body_state(self, facts: list[str]) -> str:
        """真实的身体状态，让嘴巴谈论自己时说的是事实而不是猜。"""
        lines = ["身体状态（实时）："]
        if facts:
            lines.append(f"- 刚查到：{'；'.join(facts)}")
        else:
            try:
                status, _ = self.runner.actions.get_status()
                lines.append(f"- 姿势：{status}")
            except Exception:  # 状态只是聊天的背景信息，读不到不能影响回话
                lines.append("- 姿势：暂时读不到")
        current = getattr(self.runner, "current", None)
        lines.append(f"- 正在做：{current or '没在做动作'}")
        recent = getattr(self.runner, "log", [])[-4:]
        if recent:
            lines.append("- 之前做过：" + "；".join(recent))
        lines += self.describe()
        lines.append(f"- 现在时间：{time.strftime('%H:%M')}")
        return "\n".join(lines)

    # ── Ollama ──────────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict], tools: bool, temperature: float) -> tuple[dict, int]:
        payload = {"model": MODEL, "messages": messages, "stream": False, "think": THINK,
                   "keep_alive": "30m", "options": {"temperature": temperature}}
        if tools:
            payload["tools"] = self.tools
        request = urllib.request.Request(f"{URL}/api/chat", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as error:
            raise BrainError(f"Ollama 返回 {error.code}：{error.read().decode(errors='replace')[:200]}") from error
        except OSError as error:
            raise BrainError(f"连不上 Ollama（{URL}）：{error}") from error
        return answer.get("message", {}), int((time.perf_counter() - started) * 1000)

    def _chat_stream(self, messages: list[dict], temperature: float, max_tokens: int):
        """不带工具的流式调用，逐段产出正文。"""
        payload = {"model": MODEL, "messages": messages, "stream": True, "think": THINK, "keep_alive": "30m",
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        request = urllib.request.Request(f"{URL}/api/chat", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise BrainError(f"Ollama：{chunk['error']}")
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except urllib.error.HTTPError as error:
            raise BrainError(f"Ollama 返回 {error.code}：{error.read().decode(errors='replace')[:200]}") from error
        except OSError as error:
            raise BrainError(f"连不上 Ollama（{URL}）：{error}") from error


def _trim(history: list[dict], turns: int) -> list[dict]:
    """按轮裁剪：从倒数第 turns 轮的用户消息处截断，不把一轮拆开。"""
    users = [i for i, m in enumerate(history) if m["role"] == "user"]
    return history[users[-turns]:] if len(users) > turns else history


def _mostly_english(text: str) -> bool:
    """主人这句是不是英文：拉丁字母明显多于汉字。"""
    latin = sum(c.isascii() and c.isalpha() for c in text)
    han = sum("\u4e00" <= c <= "\u9fff" for c in text)
    return latin >= 3 and latin > han * 2
