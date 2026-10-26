"""The brain against a scripted fake Ollama: what reaches the body, what reaches the mouth.

No model is run. The fake answers the planner (tools, non-streamed) from a script and streams a
fixed reply for the talker, and records every request so the tests can check what the brain sent.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import brain


class FakeOllama:
    def __init__(self):
        self.plans: list[dict] = []  # scripted planner messages, one per planner call
        self.requests: list[dict] = []
        self.reply = ["好", "的～"]
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": brain.MODEL}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(payload)
                self.send_response(200)
                self.end_headers()
                if payload.get("stream"):
                    for piece in fake.reply:
                        self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
                    self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
                else:
                    message = fake.plans.pop(0) if fake.plans else {"content": ""}
                    self.wfile.write(json.dumps({"message": {"role": "assistant", **message}}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def planner_calls(self):
        return [r for r in self.requests if "tools" in r]

    def talker_calls(self):
        return [r for r in self.requests if r.get("stream")]


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


@pytest.fixture
def ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(brain, "URL", fake.url)
    yield fake
    fake.server.shutdown()


def test_actions_reach_the_body_and_impossible_parts_do_not(ollama, runner):
    ollama.plans = [{"content": "", "tool_calls": [call("walk_forward", steps=5), call("cannot_do", what="跳")]}]
    events = list(brain.Brain(runner).think_stream("walk 5 steps then jump"))
    assert [e["type"] for e in events][0] == "act"  # the body moves before the mouth speaks
    assert runner.names() == ["walk_forward"]
    assert runner.submitted[0].args == {"steps": 5}
    talker_note = ollama.talker_calls()[0]["messages"][-1]["content"]
    assert "身体做不到：跳" in talker_note
    assert "英文" in talker_note  # English in, English out
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "好的～"


def test_status_questions_get_a_second_planner_round(ollama, runner):
    ollama.plans = [{"content": "", "tool_calls": [call("get_status")]}, {"content": "好"}]
    thought = brain.Brain(runner).think("你现在站着吗")
    assert len(ollama.planner_calls()) == 2
    assert thought.steps == []
    assert runner.actions.queried[0].name == "get_status"
    system = ollama.talker_calls()[0]["messages"][0]["content"]
    assert "刚查到：现在站着" in system


def test_history_keeps_real_tool_calls(ollama, runner):
    # stored as prose (“reply (actions: …)”), the model imitated the prose and stopped calling tools
    duck = brain.Brain(runner)
    ollama.plans = [{"content": "", "tool_calls": [call("sit_down")]}]
    duck.think("坐下")
    assistant = [m for m in duck.plan_history if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "sit_down"
    assert any(m["role"] == "tool" for m in duck.plan_history)
    assert all("tool_calls" not in m for m in duck.talk_history)


def test_unknown_tools_are_rejected_not_run(ollama, runner):
    ollama.plans = [{"content": "", "tool_calls": [call("fly", height=3)]}]
    thought = brain.Brain(runner).think("fly")
    assert thought.steps == [] and runner.submitted == []
    assert thought.rejected and thought.rejected[0].startswith("fly")


def test_go_to_exists_only_with_a_map(ollama, runner):
    ollama.plans = [{"content": "", "tool_calls": [call("go_to", place="厨房")]}]
    brain.Brain(runner).think("去厨房")
    assert runner.submitted == []  # no map → no go_to tool → rejected
    ollama.plans = [{"content": "", "tool_calls": [call("go_to", place="厨房")]}]
    brain.Brain(runner, places=["厨房", "客厅"]).think("去厨房")
    assert runner.names() == ["go_to"]
    tools = {t["function"]["name"]: t for t in ollama.planner_calls()[-1]["tools"]}
    assert tools["go_to"]["function"]["parameters"]["properties"]["place"]["enum"] == ["厨房", "客厅"]


def test_ollama_down_is_a_brain_error(monkeypatch, runner):
    monkeypatch.setattr(brain, "URL", "http://127.0.0.1:9")
    with pytest.raises(brain.BrainError):
        brain.Brain(runner).think("你好")
