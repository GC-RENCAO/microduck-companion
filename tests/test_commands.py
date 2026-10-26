"""Keyword mode: the fallback when no LLM is running."""

import pytest

import commands


@pytest.mark.parametrize("text, step, args", [
    ("往前走5秒", "walk_forward", {"seconds": 5.0}),
    ("左转 45 度", "turn", {"direction": "left", "degrees": 45.0}),
    ("右转", "turn", {"direction": "right", "degrees": 90.0}),
    ("掉头", "turn", {"direction": "left", "degrees": 180}),
    ("坐下", "sit_down", {}),
    ("站起来", "stand_up", {}),
    ("翻滚", "do_trick", {"name": "roulade"}),
    ("踢右脚", "do_trick", {"name": "kick_right"}),
    ("看看左边", "look", {"direction": "left"}),
])
def test_commands_queue_the_right_step(runner, text, step, args):
    reply = commands.handle(runner, text)
    assert reply.ok
    assert runner.submitted[0].name == step
    assert runner.submitted[0].args == args


def test_stop_cancels_instead_of_queueing(runner):
    commands.handle(runner, "停")
    assert runner.cancelled == 1 and not runner.submitted


def test_impossible_requests_are_refused(runner):
    for text in ("后退", "横着走"):
        assert not commands.handle(runner, text).ok
    assert not runner.submitted


def test_unknown_text_asks_back(runner):
    reply = commands.handle(runner, "blah")
    assert not reply.ok
    assert runner.names() == ["make_sound"]
