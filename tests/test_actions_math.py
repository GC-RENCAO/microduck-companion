"""Unit conversion and path-following geometry: no duck, no simulator."""

import math

import pytest

from actions import (MAX_WALK_S, MIN_WALK_S, STEPS_PER_S, TURN_DEG_PER_S, WALK_SPEED_MPS, _distance_to_path,
                     _lookahead, _path_length, plan_turn, plan_walk)


def test_steps_are_converted_by_code_not_by_the_model():
    # “walk 5 steps” was once turned into 50 seconds by the LLM; the conversion lives here now
    seconds, words = plan_walk(steps=5)
    assert seconds == pytest.approx(max(MIN_WALK_S, 5 / STEPS_PER_S))
    assert "5 步" in words


def test_meters_and_seconds():
    assert plan_walk(meters=1)[0] == pytest.approx(1 / WALK_SPEED_MPS)
    assert plan_walk(seconds=4)[0] == 4


def test_walks_are_clamped_and_say_so():
    seconds, words = plan_walk(seconds=50)
    assert seconds == MAX_WALK_S
    assert "最多" in words
    assert plan_walk(seconds=0.1)[0] == MIN_WALK_S


def test_turn_direction_and_duration():
    side, degrees, seconds, _ = plan_turn("右", 90)
    assert side == "right" and degrees == 90
    assert seconds == pytest.approx(90 / TURN_DEG_PER_S["right"])
    assert plan_turn("left", 5000)[1] == 720  # clamped
    assert plan_turn("front", 90)[0] == "left"  # an invalid side from the LLM falls back to left


def test_path_helpers():
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert _path_length(path) == pytest.approx(2.0)
    assert _distance_to_path((0.5, 0.3), path) == pytest.approx(0.3)
    # lookahead walks along the path from the closest point, round the corner
    x, y = _lookahead((0.9, 0.0), path, 0.3)
    assert (x, y) == pytest.approx((1.0, 0.2))
    assert _lookahead((1.0, 0.95), path, 0.3) == path[-1]  # never past the end
