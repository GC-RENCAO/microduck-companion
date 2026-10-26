"""Gesture recognition on synthetic hand features, with a fake clock.

Each case is a stream of browser frames at 20 Hz. The point is the near misses: a raised hand,
petting and waving must not read as “stop”, and a beckon must not need the whole hand to move.
"""

import math
import random

import pytest

from tracker import Tracker

HZ = 20


def hand(**overrides):
    frame = {"kind": "hand", "x": 0.0, "y": 0.0, "size": 0.35, "gesture": "Open_Palm", "score": 0.9,
             "curl": 1.9, "icurl": 2.0, "upright": 1.6}
    frame.update(overrides)
    return frame


def play(runner, clock, seconds, frame_at):
    tracker = Tracker(runner)
    for k in range(int(seconds * HZ)):
        tracker.observe(frame_at(k / HZ))
        clock.advance(1 / HZ)
    return runner.names()


jitter = random.Random(1)


def j(a=0.02):
    return jitter.uniform(-a, a)


def test_police_stop_triggers_once(runner, clock):
    assert play(runner, clock, 4.0, lambda t: hand(x=j(), y=j())) == ["stop"]


def test_stop_needs_one_second(runner, clock):
    assert play(runner, clock, 0.6, lambda t: hand()) == []


def test_far_palm_is_not_stop(runner, clock):
    assert play(runner, clock, 1.5, lambda t: hand(size=0.12)) == []


def test_low_confidence_is_ignored(runner, clock):
    assert play(runner, clock, 1.5, lambda t: hand(score=0.4)) == []
    runner.submitted.clear()
    assert play(runner, clock, 1.5, lambda t: hand(gesture="Thumb_Up", score=0.6, curl=1.1, upright=0.2)) == []


def test_a_real_webcams_stop_palm_triggers(runner, clock):
    # measured: Open_Palm at 0.58–0.60, size 0.18–0.27, extension 1.7, upright 1.8, drift 0.03
    frames = lambda t: hand(score=0.59 + j(0.01), size=0.21 + j(0.03), curl=1.71 + j(0.01), upright=1.82 + j(0.02), x=j(0.015), y=j(0.015))
    assert play(runner, clock, 2.0, frames) == ["stop"]


def test_a_half_closed_upright_hand_is_not_stop(runner, clock):
    assert play(runner, clock, 1.5, lambda t: hand(score=0.59, curl=1.2, upright=1.1)) == []


def test_slow_flat_petting_is_not_stop(runner, clock):
    # very slow up-and-down petting barely moves in any one second; the flat hand gives it away
    assert play(runner, clock, 3.0, lambda t: hand(y=0.12 * math.sin(t * 1.5), upright=0.3)) == []


def test_side_to_side_petting_is_not_stop(runner, clock):
    assert play(runner, clock, 2.5, lambda t: hand(x=0.15 * math.sin(t * 2.5), upright=0.5)) == []


def test_waving_is_nothing(runner, clock):
    assert play(runner, clock, 2.5, lambda t: hand(x=0.4 * math.sin(t * 6), size=0.3)) == []


def test_fidgeting_near_the_camera_is_nothing(runner, clock):
    frames = lambda t: hand(x=j(0.05), curl=1.8 + j(0.08), icurl=1.9 + j(0.08), upright=0.4, gesture=None)
    assert play(runner, clock, 3.0, frames) == []


def test_one_slow_fist_is_not_a_beckon(runner, clock):
    closing = lambda t: hand(curl=1.9 - 0.8 * min(1, t / 1.5), icurl=2.0 - 0.8 * min(1, t / 1.5), upright=0.5, gesture=None)
    assert play(runner, clock, 2.0, closing) == []


def four_finger_beckon(t, drift=0.0):
    curl = 1.55 + 0.35 * math.sin(2 * math.pi * 2 * t)
    return hand(curl=curl, icurl=curl + 0.1, upright=0.5, x=drift * math.sin(t * 4) + j(), y=j(),
                gesture="Open_Palm" if curl > 1.55 else "Closed_Fist")


def test_four_finger_beckon_comes_here(runner, clock):
    assert play(runner, clock, 2.0, four_finger_beckon) == ["make_sound", "come_here"]


def test_index_finger_beckon_comes_here(runner, clock):
    frames = lambda t: hand(icurl=1.65 + 0.35 * math.sin(2 * math.pi * 1.5 * t), curl=1.3 + 0.08 * math.sin(2 * math.pi * 1.5 * t),
                            upright=0.6, gesture="Pointing_Up", x=j(), y=j())
    assert play(runner, clock, 2.2, frames) == ["make_sound", "come_here"]


def test_petting_with_a_little_finger_twitch_is_not_a_beckon(runner, clock):
    # a flat hand stroking back and forth whose extension wobbles at 1 Hz: 4 flips in 2 s, enough for the old threshold
    frames = lambda t: hand(x=0.08 * math.sin(t * 2), upright=0.16, gesture=None,
                            curl=1.85 + 0.3 * math.sin(2 * math.pi * 1.0 * t), icurl=2.0)
    assert play(runner, clock, 4.0, frames) == []


def test_beckoning_while_the_whole_hand_swings_is_nothing(runner, clock):
    assert play(runner, clock, 2.0, lambda t: four_finger_beckon(t, drift=0.5)) == []


@pytest.mark.parametrize("gesture, expected", [("Thumb_Up", "make_sound"), ("Victory", "do_trick")])
def test_other_gestures_need_0_7_seconds(runner, clock, gesture, expected):
    assert play(runner, clock, 0.4, lambda t: hand(gesture=gesture, curl=1.1, upright=0.2)) == []
    runner.submitted.clear()
    assert play(runner, clock, 1.0, lambda t: hand(gesture=gesture, curl=1.1, upright=0.2)) == [expected]


def test_presence_expires(runner, clock):
    tracker = Tracker(runner)
    tracker.observe(hand())
    assert tracker.seen().kind == "hand"
    clock.advance(1.0)
    assert tracker.seen().kind == "none"  # nobody seen for 1 s
    assert tracker.camera_on()  # but frames were arriving 1 s ago: the camera is still on
    clock.advance(1.0)
    assert not tracker.camera_on()


def test_calibration_reports_real_feature_ranges(runner, clock):
    tracker = Tracker(runner)
    for k in range(40):  # 2 s of a beckoning hand
        tracker.observe(four_finger_beckon(k / HZ))
        clock.advance(1 / HZ)
    stats = tracker.calibration()
    assert {"Open_Palm", "Closed_Fist"} <= set(stats)
    assert stats["Open_Palm"]["curl"]["p50"] > stats["Closed_Fist"]["curl"]["p50"]
    assert stats["beckon"]["curl_swing"]["p95"] > stats["beckon"]["thresholds"]["min_swing"]
    tracker.reset_calibration()
    assert set(tracker.calibration()) == {"min_score", "stop_thresholds"}


def test_calibration_keeps_low_confidence_labels(runner, clock):
    tracker = Tracker(runner)
    for _ in range(20):
        tracker.observe(hand(score=0.6))
        clock.advance(1 / HZ)
    palm = tracker.calibration()["Open_Palm"]  # not lumped into "None": the low score is the finding
    assert palm["score"]["p50"] == 0.6 and palm["confident"] == 1  # stop accepts 0.5
