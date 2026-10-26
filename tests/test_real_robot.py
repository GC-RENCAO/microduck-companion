"""Real-robot mode: the duck drawn from its own robot.state, no house, no roaming; DuckClient over the rendezvous."""

import math
import time

import pytest

import transport
from actions import Step
from conftest import FakeRunner, needs_scenes
from duck_client import DuckClient
from fake_rendezvous import FakeRendezvous
from fake_robot import FakeRobot
from test_transport import wait_for


class StubDuck:
    """Enough of DuckClient for PoseSource and Life: records listeners, answers everything benignly."""

    def __init__(self):
        self.listeners = []

    def on_state(self, listener):
        self.listeners.append(listener)

    def state(self, timeout=3.0):
        return {"policy": "stand", "safety": {"fallen": False}}

    def __getattr__(self, name):
        return lambda *args, **kwargs: {}


@needs_scenes
def test_the_robot_is_drawn_from_its_own_state(monkeypatch):
    import pose
    monkeypatch.setenv("DUCK_SIM_SCENE", "apartment")  # a real duck is not in our apartment
    duck = StubDuck()
    source = pose.PoseSource("robot", duck)
    assert source.scene_file.name == "scene.xml"
    assert not source.connected

    frame = FakeRobot.state(type("R", (), {"x": 0.4})())
    duck.listeners[0](frame)
    assert source.connected and source.frame_id == 1
    assert source.duck_xy == (0.4, 0.5) and math.isclose(source.duck_yaw, math.pi / 2)
    trunk = source.data.qpos[source._trunk : source._trunk + 3]
    assert list(trunk) == pytest.approx([0.4, 0.5, 0.12])

    duck.listeners[0]({"policy": "stand"})  # a robotd predating odometry: ignored, not a crash
    assert source.frame_id == 1


def test_world_must_be_sim_or_robot():
    import pose
    with pytest.raises(ValueError):
        pose.PoseSource("mars", StubDuck())


class Runner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.actions.duck, self.actions.world = StubDuck(), None


def make_life(real, roam=None):
    from life import Life
    from tracker import Tracker
    runner = Runner()
    pose = type("P", (), {"connected": True, "duck_xy": (0.0, 0.0), "duck_yaw": 0.0})()
    life = Life(runner, Tracker(runner), pose, None, real=real, roam=roam)
    return life, runner


def test_a_real_duck_does_not_walk_on_its_own():
    life, runner = make_life(real=True)
    life._do([Step("make_sound", {"kind": "inquire"}), Step("turn", {"direction": "left", "degrees": 90})], None, time.monotonic())
    life._do([Step("come_here", {})], "🐥 凑过来", time.monotonic())
    assert [step.name for step in runner.submitted] == ["make_sound"]


def test_a_real_duck_does_not_know_where_you_are():
    life, _ = make_life(real=True)
    life.set_viewer(1.0, 0.0, 0.3)  # the 3D camera is not a place in the robot's world
    assert life.where_is_viewer() is None
    sim, _ = make_life(real=False)
    sim.set_viewer(1.0, 0.0, 0.3)
    assert sim.where_is_viewer() == pytest.approx((0.0, 1.0))


def test_roaming_can_be_allowed():
    life, runner = make_life(real=True, roam=True)
    life._do([Step("walk_forward", {"seconds": 2})], None, time.monotonic())
    assert [step.name for step in runner.submitted] == ["walk_forward"]


def test_the_duck_client_over_the_rendezvous(monkeypatch):
    robot = FakeRobot()
    hub = FakeRendezvous(robot.path, token="secret")
    monkeypatch.setattr(transport, "RECONNECT_MIN_S", 0.05)
    link = transport.RendezvousTransport(token="secret", base=hub.url)
    try:
        assert wait_for(link._connected.is_set), link.error
        duck = DuckClient(link)
        assert duck.state()["odom"]["position"][1] == 0.5
        duck.move(0.3, 0.0, seconds=1.5)
        for _ in range(30):  # the camera retargets the head 20 times a second
            duck.hold_head({"yaw": 0.1}, 1.0)
            time.sleep(0.05)
        time.sleep(0.2)
        moves = [i for i in robot.intents if i[0] == "robot.move"]
        heads = [i for i in robot.intents if i[0] == "robot.head"]
        assert 3 <= len(moves) <= 8  # ~1.5 s at one per 0.3 s
        assert len(heads) <= 8  # coalesced to the lane's cadence, not 30
        assert duck.request("robot.do", {"skill": "roulade"})["accepted"]
        assert hub.rejected_429 == 0
    finally:
        link.close()
        hub.close()
        robot.close()


def test_an_expression_never_blocks_a_walk():
    robot = FakeRobot()
    try:
        duck = DuckClient(transport.UnixTransport(robot.path))
        duck.move(0.3, 1.0, seconds=1.0)
        duck.hold_pose(pitch=0.2, seconds=0.5)  # life thought it was idle a tick ago
        time.sleep(0.4)
        assert not [p for m, p in robot.intents if m == "robot.pose" and p.get("active")]
        assert any(m == "robot.move" for m, _ in robot.intents)
    finally:
        robot.close()
