"""Both transports against a fake robotd; the rendezvous one through a fake Space that behaves like the real one."""

import time

import pytest

import transport
from fake_rendezvous import FakeRendezvous
from fake_robot import FakeRobot
from transport import DuckError, RendezvousTransport, UnixTransport


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def robot():
    fake = FakeRobot()
    yield fake
    fake.close()


@pytest.fixture
def hub(robot):
    fake = FakeRendezvous(robot.path, token="secret")
    yield fake
    fake.close()


@pytest.fixture
def remote(hub, monkeypatch):
    monkeypatch.setattr(transport, "RECONNECT_MIN_S", 0.05)
    link = RendezvousTransport(token="secret", base=hub.url)
    assert wait_for(link._connected.is_set), link.error
    yield link
    link.close()


@pytest.fixture(params=["unix", "rendezvous"])
def link(request, robot):
    if request.param == "unix":
        yield UnixTransport(robot.path)
    else:
        yield request.getfixturevalue("remote")


def test_a_call_is_answered(link, robot):
    assert link.request("robot.do", {"skill": "roulade"})["method"] == "robot.do"
    assert robot.calls[-1] == ("robot.do", {"skill": "roulade"})


def test_an_error_is_a_duck_error(link):
    with pytest.raises(DuckError, match="unknown method"):
        link.request("robot.nosuch")


def test_intents_arrive(link, robot):
    link.notify("robot.move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0})
    assert wait_for(lambda: ("robot.move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0}) in robot.intents)


def test_state_frames_stream(link, robot):
    frames = link.subscribe("robot.subscribe", {"hz": 20})
    first = next(frames)
    robot.x = 1.5
    assert wait_for(lambda: next(frames)["odom"]["position"][0] == 1.5)
    assert len(first["joints"]) == 15
    frames.close()


def test_a_float_rate_is_refused_like_robotd_does(link):
    with pytest.raises(DuckError, match="u32"):
        next(link.subscribe("robot.subscribe", {"hz": 3.0}))


@pytest.mark.parametrize("kind", [UnixTransport, RendezvousTransport])
def test_state_rates_are_whole_numbers(kind):
    assert isinstance(kind.state_hz, int)


def test_the_remote_lane_publishes_slower_rates():
    # 3 intent channels × (1 / resend_s) a minute stays well inside the rendezvous's 1200
    assert 3 * 60 / RendezvousTransport.resend_s <= 0.5 * 1200
    assert 60 * RendezvousTransport.state_hz <= 0.5 * 400
    assert RendezvousTransport.resend_s < 0.5  # robotd drops an intent 500 ms after the last one


def test_the_link_reports_round_trip_and_usage(remote):
    remote.request("robot.policies")
    status = remote.describe()
    assert status["connected"] and status["robot"] == "sim-duck"
    assert status["rtt_ms"] is not None and status["requests_last_minute"] >= 1


def test_a_wrong_token_says_so(hub):
    link = RendezvousTransport(token="wrong", base=hub.url)
    assert wait_for(lambda: link.error is not None)
    assert "token" in link.error
    with pytest.raises(DuckError):
        link.request("robot.policies")
    link.close()


def test_a_busy_duck_is_not_taken(hub):
    hub.busy = True
    link = RendezvousTransport(token="secret", base=hub.url)
    assert wait_for(lambda: link.error is not None)
    assert "占用" in link.error
    link.close()


def test_choosing_a_duck_by_name(hub):
    link = RendezvousTransport(token="secret", base=hub.url, robot="other-duck")
    assert wait_for(lambda: link.error is not None)
    assert "other-duck" in link.error and "sim-duck" in link.error
    link.close()


def test_intents_never_spend_the_last_of_the_budget(remote, hub, monkeypatch):
    budget = transport._Budget(20)
    monkeypatch.setattr(remote, "_budget", budget)
    monkeypatch.setattr(transport, "REQUEST_BUDGET", 20)
    sent = 0
    for _ in range(40):
        try:
            remote.notify("robot.head", {"yaw": 0.1})
            sent += 1
        except DuckError:
            pass
    assert sent == 16  # a fifth is kept for calls
    assert remote.request("robot.policies")  # which still go through
    assert hub.rejected_429 == 0


def test_it_reconnects_after_the_stream_drops(remote, hub):
    frames = remote.subscribe("robot.subscribe", {"hz": 20})
    next(frames)
    hub.drop_stream()
    with pytest.raises(DuckError):
        while True:
            next(frames)
    assert wait_for(remote._connected.is_set)
    assert remote.request("robot.policies")["accepted"]
    assert wait_for(lambda: next(remote.subscribe("robot.subscribe", {"hz": 20}))["policy"] == "stand")


def test_the_token_comes_from_hf_login(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert transport.hf_token() is None
    (tmp_path / "token").write_text("hf_abc\n")
    assert transport.hf_token() == "hf_abc"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert transport.hf_token() == "hf_env"
