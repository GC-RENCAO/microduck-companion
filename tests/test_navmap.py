"""The map, rooms, places and A* on the official apartment scene (needs a microduck_rl checkout)."""

import math
import time

import pytest

from conftest import MICRODUCK_RL, needs_scenes

pytestmark = needs_scenes


@pytest.fixture(scope="module")
def apartment():
    import mujoco
    import navmap
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_RL / "src/mjlab_microduck/robot/microduck/scene_apartment.xml"))
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    loose = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)}
    return navmap.NavMap(model, {trunk} | loose)


def test_the_stairwell_is_not_walkable(apartment):
    assert not apartment.is_free(-0.2, -1.0)  # a hole in the corridor floor; a ball fell in once
    assert apartment.is_free(-0.25, 0.0)


@pytest.mark.parametrize("door", [(-2.6, 0.5), (-1.0, -1.4), (-1.0, 2.0), (0.5, 0.2), (0.5, 1.8), (0.5, -2.1)])
def test_every_doorway_is_open(apartment, door):
    assert apartment.is_free(*door)


def test_rooms_and_places(apartment):
    assert {label for label, _ in apartment.rooms} == {"厨房", "客厅", "书房", "卧室", "浴室", "走廊", "走廊南头"}
    assert {"窝（充电座）", "沙发", "床", "冰箱"} <= set(apartment.places)
    # the corridor floor strip is narrow: furniture beside it still belongs to the corridor, not the kitchen
    assert apartment.places["走廊边桌"].room == "走廊"


def _rooms_along(nav, path):
    rooms = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        for k in range(11):
            room = nav.room_at(x0 + (x1 - x0) * k / 10, y0 + (y1 - y0) * k / 10)
            if not rooms or rooms[-1] != room:
                rooms.append(room)
    return rooms


def test_plans_go_through_doors_quickly(apartment):
    sofa, bed = apartment.places["沙发"], apartment.places["床"]
    assert not apartment.clear_line((sofa.x, sofa.y), (bed.x, bed.y))  # walls in between
    started = time.perf_counter()
    path = apartment.plan((sofa.x, sofa.y), (bed.x, bed.y))
    assert time.perf_counter() - started < 0.5
    rooms = _rooms_along(apartment, path)
    # either way round is fine (via the kitchen or straight out of the living room); the bedroom is
    # only reachable from the corridor
    assert rooms[0] == "客厅" and rooms[-1] == "卧室" and "走廊" in rooms
    for (x0, y0), (x1, y1) in zip(path, path[1:]):  # every leg is walkable end to end
        assert apartment.clear_line((x0, y0), (x1, y1))


def test_a_goal_outside_the_house_snaps_to_the_nearest_reachable_spot(apartment):
    outside = (5.0, 0.0)
    assert apartment.plan((-2.0, -1.5), outside) is None  # 1 m is not enough to reach in
    path = apartment.plan((-2.0, -1.5), outside, goal_reach_m=3.0)
    assert path and math.dist(path[-1], outside) < 1.6


def test_an_open_floor_needs_no_map():
    import mujoco
    import navmap
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_RL / "src/mjlab_microduck/robot/microduck/scene.xml"))
    with pytest.raises(ValueError):
        navmap.NavMap(model, set())
