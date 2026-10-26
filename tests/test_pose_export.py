"""What the browser downloads once: the duck's parts, the house, its textures (needs microduck_rl)."""

import io

import pytest
from PIL import Image

from conftest import needs_scenes

pytestmark = needs_scenes


@pytest.fixture(scope="module")
def apartment(monkeypatch_module):
    monkeypatch_module.setenv("DUCK_SIM_SCENE", "apartment")
    import pose
    monkeypatch_module.setattr(pose, "BODY_ADDR", ("127.0.0.1", 9))  # no simulator: the stream just retries
    return pose.PoseSource()


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def test_the_duck_comes_first_and_loose_things_after(apartment):
    scene = apartment.scene
    assert scene["name"] == "scene_apartment"
    assert scene["duck_parts"] == 70
    assert [p["type"] for p in scene["parts"][scene["duck_parts"]:]].count("sphere") == 3  # the balls


def test_the_house_is_exported_once_with_walls_marked(apartment):
    statics = apartment.scene["statics"]
    assert len(statics) == 86
    assert sum(item["wall"] for item in statics) == 25
    assert all(len(item["pose"]) == 12 for item in statics)


def test_textures_are_real_pngs(apartment):
    used = {item["tex"] for item in apartment.scene["statics"] if "tex" in item}
    assert used and used <= set(apartment.textures)
    image = Image.open(io.BytesIO(apartment.textures[min(used)]))
    assert image.format == "PNG" and image.size[0] > 0


def test_the_scene_name_follows_duck_sim_rules(monkeypatch):
    import pose
    monkeypatch.setenv("DUCK_SIM_SCENE", "")
    assert pose.scene_path().name == "scene.xml"
    monkeypatch.setenv("DUCK_SIM_SCENE", "ball")
    assert pose.scene_path().name == "scene_ball.xml"
    monkeypatch.setenv("DUCK_SIM_SCENE", "/tmp/my_room.xml")
    assert str(pose.scene_path()) == "/tmp/my_room.xml"
