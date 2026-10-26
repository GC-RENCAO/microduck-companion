"""Scene export + pose stream: the browser draws the duck and its house itself with WebGL.

- At startup, from the same scene as the sim (DUCK_SIM_SCENE, same rules as microduck's scripts/duck-sim):
  · the duck's parts (meshes) and the loose things in the room (balls, boxes) — posed every frame;
  · the walls, floors and furniture that never move — sent once with their poses;
  · the textures they use (MuJoCo's raw pixels → PNG).
- At runtime, poses are read (read-only) from duck-body (TCP 7801): `read` gives the duck's joints and
  trunk, `objects` the loose things (`objects` is an op added to microduck_rl by this project's patch;
  without it the loose things stay where they started). One mj_kinematics pass, then every part's
  position and rotation go to the browser at about 50 Hz.
"""

from __future__ import annotations

import io
import json
import math
import os
import socket
import struct
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from mjlab_microduck.sim.body_server import DEFAULT_SCENE, JOINT_NAMES, PROTOCOL, SCENES
from PIL import Image

BODY_ADDR = ("127.0.0.1", 7801)
POSE_HZ = 50
VISIBLE_GROUPS = (0, 1, 2)  # groups MuJoCo shows by default; 3 is collision geometry
WALL_MIN_HEIGHT = 0.6  # static things taller than this are “walls”: the page fades them when they hide the duck

# Plain int keys: since MuJoCo 3.17 enums and numpy ints hash differently, so enum keys miss (see navmap.py)
GEOM_TYPES = {int(kind): name for kind, name in (
    (mujoco.mjtGeom.mjGEOM_PLANE, "plane"), (mujoco.mjtGeom.mjGEOM_SPHERE, "sphere"),
    (mujoco.mjtGeom.mjGEOM_CAPSULE, "capsule"), (mujoco.mjtGeom.mjGEOM_ELLIPSOID, "ellipsoid"),
    (mujoco.mjtGeom.mjGEOM_CYLINDER, "cylinder"), (mujoco.mjtGeom.mjGEOM_BOX, "box"),
    (mujoco.mjtGeom.mjGEOM_MESH, "mesh"))}


def scene_path() -> Path:
    """Like scripts/duck-sim: empty = the default scene; contains a slash = a path; otherwise microduck_rl's scene_<name>.xml."""
    name = os.environ.get("DUCK_SIM_SCENE", "")
    if not name:
        return DEFAULT_SCENE
    if "/" in name:
        return Path(name)
    return SCENES / f"scene_{name}.xml"


class PoseSource:
    def __init__(self) -> None:
        self.scene_file = scene_path()
        model = mujoco.MjModel.from_xml_path(str(self.scene_file))
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)

        trunk_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        loose_joints = [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
                        and model.jnt_bodyid[j] != trunk_body]
        self._loose = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])): int(model.jnt_qposadr[j])
                       for j in loose_joints}
        loose_bodies = {int(model.jnt_bodyid[j]) for j in loose_joints}
        self.moving_bodies = {trunk_body} | loose_bodies  # not part of the house (skipped when building the map)

        def root_of(body: int, roots: set[int]) -> int | None:
            while body != 0:
                if body in roots:
                    return body
                body = int(model.body_parentid[body])
            return None

        visible = [g for g in range(model.ngeom) if model.geom_group[g] in VISIBLE_GROUPS]
        # The duck: visible meshes only (the simplified collision shapes are not drawn)
        self.parts = [g for g in visible if root_of(int(model.geom_bodyid[g]), {trunk_body}) is not None
                      and GEOM_TYPES.get(int(model.geom_type[g])) == "mesh"]
        self.duck_parts = len(self.parts)  # the first this-many parts are the duck (the camera follows its centre, not pulled off by the balls)
        # Loose things in the room: posed every frame too
        self.parts += [g for g in visible if root_of(int(model.geom_bodyid[g]), loose_bodies) is not None]
        moving = set(self.parts) | {g for g in range(model.ngeom) if root_of(int(model.geom_bodyid[g]), {trunk_body}) is not None}
        self.statics = [g for g in visible if g not in moving and int(model.geom_type[g]) in GEOM_TYPES]

        self.textures: dict[int, bytes] = {}
        self.scene, self.meshes = self._export()

        self._trunk = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        # The wire array follows JOINT_NAMES; joints missing from the model (mouth) are skipped. Look up by name, not index.
        self._joints = []
        for wire_index, name in enumerate(JOINT_NAMES):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                self._joints.append((wire_index, model.jnt_qposadr[joint]))

        self.connected = False
        self.objects_live = False  # whether duck-body serves the loose things' poses
        # The duck's world position (m) and heading (rad, 0 = +x, counter-clockwise positive), for the “knows where you are” behaviours
        self.duck_xy = (0.0, 0.0)
        self.duck_yaw = 0.0
        self.frame: bytes | None = None  # the latest packed poses
        self.frame_id = 0
        self._cond = threading.Condition()
        threading.Thread(target=self._run, name="pose", daemon=True).start()

    # ── One-off export ──────────────────────────────────────────────────────────

    def _export(self) -> tuple[dict, bytes]:
        """The scene manifest (JSON) + the meshes it uses (binary: float32 vertices + uint32 triangles)."""
        model, data = self.model, self.data
        blob = bytearray()
        mesh_index: dict[int, dict] = {}

        def describe(geom: int) -> dict:
            kind = GEOM_TYPES[int(model.geom_type[geom])]
            item = {"type": kind, "size": [round(float(v), 5) for v in model.geom_size[geom]]}
            if kind == "mesh":
                mesh = int(model.geom_dataid[geom])
                if mesh not in mesh_index:
                    v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
                    mesh_index[mesh] = {"offset": len(blob), "verts": int(nv), "faces": int(nf)}
                    blob.extend(model.mesh_vert[v0 : v0 + nv].astype("<f4").tobytes())
                    blob.extend(model.mesh_face[f0 : f0 + nf].astype("<u4").tobytes())
                item["mesh"] = mesh
            material = int(model.geom_matid[geom])
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            item["rgba"] = [round(float(c), 4) for c in rgba]
            if material >= 0:
                tex = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
                if tex >= 0 and int(model.tex_type[tex]) == int(mujoco.mjtTexture.mjTEXTURE_2D):
                    self._texture(tex)
                    item["tex"] = tex
                    item["texrepeat"] = [round(float(v), 4) for v in model.mat_texrepeat[material]]
                    item["texuniform"] = bool(model.mat_texuniform[material])
            return item

        parts = [describe(g) for g in self.parts]
        statics = []
        for g in self.statics:
            item = describe(g)
            item["pose"] = [round(float(v), 5) for v in np.concatenate([data.geom_xpos[g], data.geom_xmat[g]])]
            top = float(data.geom_xpos[g][2] + model.geom_rbound[g])
            item["wall"] = item["type"] != "plane" and top > WALL_MIN_HEIGHT and float(model.geom_size[g][2]) > 0.2
            statics.append(item)
        scene = {
            "name": self.scene_file.stem,
            "parts": parts,
            "duck_parts": self.duck_parts,
            "statics": statics,
            "meshes": {str(k): v for k, v in mesh_index.items()},
            "floor": any(GEOM_TYPES[int(model.geom_type[g])] in ("plane", "box") and float(data.geom_xpos[g][2]) < 0.02
                         for g in self.statics),
            "pose_floats_per_part": 12,
        }
        return scene, bytes(blob)

    def _texture(self, tex: int) -> None:
        if tex in self.textures:
            return
        model = self.model
        w, h, c = int(model.tex_width[tex]), int(model.tex_height[tex]), int(model.tex_nchannel[tex])
        adr = int(model.tex_adr[tex])
        pixels = np.asarray(model.tex_data[adr : adr + w * h * c], dtype=np.uint8).reshape(h, w, c)
        buffer = io.BytesIO()
        Image.fromarray(pixels[:, :, :3] if c >= 3 else pixels[:, :, 0]).save(buffer, "PNG")
        self.textures[tex] = buffer.getvalue()

    # ── Pose stream ──────────────────────────────────────────────────────────────

    def wait_frame(self, last_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self.frame_id != last_id, timeout=timeout)
            return self.frame_id, self.frame

    def _run(self) -> None:
        model, data = self.model, self.data
        parts = np.array(self.parts)
        body = None
        ask_objects = bool(self._loose)
        while True:
            started = time.perf_counter()
            try:
                if body is None:
                    body = self._connect()
                body.write('{"op":"read"}\n')
                body.flush()
                sensors = json.loads(body.readline())
                data.qpos[self._trunk : self._trunk + 3] = sensors["trunk"]
                data.qpos[self._trunk + 3 : self._trunk + 7] = sensors["imu"]["quat"]
                for wire_index, adr in self._joints:
                    data.qpos[adr] = sensors["positions"][wire_index]
                if ask_objects:
                    body.write('{"op":"objects"}\n')
                    body.flush()
                    answer = json.loads(body.readline())
                    if "objects" in answer:
                        self.objects_live = True
                        for name, qpos in answer["objects"].items():
                            if name in self._loose:
                                adr = self._loose[name]
                                data.qpos[adr : adr + 7] = qpos
                    else:  # no such op (microduck_rl without the patch): loose things stay where they started
                        ask_objects = False
                self.connected = True
                w, x, y, z = sensors["imu"]["quat"]
                self.duck_xy = (sensors["trunk"][0], sensors["trunk"][1])
                self.duck_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            except (OSError, ValueError, KeyError):
                body = None
                self.connected = False
                ask_objects = bool(self._loose)
                time.sleep(1.0)  # the sim is down or restarting: retry shortly
                continue

            mujoco.mj_kinematics(model, data)
            # 12 floats per part: position 3 + row-major rotation 9, after the sim time the browser interpolates with.
            poses = np.concatenate([data.geom_xpos[parts], data.geom_xmat[parts]], axis=1)
            packed = struct.pack("<d", float(sensors["sim_time"])) + poses.astype("<f4").tobytes()
            with self._cond:
                self.frame = packed
                self.frame_id += 1
                self._cond.notify_all()

            time.sleep(max(0.0, 1.0 / POSE_HZ - (time.perf_counter() - started)))

    @staticmethod
    def _connect():
        conn = socket.create_connection(BODY_ADDR, timeout=3)
        stream = conn.makefile("rw")
        stream.write(json.dumps({"op": "hello", "protocol": PROTOCOL, "joints": len(JOINT_NAMES)}) + "\n")
        stream.flush()
        stream.readline()
        return stream
