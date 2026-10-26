"""场景导出 + 姿态流：让浏览器自己用 WebGL 画鸭子和它住的房子。

- 启动时从和仿真同一份场景（DUCK_SIM_SCENE，规则和 microduck 的 scripts/duck-sim 一样）导出：
  · 鸭子的零件（网格）和房间里会动的东西（球、箱子）——每帧推姿态；
  · 墙、地板、家具这些不动的东西——连同位置只发一次；
  · 它们用到的贴图（MuJoCo 里的原始像素 → PNG）。
- 运行时从 duck-body（TCP 7801）只读姿态：`read` 给鸭子的关节和躯干，`objects` 给会动的东西
  （`objects` 是本地给 microduck_rl 加的只读接口；没有的话会动的东西就停在初始位置）。
  做一次运动学（mj_kinematics），把每个零件的位置和旋转以约 50 Hz 推给浏览器。
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
VISIBLE_GROUPS = (0, 1, 2)  # MuJoCo 默认显示的组；3 是碰撞体
WALL_MIN_HEIGHT = 0.6  # 比这高的不动的东西算“墙”：挡住鸭子时页面会把它变透明

GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


def scene_path() -> Path:
    """和 scripts/duck-sim 一样：空 = 默认场景；带斜杠 = 路径；否则 = microduck_rl 自带的 scene_<名字>.xml。"""
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
        loose_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        and model.jnt_bodyid[j] != trunk_body]
        self._loose = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])): int(model.jnt_qposadr[j])
                       for j in loose_joints}
        loose_bodies = {int(model.jnt_bodyid[j]) for j in loose_joints}
        self.moving_bodies = {trunk_body} | loose_bodies  # 不算房子的一部分（地图生成时跳过）

        def root_of(body: int, roots: set[int]) -> int | None:
            while body != 0:
                if body in roots:
                    return body
                body = int(model.body_parentid[body])
            return None

        visible = [g for g in range(model.ngeom) if model.geom_group[g] in VISIBLE_GROUPS]
        # 鸭子：只要可见的网格（碰撞用的简化形状不画）
        self.parts = [g for g in visible if root_of(int(model.geom_bodyid[g]), {trunk_body}) is not None
                      and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.duck_parts = len(self.parts)  # 前这么多个零件是鸭子（相机盯着鸭子的中心，不被球拉偏）
        # 房间里会动的东西：也每帧推姿态
        self.parts += [g for g in visible if root_of(int(model.geom_bodyid[g]), loose_bodies) is not None]
        moving = set(self.parts) | {g for g in range(model.ngeom) if root_of(int(model.geom_bodyid[g]), {trunk_body}) is not None}
        self.statics = [g for g in visible if g not in moving and model.geom_type[g] in GEOM_TYPES]

        self.textures: dict[int, bytes] = {}
        self.scene, self.meshes = self._export()

        self._trunk = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        # 线上数组按 JOINT_NAMES 排序；模型里没有的关节（mouth）跳过。按名字查，不按下标。
        self._joints = []
        for wire_index, name in enumerate(JOINT_NAMES):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                self._joints.append((wire_index, model.jnt_qposadr[joint]))

        self.connected = False
        self.objects_live = False  # duck-body 是否提供会动物体的姿态
        # 鸭子在世界里的位置（米）和朝向（弧度，0 = +x，逆时针为正），给“知道你在哪”的行为用
        self.duck_xy = (0.0, 0.0)
        self.duck_yaw = 0.0
        self.frame: bytes | None = None  # 打包好的最新姿态
        self.frame_id = 0
        self._cond = threading.Condition()
        threading.Thread(target=self._run, name="pose", daemon=True).start()

    # ── 一次性导出 ──────────────────────────────────────────────────────────

    def _export(self) -> tuple[dict, bytes]:
        """场景清单（JSON）+ 用到的网格（二进制：float32 顶点 + uint32 三角面）。"""
        model, data = self.model, self.data
        blob = bytearray()
        mesh_index: dict[int, dict] = {}

        def describe(geom: int) -> dict:
            kind = GEOM_TYPES[model.geom_type[geom]]
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
                if tex >= 0 and model.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
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
            "floor": any(GEOM_TYPES[model.geom_type[g]] in ("plane", "box") and float(data.geom_xpos[g][2]) < 0.02
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

    # ── 姿态流 ──────────────────────────────────────────────────────────────

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
                    else:  # 没有这个接口（没打本地补丁的 microduck_rl）：会动的东西停在初始位置
                        ask_objects = False
                self.connected = True
                w, x, y, z = sensors["imu"]["quat"]
                self.duck_xy = (sensors["trunk"][0], sensors["trunk"][1])
                self.duck_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            except (OSError, ValueError, KeyError):
                body = None
                self.connected = False
                ask_objects = bool(self._loose)
                time.sleep(1.0)  # 仿真没开或重启中：过一会儿再连
                continue

            mujoco.mj_kinematics(model, data)
            # 每个零件 12 个 float：位置 3 + 行优先旋转矩阵 9。前面加仿真时间，浏览器用它插值。
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
