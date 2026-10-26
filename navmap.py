"""房子的地图和寻路：从仿真同一份场景文件生成，鸭子靠它绕过墙、穿过门。

地图（5 厘米一格）：一格能走，要同时满足
- 下面有能踩的地板：有碰撞的东西，顶面在地面高度附近（公寓走廊里有个楼梯井，没地板，会掉下去——ball_2 就掉进去了）；
- 鸭子身高以内没有障碍：有碰撞、顶面高于地面、底面低于 DUCK_HEIGHT 的东西（墙、家具腿、沙发……）。
然后把障碍“膨胀”出鸭子半个身宽，再给离墙近的格子加代价，让路线走在过道中间。

房间：场景里叫 floor_<名字> 的地板块（公寓里是不带碰撞的装饰地板，正好一块一个房间）。
地点：有名字的家具（ROOM/PLACE_LABELS 里给了中文名）；去那里 = 走到它旁边最近的能走的格子。
寻路：A*（8 邻接）+ 视线剪枝成几个路标点。
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

RESOLUTION = 0.05  # 米/格
DUCK_HEIGHT = 0.30  # 底面低于这个高度的东西会挡住鸭子
FLOOR_TOP = (-0.03, 0.03)  # 顶面在这个范围内 = 能踩的地板
INFLATE_M = 0.16  # 障碍向外膨胀：鸭子半个身宽 + 余量（门洞 0.8 米，膨胀后还剩约 0.45 米；0.14 时实测会蹭到门框）
CLEARANCE_M = 0.45  # 离障碍比这近的格子加代价，路线尽量走中间（只能边走边转，转弯半径约 0.3 米，要留出甩出去的余地）
CLEARANCE_COST = 6.0
APPROACH_M = 0.7  # 去家具旁边：在它这么远以内找最近的能走的格子

ROOM_LABELS = {
    "kitchen": "厨房", "living": "客厅", "office": "书房", "bedroom": "卧室", "bath": "浴室",
    "cor_n": "走廊", "cor_s": "走廊南头",
}
# 公寓里的家具 → 中文名。没列在这里的场景也能用（房间按 floor_ 名字，家具不起名）
PLACE_LABELS = {
    "dock_base": "窝（充电座）", "liv_sofa_seat": "沙发", "liv_table": "茶几", "liv_tv_stand": "电视柜",
    "liv_shelf": "客厅书架", "liv_chair_seat": "客厅椅子", "liv_plant": "客厅的绿植",
    "kit_fridge": "冰箱", "kit_counter": "灶台", "kit_island": "厨房中岛",
    "bed_base": "床", "bed_wardrobe": "衣柜", "bed_dresser": "梳妆台",
    "off_desk": "书桌", "off_chair": "办公椅", "off_shelf": "书房书架", "off_plant": "书房的绿植",
    "bth_tub": "浴缸", "bth_vanity": "洗手台", "bth_toilet": "马桶",
    "cor_console": "走廊边桌", "cor_shoes": "鞋柜",
}


@dataclass
class Place:
    name: str  # 中文名，给大模型和人看
    x: float  # 要走到的点（能走的格子）
    y: float
    room: str | None


class NavMap:
    def __init__(self, model: mujoco.MjModel, skip_bodies: set[int]):
        """skip_bodies：鸭子和会动的东西的根 body（它们不算地图的一部分）。"""
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def skipped(body: int) -> bool:
            while body != 0:
                if body in skip_bodies:
                    return True
                body = int(model.body_parentid[body])
            return False

        geoms = [g for g in range(model.ngeom) if not skipped(int(model.geom_bodyid[g]))]
        solid = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]

        # 地图范围：所有地板的外框
        tops = {g: float(data.geom_xpos[g][2] + _half_height(model, data, g)) for g in solid}
        bottoms = {g: float(data.geom_xpos[g][2] - _half_height(model, data, g)) for g in solid}
        floors = [g for g in solid if FLOOR_TOP[0] <= tops[g] <= FLOOR_TOP[1]
                  and model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE)]
        if not floors:
            raise ValueError("场景里没有能踩的地板，没法生成地图")
        if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and not model.geom_size[g][0] for g in floors):
            raise ValueError("无限大的空地，没有墙，不需要地图（直接朝目标走）")
        corners = np.array([c for g in floors for c in _footprint_corners(model, data, g)])
        lo, hi = corners.min(axis=0) - 0.1, corners.max(axis=0) + 0.1
        self.origin = lo
        self.shape = (int(math.ceil((hi[1] - lo[1]) / RESOLUTION)), int(math.ceil((hi[0] - lo[0]) / RESOLUTION)))
        rows, cols = self.shape
        ys = lo[1] + (np.arange(rows) + 0.5) * RESOLUTION
        xs = lo[0] + (np.arange(cols) + 0.5) * RESOLUTION
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)

        support = np.zeros(self.shape, bool)
        for g in floors:
            support |= self._covers(model, data, g)
        obstacle = np.zeros(self.shape, bool)
        for g in solid:
            if tops[g] > FLOOR_TOP[1] and bottoms[g] < DUCK_HEIGHT:
                obstacle |= self._covers(model, data, g)
        self.blocked_raw = ~support | obstacle
        clearance = ndimage.distance_transform_edt(~self.blocked_raw) * RESOLUTION
        self.free = clearance > INFLATE_M
        self.cost = 1.0 + CLEARANCE_COST * np.clip((CLEARANCE_M - clearance) / CLEARANCE_M, 0.0, 1.0)

        # 房间：floor_<名字> 的地板块（不管有没有碰撞）
        self.rooms: list[tuple[str, np.ndarray]] = []
        for g in geoms:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            key = name[len("floor_"):]
            if name.startswith("floor_") and key in ROOM_LABELS:
                self.rooms.append((ROOM_LABELS[key], self._covers(model, data, g)))

        # 每一格归哪个房间：装饰地板只铺了一部分（公寓走廊的地板只有中间 40 厘米），
        # 所以从每块地板出发，沿着没有墙的格子同时往外扩（多源 BFS）——被墙隔开的就不会归错房间。
        self._room_of = np.full(self.shape, -1, int)
        frontier = deque()
        for index, (_, mask) in enumerate(self.rooms):
            seeds = mask & (self._room_of < 0) & ~self.blocked_raw
            self._room_of[seeds] = index
            frontier.extend(zip(*np.nonzero(seeds)))
        while frontier:
            r, c = frontier.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if self.inside(nr, nc) and self._room_of[nr, nc] < 0 and not self.blocked_raw[nr, nc]:
                    self._room_of[nr, nc] = self._room_of[r, c]
                    frontier.append((nr, nc))
        if self.rooms and (self._room_of < 0).any():  # 墙里面的格子：归最近的
            _, (near_r, near_c) = ndimage.distance_transform_edt(self._room_of < 0, return_indices=True)
            self._room_of = self._room_of[near_r, near_c]

        # 地点：每个房间的中心（离墙最远的能走的格子）+ 有名字的家具旁边
        self.places: dict[str, Place] = {}
        for label, mask in self.rooms:
            if label in self.places:
                continue
            inside = np.where(mask & self.free, clearance, 0.0)
            if inside.max() > 0:
                r, c = np.unravel_index(int(np.argmax(inside)), self.shape)
                x, y = self.cell_center(r, c)
                self.places[label] = Place(label, x, y, label)
        for g in geoms:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name in PLACE_LABELS:
                spot = self.nearest_free(float(data.geom_xpos[g][0]), float(data.geom_xpos[g][1]), APPROACH_M)
                if spot:
                    label = PLACE_LABELS[name]
                    self.places[label] = Place(label, spot[0], spot[1], self.room_at(*spot))

    # ── 坐标 ────────────────────────────────────────────────────────────────

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.origin[1]) / RESOLUTION), int((x - self.origin[0]) / RESOLUTION)

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return float(self.origin[0] + (c + 0.5) * RESOLUTION), float(self.origin[1] + (r + 0.5) * RESOLUTION)

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def is_free(self, x: float, y: float) -> bool:
        r, c = self.cell(x, y)
        return self.inside(r, c) and bool(self.free[r, c])

    def room_at(self, x: float, y: float) -> str | None:
        r, c = self.cell(x, y)
        if not self.inside(r, c) or not self.rooms:
            return None
        return self.rooms[int(self._room_of[r, c])][0]

    def nearest_free(self, x: float, y: float, within_m: float = 1.0) -> tuple[float, float] | None:
        """离 (x, y) 最近的能走的点；within_m 以内找不到就返回 None。"""
        r0, c0 = self.cell(x, y)
        reach = int(within_m / RESOLUTION)
        best = None
        for r in range(max(0, r0 - reach), min(self.shape[0], r0 + reach + 1)):
            for c in range(max(0, c0 - reach), min(self.shape[1], c0 + reach + 1)):
                if self.free[r, c]:
                    d = (r - r0) ** 2 + (c - c0) ** 2
                    if best is None or d < best[0]:
                        best = (d, r, c)
        return self.cell_center(best[1], best[2]) if best and best[0] <= reach * reach else None

    # ── 寻路 ────────────────────────────────────────────────────────────────

    def plan(self, start: tuple[float, float], goal: tuple[float, float],
             goal_reach_m: float = 1.0) -> list[tuple[float, float]] | None:
        """A* 从 start 到 goal，返回路标点（世界坐标，米）；起点/终点不在能走的格子上就先挪到最近的。

        goal_reach_m：终点不能走时，在它周围多远内找最近的能走的点（主人可能站在墙外、房子外面）。
        """
        start = start if self.is_free(*start) else self.nearest_free(*start, 0.6)
        goal = goal if self.is_free(*goal) else self.nearest_free(*goal, goal_reach_m)
        if start is None or goal is None:
            return None
        s, g = self.cell(*start), self.cell(*goal)
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        def heuristic(cell):
            dr, dc = abs(cell[0] - g[0]), abs(cell[1] - g[1])
            return (dr + dc) + (1.414 - 2) * min(dr, dc)
        best = {s: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        frontier = [(heuristic(s), s)]
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == g:
                break
            for dr, dc, step in steps:
                nxt = (current[0] + dr, current[1] + dc)
                if not self.inside(*nxt) or not self.free[nxt]:
                    continue
                if dr and dc and not (self.free[current[0] + dr, current[1]] and self.free[current[0], current[1] + dc]):
                    continue  # 不斜着切墙角
                cost = best[current] + step * self.cost[nxt]
                if cost < best.get(nxt, math.inf):
                    best[nxt], came[nxt] = cost, current
                    heapq.heappush(frontier, (cost + heuristic(nxt), nxt))
        if g not in best:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        return [self.cell_center(*c) for c in self._prune(cells)]

    def clear_line(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """a 到 b 的直线上全是能走的格子吗（叫“过来”时判断要不要绕路）。"""
        return self._line_free(self.cell(*a), self.cell(*b))

    def _prune(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """视线剪枝：能直接看见的中间点都去掉，只留拐弯处。"""
        out, anchor, i = [cells[0]], 0, 1
        while i < len(cells):
            if not self._line_free(cells[anchor], cells[i]):
                out.append(cells[i - 1])
                anchor = i - 1
            i += 1
        out.append(cells[-1])
        return out

    def _line_free(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for k in range(n + 1):
            r = round(a[0] + (b[0] - a[0]) * k / n)
            c = round(a[1] + (b[1] - a[1]) * k / n)
            if not self.inside(r, c) or not self.free[r, c]:
                return False
        return True

    # ── 把几何体投影到地图上 ────────────────────────────────────────────────

    def _covers(self, model, data, g: int) -> np.ndarray:
        """这个几何体在地上的投影盖住了哪些格子（格子中心落在里面就算）。"""
        center = data.geom_xpos[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        dx, dy = self._grid_x - center[0], self._grid_y - center[1]
        lx = dx * rot[0, 0] + dy * rot[1, 0]  # 世界 → 几何体本地（只看水平面）
        ly = dx * rot[0, 1] + dy * rot[1, 1]
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return (np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.ones(self.shape, bool) if not size[0] else (np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])
        if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return (lx / size[0]) ** 2 + (ly / size[1]) ** 2 <= 1.0
        radius = size[0]  # 球、竖着的圆柱和胶囊
        return lx ** 2 + ly ** 2 <= radius ** 2

    def to_json(self) -> dict:
        """给页面画地图/调试用。"""
        return {"origin": [float(v) for v in self.origin], "resolution": RESOLUTION, "shape": list(self.shape),
                "places": {p.name: [round(p.x, 2), round(p.y, 2), p.room] for p in self.places.values()}}


def _half_height(model, data, g: int) -> float:
    """竖直方向的半高（只处理这里会遇到的：轴竖直的形状）。"""
    size, kind = model.geom_size[g], model.geom_type[g]
    if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        rot = data.geom_xmat[g].reshape(3, 3)
        return float(np.abs(rot[2]) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[1])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[1] + size[0])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return 0.0
    return float(model.geom_rbound[g])


def _footprint_corners(model, data, g: int) -> list[np.ndarray]:
    center, rot, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
    return [center[:2] + rot[:2, 0] * sx * size[0] + rot[:2, 1] * sy * size[1] for sx in (-1, 1) for sy in (-1, 1)]
