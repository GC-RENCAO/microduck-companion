"""The house's map and path planning, generated from the same scene file the simulator runs.

Map (5 cm cells). A cell is walkable when both hold:
- there is floor to stand on: something with collision whose top is at floor height (the apartment's
  corridor has a stairwell with no floor — ball_2 fell into it);
- nothing blocks the duck's height: no colliding geom with its top above the floor and its bottom
  below DUCK_HEIGHT (walls, furniture legs, the sofa…).
Obstacles are then inflated by half the duck's width, and cells near them cost more, so routes keep
to the middle of passages.

Rooms: the scene's `floor_<name>` floor patches (decorative, collision-free floors in the apartment, one per room).
Places: named furniture (Chinese names in ROOM/PLACE_LABELS); going there = the nearest walkable cell beside it.
Planning: A* on 8-connected cells, then line-of-sight pruning down to a few waypoints.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

# Geom types are compared as plain ints: from MuJoCo 3.17, `np.int32 in (enum, …)` is False although
# `==` is True, so enums as dict keys or inside tuples fail silently.
BOX, PLANE, SPHERE = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_SPHERE)
CYLINDER, CAPSULE = int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)

RESOLUTION = 0.05  # metres per cell
DUCK_HEIGHT = 0.30  # anything whose bottom is below this blocks the duck
FLOOR_TOP = (-0.03, 0.03)  # a top surface in this range is floor to stand on
INFLATE_M = 0.16  # obstacle inflation: half the duck's width + margin (0.8 m doors keep ~0.45 m; 0.14 scraped door frames)
CLEARANCE_M = 0.45  # cells closer than this to an obstacle cost more (it turns while walking, ~0.3 m radius, and swings wide)
CLEARANCE_COST = 6.0
APPROACH_M = 0.7  # going to a piece of furniture: the nearest walkable cell within this distance

ROOM_LABELS = {
    "kitchen": "厨房", "living": "客厅", "office": "书房", "bedroom": "卧室", "bath": "浴室",
    "cor_n": "走廊", "cor_s": "走廊南头",
}
# The apartment's furniture → Chinese names. Other scenes work too (rooms from floor_ names, furniture unnamed).
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
    name: str  # Chinese name, for the LLM and for people
    x: float  # where to walk to (a walkable cell)
    y: float
    room: str | None


class NavMap:
    def __init__(self, model: mujoco.MjModel, skip_bodies: set[int]):
        """skip_bodies: root bodies of the duck and of loose objects (not part of the map)."""
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

        # The map covers the bounding box of every floor
        tops = {g: float(data.geom_xpos[g][2] + _half_height(model, data, g)) for g in solid}
        bottoms = {g: float(data.geom_xpos[g][2] - _half_height(model, data, g)) for g in solid}
        floors = [g for g in solid if FLOOR_TOP[0] <= tops[g] <= FLOOR_TOP[1]
                  and int(model.geom_type[g]) in (BOX, PLANE)]
        if not floors:
            raise ValueError("no floor to stand on in this scene, so no map")
        if any(int(model.geom_type[g]) == PLANE and not model.geom_size[g][0] for g in floors):
            raise ValueError("an infinite open floor has no walls and needs no map (walk straight to the goal)")
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

        # Rooms: the floor_<name> patches (with or without collision)
        self.rooms: list[tuple[str, np.ndarray]] = []
        for g in geoms:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            key = name[len("floor_"):]
            if name.startswith("floor_") and key in ROOM_LABELS:
                self.rooms.append((ROOM_LABELS[key], self._covers(model, data, g)))

        # Which room each cell belongs to. The decorative floors cover only part of a room (the apartment's
        # corridor strip is 40 cm wide), so grow every room outward through wall-free cells at once
        # (multi-source BFS): a wall in between keeps a cell out of the wrong room.
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
        if self.rooms and (self._room_of < 0).any():  # cells inside walls: the nearest room
            _, (near_r, near_c) = ndimage.distance_transform_edt(self._room_of < 0, return_indices=True)
            self._room_of = self._room_of[near_r, near_c]

        # Places: each room's centre (its walkable cell farthest from walls) + beside named furniture
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

    # ── coordinates ────────────────────────────────────────────────────────

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
        """The walkable point nearest (x, y), or None if there is none within `within_m`."""
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

    # ── planning ───────────────────────────────────────────────────────────

    def plan(self, start: tuple[float, float], goal: tuple[float, float],
             goal_reach_m: float = 1.0) -> list[tuple[float, float]] | None:
        """A* from start to goal; returns waypoints in world metres. An unwalkable start or goal is first
        moved to the nearest walkable cell.

        goal_reach_m: how far around an unwalkable goal to look for one (the owner may stand behind a wall
        or outside the house).
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
                    continue  # no cutting wall corners diagonally
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
        """Is the straight line from a to b walkable all the way?"""
        return self._line_free(self.cell(*a), self.cell(*b))

    def _prune(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Line-of-sight pruning: drop every cell visible from the last kept one; keep the corners."""
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

    # ── projecting geoms onto the map ──────────────────────────────────────

    def _covers(self, model, data, g: int) -> np.ndarray:
        """The cells this geom's footprint covers (a cell counts when its centre is inside)."""
        center = data.geom_xpos[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        dx, dy = self._grid_x - center[0], self._grid_y - center[1]
        lx = dx * rot[0, 0] + dy * rot[1, 0]  # world → geom frame (horizontal plane only)
        ly = dx * rot[0, 1] + dy * rot[1, 1]
        size = model.geom_size[g]
        kind = int(model.geom_type[g])
        if kind == BOX:
            return (np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])
        if kind == PLANE:
            return np.ones(self.shape, bool) if not size[0] else (np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])
        if kind == ELLIPSOID:
            return (lx / size[0]) ** 2 + (ly / size[1]) ** 2 <= 1.0
        radius = size[0]  # spheres, upright cylinders and capsules
        return lx ** 2 + ly ** 2 <= radius ** 2

    def to_json(self) -> dict:
        """For drawing the map in the page / debugging."""
        return {"origin": [float(v) for v in self.origin], "resolution": RESOLUTION, "shape": list(self.shape),
                "places": {p.name: [round(p.x, 2), round(p.y, 2), p.room] for p in self.places.values()}}


def _half_height(model, data, g: int) -> float:
    """Vertical half-height (for the shapes met here: upright axes)."""
    size, kind = model.geom_size[g], int(model.geom_type[g])
    if kind in (BOX, ELLIPSOID):
        rot = data.geom_xmat[g].reshape(3, 3)
        return float(np.abs(rot[2]) @ size)
    if kind == CYLINDER:
        return float(size[1])
    if kind == CAPSULE:
        return float(size[1] + size[0])
    if kind == SPHERE:
        return float(size[0])
    if kind == PLANE:
        return 0.0
    return float(model.geom_rbound[g])


def _footprint_corners(model, data, g: int) -> list[np.ndarray]:
    center, rot, size = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
    return [center[:2] + rot[:2, 0] * sx * size[0] + rot[:2, 1] * sy * size[1] for sx in (-1, 1) for sy in (-1, 1)]
