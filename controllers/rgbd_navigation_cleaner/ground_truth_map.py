"""Ground-truth obstacle map built from the Webots .wbt world geometry.

This module is used ONLY for evaluation metrics (false-occupied / false-free cell
ratios) for the diploma report.  The navigation/mapping stack never imports it.

It parses the static world description, keeps the Solids that have a
``boundingObject`` (i.e. real collision geometry) whose vertical extent overlaps
the robot's body height, and rasterises their footprints into an occupancy grid
in the SAME map frame the controller uses.  The robot's learned obstacle mask is
then compared against this reference.

Alignment: the controller's map frame lives in the odometry frame (pose starts at
0,0).  The world frame (where .wbt coordinates live) is tied to it through the
ground-truth GPS sample taken at t0, so a world point maps to a cell via::

    odom = (world - gps_origin) + odom_origin      # axis-aligned, robot starts yaw=0
    mx   = MAP_ORIGIN_X + odom_x * MAP_SCALE
    my   = MAP_ORIGIN_Y - odom_y * MAP_SCALE        # y is flipped in the map
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present in this project
    cv2 = None


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _find_top_level_solids(text: str):
    """Yield (block_text) for every top-level ``Solid { ... }`` in the world."""
    i = 0
    n = len(text)
    while True:
        idx = text.find("Solid", i)
        if idx < 0:
            return
        brace = text.find("{", idx)
        if brace < 0:
            return
        depth = 0
        j = brace
        while j < n:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[idx:j + 1]
                    break
            j += 1
        i = j + 1


def _first_float_tuple(block: str, key: str, count: int) -> Optional[List[float]]:
    m = re.search(key + r"\s+((?:" + _NUM + r"\s+){" + str(count - 1) + r"}" + _NUM + r")", block)
    if not m:
        return None
    return [float(v) for v in m.group(1).split()]


def _bounding_shape(block: str) -> Optional[Dict]:
    """Extract the boundingObject collision shape (Box or Cylinder)."""
    bo = re.search(r"boundingObject\s+(Box|Cylinder)\s*\{([^}]*)\}", block)
    if not bo:
        return None
    kind, body = bo.group(1), bo.group(2)
    if kind == "Box":
        size = _first_float_tuple(body, "size", 3)
        if size is None:
            return None
        return {"shape": "box", "sx": size[0], "sy": size[1], "sz": size[2]}
    radius = re.search(r"radius\s+(" + _NUM + r")", body)
    height = re.search(r"height\s+(" + _NUM + r")", body)
    if radius is None or height is None:
        return None
    return {"shape": "cylinder", "r": float(radius.group(1)), "sz": float(height.group(1))}


def parse_world_obstacles(wbt_path) -> Dict:
    """Parse the .wbt and return arena size + collision obstacle solids.

    Returns ``{"arena": (floor_x, floor_y) or None, "obstacles": [ ... ]}`` where
    each obstacle has cx, cy, cz (center), yaw (rad about z), shape and dims.
    """
    text = Path(wbt_path).read_text(encoding="utf-8", errors="ignore")
    arena = None
    am = re.search(r"RectangleArena\s*\{[^}]*?floorSize\s+(" + _NUM + r")\s+(" + _NUM + r")", text, re.DOTALL)
    if am:
        arena = (float(am.group(1)), float(am.group(2)))

    obstacles: List[Dict] = []
    for block in _find_top_level_solids(text):
        shape = _bounding_shape(block)
        if shape is None:
            # No collision geometry -> visual only (carpet, wall markers, dock).
            continue
        tr = _first_float_tuple(block, "translation", 3)
        if tr is None:
            continue
        rot = _first_float_tuple(block, "rotation", 4)
        # Yaw about +Z only; tilts about X/Y (e.g. wall markers) are filtered out
        # by height anyway, so treat their footprint as axis-aligned.
        yaw = 0.0
        if rot is not None and abs(rot[0]) < 1e-6 and abs(rot[1]) < 1e-6 and abs(rot[2]) > 1e-6:
            yaw = float(rot[3]) * (1.0 if rot[2] > 0 else -1.0)
        nm = re.search(r'name\s+"([^"]*)"', block)
        obstacles.append({
            "name": nm.group(1) if nm else "?",
            "cx": tr[0], "cy": tr[1], "cz": tr[2],
            "yaw": yaw,
            **shape,
        })
    return {"arena": arena, "obstacles": obstacles}


def _rotated_box_polygon(cx, cy, sx, sy, yaw):
    hx, hy = sx / 2.0, sy / 2.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    return [(cx + dx * cyaw - dy * syaw, cy + dx * syaw + dy * cyaw) for dx, dy in corners]


def build_ground_truth_grid(
    parsed: Dict,
    world_to_map,
    map_size: int,
    robot_height_m: float = 0.12,
    floor_eps_m: float = 0.02,
    wall_thickness_m: float = 0.06,
    map_scale: float = 85.0,
):
    """Rasterise collision footprints into a bool occupancy grid (map frame).

    ``world_to_map`` is ``f(wx, wy) -> (mx, my)`` supplied by the controller so the
    reference uses exactly the controller's alignment.  Only obstacles whose
    vertical extent overlaps ``[floor_eps_m, robot_height_m]`` are drawn.
    """
    if cv2 is None:
        raise RuntimeError("cv2 required for ground-truth rasterisation")
    grid = np.zeros((map_size, map_size), dtype=np.uint8)

    def draw_box(cx, cy, sx, sy, yaw):
        poly = _rotated_box_polygon(cx, cy, sx, sy, yaw)
        pts = np.array([world_to_map(wx, wy) for wx, wy in poly], dtype=np.int32)
        cv2.fillConvexPoly(grid, pts, 1)

    # Arena perimeter walls (a ring of wall_thickness around the floor rectangle).
    arena = parsed.get("arena")
    if arena is not None:
        fx, fy = arena
        hx, hy = fx / 2.0, fy / 2.0
        t = wall_thickness_m
        # four wall strips centred on each edge
        draw_box(0.0, hy, fx + 2 * t, t, 0.0)   # north (+y)
        draw_box(0.0, -hy, fx + 2 * t, t, 0.0)  # south (-y)
        draw_box(hx, 0.0, t, fy + 2 * t, 0.0)   # east (+x)
        draw_box(-hx, 0.0, t, fy + 2 * t, 0.0)  # west (-x)

    drawn = 0
    for o in parsed.get("obstacles", []):
        z_bottom = o["cz"] - o["sz"] / 2.0
        z_top = o["cz"] + o["sz"] / 2.0
        # Keep only obstacles that actually intrude into the robot's height band.
        if z_bottom > robot_height_m or z_top < floor_eps_m:
            continue
        if o["shape"] == "box":
            draw_box(o["cx"], o["cy"], o["sx"], o["sy"], o["yaw"])
        else:  # cylinder
            cx_px, cy_px = world_to_map(o["cx"], o["cy"])
            rad_px = max(1, int(round(o["r"] * map_scale)))
            cv2.circle(grid, (int(cx_px), int(cy_px)), rad_px, 1, -1)
        drawn += 1
    return grid.astype(np.bool_), drawn


def false_cell_metrics(robot_obstacles, gt_obstacles, evaluation_mask=None) -> Dict:
    """Compare the learned obstacle mask against the ground-truth reference.

    ``evaluation_mask`` restricts the comparison to explored/known cells so the
    still-unknown part of the map is not unfairly counted as error.  Returns
    false-positive (occupied but truly free) and false-negative ratios.
    """
    robot = np.asarray(robot_obstacles, dtype=bool)
    gt = np.asarray(gt_obstacles, dtype=bool)
    if evaluation_mask is not None:
        m = np.asarray(evaluation_mask, dtype=bool)
        robot = robot & m
        gt = gt & m
    robot_occ = int(np.count_nonzero(robot))
    gt_occ = int(np.count_nonzero(gt))
    false_pos = int(np.count_nonzero(robot & (~gt)))   # robot says obstacle, truly free
    false_neg = int(np.count_nonzero((~robot) & gt))   # robot missed a real obstacle
    true_pos = int(np.count_nonzero(robot & gt))
    return {
        "gt_obstacle_cells": gt_occ,
        "robot_obstacle_cells": robot_occ,
        "false_occupied_cells": false_pos,
        "false_free_cells": false_neg,
        "true_occupied_cells": true_pos,
        "false_occupied_ratio": (false_pos / robot_occ) if robot_occ else 0.0,
        "obstacle_precision": (true_pos / robot_occ) if robot_occ else 0.0,
        "obstacle_recall": (true_pos / gt_occ) if gt_occ else 0.0,
    }
