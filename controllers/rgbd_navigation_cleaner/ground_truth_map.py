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


def _crop_bounds(mask, pad: int):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    return y0, y1, x0, x1


def save_ground_truth_png(gt_grid, path, pad: int = 24) -> bool:
    """Save the static ground-truth obstacle map as a PNG (dark = obstacle)."""
    if cv2 is None:
        return False
    gt = np.asarray(gt_grid, dtype=bool)
    bounds = _crop_bounds(gt, pad)
    if bounds is None:
        return False
    y0, y1, x0, x1 = bounds
    crop = gt[y0:y1, x0:x1]
    img = np.full((crop.shape[0], crop.shape[1], 3), 245, dtype=np.uint8)
    img[crop] = (60, 60, 60)
    cv2.putText(img, "ground truth (from .wbt)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return True


def _dilate_mask(mask, tolerance_px: int):
    """Grow a boolean mask by ``tolerance_px`` cells (a match-tolerance band).

    The learned obstacle map is rarely aligned to the ground-truth geometry to
    the exact cell: depth/odometry noise and the finite map resolution put the
    robot's obstacle one or a few cells off the true edge.  Counting that as a
    full error is too strict, so a cell counts as a match when a real obstacle
    sits within this tolerance.  Returns the mask unchanged if tolerance<=0 or
    cv2 is unavailable.
    """
    if tolerance_px <= 0 or cv2 is None:
        return mask
    r = int(tolerance_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1) > 0


def save_comparison_png(robot_obstacles, gt_grid, path, evaluation_mask=None, pad: int = 24,
                        tolerance_px: int = 0) -> bool:
    """Save an overlay of the learned obstacle map vs ground truth.

    Colours (BGR): grey = correct obstacle (matched within ``tolerance_px``),
    red = false-occupied (robot says obstacle, none within tolerance), blue =
    false-free (real obstacle the robot missed within tolerance).  Restricted to
    ``evaluation_mask`` (explored cells) when given.
    """
    if cv2 is None:
        return False
    robot = np.asarray(robot_obstacles, dtype=bool)
    gt = np.asarray(gt_grid, dtype=bool)
    if evaluation_mask is not None:
        m = np.asarray(evaluation_mask, dtype=bool)
        robot = robot & m
    bounds = _crop_bounds(gt | robot, pad)
    if bounds is None:
        return False
    y0, y1, x0, x1 = bounds
    r = robot[y0:y1, x0:x1]
    g = gt[y0:y1, x0:x1]
    # Tolerance bands: a robot cell is correct if a real obstacle is within
    # tolerance, and a real obstacle is covered if the robot built one nearby.
    g_dil = _dilate_mask(g, tolerance_px)
    r_dil = _dilate_mask(r, tolerance_px)
    matched = (r & g_dil) | (g & r_dil)   # grey: agreement within tolerance
    img = np.full((r.shape[0], r.shape[1], 3), 245, dtype=np.uint8)
    img[matched] = (110, 110, 110)        # true positive  - grey
    img[g & (~r_dil)] = (200, 90, 0)      # false free      - blue (missed)
    img[r & (~g_dil)] = (0, 0, 220)       # false occupied  - red
    label = "grey=ok  red=false-occupied  blue=missed"
    if tolerance_px > 0:
        label += f"  (tol={int(tolerance_px)}px)"
    cv2.putText(img, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return True


def false_cell_metrics(robot_obstacles, gt_obstacles, evaluation_mask=None,
                       tolerance_px: int = 0) -> Dict:
    """Compare the learned obstacle mask against the ground-truth reference.

    ``evaluation_mask`` restricts the comparison to explored/known cells so the
    still-unknown part of the map is not unfairly counted as error.
    ``tolerance_px`` allows a match band so a robot obstacle that is a few cells
    off the true edge still counts as correct.  Returns false-positive (occupied
    but truly free) and false-negative ratios.
    """
    robot = np.asarray(robot_obstacles, dtype=bool)
    gt = np.asarray(gt_obstacles, dtype=bool)
    if evaluation_mask is not None:
        m = np.asarray(evaluation_mask, dtype=bool)
        robot = robot & m
        gt = gt & m
    robot_occ = int(np.count_nonzero(robot))
    gt_occ = int(np.count_nonzero(gt))
    gt_dil = _dilate_mask(gt, tolerance_px)
    robot_dil = _dilate_mask(robot, tolerance_px)
    # A robot cell is a true positive if a real obstacle is within tolerance.
    robot_match = robot & gt_dil
    false_pos = int(np.count_nonzero(robot & (~gt_dil)))   # obstacle, none truly near
    # A real obstacle is covered if the robot built something within tolerance.
    gt_covered = int(np.count_nonzero(gt & robot_dil))
    false_neg = int(np.count_nonzero(gt & (~robot_dil)))   # real obstacle missed
    true_pos = int(np.count_nonzero(robot_match))
    return {
        "gt_obstacle_cells": gt_occ,
        "robot_obstacle_cells": robot_occ,
        "false_occupied_cells": false_pos,
        "false_free_cells": false_neg,
        "true_occupied_cells": true_pos,
        "false_occupied_ratio": (false_pos / robot_occ) if robot_occ else 0.0,
        "obstacle_precision": (true_pos / robot_occ) if robot_occ else 0.0,
        "obstacle_recall": (gt_covered / gt_occ) if gt_occ else 0.0,
    }
