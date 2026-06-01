"""
Mapping-level pure helpers for the Webots RGB-D navigation prototype.

This module intentionally contains only small stateless helpers in the first
current implementation. Persistent grid state remains in the main controller.
"""

import math


def clamp(value, low, high):
    """Clamp a numeric value into [low, high]."""
    return max(low, min(high, value))


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def snap_to_right_angle(angle):
    """Snap heading to the nearest 90-degree grid direction."""
    return normalize_angle(round(angle / (math.pi / 2.0)) * (math.pi / 2.0))


def world_to_map_cell(wx, wy, origin_x, origin_y, scale):
    """Convert world coordinates in meters to occupancy-map cell coordinates."""
    mx = int(origin_x + wx * scale)
    my = int(origin_y - wy * scale)
    return mx, my


def map_inside_cell(mx, my, map_size):
    """Return True if a map cell is inside a square map."""
    return 0 <= mx < map_size and 0 <= my < map_size


def bresenham_cells(x0, y0, x1, y1):
    """Yield integer line cells from start to end using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def update_log_odds_value(current_value, delta, low, high):
    """Return a saturated log-odds value after applying delta."""
    return clamp(current_value + delta, low, high)
