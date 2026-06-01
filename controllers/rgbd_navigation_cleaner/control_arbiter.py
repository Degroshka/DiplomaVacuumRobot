"""Deterministic control-ownership helpers for the Webots RGB-D cleaner.

This module is intentionally small: it does not plan paths and it does not read
Webots devices.  The monolithic legacy controller can use it to make one layer
own the wheels for a short atomic window, while safety/recovery remains allowed
to pre-empt everything.
"""

from dataclasses import dataclass
from enum import Enum
import math


class ControlOwner(str, Enum):
    NONE = "NONE"
    SAFETY = "SAFETY"
    RECOVERY = "RECOVERY"
    GRID_REALIGN = "GRID_REALIGN"
    PIVOT_90 = "PIVOT_90"
    LANE_SHIFT = "LANE_SHIFT"
    ROW_FORWARD = "ROW_FORWARD"
    SIMPLE_SWEEP = "SIMPLE_SWEEP_FSM"
    ROUTE_COMMIT = "ROUTE_COMMIT"
    SCAN_AROUND = "SCAN_AROUND"
    RGBD_SNAPSHOT = "RGBD_SNAPSHOT"
    PLANNER = "PLANNER"


@dataclass
class OwnershipLock:
    owner: str = ControlOwner.NONE.value
    reason: str = "startup"
    start_time: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    min_until: float = 0.0
    min_distance: float = 0.0

    def acquire(self, owner: str, *, now: float, x: float, y: float, min_time: float = 0.0, min_distance: float = 0.0, reason: str = "") -> None:
        self.owner = str(owner)
        self.reason = str(reason or owner)
        self.start_time = float(now)
        self.start_x = float(x)
        self.start_y = float(y)
        self.min_until = float(now) + max(0.0, float(min_time))
        self.min_distance = max(0.0, float(min_distance))

    def release(self, reason: str = "released") -> None:
        self.owner = ControlOwner.NONE.value
        self.reason = str(reason)
        self.min_until = 0.0
        self.min_distance = 0.0

    def distance_from_start(self, x: float, y: float) -> float:
        return math.hypot(float(x) - self.start_x, float(y) - self.start_y)

    def active(self, *, now: float, x: float, y: float) -> bool:
        if self.owner == ControlOwner.NONE.value:
            return False
        time_locked = float(now) < self.min_until
        dist_locked = self.min_distance > 0.0 and self.distance_from_start(x, y) < self.min_distance
        return bool(time_locked or dist_locked)

    def debug(self, *, now: float, x: float, y: float) -> str:
        d = self.distance_from_start(x, y)
        remain = max(0.0, self.min_until - float(now))
        return f"{self.owner}:{self.reason[:18]} t={remain:.1f}s d={d:.2f}/{self.min_distance:.2f}"
