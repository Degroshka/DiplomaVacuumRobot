"""
Shared lightweight state types for future refactoring.

These types are deliberately not wired into the controller yet. The next safe
step is to replace large groups of globals with these explicit objects after a
baseline Webots run is recorded.
"""

from dataclasses import dataclass


@dataclass
class RobotPose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class WheelCommand:
    left: float = 0.0
    right: float = 0.0


@dataclass
class BumperState:
    left: bool = False
    center: bool = False
    right: bool = False
    raw_left: float = 0.0
    raw_center: float = 0.0
    raw_right: float = 0.0
