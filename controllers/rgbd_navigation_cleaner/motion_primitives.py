"""
Motion primitive contract for the Webots RGB-D cleaner prototype.

The old controller can output arbitrary left/right wheel speeds. This module
adds a safety/clarity layer: normal coverage/exploration motion should be either
straight or an in-place turn. Curved motion is reserved for explicit exceptions
such as wall following, narrow-passage centering and recovery.
"""

from dataclasses import dataclass
from enum import Enum


class MotionPrimitive(str, Enum):
    STOP = "STOP"
    MOVE_FORWARD_CELL = "MOVE_FORWARD_CELL"
    BACKUP_SHORT = "BACKUP_SHORT"
    TURN_IN_PLACE = "TURN_IN_PLACE"
    LANE_SHIFT = "LANE_SHIFT"
    WALL_FOLLOW_STEP = "WALL_FOLLOW_STEP"
    FINE_ALIGN = "FINE_ALIGN"
    CONTACT_BACKUP = "CONTACT_BACKUP"
    CONTACT_RELEASE_TURN = "CONTACT_RELEASE_TURN"
    LEG_ESCAPE = "LEG_ESCAPE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MotionContractResult:
    left: float
    right: float
    primitive: str
    reason: str
    adjusted: bool = False


ARC_ALLOWED_STATES = {
    "LANE_SHIFT",
    "LEG_PASS_FORWARD",
    "LEG_PASS_ALIGN",
    "CONTACT_FORWARD",
}

ARC_ALLOWED_STATUS_KEYWORDS = (
    # Broad exceptions like generic wall/post-recovery/line-acquire reintroduced
    # visible diagonal motion.  Keep only named, bounded micro-manoeuvres here.
    "wall-follow",
    "narrow",
    "gap mouth",
    "gap commit",
    "under furniture",
    "frontier pursuit",
)

RECOVERY_STATE_PREFIXES = ("CONTACT_", "LEG_ESCAPE", "RECOVERY_", "PRE_PIVOT")


def classify_motion(nav_state: str, left: float, right: float, status: str = "") -> str:
    nav = str(nav_state or "")
    l = float(left)
    r = float(right)
    avg = 0.5 * (l + r)
    diff = 0.5 * (r - l)
    if abs(l) < 1e-6 and abs(r) < 1e-6:
        return MotionPrimitive.STOP.value
    if nav == "CONTACT_BACKUP":
        return MotionPrimitive.CONTACT_BACKUP.value
    if nav == "CONTACT_ROTATE":
        return MotionPrimitive.CONTACT_RELEASE_TURN.value
    if nav.startswith("LEG_ESCAPE"):
        return MotionPrimitive.LEG_ESCAPE.value
    if nav == "LANE_SHIFT":
        return MotionPrimitive.LANE_SHIFT.value
    if abs(avg) < 0.20 and abs(diff) > 0.05:
        return MotionPrimitive.TURN_IN_PLACE.value
    if avg < -0.05 and abs(diff) < max(0.08, abs(avg) * 0.35):
        return MotionPrimitive.BACKUP_SHORT.value
    if _arc_allowed(nav, status):
        return MotionPrimitive.WALL_FOLLOW_STEP.value if "wall" in str(status).lower() else MotionPrimitive.FINE_ALIGN.value
    if avg > 0.05:
        return MotionPrimitive.MOVE_FORWARD_CELL.value
    return MotionPrimitive.UNKNOWN.value


def _arc_allowed(nav_state: str, status: str) -> bool:
    nav = str(nav_state or "")
    if nav in ARC_ALLOWED_STATES:
        return True
    if any(nav.startswith(prefix) for prefix in RECOVERY_STATE_PREFIXES):
        return True
    low = str(status or "").lower()
    return any(key in low for key in ARC_ALLOWED_STATUS_KEYWORDS)


def apply_strict_motion_contract(
    *,
    nav_state: str,
    phase: str,
    left: float,
    right: float,
    status: str,
    enabled: bool = True,
    curve_diff_threshold: float = 0.10,
    pivot_min_speed: float = 0.55,
    pivot_max_speed: float = 1.75,
) -> MotionContractResult:
    """Convert accidental arcs into straight steps or in-place pivots.

    Normal coverage/exploration should not draw diagonal arcs. If the legacy
    controller asks for a forward curve, this function either makes it straight
    when the correction is tiny or turns it into an in-place alignment pivot when
    the yaw correction is large enough to be visible.
    """
    l = float(left)
    r = float(right)
    primitive = classify_motion(nav_state, l, r, status)
    if not enabled:
        return MotionContractResult(l, r, primitive, "strict contract disabled", False)

    nav = str(nav_state or "")
    if _arc_allowed(nav, status):
        return MotionContractResult(l, r, primitive, f"arc allowed by {nav or 'status'}", False)

    avg = 0.5 * (l + r)
    diff = 0.5 * (r - l)
    same_direction = (l > 0 and r > 0) or (l < 0 and r < 0)

    if same_direction and abs(diff) > curve_diff_threshold:
        # Curved forward/backward motion is the main source of diagonal tracks.
        # Convert substantial yaw correction to a pivot; keep tiny corrections as
        # a straight primitive so the controller does not arc across the map.
        if abs(avg) >= 0.18:
            sign = 1.0 if diff > 0.0 else -1.0
            pivot = max(pivot_min_speed, min(pivot_max_speed, abs(diff) * 1.20))
            return MotionContractResult(
                -sign * pivot,
                sign * pivot,
                MotionPrimitive.FINE_ALIGN.value,
                f"strict: curve->pivot phase={phase} nav={nav} avg={avg:.2f} diff={diff:.2f}",
                True,
            )

    if same_direction and abs(diff) <= curve_diff_threshold and abs(diff) > 1e-6:
        return MotionContractResult(
            avg,
            avg,
            MotionPrimitive.MOVE_FORWARD_CELL.value if avg > 0 else MotionPrimitive.BACKUP_SHORT.value,
            f"strict: tiny curve->straight phase={phase} nav={nav} diff={diff:.2f}",
            True,
        )

    return MotionContractResult(l, r, primitive, f"strict: accepted {primitive}", False)
