"""Small recovery-policy helpers split out from the main controller."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SideReleaseDecision:
    force_release_turn: bool
    turn_side: float
    reason: str


def decide_side_release_after_backup(
    *,
    bumper_left: bool,
    bumper_center: bool,
    bumper_right: bool,
    moved_m: float,
    backup_elapsed_s: float,
    front_m: float,
    center_m: float,
    body_clearance_m: float,
    min_moved_m: float = 0.18,
    max_elapsed_s: float = 1.25,
    min_front_clear_m: float = 0.16,
    min_center_clear_m: float = 0.16,
    min_body_clear_m: float = 0.030,
) -> SideReleaseDecision:
    """Allow a slow in-place side-release turn if reverse did not clear a side bumper.

    This is not a normal navigation turn. It is a bounded recovery primitive for
    the specific case where a side/front-arc bumper remains compressed while the
    robot has already reversed far enough. Keeping reverse forever makes the
    robot visibly drive backward down the wall.
    """
    l = bool(bumper_left)
    c = bool(bumper_center)
    r = bool(bumper_right)
    if not (l or c or r):
        return SideReleaseDecision(False, 0.0, "bumper already clear")
    if c or (l and r):
        return SideReleaseDecision(False, 0.0, "front/center contact: keep reverse/hold")
    if moved_m < min_moved_m and backup_elapsed_s < max_elapsed_s:
        return SideReleaseDecision(False, 0.0, f"backup not exhausted moved={moved_m:.2f} elapsed={backup_elapsed_s:.2f}")
    if front_m < min_front_clear_m or center_m < min_center_clear_m or body_clearance_m < min_body_clear_m:
        return SideReleaseDecision(False, 0.0, f"not enough clearance F={front_m:.2f} C={center_m:.2f} body={body_clearance_m:.2f}")
    # Turn away from the pressed side. If right bumper is pressed, rotate left;
    # if left is pressed, rotate right. This matches the controller convention
    # where positive side increases heading target.
    turn_side = -1.0 if r else 1.0
    return SideReleaseDecision(True, turn_side, f"side bumper release after backup moved={moved_m:.2f} elapsed={backup_elapsed_s:.2f}")
