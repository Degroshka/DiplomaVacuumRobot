"""Local wall-follow controller for the Webots RGB-D vacuum prototype.

This module is intentionally small and stateless.  It does not plan a route and
it does not decide when a row ends.  It only converts a local wall-clearance
estimate into a tiny differential-wheel correction while the higher-level
controller is already in ordinary FORWARD motion.

Priority in the main controller must stay:
    bumper/safety -> recovery -> wall follow -> row forward / coverage
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass(frozen=True)
class WallFollowConfig:
    enabled: bool = True
    target_clearance_m: float = 0.105
    too_close_m: float = 0.065
    acquire_max_m: float = 0.34
    release_max_m: float = 0.46
    min_front_m: float = 0.48
    min_center_m: float = 0.42
    min_upper_m: float = 0.42
    max_yaw: float = 0.22
    kp_distance: float = 1.45
    kp_heading: float = 0.72
    speed: float = 0.90
    slow_speed: float = 0.48
    deadband_m: float = 0.012
    side_switch_margin_m: float = 0.055
    # The RangeFinder side thirds are frontal oblique rays, not true side
    # distance sensors.  They are useful for near-wall correction, but a
    # depth-only reading around 25-35 cm must not make the robot deliberately
    # dive toward the wall/table leg.  Long-range boundary acquisition should
    # come from the remembered occupancy/contact map or from explicit corner
    # logic, not from continuous differential yaw.
    depth_follow_max_m: float = 0.195
    depth_far_pull_enabled: bool = False
    map_target_center_m: float = 0.305
    map_too_close_center_m: float = 0.245
    map_acquire_max_m: float = 0.46
    map_weight: float = 0.65
    allow_map_scrape_pivot: bool = False


@dataclass(frozen=True)
class WallFollowInput:
    front_m: float
    center_m: float
    upper_front_m: float
    left_depth_m: float
    right_depth_m: float
    body_clearance_m: float
    body_lateral_m: float
    left_map_m: float
    right_map_m: float
    current_heading_rad: float
    target_heading_rad: float
    base_speed: float
    active_side: float = 0.0  # +1 left wall, -1 right wall, 0 not locked
    bumper_active: bool = False


@dataclass(frozen=True)
class WallFollowCommand:
    left_speed: float
    right_speed: float
    side: float
    yaw: float
    clearance_m: float
    source: str
    reason: str


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _side_metric(depth_m: float, map_m: float, cfg: WallFollowConfig) -> tuple[float, str]:
    """Merge depth and map clearance into one side-distance estimate.

    The front RangeFinder side thirds are *not* true side sensors: along a wall
    they often measure an oblique front-wall point and can say "far" while the
    circular shell is already scraping.  Therefore remembered occupancy/contact
    clearance has veto priority when it says the wall is too close.  Depth is
    still useful for live acquisition, but it must not override a close map ray.
    """
    if _finite_positive(map_m) and map_m < cfg.map_too_close_center_m:
        return float(map_m), "map"

    candidates: list[tuple[float, str, float]] = []
    depth_limit = min(cfg.release_max_m, cfg.depth_follow_max_m)
    if _finite_positive(depth_m) and depth_m < depth_limit:
        candidates.append((float(depth_m), "depth", 1.0))
    if _finite_positive(map_m) and map_m < cfg.map_acquire_max_m:
        candidates.append((float(map_m), "map", cfg.map_weight))
    if not candidates:
        return 999.0, "none"

    # Prefer the source that indicates the smaller risk.  If the two are close,
    # keep depth because it is live; if map is much closer, use map to avoid
    # scraping a wall that is parallel to the camera view.
    candidates.sort(key=lambda item: (item[0], -item[2]))
    return candidates[0][0], candidates[0][1]


def compute_wall_follow_command(inp: WallFollowInput, cfg: WallFollowConfig = WallFollowConfig()) -> Optional[WallFollowCommand]:
    if not cfg.enabled or inp.bumper_active:
        return None
    if inp.front_m < cfg.min_front_m or inp.center_m < cfg.min_center_m or inp.upper_front_m < cfg.min_upper_m:
        return None

    left_metric, left_source = _side_metric(inp.left_depth_m, inp.left_map_m, cfg)
    right_metric, right_source = _side_metric(inp.right_depth_m, inp.right_map_m, cfg)

    side = 0.0
    source = "none"
    clearance = 999.0

    if inp.active_side > 0 and left_metric < cfg.release_max_m:
        side = 1.0
        clearance = left_metric
        source = left_source
    elif inp.active_side < 0 and right_metric < cfg.release_max_m:
        side = -1.0
        clearance = right_metric
        source = right_source
    elif left_metric < cfg.acquire_max_m or right_metric < cfg.acquire_max_m:
        if left_metric + cfg.side_switch_margin_m < right_metric:
            side = 1.0
            clearance = left_metric
            source = left_source
        elif right_metric + cfg.side_switch_margin_m < left_metric:
            side = -1.0
            clearance = right_metric
            source = right_source
        else:
            return None
    else:
        return None

    # Body-corridor evidence is a last-resort side warning.  If it says the shell
    # is almost touching on one side, force a correction away from that side even
    # when the side depth ray is noisy.
    if _finite_positive(inp.body_clearance_m) and inp.body_clearance_m < cfg.too_close_m + 0.030 and abs(inp.body_lateral_m) > 0.025:
        side = 1.0 if inp.body_lateral_m > 0.0 else -1.0
        clearance = min(clearance, inp.body_clearance_m)
        source = "body"

    # Depth clearances are shell-like estimates; map clearances are centre-to-wall
    # estimates.  Use a different target if the selected evidence is mostly map.
    target = cfg.map_target_center_m if source == "map" else cfg.target_clearance_m
    too_close = cfg.map_too_close_center_m if source == "map" else cfg.too_close_m

    err_dist = clearance - target
    if abs(err_dist) < cfg.deadband_m:
        err_dist = 0.0

    err_heading = normalize_angle(inp.target_heading_rad - inp.current_heading_rad)
    yaw = cfg.kp_heading * err_heading + side * cfg.kp_distance * err_dist
    yaw = clamp(yaw, -cfg.max_yaw, cfg.max_yaw)

    base = min(inp.base_speed, cfg.speed)
    reason_state = "hold"
    hard_scrape_risk = bool(
        (source == "map" and cfg.allow_map_scrape_pivot and clearance < max(0.02, too_close - 0.035))
        or (source == "body" and clearance < cfg.too_close_m + 0.006)
    )
    if clearance < too_close:
        # Do not drag the shell along the wall.  If the remembered map/body
        # clearance says we are already in the scrape band, first pivot away with
        # almost no forward component; then resume parallel trace.
        if hard_scrape_risk:
            base = 0.0
            yaw = -side * min(cfg.max_yaw, max(0.11, abs(yaw)))
            reason_state = "scrape-pivot"
        else:
            # Map-only proximity is not a reliable reason for an in-place pivot:
            # map thickness and pose drift can put an occupied cell inside the
            # side ray even when the shell is only near furniture.  Keep moving
            # slowly away; real physical contact is handled by bumper/recovery.
            base = max(0.22, min(base, cfg.slow_speed))
            yaw = -side * max(abs(yaw), min(cfg.max_yaw, 0.10))
            reason_state = "too-close-crawl"
    elif clearance < target:
        base = min(base, cfg.slow_speed)
        reason_state = "near"
    elif err_dist > 0.0:
        # A depth-side "far" reading often means the wall is merely visible in
        # the frontal RGB-D cone, not that it is a reliable side wall.  In step29
        # this produced the visible pattern "go a little forward -> turn right"
        # because right-hand perimeter tracing kept pulling toward a 28-30 cm
        # oblique depth return.  Unless explicitly enabled, do not use depth-only
        # far evidence to acquire a wall; keep only the heading lock and let the
        # row/recovery logic handle the next real boundary.
        if source == "depth" and not cfg.depth_far_pull_enabled:
            yaw = clamp(cfg.kp_heading * err_heading, -0.025, 0.025)
            base = min(base, cfg.slow_speed)
            reason_state = "depth-far-ignore"
        else:
            if source == "depth":
                yaw = clamp(yaw, -0.050, 0.050)
                base = min(base, cfg.slow_speed)
            reason_state = "far"

    left_speed = base - yaw
    right_speed = base + yaw
    return WallFollowCommand(
        left_speed=left_speed,
        right_speed=right_speed,
        side=side,
        yaw=yaw,
        clearance_m=clearance,
        source=source,
        reason=(
            f"wall-follow {reason_state} side={'L' if side > 0 else 'R'} "
            f"src={source} clr={clearance:.2f} tgt={target:.2f} yaw={yaw:.2f}"
        ),
    )
