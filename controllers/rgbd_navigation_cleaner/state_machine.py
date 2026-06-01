"""
High-level mission phase arbitration for the Webots RGB-D cleaner prototype.

This is deliberately small and deterministic. It does not replace the existing
planner; it gives the monolithic controller an explicit phase label so decisions
can be gated and debugged as:

    safety/recovery > exploration > coverage > final cleanup

The values are strings to keep logging and old controller code simple.
"""

from enum import Enum


class NavigationPhase(str, Enum):
    EXPLORE = "EXPLORE"
    COVERAGE = "COVERAGE"
    PLANNED_COVERAGE = "PLANNED_COVERAGE"
    FINISH_CLEANUP = "FINISH_CLEANUP"
    RETURN_HOME = "RETURN_HOME"
    RECOVERY = "RECOVERY"


RECOVERY_STATE_NAMES = {
    "RECOVERY_BACKUP",
    "LEG_ESCAPE_BACKUP",
    "LEG_ESCAPE_TURN",
    "LEG_ESCAPE_FORWARD",
    "PRE_PIVOT_BACKUP",
    "CONTACT_BACKUP",
    "CONTACT_WAIT_CLEAR",
    "CONTACT_ROTATE",
    "CONTACT_FORWARD",
}


def update_navigation_phase(
    *,
    nav_state: str,
    coverage_percent: float,
    frontier_cells: int,
    uncleaned_cells: int,
    sim_time: float,
    route_kind: str,
    explore_percent: float = 38.0,
    finish_percent: float = 68.0,
    min_explore_seconds: float = 35.0,
) -> tuple[str, str]:
    """Return (phase, reason) for the current robot state.

    The thresholds are intentionally conservative: a sparse early map should be
    explored with stable strips/wall-boundary behaviour, not with a global route
    that chases every new residual island.  Late cleanup starts only after most
    reachable known floor has already been covered.
    """
    nav = str(nav_state or "")
    if nav in RECOVERY_STATE_NAMES or nav.startswith("CONTACT_"):
        return NavigationPhase.RECOVERY.value, f"recovery nav={nav}"

    cov = float(coverage_percent or 0.0)
    frontiers = int(frontier_cells or 0)
    uncleaned = int(uncleaned_cells or 0)
    t = float(sim_time or 0.0)

    if cov >= finish_percent:
        return NavigationPhase.FINISH_CLEANUP.value, f"coverage {cov:.1f}% >= {finish_percent:.1f}%"

    # If the map is still very young, stay in exploration even if the raw
    # coverage percentage jumps because only a small region is known.
    if cov < explore_percent or t < min_explore_seconds:
        return NavigationPhase.EXPLORE.value, (
            f"map build cov={cov:.1f}%/{explore_percent:.1f}% "
            f"t={t:.0f}s/{min_explore_seconds:.0f}s frontiers={frontiers}"
        )

    # A mostly-covered but still frontier-rich map is still ordinary coverage:
    # continue strips and boundary expansion instead of chasing tiny leftovers.
    if frontiers > max(80, uncleaned // 18):
        return NavigationPhase.COVERAGE.value, f"coverage with frontiers={frontiers} uncleaned={uncleaned} route={route_kind}"

    return NavigationPhase.COVERAGE.value, f"coverage cov={cov:.1f}% route={route_kind}"
