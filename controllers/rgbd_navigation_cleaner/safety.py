"""Safety arbitration helpers for physical bumper events."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BumperSafetyEvent:
    active: bool
    left: bool
    center: bool
    right: bool
    single_side: bool
    both_or_center: bool
    side_sign: float
    description: str


def bumper_event_from_raw(left: bool, center: bool, right: bool, ignored_as_floor: bool = False) -> BumperSafetyEvent:
    if ignored_as_floor:
        return BumperSafetyEvent(False, False, False, False, False, False, 0.0, "ignored floor/rug grace")
    l = bool(left)
    c = bool(center)
    r = bool(right)
    active = bool(l or c or r)
    both_or_center = bool(c or (l and r))
    single_side = bool((l ^ r) and not c)
    side_sign = -1.0 if l and not r else (1.0 if r and not l else 0.0)
    if not active:
        desc = "clear"
    elif both_or_center:
        desc = "front/center physical contact"
    else:
        desc = "single-side physical contact"
    return BumperSafetyEvent(active, l, c, r, single_side, both_or_center, side_sign, desc)
