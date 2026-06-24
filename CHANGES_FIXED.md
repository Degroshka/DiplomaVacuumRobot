# cleaning_robot_fixed — working version

Forked from `rollback_atomic_work` (clean source copy; run artifacts `maps/`,
`metrics/`, `__pycache__/` and `.git` excluded — they regenerate on first run).

Goal of this fork: fix the four symptoms seen after ~40 min runs:
1. Robot camps the **right wall** (x≈8.1) for minutes, oscillating in y.
2. Room is **never closed / looped**.
3. **Obstacles are never "finished"** — interior furniture stays half-mapped.
4. Lots of **noisy floating points**.

Symptoms 1–3 share one root cause. See below.

---

## Root cause (symptoms 1, 2, 3)

The gray-gap shadow classifier in `build_gap_frontier()`
(`navigation_controller.py`) used this test:

```python
shadow_like = (clean_ratio < 0.16) or (obs_ratio > 0.50) or (...)
```

The first clause flagged **any** unknown-gray patch not yet bordered by *cleaned*
floor as a "furniture shadow". The room has ~42 000 such gray cells
(`grayGap=42058/18` in the log), so huge open interior pockets were misclassified.
A patch marked "shadow" is then:

- **rejected** as a frontier target (`continue` / `rejected_shadow`), and
- **stamped** into `hypothesis_obstacle_log_odds`, which is OR-ed into the planning
  obstacle mask whenever `OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING = True`
  (`actual_physical_obstacle_mask()`).

Effect chain, visible in the log:
`huge gray → mass shadow stamps → noGo balloons to 2500–6100 (actual=5374 vs
rawObs=2595) → planner can't commit → frontierGate=holdDrive / frontier-only →
only the right corridor stays reachable → robot pilots the right wall, interior
never gets explored, loop never closes, furniture never finished.`

The existing `RIGHT_WALL_REPEAT_SUPPRESS` penalty (−42) could not help, because the
right-wall target was the *only reachable* frontier — best of a blocked set.

---

## Fixes applied (all behind config flags, A/B-comparable)

### 1. Shadow classifier — the main fix
`navigation_controller.py`, `build_gap_frontier()`:

A gray component is now a furniture shadow **only if it is obstacle-dominated on
its boundary AND bounded in size**:

```python
bounded_shadow    = area <= MAX_COMPONENT_CELLS * GRAY_REVISIT_SHADOW_MAX_AREA_RATIO
obstacle_dominated = obs_ratio > GRAY_REVISIT_MAX_OBSTACLE_EDGE_RATIO
                     or obs_edge > clean_edge * GRAY_REVISIT_OBSTACLE_SHADOW_RATIO
shadow_like = bounded_shadow and obstacle_dominated
```

The `clean_ratio < 0.16` clause is gone. A region merely surrounded by *unknown*
gray (low clean AND low obstacle edge) is now treated as an **unexplored floor
pocket → frontier**, not a wall.

New config (`config.py`):
- `GRAY_REVISIT_SHADOW_MAX_AREA_RATIO = 0.60` — max shadow size, as a fraction of
  `GRAY_REVISIT_MAX_COMPONENT_CELLS`. **Set to `0` to restore the exact legacy
  heuristic** (for A/B comparison).
- `GRAY_REVISIT_SHADOW_REQUIRE_OBSTACLE_EDGE = True` — reserved/documentation flag.

### 2. Shadow stamp size cap
Shadow stamps are capped at `OBSTACLE_HYPOTHESIS_MAX_COMPONENT_CELLS` (3600), so a
single mis-stamp can no longer dump ~9000 cells into the no-go layer (the original
`hypObs 1018 → 10044` spike at t≈658 s).

### 3. Slow decay for hypObs in unknown space
`_hypothesis_clear_on_free_observations()` previously only decayed hypObs cells
that were later observed *free* (`log_odds < -LO_UNKNOWN_EPS`). Shadow cells sit in
**unknown** space (`log_odds ≈ 0`) and never met that condition → permanent walls.
Added a slow global decay so a false stamp clears itself in ~200 s of sim time:
- `OBSTACLE_HYPOTHESIS_UNKNOWN_DECAY_MULT = 2.0` (decay −0.055 × 2.0 per cycle).

### 4. Noisy points — intentionally NOT filtered
The scattered points are raw CV evidence (`visual_log_odds` from Canny/Hough). This
project is **CV-FIRST** — computer vision is the core of the work — so erasing
visual-only pixels would cut the sensor model that matters most. Instead, symptom 4
improves *indirectly*: once the robot actually enters the interior (fix 1), it gets
many more viewpoints and the furniture reconstructs more densely, so fewer points
look like floating noise. If true denoise is still wanted, the safe next step is a
**temporal-persistence** filter (keep a point only after N consecutive confirming
frames) rather than a single-frame morphological wipe.

---

## Escape hatches if the interior is still under-explored

- `OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING = False` — turns the whole hypothesis
  layer into a **visual-only** overlay that no longer blocks the planner. Most
  aggressive "explore everything" mode.
- `GRAY_REVISIT_SHADOW_MAX_AREA_RATIO` — lower it to wall off fewer patches, raise
  toward 0.75 to be closer to the old behavior, set 0 for exact legacy.

---

## What to expect in the next run log (vs. the 40-min baseline)

- `noGo` stays ~2200–2600 instead of spiking to 5000–6100.
- `frontierGate=holdDrive` / `frontier-only` becomes rare instead of dominating.
- Frontier candidates move off `cx≈860–890` (right wall) into the interior
  (`cx≈300–700`).
- Coverage breaks past the ~48 % ceiling and the loop starts to close.

If `noGo` still balloons or the robot still camps the right wall, flip
`OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING = False` and re-run — that isolates whether
remaining blockage is hypObs or real geometry.

---

## Round 2 — clean motion + dock delivers the map (mapping mission)

Mission goal clarified: this is **exploration to BUILD A MAP** (обход для карты), not
cleaning coverage. The frontier explorer is the right paradigm; the problems were
motion quality and the end-of-mission deliverable. The disabled `SIMPLE_SWEEP_FSM`
lawnmower is a *cleaning* pattern and is intentionally left OFF.

### 5. Straight-line motion instead of wobble  ← main motion fix
`config.py`: `EXPLORATION_FRONTIER_POINT_PURSUIT_ENABLED = False` (was `True`).

The route executor `route_commit_speeds()` had two modes:
- **pure-pursuit** (old frontier default): continuously steer toward the next
  waypoint → smooth arcs → the wobbly, curved tracks.
- **Manhattan segment-follow**: align once to the route's cardinal heading, drive
  straight, pivot in place only at real corners.

The planner is **4-connected**, so its routes are already pure cardinal. Turning
point-pursuit off routes frontier motion through `route_commit_grid_segment_heading()`
→ clean **straight lines + crisp 90° pivots** for the whole mapping traversal.
(The strict motion contract also explicitly *allowed arcs* only for the
`"frontier pursuit"` status; with this flag off that status is never produced, so
arcs are converted to straight/pivot everywhere.)

**45° turns:** not produced yet — the planner is 4-connected (cardinal only), so the
robot drives axis-aligned segments. True 45° diagonal traversal needs an 8-connected
(octile) planner with safe-diagonal corner checks; that is a larger, riskier change
I can add as a follow-up after you confirm the cardinal motion looks right.

### 6. Dock auto-emits the finished map
`config.py`: `AUTO_SAVE_MAP_ON_DOCK = True`. `complete_map_return_to_dock()`
(map-only branch) now calls `save_map()` when the robot parks on the dock, writing
`maps/occupancy_map_*.png` + `maps/coverage_objective_map_*.png` (+ metrics)
automatically — so the mission ends with a saved map without the manual `S` key.
`AUTO_LEARNED_MAP_CLEANING_ENABLED` stays `False` (explore → dock → stop; no cleaning).

### What to expect now
- Tracks are straight segments with in-place 90° turns at corners — no curved
  drifting. Far fewer `FINE_ALIGN` ticks; `seg=E/W/N/S` appears in route debug.
- When exploration completes (`AUTO_MAP_COMPLETE_*` thresholds), the robot routes to
  the dock and, on arrival, prints `[MISSION] map auto-saved on dock` and writes the
  map PNGs.

To A/B the motion: set `EXPLORATION_FRONTIER_POINT_PURSUIT_ENABLED = True` to get the
old wobble back.

---

## Round 3 — hypObs is no longer a planning wall (mapping mission)

`config.py`: `OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING = False` (was `True`).

Run log showed `hypObs=626 add=626 comps=1` from t=18s onward — the *regular*
shape-completion path stamped a persistent ~626-cell phantom obstacle, so
`actual = rawObs + 626` and `noGo` was inflated from the very start (this is a
different path from the shadow stamps fixed in Round 1, so the shadow fix did not
touch it). For a mission whose goal is to **map observed reality**, hallucinated
obstacle interiors should not block the planner at all. With this flag off,
`actual_physical_obstacle_mask()` no longer ORs the hypothesis layer into planning;
hypObs is kept purely as a visual overlay. This removes the 626 phantom AND any
remaining shadow walls at the root.

### Diagnosis notes from that run (not yet fixed)
- The dominant motion driver during exploration is `hybrid-row` / `frontier-direct`
  (`gate=frontierGate=hybrid-row`), **not** ROUTE_COMMIT — so the Round-2 point-pursuit
  flag had little effect. Forward headings are actually mostly cardinal
  (0/−91/180/91); the odd angles (49°, 143°, −36°) come from turns/scan/recovery.
- **Contact wedge at an invisible low box** (t≈665→796, ~130 s of `CONTACT_ROTATE`,
  pose stuck 7.6→8.4): the box is below the RGB-D view, the robot bumps it, does a
  compact backup + 90° pivot + forward, and re-hits. There is no repeated-contact
  escalation, and the progress governor keeps resetting because coverage ticks up a
  fraction. → next candidate fix: after N contacts in a small radius/time, do a large
  backup + ~150° turn away + temporary avoid, so the robot leaves decisively.

### What to look for in the re-run
- `actual` should equal `rawObs` (no `+626`); `noGo` much lower from the start.
- Robot should reach interior frontiers instead of being fenced; coverage should
  climb past the old plateau.
- The contact wedge may or may not recur (less no-go can change the trap); if it
  still wedges at a low box for >~30 s, the next step is the contact-escape escalation.

---

## Round 4 — shape-completion no longer paints over observed floor

Re-run after Round 3 confirmed the hypObs no-go fix worked: the robot explored the
**interior** (frontiers at cx≈219–464, not just the right corridor), coverage 9→48 %,
and `actual = rawObs` (no phantom inflation). hypObs is now visual-only.

But the orange shape-completion blob over-filled: it covered cells that had already
been observed as floor and over-extended the rectangular bbox. Cause in
`update_obstacle_hypothesis_cache()`: the per-cell fill excluded only *strong*-free
cells (`log_odds < -2.25`); weakly-observed-free cells were repainted as obstacle.

Fix: the fill now also excludes `observed_free` (`log_odds < -LO_UNKNOWN_EPS` or
cleaned), so a cell the sensors have seen as floor is never repainted. Obstacle
completion still fills genuinely-unknown interior.

Note: because `OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING = False`, hypObs is now purely a
visual overlay. If the orange is still unwanted, `OBSTACLE_HYPOTHESIS_ENABLED = False`
turns it off entirely and the map shows only observed obstacles.

### Open: noise / garbage points
Scattered black points come from (a) close-range RGB-D depth near obstacles (grazing
angles at <~0.35 m project scattered occupied cells — the snapshots-near-obstacles
the user suspected), (b) CV (Canny/Hough) edge speckle, and (c) contact thrashing
(many frames from bad poses while bumping). Candidate fix (not yet applied):
temporal persistence — commit an occupied cell only after N confirming frames — which
denoises without abandoning the CV-FIRST sensor model.

---

## Round 5 — drop "beyond-the-wall" ghost frontiers (user's key insight)

User observed: the map is essentially built (blue free floor defines the room, walls
read off its edge), yet the robot keeps camping the right wall and looping, and
`frontier` stays at 11000–16000 even at 51 % coverage. Hypothesis: the robot is
chasing frontiers that lie **beyond the walls** (за застенки).

Confirmed in code. The base frontier is:

```python
base_frontier = frontier_unknown & (dilate(cleanable, 7x7) > 0)   # unknown next to free floor
```

Free floor extends right up to the wall; the wall is often not a solid obstacle line,
so "unknown next to free floor" also fires along the wall pointing **out of the room**.
Those outward frontiers are unreachable, never get mapped, so `frontier` never drops,
the planner keeps routing toward them, and the robot camps the wall / loops forever —
and `AUTO_MAP_COMPLETE` (needs `frontier < ~1150`) never triggers, so it never docks.

Fix (`frontier_outside_room_mask` + clip in `build_frontier_revisit_mask`): derive the
room envelope from **sensor evidence only** (observed free floor + obstacles — NOT the
known arena rectangle, which the code forbids using as a prior), close small wall gaps,
flood-fill the non-boundary space inward from the map border, and treat everything the
flood reaches as *outside the walls*. Frontiers there are dropped; enclosed interior
unknown pockets are kept.

New config:
- `FRONTIER_INTERIOR_CLIP_ENABLED = True`
- `FRONTIER_INTERIOR_CLIP_MIN_COVERAGE = 38.0` — only active once enough of the room is
  mapped, so early outward exploration is not blocked (below this, frontiers are
  unrestricted as before).
- `FRONTIER_INTERIOR_CLIP_CLOSE_M = 0.30` — bridges small unmapped wall gaps; a real
  doorway wider than ~2×this is still explored.

### Expected effect
- `frontier` collapses toward the genuine interior gray once coverage ≥ 38 %.
- Right-wall camping / endless perimeter loops stop (no ghost targets out there).
- With `frontier` falling below the `AUTO_MAP_COMPLETE` threshold, the robot finally
  recognizes the map is done, routes to the dock, and auto-saves the map (Round 6).

If a genuinely unmapped area gets clipped (real wall section never seen), raise
`FRONTIER_INTERIOR_CLIP_MIN_COVERAGE` or `FRONTIER_INTERIOR_CLIP_CLOSE_M`, or set
`FRONTIER_INTERIOR_CLIP_ENABLED = False` to compare.

---

## Round 5b — gate the clip on LOOP CLOSURE, not coverage % (user's correction)

User correctly pointed out that coverage % is the wrong gate: coverage = cleaned /
*known* floor, and the known floor grows as you explore, so the % is relative to a
moving denominator and never tells you how much map is left. The right question is
"**has the room closed**, and is there still unknown / unfilled-obstacle area inside it."

The same border flood already measures this: the **enclosed interior** (gray the flood
can't reach) becomes large the moment the free-floor+obstacle ring closes. So:

- `frontier_outside_room_mask` now also returns the enclosed-interior cell count.
- Once that exceeds `FRONTIER_ROOM_CLOSED_MIN_AREA_M2` (3 m²), a **latched** flag
  `frontier_room_closed` is set and never reset.
- The clip activates only after closure (coverage is kept only as a low 25 % safety
  floor, no longer the deciding factor).
- Before closure the robot expands outward freely to actually close the room.

HUD now shows `roomClosed=yes/no encl=<cells>` in the gray-gap debug so you can watch
the flip. After it flips to `yes`, ghost frontiers are dropped, `frontier` collapses to
the real interior, and `AUTO_MAP_COMPLETE` can finally fire → dock + saved map.

Still open (separate): the completion check could additionally require
`frontier_room_closed` and "no obstacle-like unfilled interior" for a fully
closure-driven end condition; and noise/garbage points (close RGB-D snapshots + contact
thrashing) still want a temporal-persistence denoise.

---

## Round 6 — stop camping the right obstacle: honest completion signal + real give-up bite

**Symptom (28-min run):** after Round 5/5b the robot explores the whole room and reaches
~49 % coverage, then **grinds the right obstacle's hidden face for the last 7–8 minutes** —
`cand=frontier@(8xx,4xx-6xx)` over and over, `RGBD_SNAPSHOT`/`SCAN_AROUND` returning
`scan=done timeout nogain`, `frontier` pinned at ~10–11.5k, `grayGap≈32k`, `cov` flat at
48.6–49.3 %. It never docks.

**Root cause is NOT lack of active perception — it already drives up and looks.** Two
concrete bugs make the robot believe it's unfinished and never let it give up:

1. **The completion signal counted beyond-the-wall cells.** In `build_frontier_revisit_mask`
   the interior clip (`interior_outside`) was applied only to the *routed frontier mask*,
   while `last_gray_gap_cells` / `last_completion_gap_cells` were summed **before** the clip.
   So `grayGap≈32k` included unreachable outside-room gray. That number gates
   `AUTO_MAP_COMPLETE` (needs `≤220`) and the stall→dock give-up (blocked while `gray≥650`),
   so the room could **never** read as finished → infinite camp.
   **Fix:** recount gray + completion on the enclosed interior only (`& ~interior_outside`)
   before writing the globals. The HUD `grayGap=` now reflects real interior work left.

2. **The give-up had no bite.** The stall watchdog *did* blacklist the right-obstacle target
   during the plateau, but radius `0.68 m` / TTL `95 s` just let the robot step around the
   disc and re-approach the same multi-cell silhouette. And the coverage-stall timer was
   reset by sub-0.5 % coverage micro-creep (the camp inched cov up in +0.2/+0.4 % steps),
   so the watchdog often never matured.
   **Fix (config):** `..._STALL_BLACKLIST_RADIUS_M 0.68→1.05`, `..._STALL_BLACKLIST_SEC
   95→200`, `..._STALL_COV_GAIN_PERCENT 0.35→0.50`.

**Expected log deltas next run:**
- `grayGap=` drops sharply once `roomClosed=yes` (now interior-only) instead of sitting ~30k+.
- When the robot plateaus on the right obstacle, the watchdog matures
  (`stall=covFlat`/`stall=blacklist`) and a **wide** blacklist pushes it off to the
  unmapped interior instead of re-snapshotting the hidden face.
- Either it maps the remaining interior (frontier genuinely falls → `AUTO_MAP_COMPLETE` →
  dock + saved map), or, once interior gray/frontier are low, the stall→dock give-up fires.

**A/B:** revert by restoring the three constants above and removing the interior-clip
recount block (the `else:` branch keeps the pre-Round-6 behavior).

**Still open / next step (deliberate-vantage NBV layer):** the give-up stops the *camping*;
if the robot now *wanders* inefficiently between interior frontiers, the next increment is a
positive next-best-view bias — prefer the farthest least-recently-visited frontier component
and approach uncertain obstacle faces from a chosen standoff with line-of-sight, rather than
nearest-first. Deferred until this run confirms the camp is broken (the log shows looking was
never the problem — *not stopping* was).

---

## Round 6b — odometry no longer shears when the robot wedges by the low red box

**Round 6 confirmed working** in the next run: `grayGap` dropped to ~5–9k (was ~32k),
the right-wall camping broke, and the robot spread across the whole interior (poses at
x≈3–5, both halves) instead of grinding one obstacle.

**New symptom that run:** the robot drove into the gap by the **low red box**, wedged, and
spent ~86 s (t≈863–949 s) looping `CONTACT_ROTATE/CONTACT_RELEASE_TURN owner=NONE:route
abort: pa… ownerSource=recovery` with `wallTrap=wait 22/22s` saturated. Afterwards the map
sheared — "одометрия снова сломалась из-за того что застряла в дырке рядом с красной
коробкой."

**Root cause:** `update_odometry()` takes heading from the IMU (slip-proof) but **translation
from the wheel encoders** (`pose_x += dc·cos(mid)`). It already zeroes `dc` during commanded
in-place pivots (`NAV_TURN_90`, `NAV_SETTLE`, `NAV_LEG_ESCAPE_TURN`) — but
**`NAV_CONTACT_ROTATE` was missing from that list.** When the wedged robot pivoted in place
against the box, the slipping wheels still clocked encoder counts → a fake arc of translation
was integrated into the pose → the whole map sheared.

**Fix (1 line):** add `NAV_CONTACT_ROTATE` to the translation-zeroing guard in
`update_odometry()`. Its intended `dc` is ~0 (it's a pivot), so any encoder `dc` there is
slip and must not move the pose. Heading still comes from the IMU, so the rotate still works.

Note: `NAV_CONTACT_BACKUP`/`NAV_CONTACT_FORWARD` are deliberately left untouched — they must
measure real reverse/verify translation. Pinned-wheel slip there can't be detected from
encoders alone (the wheels really spin at the commanded speed), so it's out of scope here.

**Expected next run:** even if the robot briefly wedges by the red box, the map should stay
square (no shear / no teleport). Separate, still-open lever if it *also* wastes time there:
a decisive escape after prolonged contact-recovery (longer straight reverse + blacklist the
gap so the planner routes around it) — held back to validate the odometry fix in isolation.

---

## Round 6c — obstacles fill solid again (grazing rays no longer hollow them out)

**Symptom:** map + motion now good, but the robot "перестал закрашивать препятствия" — the
orange obstacle shape-completion stopped persisting; the log shows `update_obstacle_hypothesis_cache`
returning `add=0 comps=0` most cycles, so silhouettes go hollow.

**Root cause (a side effect of finally circling obstacles):** shape completion fills
`roi_unknownish = (~confirmed) & (~free_strong) & (~observed_free)`, and `observed_free` used
the *barely-free* unknown epsilon: `log_odds < -LO_UNKNOWN_EPS` = `< -0.18`. A single grazing
depth ray near an obstacle pushes interior cells just past -0.18, which **permanently
disqualifies the whole silhouette from filling**. While the robot camped (Round 5) it kept
re-stamping a few obstacles so this was masked; once Round 6 made it drive *all the way
around* every obstacle, every interior got grazed → hollow obstacles.

This `-0.18` exclusion is the very thing added earlier to stop the *over*-fill complaint
("закрасило больше чем нужно… известные точки внутри"). The two are the same cells — the fix
is a **moderate** threshold between barely-free and strong-free, not removing the exclusion:

- New `OBSTACLE_HYPOTHESIS_OBSERVED_FREE_LO = -1.2` (≈ midpoint of -0.18 and the strong-free
  -2.25). `observed_free = (log_odds < -1.2) | (cleaned_mask > 0)`.
- Weak/noisy grazing free (−0.18 … −1.2) → **still fills** as obstacle interior.
- Confidently/repeatedly free, and cleaned floor → still excluded, so it does **not** re-open
  the old over-fill onto real floor (e.g. an under-table gap seen reliably stays unpainted).

**Expected next run:** `hypObs` stops collapsing to 0 between obstacles; silhouettes stay
filled orange as the robot circles them. **A/B:** raise `OBSTACLE_HYPOTHESIS_OBSERVED_FREE_LO`
toward -0.18 if it over-fills again, or lower toward -2.25 to fill more aggressively.

---

## Round 6d — obstacle fill stops flickering (grazed obstacles stay filled)

**Symptom (30-min run):** Round 6c helped — `hypObs` now holds a steady baseline (~372)
and fills up to 3400–4000 — but it **flickers**: an obstacle fills, then a tick later
collapses back to baseline. User's key observation: where there's a cluster of black points
on blue floor it's *clearly* an obstacle, but the fill won't stay.

**Two flicker drivers in the hypothesis layer:**

1. **Qualify gate too strict.** `OBSTACLE_HYPOTHESIS_MAX_FREE_RATIO = 0.10`: an obstacle
   island is rejected outright if >10% of its bbox reads observed-free. A low/thin obstacle
   on open floor is grazed by side depth rays, exceeds 10%, drops out of filling that tick,
   decays, then re-appears → flicker. **Fix:** raise to `0.30`. Safe: this is only a *qualify*
   gate; the fill is still `roi_unknownish` (unknown cells only), so a looser gate never
   paints real floor.
2. **Single-tick clear too hard.** `_hypothesis_clear_on_free_observations` knocked
   strong-free cells down by `DECAY*6.0` (−0.33) per tick — enough to drop a freshly filled
   cell below the display threshold in one tick. **Fix:** `*6.0 → *3.0` (still clears genuine
   floor in a few seconds; a brief grazing ray no longer erases a real obstacle's fill).

**Expected next run:** obstacle silhouettes stay filled steadily instead of blinking; the
point-cluster-on-blue-floor obstacles read as solid. **A/B:** if it now over-fills, drop
`OBSTACLE_HYPOTHESIS_MAX_FREE_RATIO` back toward 0.10 and/or restore the `*6.0` clear; if it
still flickers, the next suspect is the instantaneous strong-free mask
`cache &= ~(log_odds < FREE_CLEAR_LO)` (line ~4951), which hides orange the moment the base
cell reads strong-free — softening that is the deeper fix but risks lying about cleared floor.

---

## Round 6e — kill the obstacle-fill flicker at its source (sticky strong hypotheses)

**Status:** map is good (obstacles found + marked); the only remaining complaint is the fill
still blinks on/off. 6c/6d reduced it but didn't kill it — the log still shows `hypObs` drop
to `0` for a whole tick (`hypObs=0 add=0 comps=0 clr=8`) then pop back to `346`/`2551`.

**Root cause (the suspect flagged in 6d):** `update_obstacle_hypothesis_cache` ended with
```
cache = hyp_log_odds > OCC_EPS
cache &= ~(log_odds < FREE_CLEAR_LO)   # "confirmed free always wins"
```
That second line **instantly masks the entire orange blob** the moment the obstacle's *base*
occupancy dips below strong-free (−2.25), which happens for a single frame whenever depth rays
graze a low/thin obstacle. The hypothesis evidence is still strong underneath — it's just
hidden — so the fill blinks off then back on.

**Fix:** confirmed-free now overrides only a **weak** hypothesis:
```
weak_hyp = hyp_log_odds < OBSTACLE_HYPOTHESIS_STICKY_KEEP_LO   # new, = 2.5
cache &= ~((log_odds < FREE_CLEAR_LO) & weak_hyp)
```
A cell only reaches 2.5 by being shape-completed several times (`SHAPE_UPDATE=1.10` each), so
noise stamps never stick, but a real (repeatedly-confirmed) obstacle survives a transient free
ray. The Round-6d gentle free-clear (`*3.0`) still erodes genuinely-free cells below 2.5 over
a few seconds, so **sustained** free still wins — honesty preserved, flicker gone.

**Expected next run:** obstacle silhouettes stay solid; no per-tick blink. **A/B:** raise
`OBSTACLE_HYPOTHESIS_STICKY_KEEP_LO` toward `OBSTACLE_HYPOTHESIS_MAX` (4.0) for stricter
honesty (more masking, possible flicker return), or lower toward `OCC_EPS` (0.85) for maximum
stickiness (fills essentially never masked by free).

---

## Round 6f — deep dive: TWO separate bugs (flicker + never-filled), both from shape-completion qualification

User asked to think harder: (1) why still flickering, (2) why several clearly-outlined
obstacles (bottom-left big rectangle, top-middle 4-sided point ring) never fill at all.
These are **different** bugs with the same origin — the per-recompute shape-completion in
`update_obstacle_hypothesis_cache`.

### Bug 1 — flicker (intermittent qualifiers never get sticky)
A cell becomes "sticky" (immune to the strong-free display mask, Round 6e) only at
`hyp_log_odds ≥ 2.5`. It climbs by `SHAPE_UPDATE` per *qualifying* recompute. The first
obstacle qualifies every recompute, so it reached MAX in ~4 adds and never flickered. But a
big obstacle whose `free_ratio` crosses the gate tick-to-tick qualifies only *intermittently*:
+1.10, decays before next qualify, +1.10 again — it lives in the maskable [0.85, 2.5] band and
blinks with every grazing free ray. The HUD shows this exactly: `hypObs` swinging 632 ↔ 1632 ↔
0 between recomputes.
**Fix:** `OBSTACLE_HYPOTHESIS_SHAPE_UPDATE 1.10 → 4.0` — one qualification jumps the cell
straight to MAX, instantly sticky, so it stops flickering. Truly-free cells still erode from
MAX over ~9 sustained-free recomputes (honesty kept).

### Bug 2 — never-filled obstacles (they never QUALIFY; `add=0`)
Two qualification gates silently drop whole obstacles:
- **Size caps:** `MAX_SPAN_M=1.85m` / `MAX_COMPONENT_CELLS=3600` reject large furniture as
  "too big, probably a wall." The bottom-left rectangle exceeds them. → `2.6m` / `9000`. Walls
  are still excluded by the separate arena-wall-touch + map-edge test, so the room perimeter is
  not filled.
- **Stud-bridging:** the MORPH_CLOSE kernel was `BOX_PAD_M=0.055m` (~3px) — too small to bridge
  gaps between SPARSE studded points, so a 4-sided ring of dots never connected into one island
  with an enclosed interior, so it never qualified. → `0.14m` (~7px) bridges the studs.

**Expected next run:** big/outlined obstacles (bottom-left, top-middle) fill solid; fills stay
put instead of blinking. **A/B knobs:** flicker→`SHAPE_UPDATE` back to 1.10; over-fill of room
→ lower `MAX_SPAN_M`/`MAX_COMPONENT_CELLS`/`BOX_PAD_M`. **Deeper option if still unsatisfying:**
since hypObs is visual-only in this mapping mission (`OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING =
False`), the obstacle *body* could be rendered as a deterministic geometric fill of confirmed
clusters (morph-close + per-component hole-fill, recomputed each frame = no decay, no flicker),
fully decoupled from the inference layer — bigger change, held unless these knobs fall short.

---

## Round 6g — robust fix: deterministic obstacle BODY fill (display), decoupled from inference

User asked for the durable fix instead of more threshold tuning. Root problem: the orange
obstacle fill was driven by a per-recompute **speculative log-odds layer** that accumulates and
decays — inherently flicker-prone and gate-dependent, so no constant set fully fixed it.

Because hypObs is **visual-only** in this mapping mission (`OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING
= False`, and `actual_physical_obstacle_mask` only merges hypObs into planning when that flag is
True), the obstacle *body shown on the maps* can be rebuilt geometrically every frame, fully
decoupled from the inference layer:

`build_obstacle_body_fill()` (new, display only):
1. take CONFIRMED obstacle cells (`base_physical_obstacle_mask` + structural + dense-CV),
2. morphological CLOSE (`OBSTACLE_BODY_FILL_CLOSE_M=0.14m`) to bridge sparse studded points
   into one island,
3. per connected component, solid-fill the enclosed interior (`_fill_enclosed_holes`, a
   border-floodfill hole fill) — a 4-sided ring of dots becomes a solid body,
4. exclude walls/room: skip components whose bbox span > `OBSTACLE_BODY_FILL_MAX_SPAN_M=2.6m`,
   that touch the map edge, or that overlap `arena_wall_touch_mask()`,
5. never paint confidently-free (`< FREE_CLEAR_LO`) or cleaned cells.

It's cached by `step_id` (computed once per step, both map windows reuse it).
`draw_hypothesis_obstacle_overlay` now draws this body when `OBSTACLE_BODY_FILL_ENABLED` (raw
hypObs overlay kept only as fallback). **Planning is untouched** — the inference layer and all
its consumers (frontier/closure/no-go) still use `hypothesis_obstacle_mask` exactly as before.

**Why this is robust:** rebuilt from confirmed evidence each frame → no accumulation, no decay,
**no flicker**; large/sparse silhouettes fill because there is no size/free-ratio qualification
to fail — only walls and real floor are excluded. Legend updated: "orange = filled obstacle body".

**A/B:** `OBSTACLE_BODY_FILL_ENABLED = False` restores the old hypObs overlay. Tune
`OBSTACLE_BODY_FILL_CLOSE_M` (smaller = less stud-bridging), `OBSTACLE_BODY_FILL_MAX_SPAN_M`
(smaller = more conservative about large clusters).

---

## Round 6h — revert the display-tuning that leaked into the planner (path regression)

**Symptom:** after 6c–6g the path went weird again. **Cause (my miss):** rounds 6c–6f tuned
the `OBSTACLE_HYPOTHESIS_*` constants to fix the *orange display*, but that hypothesis layer is
**also read by the planner** — `build_frontier_revisit_mask` does
`frontier_unknown = unknown & (~hyp_shadow)` and closure does `known |= hypothesis_obstacle_mask`.
Cranking `SHAPE_UPDATE 1.10→4.0`, `MAX_SPAN 1.85→2.6`, `MAX_COMPONENT 3600→9000`,
`BOX_PAD 0.055→0.14`, `MAX_FREE_RATIO 0.10→0.30`, the sticky-mask and gentle-clear inflated
hypObs ~5–10× (logs showed `hypObs` 1743–6048 vs ~325 in the good run). That subtracted far
more cells from the frontier and shifted closure → different/weirder frontier selection.

**Fix:** revert all 6c–6f inference knobs to their good-run values:
`SHAPE_UPDATE=1.10`, `MAX_FREE_RATIO=0.10`, `MAX_SPAN_M=1.85`, `MAX_COMPONENT_CELLS=3600`,
`BOX_PAD_M=0.055`, `OBSERVED_FREE_LO=-0.18`, free-clear back to `*6.0`, and the cache mask back
to `cache &= ~(log_odds < FREE_CLEAR_LO)`. **Kept:** 6g body-fill (display, planning-independent,
uses its own `OBSTACLE_BODY_FILL_*` constants), 6b odometry guard, 6a interior-clipped completion
counts.

**Net architecture now:** planning uses the original small/conservative hypObs inference;
display uses the deterministic body-fill. The two no longer fight. Path should match the earlier
good run, and obstacle bodies still render solid + non-flicker.

**Lesson:** the hypothesis layer is dual-use (display + planner). Any change meant for the
visualization must go in `build_obstacle_body_fill` / `OBSTACLE_BODY_FILL_*`, never in
`OBSTACLE_HYPOTHESIS_*`.

---

## Round 6i — right-wall pacing: engage the wall-repeat penalty earlier

After 6h the path is clean (straight lines), but the robot paces the right wall up/down at
cov~31% instead of committing to the large unexplored LEFT half. **Why:** the right wall is the
arena boundary; depth keeps spawning "unknown beyond the wall" frontier candidates right next to
the robot, and the selector takes the NEAREST committable frontier → it oscillates on the wall.
The two mechanisms that break this are both gated off early: the beyond-wall ghost clip
(`frontier_outside_room_mask`, needs loop closure) and `RIGHT_WALL_REPEAT_SUPPRESS` (was gated
`cov≥36%` AND `gray≥650`; here cov=31%, gray=0). Also the 6h revert removed an accidental side
effect — the inflated hypObs (6c–6f) had been masking those wall ghosts, so the weird-path runs
left the wall sooner.

**Fix (pure scoring, no hypObs/planner/wall changes):** engage the wall-repeat penalty earlier —
`RIGHT_WALL_REPEAT_SUPPRESS_MIN_COVERAGE_PERCENT 36→24`, `MIN_GRAY_CELLS 650→0`. It only penalizes
a RE-visited edge band near the robot (first pass is unpunished), so it nudges the robot to pick
the far left frontier instead of re-pacing the right wall, without removing the wall or blocking
genuine wall-only situations.

**A/B / if still pacing:** raise `RIGHT_WALL_REPEAT_SUPPRESS_PENALTY` (42 → 60+) so it outweighs
the near-frontier distance advantage; revert the two gates to 36/650 to disable early engagement.

---

## Tooling — per-run debug log file

For easier debugging, the controller now writes a fresh, full-detail telemetry trace to
`controllers/rgbd_navigation_cleaner/debug_log.txt` on every run (opened with `"w"`, so it is
**replaced each startup**). It logs the **verbose** status line (much richer than the compact
console line that gets pasted into chat) plus extra raw counts — `frontierCells`, `gray`,
`completion`, `bodyFill`, `wallTrap`, `prog` — every `DEBUG_LOG_INTERVAL_SEC` (default 4 s, vs
the 18 s console cadence), line-buffered so it survives a crash.

Config (config.py): `DEBUG_LOG_TO_FILE=True`, `DEBUG_LOG_FILE_NAME="debug_log.txt"`,
`DEBUG_LOG_INTERVAL_SEC=4.0`. Set `DEBUG_LOG_TO_FILE=False` to disable, or raise the interval if
the file grows too large. The write path is guarded (try/except) so a bad field can never crash
navigation. Just open `debug_log.txt` in Notepad after a run, or paste it here.

---

## Round 6k — cleaner scan capture: settle before fusing the dwell + strict still-gate

Mapping was already frozen during the scan *rotation* (`route_or_scan_rotation_mapping_reason`
returns `scan-turn` unless in a dwell). The remaining smear source was the dwell **entry**: the
moment the robot hit the ±3° yaw tolerance it commanded stop but was still **coasting**, and the
dwell allowed fusion immediately — guarded only by the lenient global `omega ≤ 0.12 rad/s` gate.
Two tightenings (user-requested):

1. **Strict still-gate for scan-dwell fusion** — new `ACTIVE_SCAN_DWELL_OMEGA_LIMIT = 0.06`
   (~3.4°/s) instead of the global 0.12, applied only inside the scan dwell. Frozen frames now
   show `cvFreeze=scan-settle` on the HUD.
2. **Settle-before-fuse** — `active_scan_mapping_dwell_ready` now excludes the first
   `ACTIVE_SCAN_DWELL_SETTLE_SEC = 0.12 s` of the dwell (the coast-out), so it fuses only the
   settled tail. `ACTIVE_SCAN_DWELL_SEC` bumped `0.42 → 0.55` so the clean window stays ~0.35 s.

Net: depth is fused only when the robot is genuinely still at each scan yaw → less smear / fewer
junk points from active scans. Planning untouched; global `TURN_MAPPING_OMEGA_LIMIT` (forward
arcs) untouched. **A/B:** raise `ACTIVE_SCAN_DWELL_OMEGA_LIMIT`/lower `..._SETTLE_SEC` to capture
more (riskier smear); the dwell-entry smear was small to begin with, so this is a refinement, not
the main junk-point source (close grazing `RGBD_SNAPSHOT` near obstacles is separate).

## Round 6l — instrumentation: mapping (explored) metric + penalty/watchdog telemetry

Telemetry-only, **no behaviour change**. Two motivations:

1. **`coverage` is a cleaning metric, not a mapping metric.** `last_coverage_percent =
   100*done/total` where `done = cleaned & cleanable` (floor physically **driven over**) and
   `total = cleanable` (known-free floor). For an exploration/mapping mission this is the wrong
   yardstick — RGB-D maps cells from a distance without driving over them, so a map can be ~done
   while `coverage` is low. Added a real exploration metric to the file log:
   `map=explored=X% known=K unkInt=G driven=d/t`, where `explored = 100*known/(known+grayInt)`,
   `known = cleanable(known-free) + obstacles(known-occupied)`, `grayInt = last_gray_gap_cells`
   (interior-clipped). New global `last_coverage_obstacle_cells` (set in
   `update_coverage_objective`). NOTE: first version used `(encl-grayInt)/encl` with
   `encl=frontier_outside_room_mask` enclosed count — that went **negative** in the overnight run
   (encl is the residual enclosed-unknown region, not the room area; gray > encl). Corrected to
   the known/(known+gray) form above.
2. **Penalties / give-up were invisible in the log.** Added to `debug_log_file_line`:
   - `stall=...` — `frontier_only_last_stall_debug` from the stall watchdog
     (`reset/covFlat/blacklist/dock/noDock`). This is the give-up state machine; it was never
     logged.
   - `rwSupp=hits/pen` — how many candidates got the RIGHT_WALL_REPEAT_SUPPRESS penalty this
     route plan, and the max penalty applied. New globals `right_wall_suppress_hits`,
     `right_wall_suppress_max_pen`, reset at the top of `plan_best_coverage_route`, incremented in
     the suppress block.

### Suspected bug surfaced while instrumenting (NOT yet fixed — awaiting A/B go-ahead)
`frontier_only_stall_watchdog_update` blocks the return-to-dock with, at line ~8548:
`remaining_gray >= int(RIGHT_WALL_REPEAT_SUPPRESS_MIN_GRAY_CELLS)`. Round 6i set that constant
`650 → 0` to make the right-wall suppress engage earlier — but the **same** constant is reused
here, so `remaining_gray >= 0` is **always true** → the dock branch is permanently blocked →
watchdog can only ever emit `stall=noDock` → the robot never gives up / never docks, and keeps
re-chasing unreachable near-wall frontiers (matches the "penalties don't work / strange path" +
flat-coverage plateau seen at 22 min). The new `stall=` log field will confirm this on the next
run. Fix candidate: give the dock-block its own constant (e.g. `STALL_DOCK_MIN_GRAY_CELLS`)
decoupled from the suppress threshold.

## Round 6m — stop chasing already-mapped floor: fix the cleanup-unlock ratio bug

Overnight run (2 h) diagnosis via the new 6l telemetry:
- Map reached **~97% explored** (`map=explored`) while `coverage` was only 83.7% — confirms the
  mapping/coverage split: the robot mapped almost everything from a distance without driving over
  it.
- Mission could not end: `stall=noDock` x72, **119 blacklists, 0 docks** (the `MIN_GRAY_CELLS=0`
  dock-block bug, plus `completion`/`gray` block).  `rwSupp=0/0` all run (right-wall penalty never
  fired). Residual `gray≈14.5k` is unreducible (furniture interiors / occlusions) and plateaued.
- **Root inefficiency:** `route=uncleaned` 2009 vs `route=frontier` 2521 — ~38% of planning chased
  *already-known* floor.  `cleanupLock` **unlocked at ~42% into the run** and stayed unlocked for
  ~64% of it (`MAP_STABILIZATION`), so the robot spent most of the night driving over mapped floor
  (coverage) instead of mapping frontiers.

Cause: `exploration_cleanup_unlocked()` unlock = `cov>=68 AND t>=180 AND (frontiers<=900 OR
ratio<=0.030)`.  The **ratio branch** fired: at run end `fr=10396`, `total=423175`,
`ratio=0.0246<=0.030` → unlocked despite 10396 frontier cells (11x the absolute 900 gate).  As the
map grows the `cleanable` denominator inflates, so a 3% ratio = ~12.7k cells — the ratio test is an
absolute count in disguise and defeats the `MAX_FRONTIER_CELLS` gate.

Fix (config, one lever, reversible): `EXPLORE_CLEANUP_UNLOCK_MAX_FRONTIER_RATIO 0.030 -> 0.0`.
The ratio branch is now inert; unlock requires the absolute `frontiers<=900`.  EXPAND_MAP stays
frontier-first until frontiers are genuinely exhausted, instead of switching to uncleaned-chasing
while 10k frontiers remain.

**A/B / watch next run:** look at the `route=uncleaned` share and the time in
`plannerMode=MAP_STABILIZATION` — both should drop sharply.  Risk: if the residual ~10k frontier
cells are mostly unreachable beyond-wall/occluded ghosts, the robot may now thrash on *frontiers*
instead (blacklists near boundaries) — that would point the next lever at §7.1 (beyond-wall ghost
frontier suppression).  Completion (dock+save) is still the deferred lever.

## Round 6n — break the greedy-nearest frontier trap: restore recency penalty for frontiers

6m worked: 30-min run showed `route=uncleaned`=0 (was ~38%), `cleanupLock=locked` and
`EXPLORATION_FRONTIER` the whole run — no more uncleaned-chasing.  `map=explored` reached ~92%.

But the predicted §7.1 risk materialised: the robot now thrashes on **frontiers**.  Telemetry:
`explored` plateaued at ~92.1-92.5% and `gray≈33k` for the last ~15 min (no unknown-collapse) while
it blacklisted 8x (bl 3->11).  Late poses clustered on the right (x~7-8) and late frontier targets
clustered on the right (x=760-900, y=468-947) — a vertical column by the right wall / big right
obstacle.  Early targets were all over the map; the robot got stuck re-picking the nearby right
cluster instead of leaving for resolvable frontiers elsewhere.  `rwSupp=0/0` (the right-wall
band-aid is per-cell and too narrow — the exact cells shift, so its recency never accumulates).

Root cause (planner-relevant frontier scoring in `plan_best_coverage_route`): the recency penalty
for FRONTIER targets was a hardcoded `- 0.25 * recent_penalty`.  `recent_penalty =
RECENT_VISIT_TARGET_PENALTY(10.5) * recent_target(0..1)`, so frontier recency maxed at **2.6** vs
frontier scores of ~150-180 — effectively off.  (uncleaned targets get the full `- recent_penalty`.)
Frontier near-vs-far differs by ~`EXPLORATION_ROUTE_COST_PENALTY(5.2)/m * dcost` ~= 8-13 pts, which a
2.6 recency penalty can never overcome → greedy-nearest trap.

Fix (config, one lever, reversible): new `EXPLORATION_FRONTIER_RECENT_VISIT_PENALTY_SCALE = 1.5`
replaces the hardcoded `0.25`.  A thrashed cell (`recent_target~0.5-0.8`) now costs ~8-17 pts and
loses to a fresh far frontier, pushing the robot to leave a re-treaded area.  Also added `recPen=`
(selected target's raw recency penalty) to `debug_log.txt`.

**A/B / watch next run:** late-run pose spread should widen (robot leaves the right cluster);
`explored` should climb past ~92% and `gray` keep dropping; `recPen` should be >0 on thrashed
targets.  Risk if too strong: robot abandons genuinely-needed nearby frontiers and oscillates — if so
dial `..._SCALE` down toward 0.8-1.0.  Completion (dock+save) still deferred.

## Round 6o — wall-trap watchdog: reset on mapping progress (gray), not coverage

5-min run after 6n: the robot traced the perimeter, **closed the room** (`roomClosed=yes
encl~29k`) and reached `explored~92%` fast — but then oscillated up/down the right wall.  `recPen=0.0`
the entire run: the 6n recency penalty is **structurally inert** for this mode because the robot
never reaches the (occluded) target cells, so they never accumulate recency.  The targets were
frontier pockets along / behind the big right obstacle (`route abort: path` -> `nearHyp` marked),
i.e. occluded interior cells the robot can't resolve from the corridor.

The `wall_trapped_frontier_watchdog` is the right mechanism (anchor on robot column + progress;
look-around scan; then escalating blacklist of the strip) and it *was* firing (`wallTrap=wait
21/22s -> zoneEscape`).  But it reset its anchor on **coverage** growth
(`WALL_TRAP_FRONTIER_ZONE_COV_GAIN_PERCENT=0.4`): driving up/down the occluded strip cleans floor
(coverage +0.4%) while mapping nothing (gray flat, cells are behind the obstacle), so coverage-creep
kept resetting the trap and it cleared the strip only very slowly -> prolonged oscillation.  Same
coverage-is-not-progress trap as everywhere else.

Fix (one lever): after the room has closed, the wall-trap resets only on real mapping progress —
interior `gray` shrinking by >= `WALL_TRAP_FRONTIER_ZONE_GRAY_DROP_CELLS=300` — instead of coverage
growth.  Before closure, coverage growth is still ~ genuine discovery, so the cov-gain reset is kept.
New global `wall_trap_anchor_gray`; debug now prints `dgray=...` (closed) or `dcov=...` (open).  Net:
on an occluded strip gray stays flat -> no reset -> trap fires at the 22s mark -> strip blacklisted
(radius escalates to 1.2 m) -> picker must choose a far region -> robot leaves the wall.

(6n kept: harmless, and `recPen` will show if it ever bites in a reachable-revisit case.)

**Watch next run:** the robot should leave the right wall after ~one stall period instead of
sweeping it repeatedly; `explored`/`gray` should keep moving rather than plateauing while the robot
is pinned; `wallTrap=` should reach `blacklist`/`zoneEscape` promptly and `dgray` should sit near 0
while trapped.  Risk if too eager: blacklists a strip that had a genuinely reachable thin opening —
then raise `..._GRAY_DROP_CELLS` or lower the blacklist TTL.

## Round 6p — robot plows the row instead of turning to its chosen frontier

**Symptom (user):** "почему робот не едет куда хочет а просто едет дальше вперед" — the HUD
shows a real target (`cand=frontier@(204,772)`, then `(357,660)`…) but the robot just drives
straight along the bottom wall: pose x 1.15 -> 5.83 (~4.7 m, y≈−4.07) while the candidate is
recomputed every cycle to "nearest ahead". Gate field: `gate=frontierGate=holdDrive row-pri`.

**Diagnosis:** not a bug — the `row_primary_frontier_should_hold_forward` guard
(`navigation_controller.py:12047`) is overriding the turn-to-candidate. It holds the current row
(refuses to turn to the frontier) whenever **front is clear ≥ `FRONT_CLEAR_M=0.42`** AND reaching
the candidate needs a **sharp first turn** (`first_turn_frac > MAX_FIRST_TURN_FRAC=0.24`) or
> `MAX_CORNERS=3` corners. Combined with `frontierGate=holdDrive`
(`EXPLORATION_FRONTIER_ONLY_HOLD_DRIVE_INSTEAD=True`, line 8809) which hands motion to ROW_FORWARD
when the frontier route is non-committable. This is the same anti-thrash that killed the single-wall
oscillation (good), but its side effect now: the perimeter is already mapped and the remaining
frontiers are **interior** — to reach them the robot must turn off the row, which `row-pri` suppresses.
Hence the endless perimeter re-trace.

**Fix (one lever):** `EXPLORATION_FRONTIER_ROW_PRIMARY_MAX_FIRST_TURN_FRAC` `0.24 -> 0.45`
(`config.py:468`). The robot now tolerates a bigger first turn before invoking row-hold, so a
side/interior frontier wins over plowing the row. `FRONT_CLEAR_M` and `MAX_CORNERS` left unchanged
(single-variable step).

**Watch next run:** with a clear front the robot should now **turn off the bottom row toward
interior frontiers** instead of driving it end-to-end; `explored`/`gray` should resume moving past
the ~92% plateau as interior pockets get serviced; `gate=` should show fewer consecutive
`holdDrive row-pri` lines. Risk if too eager: more zig-zag / partial return of nearest-frontier
thrash — then back off toward ~0.35, or add a "candidate is close & valuable" exception to
`row_primary_frontier_should_hold_forward` instead of a flat threshold.

## Round 6q — big furniture not filled (obstacle body fill)

**Symptom (user):** circled in red the obstacles that should be solid-filled (bottom-left big table,
two on the right, one small center) and in green the one still to be observed. "у нас ведь есть там
точки которые огораживают серое пространство то есть это буквально препятствие и мы как будто должны
их закрасить" — the black points fence a gray interior = a furniture body, fill it.

**Diagnosis:** `build_obstacle_body_fill` (`navigation_controller.py:19382`, display-only, separate
from the planner's `OBSTACLE_HYPOTHESIS_*`) gates each cluster two ways: (1) `_fill_enclosed_holes`
fills only a topologically **closed** ring; (2) a **span cap** drops any cluster whose bbox is
larger than `OBSTACLE_BODY_FILL_MAX_SPAN_M = 2.6 m` as "wall/room-sized" (line 19420). The red
furniture is fully outlined (enclosure OK) but **bigger than 2.6 m**, so it was rejected as wall.
The green one is correctly skipped — its outline is still open (C-shape, occluded far side), so
there is no enclosed interior to fill yet.

**Fix (one lever):** `OBSTACLE_BODY_FILL_MAX_SPAN_M` `2.6 -> 4.5 m` (`config.py:1615`). Real tables
(~3-4 m) are now filled; the arena wall ring (~14-18 m span) is still excluded by this cap **and**
by the arena-wall-touch and map-edge per-component checks, and `body &= ~free`, `body &= ~cleaned`
still prevent any bleed onto real floor. Display only — planner geometry unchanged.

**Watch next run:** the red obstacles should show solid orange bodies; the room interior must NOT
flood orange and the perimeter wall must stay unfilled. If a fully-outlined obstacle still shows
hollow, the remaining cause is an open silhouette (enclosure) — next lever would be a larger
`OBSTACLE_BODY_FILL_CLOSE_M` or an open-pocket fill, not the span cap. If anything large/free floods,
lower the cap back toward ~3.5 m.

## Round 6r — partial obstacles still hollow + coverage metric misleading (display, two levers)

**Symptoms (user, with screenshot + debug_log.txt):**
1. "coverage это не то что нам помогает... препятствия все еще не закрасились кроме одного" — the
   HUD leads with `Coverage: 76.0%` (driven floor) which says nothing about map exploredness, and
   after a very long run only **one** obstacle body is orange-filled.
2. "ножки стола построились много где но они буквально не нужны" — scattered leg-like dots around
   furniture.
3. "робот плохо осмотрел препятствия и не закрасил хотя уже надо было."

**Diagnosis (all one root + one extra blocker):**
- The robot has geometrically finished: `roomClosed=yes encl=18027`, `map=explored=96.2%`. It drove
  everywhere; what it lacks is clean line-of-sight to the back/under sides of furniture.
- **Fill (1):** `bodyFill=4408` ≈ a single object. `build_obstacle_body_fill` filled clusters only
  through `_fill_enclosed_holes`, which needs a topologically **closed** ring. Furniture seen from
  1-2 sides is an open arc → no enclosed interior → no fill. Only the one piece the robot happened
  to circle closed. (6q already raised the span cap; this is the open-silhouette lever 6q predicted.)
- **Legs (3):** `legQuad=0 c=70 g=1 s=0` — leg-quad stamps **0**; the "legs" are 70 raw RGB-D noise
  studs near furniture, sparse points the fill could not bridge. Same root as fill.
- **Metric (1):** `explored%` (unknown-collapse) is already computed in `debug_log_file_line` but was
  not on the HUD; `coverage` is driven-floor cleaning %, the wrong number to judge mapping by.

**Fix — lever 1 (obstacle body fill, display only, planner untouched):** new
`_fill_obstacle_body(comp, free)` (`navigation_controller.py`) fills the silhouette plus the unknown
cells it walls off **together with the confidently-free floor around it** — i.e. floor the robot
drove around the obstacle is used as an extra flood barrier (never painted). A partially-seen object
the robot circled now gets its interior body; a lone unobserved face stays connected to the open room
and is never claimed. Per-component window is padded by `OBSTACLE_BODY_FILL_BARRIER_PAD_M = 0.30 m`
so surrounding floor is in scope; span cap / wall-touch / map-edge / `~free` / `~cleaned` guards all
unchanged. Toggle `OBSTACLE_BODY_FILL_FLOOR_BARRIER = True` (False = legacy enclosed-ring-only).
This deliberately ties fill quality to inspection quality.

**Fix — lever 3 (HUD metric):** new `exploration_explored_pct()` (single source of truth, reused by
`debug_log_file_line`). First cut led the HUD with `Explored: NN% (unk K)` — but a test run showed
**`Explored: 100.0% (unk 0)` on a barely-mapped open wedge** (driven 25.8%). Cause: `unk` =
`last_gray_gap_cells` counts only **interior** unknown pockets (enclosed by known); before the room
loop-closes the entire unmapped area is open frontier, not interior gray, so `unk=0` →
`known/(known+0)` = 100%. A true "% of room" is not computable pre-closure without an arena prior
(forbidden). So the HUD label is now **honest two-mode** via `exploration_progress_hud_text()`:
- room **open** (`frontier_room_closed=False`): `Mapping: open | mapped X.X m2 | frontier N` — monotone
  mapped area + open frontier, no fake %.
- room **closed**: `Explored: NN% (unk K)` — bounded-room resolved %, now meaningful.
Driven-floor coverage stays as the secondary `driven NN%`.

**Watch next run:**
- More obstacles should show solid orange bodies (every piece the robot actually drove around), the
  room interior must NOT flood orange, the perimeter wall stays unfilled, and `bodyFill=` should rise
  well above ~4400. If a piece the robot clearly circled is still hollow → raise
  `OBSTACLE_BODY_FILL_BARRIER_PAD_M` or `OBSTACLE_BODY_FILL_CLOSE_M`. If any open floor gets painted →
  the free-clear threshold is too weak there (or lower the pad).
- If obstacles the robot only passed once stay hollow, that **confirms** the weak-inspection
  hypothesis and the next lever is active-perception / vantage standoff, not display.
- HUD: while the boundary is open it should read `Mapping: open | mapped … | frontier …`; it must NOT
  show a 100% until the room actually closes, then it flips to `Explored: NN%`.

## Round 6s — robot camps walls, never diverts to inspect/fill obstacles

**Symptom (user + pasted debug_log):** robot drives the perimeter/rows and never goes to map &
fill the furniture in the room interior. Suspected the penalties.

**Diagnosis (from the log, NOT penalties):** `rwSupp=0/0`, `recPen=0.0` — right-wall and recent
penalties are inactive, so they are not the cause. Real causes:
1. **hybrid-row gate** `frontierGate=hybrid-row cov=29.9/48.0` — row-primary mode plows straight rows
   until cov≥48%, deferring interior diversions.
2. **nearest-frontier scoring** — selected frontiers were `d=0.6..1.7 m`; the scorer nibbles the
   closest frontier on the path.
3. **obstacle-inspection layer is DEAD in exploration.** `build_obs_boundary_vantage_mask`
   (`OBS_BOUNDARY_FRONTIER`, vantage standpoints beside obstacles with unseen faces) produces
   `target_kind=4`, but the route scorer has `if exploration_route_only and kind_code != 2: continue`
   (`navigation_controller.py`), which **discards every kind=4 target during frontier exploration**.
   Even if it survived, its reward was `OBS_BOUNDARY_FRONTIER_REWARD=9.5 (+≤5)` vs frontier scores of
   80–185, and there was no exploration score branch for it. So obstacle faces (which are also
   shadow-rejected out of the gray-gap, `rej=360..494`) had nothing routing the robot to them.

**Fix (one lever — call the robot to obstacles; flag `OBS_BOUNDARY_FRONTIER_EXPLORATION_TARGET`):**
- Let `kind=4` through the exploration filter (line ~10053), flag-gated.
- New kind=4 exploration **score branch** on the SAME penalty scale as frontiers, with a proximity
  `OBS_BOUNDARY_FRONTIER_CLOSE_BONUS=28` decaying to 0 at `OBS_BOUNDARY_FRONTIER_NEAR_RADIUS_M=1.6 m`
  → robot diverts to a *nearby* obstacle but never crosses the room for a far one.
- Rewards raised competitive: `OBS_BOUNDARY_FRONTIER_REWARD 9.5→26`, `BONUS_MAX 5→16`,
  `BONUS_SCALE 0.04→0.06`. `MIN_COVERAGE_PERCENT` kept 28.
- The vantage is **selected** with the kind=4 obstacle reward but **committed as `kind="frontier"`**
  (return-dict remap), because every exploration commit gate keys off `kind=="frontier"` (8403,
  8700–8731, 3090, …). So the existing frontier route + RGBD-capture pipeline drives the robot to the
  standpoint. Coverage-mode behaviour unchanged (`obs-boundary` label preserved there).
- New `coarse_obs_boundary_cells` grid feeds the debug. Display body fill / hypObs untouched.

**Watch next run (grep debug_log for `obs-vantage`):**
- Expect `exploreRoute=obs-vantage obs=… close=… cost=… d=…` and
  `frontierChoice selected=(…) … kind=obs-vantage` lines, and the robot visibly diverting to circle
  furniture. With body fill (6r) this should also start filling those obstacles.
- If obs-vantage targets are *selected but the robot does not move*, an exploration commit gate is
  rejecting the (now frontier-labelled) vantage — report the `gate=`/`route=` line and we relax it.
- If it wins too aggressively (ignores real frontiers / stalls circling one object): lower
  `OBS_BOUNDARY_FRONTIER_REWARD`/`CLOSE_BONUS` or shrink `NEAR_RADIUS_M`. Revert wholesale with
  `OBS_BOUNDARY_FRONTIER_EXPLORATION_TARGET=False`.

**Still open (diagnosed, not changed this round):** the explored% metric/closure latch — `roomClosed`
latches at `encl≈23636` while `cov=27%`, and `explored=known/(known+interiorUnknown)` saturates ~99%
because the 22k open frontier cells are not in the denominator. The honest signal remains absolute
`unk`/`frontier`, not the %. Separate lever.

## Round 6t — obstacles still hollow: bridge studs + complete to a primitive

**Symptom:** `bodyFill=64` while obstacles are clearly visible as studded point outlines with
gray (unknown) floor inside. User: "по точкам видно что есть что соединять... если напоминает
стандартные геометрические фигуры — можно достроить".

**Diagnosis:** body fill builds from CONFIRMED obstacle cells morph-closed into components ≥18 cells.
Obstacles arrive as **sparse studded points** (early log `rawObs≈185`; even mature `rawObs≈5010` is
studded, not solid). At `CLOSE_M=0.14` (~7px) studs >14 cm apart never connect, so almost nothing
reached the fill floor. (Not the 6r floor-barrier — that returns ≥ enclosed-holes.)

**Fix (display only, two levers, both flag/config, planner untouched):**
1. `OBSTACLE_BODY_FILL_CLOSE_M` `0.14 → 0.28` — bridge wider stud gaps into fillable bodies
   (observed `bodyFill` 64 → ~8200 on the next run).
2. New `_fit_obstacle_primitive()` + `OBSTACLE_BODY_FILL_FIT_PRIMITIVES=True`: per cluster fit the
   tighter of `cv2.minAreaRect` / `cv2.minEnclosingCircle` and fill the **completed** shape — the
   scene's furniture is rectangular or round, so this reconstructs the body the sparse outline
   implies (and fills the gray interior). Guard `OBSTACLE_BODY_FILL_FIT_MAX_FREE_RATIO=0.55`: if the
   fitted shape lands mostly on confidently-free floor (bad fit across open space, e.g. an L-arc),
   reject and fall back to the flood fill. The existing `body &= ~free / ~cleaned` post-clip still
   trims any floor the primitive overlaps.

**Watch next run:** obstacles should now show clean solid rectangles/circles (not dotted outlines),
`bodyFill` much higher. If a primitive bulges onto open floor → lower `FIT_MAX_FREE_RATIO` (e.g. 0.4)
or raise `MIN_COMPONENT_CELLS`. If two near obstacles merge into one rect → reduce `CLOSE_M` toward
0.20. Revert primitive completion with `OBSTACLE_BODY_FILL_FIT_PRIMITIVES=False` (keeps the flood fill).

## Round 6w — raw-forward wandering + junk fills

**Symptoms (user):** (1) "raw forward очень плохо влияет на построение дороги" — robot keeps doing
ROW_FORWARD / frontier-direct / holdDrive and bounces at walls instead of following a planned route;
"со временем поиск пути работает хуже, чаще move forward". (2) some junk obstacle fills (a small
diamond/quad over near-floor) alongside the good ones.

**Diagnosis (1):** the frontier owner gate commits the planned wavefront route only when it is BOTH
committable (safety) AND, in the hybrid map-building phase (cov < `MIN_COVERAGE_PERCENT=48`), short
(`HYBRID_SHORT_COMMIT_MAX_COST_M=1.80 m`). Far frontier routes (cost 2–4.5 m) exceeded 1.80 m → never
committed → fell to frontier-direct/holdDrive = "drive forward toward the frontier" = the wall
bouncing. **Lever:** `EXPLORATION_FRONTIER_HYBRID_SHORT_COMMIT_MAX_COST_M` `1.80 → 3.00` so medium
routes commit and the robot follows the planned path (safety still gated by
`coverage_candidate_is_committable`). Deeper holdDrive at cov ≥ 48 is the committable policy — riskier,
left for a separate lever.

**Diagnosis (2):** `_build_body_floor_enclosed` accepted any enclosed region with ≥18 obstacle points;
small noise clusters in a tightly-looped pocket fit a tiny figure. **Lever:** new
`OBSTACLE_BODY_FILL_FLOOR_MIN_PTS=40` — the main point cluster must have ≥40 points, dropping junk
fills from a few scattered points while real obstacles (hundreds of points) stay.

**Watch next run:** more `owner=ROUTE_COMMIT` / fewer `frontierGate=direct`/`holdDrive`, cleaner paths,
less wall bouncing. Junk diamonds/quads gone. If a real small obstacle stops filling → lower
`FLOOR_MIN_PTS`. If route-commit over-commits to bad far routes → lower `HYBRID_SHORT_COMMIT_MAX_COST_M`.

## Round 6v — use the BLUE driveable floor as the obstacle boundary (user's idea)

**Idea (user):** obstacles are surrounded on all sides by blue floor (free + cleaned) the robot drove,
and together with the black points that forms the figure — use that instead of fitting sparse points.

**Fix (display only, flag `OBSTACLE_BODY_FILL_FLOOR_ENCLOSED`, takes precedence over 6u/6t):** new
`_build_body_floor_enclosed()`. Find the non-blue region ENCLOSED by blue floor (flood non-blue from
the map border; what it can't reach is walled off by driveable floor = an obstacle footprint). Then
per enclosed region: fit the tighter of min-area-rect / min-enclosing-circle to the LARGEST obstacle-
point cluster in it, and paint `figure ∩ non-blue-region`. Two follow-up fixes after the first cut:
- fit to the **largest point cluster** (points dilated `FLOOR_COVER_M=0.30`), so a stray far point
  can't stretch the rectangle;
- **clip the figure to the enclosed non-blue region**, so it never paints driveable (blue) floor and
  never extends past where the robot drove around the object.
Span cap + wall test on the fill. Phantom-proof: an unexplored pocket has no real point cluster → tiny
fill; a giant enclosed area → rect exceeds span cap.

**Watch next run:** circled obstacles should fill as clean rectangles/circles tight to the object, no
spill onto blue floor, no stray-point tails. Partially-circled ones fill progressively as the floor
closes around them. Revert with `OBSTACLE_BODY_FILL_FLOOR_ENCLOSED=False`.

## Round 6u-fix — gray-bridge made a GIANT phantom rectangle

**Symptom:** a huge tilted orange rectangle filled half the room over open/unexplored area.

**Cause:** the first 6u dilated obstacle points by 0.55 m and counted ALL nearby gray as "body".
Near scattered noise points there was lots of UNEXPLORED gray (open, not interior), which stitched
far-apart points into one cluster → `minAreaRect` = giant rect; evidence (`points|gray`) passed
because the interior was gray, free-check passed because gray isn't free → it painted a phantom over
unexplored space.

**Fix:** (1) bridge by closing the obstacle POINTS only (bounded, 0.40 m), never via gray; (2) fill
only the gray the outline ENCLOSES (`_fill_enclosed_holes`, walled off from the window border) — open
gray is excluded; (3) keep the fitted figure only if that enclosed body fills ≥60% of the rect/circle
(the "looks like a figure" test) — scattered points in open space have little enclosed body → rejected.

## Round 6u — gray-bridged geometric completion (big obstacles stay hollow)

**Symptom:** with 6t primitive-fitting, a small compact obstacle fills cleanly, but BIG obstacles
keep a gray interior with only stray orange bits. User: "серая зона + точки = препятствие, если
похоже на геометрическую фигуру".

**Diagnosis:** 6t formed clusters by morph-closing obstacle POINTS only. A big object's perimeter
studs are sparser, so `close 0.28m` leaves gaps → the outline fragments into separate arc-components
→ each fits a tiny rect → interior gray never enclosed. Compact objects connect fully and fill.

**Fix (display only, flag `OBSTACLE_BODY_FILL_GRAY_BRIDGE`):** new `_build_body_gray_bridge()`:
1. `seed = obstacle points ∪ (unknown cells within OBSTACLE_BODY_FILL_GRAY_BRIDGE_M=0.55m of points)`
   — the interior/shadow gray itself bridges the outline gaps, so the whole object becomes ONE
   cluster. (Only gray *near* points, never the whole unexplored map.)
2. Per cluster, fit the tighter of min-area-rect / min-enclosing-circle to the real obstacle POINTS
   (extent matches the object, not the gray halo) and fill it.
3. Keep the shape only if it is a genuine figure: `≥ OBSTACLE_BODY_FILL_GRAY_MIN_EVIDENCE=0.60` of it
   is points-or-gray, `≤ FIT_MAX_FREE_RATIO=0.55` is observed-free, and it is not mostly wall
   (relaxed wall test: reject only if >50% wall, so obstacles beside a wall survive). Span cap +
   `~free`/`~cleaned` post-clip unchanged.

**Watch next run:** big obstacles should fill as clean rectangles/circles (gray interior absorbed),
not stray bits. If a fill bleeds into open floor → lower `GRAY_BRIDGE_M` (0.55→0.4) or raise
`GRAY_MIN_EVIDENCE` (0.60→0.7). If two near obstacles merge into one rect → lower `GRAY_BRIDGE_M`.
Revert with `OBSTACLE_BODY_FILL_GRAY_BRIDGE=False` (falls back to 6t per-point fitting).

**Reconstruction-vs-ground-truth metric (the "after").** The GT comparison previously used the raw
sensed mask (`actual`) only, so the solid .wbt boxes showed mostly blue (missed interior) even though
the body-fill completes them. Added a SECOND evaluation in `collect_metrics_snapshot`: compare
`actual ∪ body_fill` against ground truth (with the filled interior added to the explored set), saved
as `metrics/obstacle_comparison_reconstructed.png` and reported as `recon_precision/recall/f1/iou` in
the JSON summary + console (`obstacleMap (recon)`). The raw-mask metric is untouched (stays the thesis
number). This gives the honest before→after: raw outline (blue interiors) → completed bodies (matched),
recall/IoU up; over-filled rects that exceed the true box correctly count as red (precision penalty).

## How to run
Open `worlds/cleaning_world.wbt` in Webots from **this** folder
(`cleaning_robot_fixed`). The controller name (`rgbd_navigation_cleaner`) is
unchanged, so it picks up these files automatically.
