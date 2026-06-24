"""Experiment metrics recorder for the RGB-D navigation prototype.

The recorder is intentionally independent from Webots.  The controller passes
plain dictionaries with already computed values; this module only handles
periodic CSV/JSON export and a small amount of event counting.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, Iterable


class MetricsRecorder:
    """Write repeatable experiment metrics for diploma evaluation.

    Files created inside controllers/rgbd_navigation_cleaner/metrics/:
      - metrics_timeseries_*.csv: coverage, route, owner and map-quality values;
      - metrics_events_*.csv: bumper/recovery/route-abort/known-map-wait events;
      - metrics_summary_*.json: aggregate values used in the report.
    """

    FIELDNAMES = [
        "sample_id",
        "time_s",
        "step_id",
        "pose_x_m",
        "pose_y_m",
        "pose_theta_deg",
        "gt_x_m",
        "gt_y_m",
        "localization_error_m",
        "path_length_m",
        "coverage_percent",
        "coverage_cleaned_cells",
        "coverage_total_cells",
        "uncleaned_cells",
        "frontier_cells",
        "gray_gap_cells",
        "gray_gap_components",
        "planner_intent",
        "planner_mode",
        "navigation_phase",
        "nav_state",
        "owner_source",
        "control_owner",
        "route_active",
        "route_kind",
        "route_len_cells",
        "route_cost_m",
        "route_status",
        "map_mature",
        "known_map_active",
        "dock_completed",
        "raw_obstacle_cells",
        "actual_obstacle_cells",
        "no_go_cells",
        "hypothesis_obstacle_cells",
        "contact_obstacle_cells",
        "structural_obstacle_cells",
        "thin_obstacle_cells",
        "weak_unconfirmed_obstacle_cells",
        "gt_obstacle_cells",
        "false_occupied_cells",
        "false_free_cells",
        "false_occupied_ratio",
        "obstacle_precision",
        "obstacle_recall",
<<<<<<< HEAD
=======
        "recon_precision",
        "recon_recall",
        "recon_obstacle_cells",
        "recon_false_occupied_cells",
        "recon_false_free_cells",
        "explored_percent",
        "known_cells",
        "interior_unknown_cells",
        "room_closed",
        "obstacle_body_fill_cells",
        "explored_area_m2",
        "coverage_area_m2",
>>>>>>> brave-merkle-main
        "quality_ok",
        "quality_debug",
        "loop_ms",
        "planner_ms",
        "planner_nodes",
        "load_shedding",
        "bumper_active",
        "low_obstacle_debug",
        "bumper_events",
        "recovery_events",
        "route_abort_events",
        "known_wait_events",
        "low_obstacle_marks",
    ]

    EVENT_FIELDNAMES = ["time_s", "step_id", "event", "detail"]

    def __init__(self, output_dir: Path, prefix: str, enabled: bool = True):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.prefix = str(prefix)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeseries_path = self.output_dir / f"{self.prefix}_timeseries.csv"
        self.events_path = self.output_dir / f"{self.prefix}_events.csv"
        self.summary_path = self.output_dir / f"{self.prefix}_summary.json"
        self.latest_summary_path = self.output_dir / "latest_summary.json"

        self.sample_id = 0
        self.first_time = None
        self.last_time = 0.0
        self.last_coverage = 0.0
        self.max_coverage = 0.0
        self.final_coverage = 0.0
        self.path_length_m = 0.0
        self.max_frontier_cells = 0
        self.max_gray_gap_cells = 0
        self.min_quality_ok = True
        self.max_weak_unconfirmed = 0
        self.max_actual_obstacles = 0
        self.max_hypothesis_obstacles = 0
        self.total_bumper_events = 0
        self.total_recovery_events = 0
        self.total_route_abort_events = 0
        self.total_known_wait_events = 0
        self.total_low_obstacle_marks = 0
        # Localization-error accumulators (ground-truth GPS vs odometry).
        self.loc_err_count = 0
        self.loc_err_sum = 0.0
        self.loc_err_sq_sum = 0.0
        self.loc_err_max = 0.0
        self.loc_err_final = 0.0
        # Ground-truth obstacle-map accuracy (last heavy sample wins for final).
        self.gt_samples = 0
        self.false_occupied_ratio_final = None
        self.false_occupied_ratio_max = 0.0
        self.obstacle_precision_final = None
        self.obstacle_recall_final = None
        self.obstacle_recall_max = 0.0
        self.false_occupied_cells_final = 0
        self.false_free_cells_final = 0
        self.gt_obstacle_cells = 0
<<<<<<< HEAD
=======
        # Exploration completeness + obstacle reconstruction (Round 6t).
        self.explored_percent_final = 0.0
        self.explored_percent_max = 0.0
        self.explored_area_final = 0.0
        self.room_closed = False
        self.time_to_room_closed = None
        self.body_fill_final = 0
        self.body_fill_max = 0
        # Reconstruction (geometric body-fill) vs ground truth (Round 6u).
        self.recon_gt_samples = 0
        self.recon_precision_final = None
        self.recon_recall_final = None
        self.recon_false_occupied_cells_final = 0
        self.recon_false_free_cells_final = 0
        self.recon_obstacle_cells_final = 0
>>>>>>> brave-merkle-main
        self.mode_seconds: Dict[str, float] = {}
        self.owner_seconds: Dict[str, float] = {}
        self._prev_time = None
        self._prev_bumper_active = False
        self._prev_recovery_active = False
        self._prev_route_abort_active = False
        self._prev_known_wait_active = False
        self._prev_low_obstacle_mark = ""

        self._csv_file = None
        self._csv_writer = None
        self._event_file = None
        self._event_writer = None
        if self.enabled:
            self._csv_file = self.timeseries_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            self._csv_writer.writeheader()
            self._event_file = self.events_path.open("w", newline="", encoding="utf-8")
            self._event_writer = csv.DictWriter(self._event_file, fieldnames=self.EVENT_FIELDNAMES, extrasaction="ignore")
            self._event_writer.writeheader()

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_text(value: Any) -> str:
        text = str(value if value is not None else "")
        return text.replace("\n", " ").replace("\r", " ")

    def _write_event(self, time_s: float, step_id: int, event: str, detail: str = "") -> None:
        if not self.enabled or self._event_writer is None:
            return
        self._event_writer.writerow({
            "time_s": f"{time_s:.3f}",
            "step_id": int(step_id),
            "event": str(event),
            "detail": self._safe_text(detail),
        })

    def record(self, snapshot: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        row = {name: snapshot.get(name, "") for name in self.FIELDNAMES}
        time_s = self._as_float(snapshot.get("time_s"), 0.0)
        step_id = self._as_int(snapshot.get("step_id"), 0)
        coverage = self._as_float(snapshot.get("coverage_percent"), 0.0)
        frontier = self._as_int(snapshot.get("frontier_cells"), 0)
        gray_gap = self._as_int(snapshot.get("gray_gap_cells"), 0)
        path_len = self._as_float(snapshot.get("path_length_m"), self.path_length_m)
        weak_unconfirmed = self._as_int(snapshot.get("weak_unconfirmed_obstacle_cells"), 0)
        actual_obstacles = self._as_int(snapshot.get("actual_obstacle_cells"), 0)
        hypothesis_obstacles = self._as_int(snapshot.get("hypothesis_obstacle_cells"), 0)

        if self.first_time is None:
            self.first_time = time_s
        if self._prev_time is not None and time_s >= self._prev_time:
            dt = time_s - self._prev_time
            mode = self._safe_text(snapshot.get("planner_mode", "unknown")) or "unknown"
            owner = self._safe_text(snapshot.get("owner_source", "unknown")) or "unknown"
            self.mode_seconds[mode] = self.mode_seconds.get(mode, 0.0) + dt
            self.owner_seconds[owner] = self.owner_seconds.get(owner, 0.0) + dt
        self._prev_time = time_s

        self.last_time = time_s
        self.last_coverage = coverage
        self.final_coverage = coverage
        self.max_coverage = max(self.max_coverage, coverage)
        self.path_length_m = max(self.path_length_m, path_len)
        self.max_frontier_cells = max(self.max_frontier_cells, frontier)
        self.max_gray_gap_cells = max(self.max_gray_gap_cells, gray_gap)
        self.max_weak_unconfirmed = max(self.max_weak_unconfirmed, weak_unconfirmed)
        self.max_actual_obstacles = max(self.max_actual_obstacles, actual_obstacles)
        self.max_hypothesis_obstacles = max(self.max_hypothesis_obstacles, hypothesis_obstacles)
        if snapshot.get("quality_ok") is False or str(snapshot.get("quality_ok", "")).lower() == "false":
            self.min_quality_ok = False

        if "localization_error_m" in snapshot:
            loc_err = self._as_float(snapshot.get("localization_error_m"), 0.0)
            self.loc_err_count += 1
            self.loc_err_sum += loc_err
            self.loc_err_sq_sum += loc_err * loc_err
            self.loc_err_max = max(self.loc_err_max, loc_err)
            self.loc_err_final = loc_err

        if "false_occupied_ratio" in snapshot:
            self.gt_samples += 1
            for_ratio = self._as_float(snapshot.get("false_occupied_ratio"), 0.0)
            self.false_occupied_ratio_final = for_ratio
            self.false_occupied_ratio_max = max(self.false_occupied_ratio_max, for_ratio)
            self.obstacle_precision_final = self._as_float(snapshot.get("obstacle_precision"), 0.0)
            rec = self._as_float(snapshot.get("obstacle_recall"), 0.0)
            self.obstacle_recall_final = rec
            self.obstacle_recall_max = max(self.obstacle_recall_max, rec)
            self.false_occupied_cells_final = self._as_int(snapshot.get("false_occupied_cells"), 0)
            self.false_free_cells_final = self._as_int(snapshot.get("false_free_cells"), 0)
            self.gt_obstacle_cells = self._as_int(snapshot.get("gt_obstacle_cells"), 0)

<<<<<<< HEAD
=======
        if "recon_precision" in snapshot:
            self.recon_gt_samples += 1
            self.recon_precision_final = self._as_float(snapshot.get("recon_precision"), 0.0)
            self.recon_recall_final = self._as_float(snapshot.get("recon_recall"), 0.0)
            self.recon_false_occupied_cells_final = self._as_int(snapshot.get("recon_false_occupied_cells"), 0)
            self.recon_false_free_cells_final = self._as_int(snapshot.get("recon_false_free_cells"), 0)
            self.recon_obstacle_cells_final = self._as_int(snapshot.get("recon_obstacle_cells"), 0)

        # Exploration completeness + obstacle reconstruction (Round 6t).
        explored_pct = self._as_float(snapshot.get("explored_percent"), self.explored_percent_final)
        self.explored_percent_final = explored_pct
        self.explored_percent_max = max(self.explored_percent_max, explored_pct)
        self.explored_area_final = self._as_float(snapshot.get("explored_area_m2"), self.explored_area_final)
        body_fill = self._as_int(snapshot.get("obstacle_body_fill_cells"), self.body_fill_final)
        self.body_fill_final = body_fill
        self.body_fill_max = max(self.body_fill_max, body_fill)
        room_closed_now = snapshot.get("room_closed", False)
        if (not self.room_closed) and (room_closed_now is True or str(room_closed_now).lower() == "true"):
            self.room_closed = True
            self.time_to_room_closed = max(0.0, time_s - (self.first_time if self.first_time is not None else time_s))

>>>>>>> brave-merkle-main
        bumper_active = bool(snapshot.get("bumper_active", False))
        if bumper_active and not self._prev_bumper_active:
            self.total_bumper_events += 1
            self._write_event(time_s, step_id, "bumper", snapshot.get("nav_state", ""))
        self._prev_bumper_active = bumper_active

        nav_state = self._safe_text(snapshot.get("nav_state", ""))
        recovery_active = any(token in nav_state for token in ("RECOVERY", "CONTACT", "LEG_ESCAPE"))
        if recovery_active and not self._prev_recovery_active:
            self.total_recovery_events += 1
            self._write_event(time_s, step_id, "recovery", nav_state)
        self._prev_recovery_active = recovery_active

        control_owner = self._safe_text(snapshot.get("control_owner", ""))
        route_status = self._safe_text(snapshot.get("route_status", ""))
        route_abort_active = ("abort" in control_owner.lower()) or ("abort" in route_status.lower())
        if route_abort_active and not self._prev_route_abort_active:
            self.total_route_abort_events += 1
            self._write_event(time_s, step_id, "route_abort", f"{control_owner} {route_status}")
        self._prev_route_abort_active = route_abort_active

        known_wait_active = "known-map wait" in control_owner.lower()
        if known_wait_active and not self._prev_known_wait_active:
            self.total_known_wait_events += 1
            self._write_event(time_s, step_id, "known_map_wait", control_owner)
        self._prev_known_wait_active = known_wait_active

        low_mark = self._safe_text(snapshot.get("low_obstacle_debug", ""))
        if "lowObs=mark" in low_mark and low_mark != self._prev_low_obstacle_mark:
            self.total_low_obstacle_marks += 1
            self._write_event(time_s, step_id, "low_obstacle_mark", low_mark)
        self._prev_low_obstacle_mark = low_mark

        row["sample_id"] = self.sample_id
        row["bumper_events"] = self.total_bumper_events
        row["recovery_events"] = self.total_recovery_events
        row["route_abort_events"] = self.total_route_abort_events
        row["known_wait_events"] = self.total_known_wait_events
        row["low_obstacle_marks"] = self.total_low_obstacle_marks
        # Normalize values for CSV readability.
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.6g}"
            elif isinstance(value, bool):
                row[key] = int(value)
            else:
                row[key] = self._safe_text(value)
        self._csv_writer.writerow(row)
        self.sample_id += 1
        if self.sample_id % 10 == 0:
            self.flush()

    def summary(self) -> Dict[str, Any]:
        duration = max(0.0, (self.last_time - self.first_time) if self.first_time is not None else 0.0)
<<<<<<< HEAD
=======
        # Derived map-quality metrics (Round 6t): F1 and IoU of the obstacle map vs ground
        # truth, from the already-collected precision/recall and TP/FP/FN cell counts.
        p = self.obstacle_precision_final
        r = self.obstacle_recall_final
        obstacle_f1 = round(2.0 * p * r / (p + r), 4) if (p is not None and r is not None and (p + r) > 0) else None
        tp = max(0, int(self.gt_obstacle_cells) - int(self.false_free_cells_final))
        iou_denom = tp + int(self.false_occupied_cells_final) + int(self.false_free_cells_final)
        obstacle_iou = round(tp / iou_denom, 4) if (self.gt_samples and iou_denom > 0) else None
        # Reconstruction (geometric body-fill completion) vs ground truth — the "after" metric.
        rp = self.recon_precision_final
        rr = self.recon_recall_final
        recon_f1 = round(2.0 * rp * rr / (rp + rr), 4) if (rp is not None and rr is not None and (rp + rr) > 0) else None
        recon_tp = max(0, int(self.gt_obstacle_cells) - int(self.recon_false_free_cells_final))
        recon_iou_denom = recon_tp + int(self.recon_false_occupied_cells_final) + int(self.recon_false_free_cells_final)
        recon_iou = round(recon_tp / recon_iou_denom, 4) if (self.recon_gt_samples and recon_iou_denom > 0) else None
        exploration_efficiency = (
            round(self.explored_area_final / self.path_length_m, 4)
            if (self.path_length_m > 0 and self.explored_area_final > 0) else None
        )
>>>>>>> brave-merkle-main
        return {
            "duration_s": round(duration, 3),
            "samples": int(self.sample_id),
            "final_coverage_percent": round(float(self.final_coverage), 3),
            "max_coverage_percent": round(float(self.max_coverage), 3),
            "path_length_m": round(float(self.path_length_m), 3),
            "max_frontier_cells": int(self.max_frontier_cells),
            "max_gray_gap_cells": int(self.max_gray_gap_cells),
            "max_actual_obstacle_cells": int(self.max_actual_obstacles),
            "max_hypothesis_obstacle_cells": int(self.max_hypothesis_obstacles),
            "max_weak_unconfirmed_obstacle_cells": int(self.max_weak_unconfirmed),
            "map_quality_ok_all_samples": bool(self.min_quality_ok),
            "localization_samples": int(self.loc_err_count),
            "localization_error_mean_m": round(self.loc_err_sum / self.loc_err_count, 4) if self.loc_err_count else None,
            "localization_error_rmse_m": round(math.sqrt(self.loc_err_sq_sum / self.loc_err_count), 4) if self.loc_err_count else None,
            "localization_error_max_m": round(float(self.loc_err_max), 4) if self.loc_err_count else None,
            "localization_error_final_m": round(float(self.loc_err_final), 4) if self.loc_err_count else None,
            "gt_samples": int(self.gt_samples),
            "gt_obstacle_cells": int(self.gt_obstacle_cells) if self.gt_samples else None,
            "false_occupied_cells_final": int(self.false_occupied_cells_final) if self.gt_samples else None,
            "false_free_cells_final": int(self.false_free_cells_final) if self.gt_samples else None,
            "false_occupied_ratio_final": round(float(self.false_occupied_ratio_final), 4) if self.false_occupied_ratio_final is not None else None,
            "false_occupied_ratio_max": round(float(self.false_occupied_ratio_max), 4) if self.gt_samples else None,
            "obstacle_precision_final": round(float(self.obstacle_precision_final), 4) if self.obstacle_precision_final is not None else None,
            "obstacle_recall_final": round(float(self.obstacle_recall_final), 4) if self.obstacle_recall_final is not None else None,
            "obstacle_recall_max": round(float(self.obstacle_recall_max), 4) if self.gt_samples else None,
<<<<<<< HEAD
=======
            "obstacle_f1": obstacle_f1,
            "obstacle_iou": obstacle_iou,
            "recon_precision_final": round(float(self.recon_precision_final), 4) if self.recon_precision_final is not None else None,
            "recon_recall_final": round(float(self.recon_recall_final), 4) if self.recon_recall_final is not None else None,
            "recon_f1": recon_f1,
            "recon_iou": recon_iou,
            "recon_obstacle_cells_final": int(self.recon_obstacle_cells_final) if self.recon_gt_samples else None,
            "explored_percent_final": round(float(self.explored_percent_final), 3),
            "explored_percent_max": round(float(self.explored_percent_max), 3),
            "explored_area_m2_final": round(float(self.explored_area_final), 3),
            "exploration_efficiency_m2_per_m": exploration_efficiency,
            "room_closed": bool(self.room_closed),
            "time_to_room_closed_s": round(float(self.time_to_room_closed), 2) if self.time_to_room_closed is not None else None,
            "obstacle_body_fill_cells_final": int(self.body_fill_final),
            "obstacle_body_fill_cells_max": int(self.body_fill_max),
>>>>>>> brave-merkle-main
            "bumper_events": int(self.total_bumper_events),
            "recovery_events": int(self.total_recovery_events),
            "route_abort_events": int(self.total_route_abort_events),
            "known_map_wait_events": int(self.total_known_wait_events),
            "low_obstacle_marks": int(self.total_low_obstacle_marks),
            "mode_seconds": {k: round(v, 3) for k, v in sorted(self.mode_seconds.items())},
            "owner_seconds": {k: round(v, 3) for k, v in sorted(self.owner_seconds.items())},
            "timeseries_csv": str(self.timeseries_path),
            "events_csv": str(self.events_path),
        }

    def export_summary(self) -> Dict[str, Any]:
        data = self.summary()
        if self.enabled:
            self.flush()
            self.summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.latest_summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    def flush(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
        if self._event_file is not None:
            self._event_file.flush()

    def close(self) -> Dict[str, Any]:
        data = self.export_summary()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._event_file is not None:
            self._event_file.close()
            self._event_file = None
        return data
