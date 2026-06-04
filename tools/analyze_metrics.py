"""Build diploma-friendly tables and plots from metrics_timeseries_*.csv.

Usage from project root:
    python tools/analyze_metrics.py controllers/rgbd_navigation_cleaner/metrics/metrics_XXXXXXXX_timeseries.csv

Outputs are written next to the CSV:
    *_summary_table.csv
    *_coverage_time.png
    *_trajectory.png
    *_owner_distribution.png
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    csv_path = Path(argv[1]).resolve()
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"empty metrics file: {csv_path}")
    out_prefix = csv_path.with_suffix("")

    time_s = [to_float(r, "time_s") for r in rows]
    t0 = time_s[0]
    rel_t = [t - t0 for t in time_s]
    cov = [to_float(r, "coverage_percent") for r in rows]
    xs = [to_float(r, "pose_x_m") for r in rows]
    ys = [to_float(r, "pose_y_m") for r in rows]
    path_len = [to_float(r, "path_length_m") for r in rows]
    loop_ms = [to_float(r, "loop_ms") for r in rows]
    planner_ms = [to_float(r, "planner_ms") for r in rows]
    # Localization error (ground-truth GPS vs odometry) is optional: present only
    # when the world has the metrics GPS node and rows carry the column.
    loc_pairs = [(t, to_float(r, "localization_error_m")) for t, r in zip(rel_t, rows)
                 if str(r.get("localization_error_m", "")).strip() not in ("", "None")]
    has_loc = len(loc_pairs) > 0

    duration = max(0.0, rel_t[-1])
    final_cov = cov[-1]
    max_cov = max(cov)
    final_path = max(path_len)
    mean_loop = sum(loop_ms) / max(1, len(loop_ms))
    max_loop = max(loop_ms)
    mean_planner = sum(planner_ms) / max(1, len(planner_ms))
    bumper_events = int(float(rows[-1].get("bumper_events", 0) or 0))
    recovery_events = int(float(rows[-1].get("recovery_events", 0) or 0))
    route_abort_events = int(float(rows[-1].get("route_abort_events", 0) or 0))
    known_wait_events = int(float(rows[-1].get("known_wait_events", 0) or 0))
    low_marks = int(float(rows[-1].get("low_obstacle_marks", 0) or 0))

    summary_rows = [
        ("duration_s", f"{duration:.2f}"),
        ("final_coverage_percent", f"{final_cov:.2f}"),
        ("max_coverage_percent", f"{max_cov:.2f}"),
        ("path_length_m", f"{final_path:.2f}"),
        ("mean_loop_ms", f"{mean_loop:.2f}"),
        ("max_loop_ms", f"{max_loop:.2f}"),
        ("mean_planner_ms", f"{mean_planner:.2f}"),
        ("bumper_events", str(bumper_events)),
        ("recovery_events", str(recovery_events)),
        ("route_abort_events", str(route_abort_events)),
        ("known_map_wait_events", str(known_wait_events)),
        ("low_obstacle_marks", str(low_marks)),
    ]
    if has_loc:
        errs = [e for _, e in loc_pairs]
        n = len(errs)
        mean_err = sum(errs) / n
        rmse_err = math.sqrt(sum(e * e for e in errs) / n)
        summary_rows.extend([
            ("localization_samples", str(n)),
            ("localization_error_mean_m", f"{mean_err:.4f}"),
            ("localization_error_rmse_m", f"{rmse_err:.4f}"),
            ("localization_error_max_m", f"{max(errs):.4f}"),
            ("localization_error_final_m", f"{errs[-1]:.4f}"),
        ])

    # Ground-truth obstacle accuracy (false-occupied cells), optional column.
    foc_pairs = [(t, to_float(r, "false_occupied_ratio")) for t, r in zip(rel_t, rows)
                 if str(r.get("false_occupied_ratio", "")).strip() not in ("", "None")]
    has_foc = len(foc_pairs) > 0
    if has_foc:
        last_row = next(r for r in reversed(rows) if str(r.get("false_occupied_ratio", "")).strip() not in ("", "None"))
        summary_rows.extend([
            ("gt_obstacle_cells", str(int(to_float(last_row, "gt_obstacle_cells")))),
            ("false_occupied_cells_final", str(int(to_float(last_row, "false_occupied_cells")))),
            ("false_free_cells_final", str(int(to_float(last_row, "false_free_cells")))),
            ("false_occupied_ratio_final", f"{to_float(last_row, 'false_occupied_ratio'):.4f}"),
            ("obstacle_precision_final", f"{to_float(last_row, 'obstacle_precision'):.4f}"),
            ("obstacle_recall_final", f"{to_float(last_row, 'obstacle_recall'):.4f}"),
        ])
    summary_path = out_prefix.with_name(out_prefix.name + "_summary_table.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Summary written:", summary_path)
        print("matplotlib is not installed, plots skipped:", exc)
        return 0

    plt.figure()
    plt.plot(rel_t, cov)
    plt.xlabel("Time, s")
    plt.ylabel("Coverage, %")
    plt.title("Coverage over time")
    plt.grid(True)
    coverage_png = out_prefix.with_name(out_prefix.name + "_coverage_time.png")
    plt.savefig(coverage_png, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.title("Robot trajectory")
    plt.axis("equal")
    plt.grid(True)
    trajectory_png = out_prefix.with_name(out_prefix.name + "_trajectory.png")
    plt.savefig(trajectory_png, dpi=160, bbox_inches="tight")
    plt.close()

    localization_png = None
    if has_loc:
        lt = [t for t, _ in loc_pairs]
        le = [e for _, e in loc_pairs]
        plt.figure()
        plt.plot(lt, le)
        plt.xlabel("Time, s")
        plt.ylabel("Localization error, m")
        plt.title("Localization error over time (odometry vs ground truth)")
        plt.grid(True)
        localization_png = out_prefix.with_name(out_prefix.name + "_localization_error.png")
        plt.savefig(localization_png, dpi=160, bbox_inches="tight")
        plt.close()

    false_occupied_png = None
    if has_foc:
        ft = [t for t, _ in foc_pairs]
        fr = [100.0 * v for _, v in foc_pairs]
        plt.figure()
        plt.plot(ft, fr)
        plt.xlabel("Time, s")
        plt.ylabel("False-occupied cells, %")
        plt.title("False-occupied obstacle cells over time (vs ground truth)")
        plt.grid(True)
        false_occupied_png = out_prefix.with_name(out_prefix.name + "_false_occupied.png")
        plt.savefig(false_occupied_png, dpi=160, bbox_inches="tight")
        plt.close()

    owner_time = defaultdict(float)
    for prev, cur in zip(rows[:-1], rows[1:]):
        dt = max(0.0, to_float(cur, "time_s") - to_float(prev, "time_s"))
        owner_time[prev.get("owner_source", "unknown") or "unknown"] += dt
    labels = list(owner_time.keys())
    values = [owner_time[k] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Time, s")
    plt.title("Motion owner distribution")
    plt.xticks(rotation=30, ha="right")
    owner_png = out_prefix.with_name(out_prefix.name + "_owner_distribution.png")
    plt.savefig(owner_png, dpi=160, bbox_inches="tight")
    plt.close()

    print("Summary written:", summary_path)
    plots = [coverage_png, trajectory_png, owner_png]
    if localization_png is not None:
        plots.append(localization_png)
    if false_occupied_png is not None:
        plots.append(false_occupied_png)
    print("Plots written:", *plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
