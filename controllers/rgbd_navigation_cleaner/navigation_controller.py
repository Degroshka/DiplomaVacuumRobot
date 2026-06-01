"""
Webots controller for the diploma RGB-D indoor navigation prototype.

The controller implements:
- RGB camera processing through OpenCV;
- RangeFinder as the RGB-D depth channel;
- wheel-encoder and IMU-based pose estimation;
- bumper/contact handling for low or poorly visible obstacles;
- occupancy, coverage, frontier and planning-map layers;
- frontier-based exploration and learned-map preparation for coverage planning.

The RangeFinder is treated as a depth channel, not as a lidar.  The project does
not use ORB-SLAM, RTAB-Map or an industrial SLAM framework.
"""

from controller import Robot
from pathlib import Path
import math
import time
import heapq

from mapping import (
    bresenham_cells as _bresenham_cells,
    clamp as _clamp,
    map_inside_cell as _map_inside_cell,
    normalize_angle as _normalize_angle,
    snap_to_right_angle as _snap_to_right_angle,
    update_log_odds_value as _update_log_odds_value,
    world_to_map_cell as _world_to_map_cell,
)
from robot_io import decode_camera_frame, decode_depth_image
from state_machine import NavigationPhase, update_navigation_phase as _update_navigation_phase
from motion_primitives import MotionPrimitive, apply_strict_motion_contract
from safety import bumper_event_from_raw
from recovery import decide_side_release_after_backup
from control_arbiter import ControlOwner, OwnershipLock
from wall_follow_controller import WallFollowConfig, WallFollowInput, compute_wall_follow_command
from async_debug_viewer import AsyncDebugViewerClient

try:
    import cv2
    import numpy as np
except Exception as exc:
    raise RuntimeError(
        "Install dependencies for the Python used by Webots: python -m pip install opencv-python numpy"
    ) from exc

from config import *


last_rgbd_occlusion_debug = "occGuard=0/0"

map_view_zoom = 1.0
map_view_auto_crop = True
map_view_follow_robot = False
map_view_pan_x = 0.0
map_view_pan_y = 0.0
last_map_view_crop_debug = "full"


MAP_BUILD_WALL_FOLLOW_CFG = WallFollowConfig(
    enabled=True,
    target_clearance_m=0.125,  # close, but not scraping; keep the wall row useful
    too_close_m=0.078,
    acquire_max_m=0.32,
    release_max_m=0.43,
    min_front_m=0.50,
    min_center_m=0.43,
    min_upper_m=0.43,
    max_yaw=0.16,
    kp_distance=0.85,
    kp_heading=0.66,
    speed=0.72,
    slow_speed=0.36,
    deadband_m=0.016,
    side_switch_margin_m=0.055,
    depth_follow_max_m=0.185,
    depth_far_pull_enabled=False,
    map_target_center_m=ROBOT_BODY_RADIUS + 0.075,
    map_too_close_center_m=ROBOT_BODY_RADIUS + 0.035,
    map_acquire_max_m=ROBOT_BODY_RADIUS + 0.230,
    map_weight=0.70,
)
wall_follow_active_side = 0.0
wall_follow_last_time = -999.0
wall_follow_last_reason = ""


robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_sensor = robot.getDevice("left wheel sensor")
right_sensor = robot.getDevice("right wheel sensor")
if left_sensor is None or right_sensor is None:
    raise RuntimeError("Wheel position sensors are missing. Add PositionSensor devices to both wheel HingeJoints.")
left_sensor.enable(timestep)
right_sensor.enable(timestep)

camera = robot.getDevice("front_camera")
if camera is None:
    raise RuntimeError('Camera device "front_camera" was not found.')
camera.enable(timestep)
CAM_W = camera.getWidth()
CAM_H = camera.getHeight()

range_finder = robot.getDevice("front_range_finder")
if range_finder is None:
    raise RuntimeError('RangeFinder device "front_range_finder" was not found. Add it to the Robot node.')
range_finder.enable(timestep)
RF_W = range_finder.getWidth()
RF_H = range_finder.getHeight()
try:
    RF_FOV = range_finder.getFov()
except Exception:
    pass

inertial_unit = robot.getDevice("inertial_unit")
if inertial_unit is not None:
    inertial_unit.enable(timestep)
else:
    print('WARNING: InertialUnit "inertial_unit" not found; falling back to pure wheel odometry heading.')

front_left_bumper = robot.getDevice("front_left_bumper")
front_center_bumper = robot.getDevice("front_center_bumper")
front_right_bumper = robot.getDevice("front_right_bumper")
for bumper_name, bumper in (
    ("front_left_bumper", front_left_bumper),
    ("front_center_bumper", front_center_bumper),
    ("front_right_bumper", front_right_bumper),
):
    if bumper is not None:
        bumper.enable(timestep)
    else:
        print(f'WARNING: TouchSensor "{bumper_name}" not found; bumper recovery disabled for that contact zone.')

out_dir = Path(__file__).resolve().parent
frames_dir = out_dir / "camera_frames"
maps_dir = out_dir / "maps"
frames_dir.mkdir(exist_ok=True)
maps_dir.mkdir(exist_ok=True)


pose_x = 0.0
pose_y = 0.0
pose_theta = 0.0
imu_yaw_zero = None
prev_imu_theta = None
desired_grid_heading = 0.0
grid_realign_target = 0.0
grid_realign_until = 0.0
grid_realign_after_status = "row forward after grid realign"
grid_realign_start_time = -999.0
grid_realign_best_abs_error = float("inf")
grid_realign_last_progress_time = -999.0
grid_realign_soft_finish = False
grid_realign_retry_count = 0
prev_left = None
prev_right = None
prev_cmd_left = 0.0
prev_cmd_right = 0.0
current_linear_velocity = 0.0
current_angular_velocity = 0.0
odom_arena_guard_until = -999.0
odom_arena_clamp_count = 0
odom_arena_clamp_debug = "arenaClamp=idle"
runtime_arena_guard_until = -999.0
runtime_arena_guard_debug = "arenaGuard=idle"
frontier_route_abort_hold_until = -999.0
frontier_route_abort_hold_debug = "routeHold=idle"
trajectory = []

log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
visual_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
thin_obstacle_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
contact_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
structural_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
under_surface_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
hypothesis_obstacle_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
hypothesis_obstacle_cache = None
hypothesis_obstacle_cache_step = -999999
last_hypothesis_obstacle_debug = "hypObs=init"
last_near_collision_hypothesis_time = -999.0
last_near_collision_hypothesis_debug = "nearHyp=idle"
step_id = 0
last_debug_time = time.time()
last_min_left = float("inf")
last_min_center = float("inf")
last_min_right = float("inf")
last_front_narrow = float("inf")
row_end_candidate_count = 0
last_rgb_contours = 0
last_rgb_lines = 0
last_cv_map_hits = 0
last_cv_depth_confirmed = 0
last_cv_features = {"lines": [], "boxes": [], "points": []}
last_cv_floor_rejected = 0
last_cv_shadow_rejected = 0
last_cv_left_obstacle = MAX_VALID_RANGE
last_cv_front_obstacle = MAX_VALID_RANGE
last_cv_right_obstacle = MAX_VALID_RANGE
map_freeze_until = 0.0
last_map_frozen = False
last_thin_hits = 0
last_depth_map_hits = 0
last_bumper_left = False
last_bumper_center = False
last_bumper_right = False
last_bumper_left_corner = False
last_bumper_right_corner = False
last_bumper_ignored_as_floor = False
last_floor_front_ignore = False
last_front_upper = float("inf")
last_body_corridor_clearance = MAX_VALID_RANGE
last_body_corridor_lateral = 0.0
last_left_side_bumper = False
last_right_side_bumper = False
last_contact_trap_reason = ""
last_contact_trap_time = -999.0
last_contact_map_cells = 0
last_sensor_stall_time = -999.0
last_stall_depth_signature = None
last_stall_rgb_signature = None
last_sensor_stall_reason = ""
last_motion_stall_time = -999.0
last_motion_stall_x = 0.0
last_motion_stall_y = 0.0
last_bumper_raw_left = 0.0
last_bumper_raw_center = 0.0
last_bumper_raw_right = 0.0
last_bumper_any_raw_active = False
last_contact_latch_time = -999.0
last_contact_latch_x = 0.0
last_contact_latch_y = 0.0
last_contact_latch_theta = 0.0
last_contact_latch_left = False
last_contact_latch_center = False
last_contact_latch_right = False
last_contact_latch_used = False
last_side_risk_escape_time = -999.0
under_furniture_suppressed_until = 0.0
last_local_pocket_fill_time = -999.0
last_local_pocket_side = 0.0
last_local_pocket_score = 0.0
last_local_pocket_uncleaned = 0
last_local_pocket_obstacle_ratio = 0.0
last_local_pocket_outer_uncleaned_ratio = 0.0

nav_state = NAV_FORWARD
turn_target_theta = 0.0
turn_direction = 1.0
turn_settle_until = 0.0
turn_start_left = 0.0
turn_start_right = 0.0
turn_start_time = 0.0
turn_best_abs_error = float("inf")
turn_last_progress_time = 0.0
pivot_watchdog_retry_count = 0
last_turn_reason = ""
last_turn_variant = "none"
pre_pivot_start_x = 0.0
pre_pivot_start_y = 0.0
pre_pivot_until = 0.0
pre_pivot_side = 1.0
pre_pivot_reason = ""

cleaned_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
recent_visit_log_odds = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
last_recent_visit_cells = 0
last_recent_target_penalty = 0.0
nav_action_queue = []
lane_side = 1.0               # +1: next lane to the left, -1: next lane to the right
simple_sweep_initialized = False
simple_sweep_completed = False
simple_sweep_completion_reason = "not started"
simple_sweep_state = "INIT"
simple_sweep_queue = []
simple_sweep_axis_heading = 0.0
simple_sweep_lane_direction = 1.0
simple_sweep_lane_shift_direction = 1.0
simple_sweep_lane_index = 0
simple_sweep_shift_attempt_side = 0.0
simple_sweep_planned_side_blocked = False
simple_sweep_opposite_side_blocked = False
simple_sweep_escape_start_x = 0.0
simple_sweep_escape_start_y = 0.0
simple_sweep_escape_until = 0.0
simple_sweep_align_target_heading = 0.0
simple_sweep_align_started_at = 0.0
simple_sweep_align_until = 0.0
simple_sweep_last_align_time = -999.0
simple_sweep_current_lane_lateral = 0.0
simple_sweep_lane_lateral_history = []
simple_sweep_shift_start_lateral = 0.0
simple_sweep_shift_retry_count = 0
simple_sweep_last_shift_global_side = 0.0
simple_sweep_initial_lateral = 0.0
simple_sweep_lane_target_lateral = 0.0
simple_sweep_pending_lane_target_lateral = None
simple_sweep_shift_target_lateral = 0.0
simple_sweep_active_shift_global_side = 0.0
simple_sweep_lane_target_history = []
simple_sweep_pending_shift_global_side = 0.0
simple_sweep_lane_score_debug = "laneScore=init"
simple_sweep_last_debug = "exploreSweep=init"
simple_sweep_churn_anchor_x = 0.0
simple_sweep_churn_anchor_y = 0.0
simple_sweep_churn_start_time = -999.0
simple_sweep_churn_action_count = 0
simple_sweep_churn_last_debug = "churn=idle"
simple_sweep_best_coverage_percent = 0.0
simple_sweep_best_coverage_time = -999.0
simple_sweep_best_coverage_x = 0.0
simple_sweep_best_coverage_y = 0.0
last_row_change_time = -999.0
last_revisit_lane_change_time = -999.0
last_explore_anti_revisit_turn_time = -999.0
post_lane_forward_lock_until = -999.0
post_lane_forward_lock_start_x = 0.0
post_lane_forward_lock_start_y = 0.0
lane_shift_start_x = 0.0
lane_shift_start_y = 0.0
lane_shift_start_time = 0.0
lane_shift_target_dist = LANE_SPACING
lane_shift_heading_target = 0.0
backup_start_x = 0.0
backup_start_y = 0.0
backup_until = 0.0
backup_after_side = 1.0
leg_escape_side = 1.0
leg_escape_start_x = 0.0
leg_escape_start_y = 0.0
leg_escape_until = 0.0
leg_escape_turn_target = 0.0
leg_escape_forward_start_x = 0.0
leg_escape_forward_start_y = 0.0
leg_escape_forward_until = 0.0
leg_escape_backup_distance_current = LEG_ESCAPE_BACKUP_DISTANCE
leg_escape_turn_angle_current = LEG_ESCAPE_TURN_ANGLE
leg_escape_forward_distance_current = LEG_ESCAPE_FORWARD_DISTANCE
leg_escape_forward_speed_current = LEG_ESCAPE_FORWARD_SPEED
leg_escape_replan_after = False

contact_recovery_kind = "none"
contact_recovery_reason = ""
contact_recovery_side = 1.0
contact_recovery_start_x = 0.0
contact_recovery_start_y = 0.0
contact_recovery_backup_distance = FRONT_CONTACT_TRAP_BACKUP_DISTANCE
contact_recovery_backup_until = 0.0
contact_recovery_clear_since = -999.0
contact_recovery_wait_until = 0.0
contact_recovery_turn_angle = FRONT_CONTACT_TRAP_TURN_ANGLE
contact_recovery_turn_target = 0.0
contact_recovery_forward_start_x = 0.0
contact_recovery_forward_start_y = 0.0
contact_recovery_forward_distance = FRONT_CONTACT_TRAP_FORWARD_DISTANCE
contact_recovery_forward_speed = CONTACT_ESCAPE_VERIFY_SPEED
contact_recovery_forward_until = 0.0
contact_recovery_forward_timeout = FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC
contact_recovery_replan_after = False
contact_recovery_bumper_hold_since = -999.0
last_leg_escape_time = -999.0
last_low_obstacle_guard_time = -999.0
last_contact_route_kill_until = -999.0
last_contact_escape_kind = "none"
last_rear_guard_clearance = 999.0
last_rear_guard_reason = "clear"
last_contact_cluster_time = -999.0
last_contact_cluster_x = 1e9
last_contact_cluster_y = 1e9
last_contact_cluster_count = 0
last_narrow_passage_time = -999.0
last_narrow_passage_start_time = -999.0
last_narrow_passage_side = 0.0
last_narrow_passage_reason = ""
last_gap_mouth_align_time = -999.0
last_gap_mouth_align_side = 0.0
last_gap_mouth_align_reason = ""
post_gap_commit_until = -999.0
post_gap_commit_start_x = 0.0
post_gap_commit_start_y = 0.0
post_recovery_stabilize_until = -999.0
post_recovery_stabilize_start_x = 0.0
post_recovery_stabilize_start_y = 0.0
post_recovery_stabilize_heading = 0.0
post_recovery_stabilize_distance = POST_RECOVERY_STABILIZE_DISTANCE_M
post_recovery_stabilize_kind = "none"
under_furniture_confirm_count = 0
leg_pass_side = 1.0
leg_pass_start_x = 0.0
leg_pass_start_y = 0.0
leg_pass_until = 0.0
leg_pass_target_heading = 0.0
leg_pass_original_heading = 0.0
leg_pass_grace_until = 0.0
last_leg_pass_time = -999.0
last_leg_pass_detected = False
last_leg_pass_side = 0.0
last_under_furniture_time = -999.0
under_furniture_until = 0.0
under_furniture_active = False
last_under_surface_cells = 0
last_under_surface_target_cells = 0
last_under_surface_marked = 0
last_residual_route_deferred_cells = 0
last_line_acquire_time = -999.0
last_line_acquire_side = 0.0
last_line_acquire_reason = ""
last_edge_trim_cells = 0
last_wall_fringe_trim_cells = 0
last_unreachable_target_trim_cells = 0
last_parallel_aperture_time = -999.0
last_parallel_aperture_count = 0
last_parallel_aperture_side = 0.0
last_parallel_aperture_score = 0
last_parallel_aperture_blocked_ratio = 1.0
last_parallel_aperture_x = 1e9
last_parallel_aperture_y = 1e9
coverage_status = "grid"
row_start_x = 0.0
row_start_y = 0.0
row_start_time = 0.0

coverage_goal_map = None
coverage_goal_world = None
coverage_goal_kind = "none"
coverage_route_map = []
coverage_route_world = []
coverage_route_waypoint_map = None
coverage_route_waypoint_world = None
coverage_route_cost = float("inf")
coverage_route_score = float("-inf")
coverage_route_len = 0
coverage_route_kind = "none"
coverage_route_component_id = -1
coverage_route_component_cells = 0
coverage_route_status = "no route"
coverage_route_top_debug = "none"
coverage_route_commit_class = "none"
coverage_route_straight_dist = float("inf")
coverage_route_lateral_abs = float("inf")
coverage_route_turn_need = float("inf")
coverage_route_continuity_bonus = 0.0
coverage_route_wall_strip_bonus = 0.0
coverage_route_missed_strip_bonus = 0.0
coverage_route_segment_gain = 0
coverage_route_segment_bonus = 0.0
coverage_route_footprint_gain_cells = 0
coverage_route_first_turn_frac = 0.0
coverage_route_corner_count = 0
coverage_route_geometry_debug = "geom=none"
coverage_target_sticky_map = None
coverage_target_sticky_kind = "none"
coverage_target_sticky_component_id = -1
coverage_target_sticky_score = float("-inf")
coverage_target_sticky_until = -999.0
coverage_target_sticky_debug = "none"
last_route_dock_loiter_guard_debug = "none"
last_missed_strip_recovery_debug = "none"
last_rejected_components_debug = "none"
last_missed_strip_promoted_cells = 0
last_coverage_footprint_debug = "none"
last_coverage_footprint_gain_cells = 0
last_coverage_footprint_reclean_ratio = 0.0
last_coverage_segment_debug = "none"
last_explore_arbitration_debug = "exploreMode=unknown"
last_exploration_route_debug = "exploreRoute=init"
last_frontier_target_debug = "frontierTarget=none"
last_known_backtrack_debug = "knownBacktrackPenalty=0"
last_explore_gain_debug = "exploreGain=0 coverageGain=0"
last_commit_type_debug = "commitType=none"
last_objective_noise_filter_debug = "noise off"
last_raw_map_speckle_debug = "rawNoise=idle"
last_furniture_zone_debug = "furnitureZone=init"
last_planning_layer_debug = "planningMap=init"
furniture_zone_core_cache = None
furniture_zone_inflated_cache = None
furniture_zone_cache_step = -999999
last_perf_debug = "perf startup"
last_perf_loop_ms = 0.0
last_planner_update_step = -999999
last_planner_update_x = 0.0
last_planner_update_y = 0.0
last_planner_update_reason = "startup"
perf_ema_ms = {}
perf_last_ms = {}
objective_noise_filter_cache = None
objective_noise_filter_cache_step = -999999
objective_noise_filter_cache_debug = "empty"
footprint_passability_cache = None
footprint_passability_cache_step = -999999
footprint_passability_cache_margin = None
footprint_value_cache = None
footprint_value_cache_step = -999999

active_scan_target_yaws = []
active_scan_index = 0
active_scan_dwell_until = -999.0
active_scan_started_at = -999.0
active_scan_started_x = 0.0
active_scan_started_y = 0.0
active_scan_reason = "none"
active_scan_last_completed_at = -999.0
active_scan_last_x = 1e9
active_scan_last_y = 1e9
active_scan_start_frontier_local = 0
active_scan_start_unknown_local = 0
active_scan_area_memory = []
last_active_scan_debug = "scan=idle"
last_frontier_only_gate_debug = "frontierGate=idle"
frontier_only_last_snapshot_at = -999.0
frontier_only_stall_since = -999.0
frontier_only_stall_x = 0.0
frontier_only_stall_y = 0.0
frontier_only_stall_theta = 0.0
frontier_only_stall_blacklists = 0
frontier_only_last_blacklist_time = -999.0
frontier_only_last_stall_debug = "stall=idle"
frontier_direct_last_unsafe_recovery_at = -999.0
frontier_direct_last_unsafe_recovery_debug = "unsafeRecovery=idle"
post_turn_rgbd_snapshot_until = -999.0
post_turn_rgbd_snapshot_started_at = -999.0
post_turn_rgbd_snapshot_frames = 0
post_turn_rgbd_snapshot_write_frames = 0
post_turn_rgbd_snapshot_reason = "none"
post_turn_rgbd_snapshot_pending = False
last_post_turn_snapshot_debug = "snap=idle"
last_mapping_write_debug = "mapWrite=init"
last_mapping_reason = "startup"
last_mapping_depth_hits_debug = 0
last_mapping_cv_hits_debug = 0
route_commit_turn_snapshot_armed = False
route_commit_turn_snapshot_started_at = -999.0
route_commit_turn_snapshot_peak_err = 0.0
route_commit_turn_snapshot_reason = "none"
last_exploration_cleanup_lock_debug = "cleanupLock=init"
known_map_eval_runtime_enabled = bool(KNOWN_MAP_COVERAGE_EVAL_DEFAULT)
known_map_eval_status = "knownMap=off"
known_map_eval_last_seed_time = -999.0

known_map_primary_sweep_completed = False
known_map_primary_sweep_finish_time = -999.0
known_map_primary_sweep_finish_coverage = 0.0
known_map_residual_cleanup_commits_started = 0
known_map_residual_cleanup_started_at = -999.0
known_map_residual_policy_status = "residualPolicy=primary"

map_mature = False
map_mature_soft = False
map_mature_reason = "startup"
planner_confidence = "EARLY"
planner_mode = "ROW_COVERAGE"
planner_intent = PLANNER_INTENT_EXPAND_MAP
planner_intent_reason = "startup"
route_commit_active = False
route_commit_target_map = None
route_commit_target_world = None
route_commit_kind = "none"
route_commit_route_map = []
route_commit_route_world = []
route_commit_waypoint_map = None
route_commit_waypoint_world = None
route_commit_progress_idx = 0
route_commit_waypoint_idx = 0
route_commit_waypoint_is_corner = False
route_commit_cost = float("inf")
route_commit_score = float("-inf")
route_commit_component_id = -1
route_commit_component_cells = 0
route_commit_wall_strip_bonus = 0.0
route_commit_missed_strip_bonus = 0.0
route_commit_segment_gain = 0
route_commit_footprint_gain_cells = 0
route_commit_first_turn_frac = 0.0
route_commit_corner_count = 0
route_commit_geometry_debug = "geom=none"
route_commit_started_at = -999.0
route_commit_last_abort_time = -999.0
route_commit_best_target_dist = float("inf")
route_commit_last_progress_time = -999.0
route_commit_target_blacklist = []
last_route_target_blacklist_debug = "none"
route_commit_reason = "none"
route_abort_reason = "none"
last_route_commit_debug = "inactive"

dock_return_active = False
dock_return_completed = False
dock_return_reason = "none"
dock_return_status = "inactive"
dock_return_last_plan_time = -999.0
dock_return_last_start_time = -999.0
dock_route_cost = float("inf")
auto_map_mission_phase = "EXPLORE_MAP"
auto_map_return_to_dock_active = False
auto_map_cleaning_started = False
auto_map_final_dock_requested = False
auto_map_ready_best_coverage = -1.0
auto_map_ready_best_time = -999.0
auto_map_ready_debug = "autoMap=init"
learned_map_sanitized_once = False
learned_map_sanitize_debug = "learnedMap=raw"
last_nearby_route_cleanup_time = -999.0
last_nearby_route_cleanup_reason = ""
last_side_strip_cleanup_time = -999.0
last_side_strip_cleanup_reason = ""
last_coverage_total_cells = 0
last_coverage_cleaned_cells = 0
last_coverage_percent = 0.0
last_frontier_cells = 0
last_gray_gap_cells = 0
last_gray_gap_components = 0
last_gray_gap_debug = "grayGap=init"
last_uncleaned_cells = 0
last_footprint_clearance_m = 0.0
last_footprint_route_blocked = 0
last_footprint_passable_cells = 0

navigation_phase = NavigationPhase.EXPLORE.value
last_navigation_phase_reason = "startup"
last_motion_primitive = MotionPrimitive.STOP.value
last_motion_contract_reason = "startup"
last_requested_left = 0.0
last_requested_right = 0.0
last_safety_event_desc = "clear"
contact_recovery_start_time = 0.0
contact_recovery_forced_side_release = False
last_contact_side_release_reason = ""
control_lock = OwnershipLock()
last_control_owner_debug = "NONE"
last_owner_source_debug = "ownerSource=none"
last_optional_block_reason = "startup"
debug_viewer_client = None
last_debug_viewer_status = "viewer=init"
last_load_shedding_debug = "load=normal"
last_render_throttle_debug = "render=init"

orb = cv2.ORB_create(nfeatures=350)
prev_kp = None
prev_desc = None
last_orb_matches = 0

print("Webots real-time RGB-D visual navigation started")
print("Mode: CV-FIRST mapping: RGB/OpenCV features + depth metric confirmation + wheel/IMU odometry")
print("RangeFinder is used as a depth channel/safety corridor, not as a lidar-style primary mapper")
print("Camera/RangeFinder in WBT: translation 0.215 0 0.06, rotation 0 -1 0 -5.31e-06; RangeFinder is the RGB-D depth channel, not a lidar")
print(f"Camera: {CAM_W}x{CAM_H}; RangeFinder: {RF_W}x{RF_H}; FOV={RF_FOV:.2f} rad")
print("OpenCV keys: S save map, R reset map, K toggle known-map eval, +/- map zoom, C auto/full map, 0 reset view, Q/Esc hide windows")
print(f"Performance mode={PERFORMANCE_MODE}: depthMap/{DEPTH_MAP_UPDATE_STEPS}->{DEPTH_ROUTE_COMMIT_MAP_UPDATE_STEPS}, CV/{CV_MAP_UPDATE_STEPS}->{CV_ROUTE_COMMIT_MAP_UPDATE_STEPS}, underSurface/{UNDER_SURFACE_UPDATE_STEPS}->{UNDER_ROUTE_COMMIT_UPDATE_STEPS}, planner/{PLANNER_UPDATE_STEPS}->{PLANNER_ROUTE_COMMIT_UPDATE_STEPS}, windows/{WINDOW_UPDATE_STEPS}->{WINDOW_ROUTE_COMMIT_UPDATE_STEPS}, ORB={'on' if ORB_DEBUG_ENABLED else 'off'}, occupancyWindow={'on' if SHOW_OCCUPANCY_MAP_WINDOW else 'off'}, asyncViewer={'on' if ASYNC_DEBUG_VIEWER_ENABLED else 'off'}, loadShed={'on' if ADAPTIVE_LOAD_SHEDDING_ENABLED else 'off'}")
# ---------------- Utility ----------------
def perf_start():
    return time.perf_counter() if PERF_PROFILER_ENABLED else 0.0


def perf_end(name, t0):
    if not PERF_PROFILER_ENABLED or not t0:
        return 0.0
    dt = max(0.0, (time.perf_counter() - t0) * 1000.0)
    try:
        prev = float(perf_ema_ms.get(name, dt))
        perf_ema_ms[name] = (1.0 - PERF_EMA_ALPHA) * prev + PERF_EMA_ALPHA * dt
        perf_last_ms[name] = dt
    except Exception:
        pass
    return dt


def perf_text(max_len=120):
    try:
        parts = []
        for key in PERF_DEBUG_INCLUDE:
            val = float(perf_ema_ms.get(key, 0.0))
            if val <= 0.05:
                continue
            parts.append(f"{key}={val:.0f}")
        txt = "perf " + " ".join(parts) if parts else "perf idle"
        return txt[:max_len]
    except Exception:
        return "perf err"


def committed_route_heavy_throttle_active():
    return bool(PERF_THROTTLE_WHILE_ROUTE_COMMIT and route_commit_active and route_commit_kind != "dock")


def adaptive_load_shedding_active():
    """True when the previous controller loop was slower than the budget.

    This does not affect collision checks or motor commands.  It only slows
    advisory work: planner, non-safety mapping and debug rendering.
    """
    if not ADAPTIVE_LOAD_SHEDDING_ENABLED:
        return False
    try:
        loop_last = float(last_perf_loop_ms or 0.0)
    except Exception:
        loop_last = 0.0
    try:
        loop_ema = float(perf_ema_ms.get("loop", 0.0))
    except Exception:
        loop_ema = 0.0
    return bool(loop_last >= float(ADAPTIVE_LOAD_SHEDDING_LOOP_MS) or loop_ema >= float(ADAPTIVE_LOAD_SHEDDING_EMA_MS))


def load_shedding_debug_text():
    try:
        if adaptive_load_shedding_active():
            return f"load=shed loop={float(last_perf_loop_ms):.0f} ema={float(perf_ema_ms.get('loop', 0.0)):.0f}"
        return f"load=normal loop={float(last_perf_loop_ms):.0f} ema={float(perf_ema_ms.get('loop', 0.0)):.0f}"
    except Exception:
        return "load=unknown"


def adaptive_cadence(base_steps, multiplier):
    base = max(1, int(base_steps))
    if adaptive_load_shedding_active():
        return max(base, int(base * max(1, int(multiplier))))
    return base


def effective_depth_map_update_steps():
    base = int(DEPTH_ROUTE_COMMIT_MAP_UPDATE_STEPS if committed_route_heavy_throttle_active() else DEPTH_MAP_UPDATE_STEPS)
    return max(1, adaptive_cadence(base, ADAPTIVE_LOAD_SHEDDING_MAPPING_MULT))


def effective_cv_map_update_steps():
    base = int(CV_ROUTE_COMMIT_MAP_UPDATE_STEPS if committed_route_heavy_throttle_active() else CV_MAP_UPDATE_STEPS)
    return max(1, adaptive_cadence(base, ADAPTIVE_LOAD_SHEDDING_MAPPING_MULT))


def effective_under_update_steps():
    base = int(UNDER_ROUTE_COMMIT_UPDATE_STEPS if committed_route_heavy_throttle_active() else UNDER_SURFACE_UPDATE_STEPS)
    return max(1, adaptive_cadence(base, ADAPTIVE_LOAD_SHEDDING_MAPPING_MULT))


def effective_structural_update_steps():
    base = int(STRUCTURAL_ROUTE_COMMIT_UPDATE_STEPS if committed_route_heavy_throttle_active() else CV_MAP_UPDATE_STEPS)
    return max(1, adaptive_cadence(base, ADAPTIVE_LOAD_SHEDDING_MAPPING_MULT))


def effective_window_update_steps():
    base = int(WINDOW_ROUTE_COMMIT_UPDATE_STEPS if committed_route_heavy_throttle_active() else WINDOW_UPDATE_STEPS)
    return max(1, adaptive_cadence(base, ADAPTIVE_LOAD_SHEDDING_WINDOW_MULT))


def effective_planner_update_steps(base_steps):
    return max(1, adaptive_cadence(base_steps, ADAPTIVE_LOAD_SHEDDING_PLANNER_MULT))


def frontier_direct_mapping_mode_active():
    """True while global frontier planning should be an occasional advisor.

    In this mode the robot keeps deterministic forward/perimeter motion and the
    expensive frontier selector is sampled by cadence or after meaningful travel.
    This avoids the active=none/frontier-hold loop: a displayed frontier candidate
    is not allowed to become the wheel owner until ROUTE_COMMIT accepts it.
    """
    if not FRONTIER_DIRECT_MAPPING_ENABLED:
        return False
    if route_commit_active or dock_return_active or dock_return_completed:
        return False
    if known_map_coverage_eval_active() or map_mature:
        return False
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return False
    try:
        cov = float(last_coverage_percent or 0.0)
    except Exception:
        cov = 100.0
    if cov >= float(FRONTIER_DIRECT_MAPPING_MAX_COVERAGE_PERCENT):
        return False
    try:
        has_frontier_candidate = bool(
            coverage_goal_kind == "frontier"
            and coverage_route_kind == "frontier"
            and coverage_goal_map is not None
            and math.isfinite(float(coverage_route_cost))
        )
    except Exception:
        has_frontier_candidate = False
    return bool(has_frontier_candidate or int(last_frontier_cells or 0) >= int(EXPLORATION_FRONTIER_ONLY_MIN_FRONTIER_CELLS))


def hybrid_local_mapping_active(cov=None):
    """True while local movement should be the default map-building owner.

    moved better because frontier did not become a hard gate on a sparse
    map.  Keep that behavior explicitly: until the coverage threshold is reached,
    a non-committed frontier candidate is only an advisor.  ROUTE_COMMIT still has
    priority because route_commit_speeds() runs before the owner gate.
    """
    if not HYBRID_LOCAL_MOTION_RESTORE_ENABLED:
        return False
    if route_commit_active or dock_return_active or dock_return_completed:
        return False
    if known_map_coverage_eval_active() or map_mature:
        return False
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return False
    try:
        c = float(last_coverage_percent if cov is None else cov)
    except Exception:
        c = 100.0
    return bool(matrix_first_explore_active() and c < float(EXPLORATION_FRONTIER_ONLY_MIN_COVERAGE_PERCENT))

def frontier_direct_mapping_safe(front, center, body_clearance):
    if not frontier_direct_mapping_mode_active():
        return False
    try:
        return bool(
            front is not None
            and center is not None
            and body_clearance is not None
            and float(front) >= float(FRONTIER_DIRECT_MAPPING_MIN_FRONT_CLEAR_M)
            and float(center) >= float(FRONTIER_DIRECT_MAPPING_MIN_CENTER_CLEAR_M)
            and float(body_clearance) >= float(FRONTIER_DIRECT_MAPPING_MIN_BODY_CLEAR_M)
        )
    except Exception:
        return False


def frontier_direct_mapping_unsafe(front, center, body_clearance):
    """True when direct mapping is active but local RGB-D says: do not drive forward."""
    if not frontier_direct_mapping_mode_active():
        return False
    try:
        f = float(front) if front is not None else 0.0
        c = float(center) if center is not None else 0.0
        b = float(body_clearance) if body_clearance is not None else 0.0
        return bool(
            f < float(FRONTIER_DIRECT_UNSAFE_RECOVERY_FRONT_M)
            or c < float(FRONTIER_DIRECT_UNSAFE_RECOVERY_FRONT_M)
            or b < float(FRONTIER_DIRECT_UNSAFE_RECOVERY_BODY_M)
        )
    except Exception:
        return False


def start_frontier_direct_unsafe_recovery(front, center, left, right, body_clearance, route_reason):
    """Recover from a route-less frontier/direct hold near an obstacle/wall.

    This is not a random fallback.  It is a controlled recovery state used only
    when the frontier/direct mapper has no committable route and local depth says
    that ordinary forward motion is unsafe.  It blacklists the current frontier
    viewpoint before turning so the planner does not select the same shadow point
    again.
    """
    global frontier_direct_last_unsafe_recovery_at, frontier_direct_last_unsafe_recovery_debug
    global frontier_only_stall_blacklists, frontier_only_last_blacklist_time, last_planner_update_step
    global coverage_status, last_optional_block_reason
    if not FRONTIER_DIRECT_UNSAFE_RECOVERY_ENABLED:
        frontier_direct_last_unsafe_recovery_debug = "unsafeRecovery=off"
        return False
    if nav_state != NAV_FORWARD or route_commit_active or dock_return_active or dock_return_completed:
        frontier_direct_last_unsafe_recovery_debug = f"unsafeRecovery=skip nav={nav_state}"
        return False
    if not frontier_direct_mapping_unsafe(front, center, body_clearance):
        frontier_direct_last_unsafe_recovery_debug = "unsafeRecovery=not-unsafe"
        return False
    try:
        now = float(robot.getTime())
    except Exception:
        now = 0.0
    if now - float(frontier_direct_last_unsafe_recovery_at) < float(FRONTIER_DIRECT_UNSAFE_RECOVERY_COOLDOWN_SEC):
        frontier_direct_last_unsafe_recovery_debug = f"unsafeRecovery=cooldown {now - float(frontier_direct_last_unsafe_recovery_at):.1f}s"
        return False

    try:
        if coverage_goal_kind == "frontier" and coverage_goal_map is not None:
            gx, gy = coverage_goal_map
            register_map_target_blacklist(
                int(gx), int(gy), "frontier",
                f"frontier unsafe hold {str(route_reason)[:24]}",
                ttl_sec=EXPLORATION_FRONTIER_ONLY_STALL_BLACKLIST_SEC,
                radius_m=FRONTIER_DIRECT_UNSAFE_BLACKLIST_RADIUS_M,
            )
            frontier_only_stall_blacklists += 1
            frontier_only_last_blacklist_time = now
    except Exception:
        pass
    try:
        maybe_mark_near_collision_hypothesis("frontier unsafe hold: " + str(route_reason)[:24])
    except Exception:
        pass

    try:
        side, side_reason = choose_explore_contact_turn_side(False, False, float(left), float(right))
    except Exception:
        try:
            side = choose_explore_turn_side_by_depth(float(left), float(right))
            side_reason = "depth fallback"
        except Exception:
            side, side_reason = 1.0, "default left"

    frontier_direct_last_unsafe_recovery_at = now
    frontier_direct_last_unsafe_recovery_debug = (
        f"unsafeRecovery=start side={'L' if side > 0 else 'R'} "
        f"F={float(front) if front is not None else -1.0:.2f} "
        f"C={float(center) if center is not None else -1.0:.2f} "
        f"B={float(body_clearance) if body_clearance is not None else -1.0:.2f}"
    )
    coverage_status = frontier_direct_last_unsafe_recovery_debug
    last_optional_block_reason = "frontier-direct unsafe -> controlled recovery"
    last_planner_update_step = -999999
    return bool(start_contact_recovery(
        side,
        "frontier-depth",
        f"frontier-direct unsafe: {side_reason}; {str(route_reason)[:40]}",
        EXPLORE_CONTACT_BACKUP_GOAL_M,
        EXPLORE_CONTACT_BACKUP_TIMEOUT_SEC,
        PIVOT_TURN_ANGLE,
        EXPLORE_CONTACT_FORWARD_VERIFY_M,
        CONTACT_ESCAPE_VERIFY_SPEED,
        FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
        True,
    ))


def planner_update_due():
    """Run heavy coverage planning only when it can actually change motion."""
    global last_planner_update_reason
    if not PLANNER_ENABLED:
        last_planner_update_reason = "disabled"
        return False
    try:
        now = robot.getTime()
    except Exception:
        now = 0.0
    if route_commit_active and route_commit_kind != "dock" and PERF_THROTTLE_WHILE_ROUTE_COMMIT:
        if known_map_coverage_eval_active() and bool(KNOWN_MAP_EVAL_FREEZE_PLANNER_DURING_COMMIT):
            # Do not search for a new "best" coverage route while the full-sweep
            # route is already the owner of motion. We still occasionally call
            # update_coverage_objective(); in known-map freeze mode it only
            # refreshes coverage percentages and mirrors the active route for HUD.
            cadence = effective_planner_update_steps(max(1, int(KNOWN_MAP_EVAL_ACTIVE_METRICS_UPDATE_STEPS)))
            due = (step_id % cadence) == 0
            last_planner_update_reason = f"known-sweep metrics {cadence}" if due else f"known-sweep frozen {cadence}"
            return bool(due)
        cadence = effective_planner_update_steps(max(1, int(PLANNER_ROUTE_COMMIT_UPDATE_STEPS)))
        due = (step_id % cadence) == 0
        last_planner_update_reason = f"commit cadence {cadence}" if due else f"commit skip {cadence}"
        return bool(due)
    if (not route_commit_active) and route_commit_last_abort_time > -900.0 and (now - route_commit_last_abort_time) <= PLANNER_POST_ABORT_FORCE_SEC:
        if step_id - last_planner_update_step >= effective_planner_update_steps(max(8, int(PLANNER_UPDATE_STEPS // 4))):
            last_planner_update_reason = "post-abort force"
            return True

    if frontier_direct_mapping_mode_active():
        cadence = effective_planner_update_steps(max(int(PLANNER_UPDATE_STEPS), int(FRONTIER_DIRECT_MAPPING_PLANNER_STEPS)))
        try:
            moved = math.hypot(float(pose_x) - float(last_planner_update_x), float(pose_y) - float(last_planner_update_y))
        except Exception:
            moved = 0.0
        # When the robot is stopped/held near a bad frontier, do not burn CPU
        # replanning the same unsafe candidate every few seconds. Recovery and
        # blacklist are the correct owner transitions; planner can wait longer.
        if moved < float(FRONTIER_DIRECT_MAPPING_STOPPED_MOVE_EPS_M):
            cadence = max(cadence, effective_planner_update_steps(int(FRONTIER_DIRECT_MAPPING_MIN_REPLAN_STEPS_WHEN_STOPPED)))
        due = bool(
            step_id - last_planner_update_step >= cadence
            or moved >= float(FRONTIER_DIRECT_MAPPING_FORCE_REPLAN_DIST_M)
        )
        last_planner_update_reason = (
            f"frontier-direct cadence {cadence} moved={moved:.2f}"
            if due else f"frontier-direct skip {cadence} moved={moved:.2f}"
        )
        return bool(due)

    cadence = effective_planner_update_steps(max(1, int(PLANNER_UPDATE_STEPS)))
    due = (step_id % cadence) == 0
    last_planner_update_reason = f"normal {cadence}" if due else f"normal skip {cadence}"
    return bool(due)


def task_due(cadence, force=False):
    return bool(force or (step_id % max(1, int(cadence))) == 0)


def profiled_call(name, fn, *args, **kwargs):
    t0 = perf_start()
    result = fn(*args, **kwargs)
    perf_end(name, t0)
    return result


def snapshot_mapping_forced():
    return bool(
        (POST_TURN_RGBD_SNAPSHOT_FORCE_EVERY_TICK and nav_state == NAV_RGBD_SNAPSHOT)
        or (ACTIVE_SCAN_FORCE_MAP_EVERY_TICK and active_scan_mapping_dwell_ready(robot.getTime()))
    )


def mapping_update_cadences():
    return {
        "depth": effective_depth_map_update_steps(),
        "cv": effective_cv_map_update_steps(),
        "under": effective_under_update_steps(),
        "struct": effective_structural_update_steps(),
        "noise": effective_planner_update_steps(max(1, int(RAW_MAP_SPECKLE_CLEANUP_STEPS))),
    }


def run_mapping_stage(pose, frame, depth):
    """Update expensive perception/map layers through one scheduler gate."""
    global last_depth_map_hits
    if known_map_coverage_eval_active():
        update_mapping_write_debug(0, 0, False)
        return 0, 0, 0

    cadence = mapping_update_cadences()
    force_snapshot_map = snapshot_mapping_forced()
    if task_due(cadence["depth"], force_snapshot_map):
        last_depth_map_hits = profiled_call("depthMap", update_map_from_depth, pose, depth)
    hit_count = last_depth_map_hits

    if task_due(cadence["cv"], force_snapshot_map):
        cv_hit_count = profiled_call("cv", update_visual_map_from_rgb_depth, pose, frame, depth)
    else:
        cv_hit_count = last_cv_map_hits

    if nav_state == NAV_RGBD_SNAPSHOT:
        note_post_turn_rgbd_snapshot_mapping(last_depth_map_hits, cv_hit_count)
    update_mapping_write_debug(last_depth_map_hits, cv_hit_count, force_snapshot_map)

    if task_due(cadence["under"]):
        under_marked = profiled_call("under", update_under_surface_from_depth, depth)
    else:
        under_marked = last_under_surface_marked

    if STRUCTURAL_OBSTACLE_MEMORY_ENABLED and task_due(cadence["struct"]):
        profiled_call("struct", update_structural_obstacle_memory)

    if RAW_MAP_SPECKLE_CLEANUP_ENABLED and task_due(cadence["noise"]):
        profiled_call("noise", cleanup_raw_map_speckles)

    return hit_count, cv_hit_count, under_marked


def append_current_trajectory_point():
    mx, my = world_to_map(pose_x, pose_y)
    if map_inside(mx, my):
        if not trajectory or abs(mx - trajectory[-1][0]) + abs(my - trajectory[-1][1]) > 2:
            trajectory.append((mx, my))
            if len(trajectory) > 5000:
                trajectory.pop(0)


def run_planner_stage():
    global last_planner_update_step, last_planner_update_x, last_planner_update_y
    refresh_navigation_phase()
    if planner_update_due():
        update_coverage_objective()
        last_planner_update_step = int(step_id)
        try:
            last_planner_update_x = float(pose_x)
            last_planner_update_y = float(pose_y)
        except Exception:
            pass
        refresh_navigation_phase()


def run_motion_stage(depth):
    t0 = perf_start()
    lv, rv = choose_motion_from_depth(depth)
    set_wheel_speeds(lv, rv)
    perf_end("motion", t0)


def save_frame_if_due(frame):
    if SAVE_FRAMES and task_due(SAVE_EVERY_N_STEPS):
        cv2.imwrite(str(frames_dir / f"frame_{step_id:06d}.png"), frame)


def heavy_debug_render_due():
    """Throttle heavy map rendering separately from the viewer process.

    moved imshow/waitKey away from the controller, but render_coverage_
    planner_map() still runs in main and calls compute_coverage_masks().  This
    function keeps debug images human-readable without treating them as real-time
    control work.
    """
    if not FAST_DEBUG_RENDER_ENABLED:
        return True
    try:
        base = int(FAST_DEBUG_RENDER_ROUTE_STEPS if route_commit_active else FAST_DEBUG_RENDER_MIN_STEPS)
    except Exception:
        base = int(FAST_DEBUG_RENDER_MIN_STEPS)
    cadence = effective_planner_update_steps(max(base, effective_window_update_steps()))
    return bool((step_id % cadence) == 0)


def show_map_windows(frame, depth):
    global last_debug_viewer_status, last_render_throttle_debug
    if task_due(effective_window_update_steps()):
        t0 = perf_start()
        rendered_windows = []
        allow_heavy_render = heavy_debug_render_due()
        if allow_heavy_render:
            if SHOW_RGB_DEBUG_WINDOW:
                rendered_windows.append(("RGB camera + depth debug", draw_cv_debug(frame, depth), (480, 360)))
            if SHOW_OCCUPANCY_MAP_WINDOW:
                rendered_windows.append(("Persistent occupancy map", render_map(auto_crop=True), (MAP_VIEW_W, MAP_VIEW_H)))
            if SHOW_COVERAGE_PLANNER_WINDOW:
                rendered_windows.append(("Coverage objective map", render_coverage_planner_map(auto_crop=True), (MAP_VIEW_W, MAP_VIEW_H + 132)))
            last_render_throttle_debug = f"render=draw step={step_id}"
        else:
            last_render_throttle_debug = f"render=skip step={step_id}"
        if ASYNC_DEBUG_VIEWER_ENABLED and debug_viewer_client is not None:
            if rendered_windows and debug_viewer_client.submit(rendered_windows):
                last_debug_viewer_status = f"viewer=async queued={len(rendered_windows)}"
            elif rendered_windows:
                last_debug_viewer_status = "viewer=async drop"
            else:
                last_debug_viewer_status = "viewer=async idle"
        else:
            for name, image, _size in rendered_windows:
                cv2.imshow(name, image)
            last_debug_viewer_status = f"viewer=sync shown={len(rendered_windows)}" if rendered_windows else "viewer=sync idle"
        perf_end("render", t0)

def set_map_zoom(factor):
    global map_view_zoom
    map_view_zoom = min(MAP_VIEW_ZOOM_MAX, max(MAP_VIEW_ZOOM_MIN, map_view_zoom * float(factor)))
    print(f"Map view zoom: {map_view_zoom:.2f}x")


def pan_map_view(dx, dy):
    global map_view_auto_crop, map_view_pan_x, map_view_pan_y
    map_view_auto_crop = True
    scale = MAP_VIEW_PAN_STEP_PX / max(0.5, map_view_zoom)
    map_view_pan_x += float(dx) * scale
    map_view_pan_y += float(dy) * scale
    print(f"Map view pan: {map_view_pan_x:.0f},{map_view_pan_y:.0f}")


def toggle_map_auto_crop():
    global map_view_auto_crop
    map_view_auto_crop = not map_view_auto_crop
    print(f"Map view: {'auto-crop' if map_view_auto_crop else 'full map'}")


def toggle_map_follow_robot():
    global map_view_auto_crop, map_view_follow_robot
    map_view_follow_robot = not map_view_follow_robot
    map_view_auto_crop = True
    print(f"Map view follow robot: {map_view_follow_robot}")


def reset_map_view():
    global map_view_zoom, map_view_auto_crop, map_view_follow_robot, map_view_pan_x, map_view_pan_y
    map_view_zoom = 1.0
    map_view_auto_crop = True
    map_view_follow_robot = False
    map_view_pan_x = 0.0
    map_view_pan_y = 0.0
    print("Map view reset: auto-crop zoom 1.00x pan=0 follow=0")


def close_map_windows():
    global SHOW_WINDOWS, debug_viewer_client, last_debug_viewer_status
    if debug_viewer_client is not None:
        try:
            debug_viewer_client.close()
        except Exception:
            pass
    else:
        cv2.destroyAllWindows()
    SHOW_WINDOWS = False
    last_debug_viewer_status = "viewer=closed"

def handle_window_key(key):
    if key == 255:
        return
    if key in (ord('k'), ord('K')) and KNOWN_MAP_EVAL_KEY_TOGGLE_ENABLED:
        toggle_known_map_coverage_eval()
        return
    key_actions = {
        ord('s'): save_map,
        ord('S'): save_map,
        ord('r'): reset_map,
        ord('R'): reset_map,
        ord('+'): lambda: set_map_zoom(MAP_VIEW_ZOOM_STEP),
        ord('='): lambda: set_map_zoom(MAP_VIEW_ZOOM_STEP),
        ord('-'): lambda: set_map_zoom(1.0 / MAP_VIEW_ZOOM_STEP),
        ord('_'): lambda: set_map_zoom(1.0 / MAP_VIEW_ZOOM_STEP),
        ord('c'): toggle_map_auto_crop,
        ord('C'): toggle_map_auto_crop,
        ord('f'): toggle_map_follow_robot,
        ord('F'): toggle_map_follow_robot,
        ord('j'): lambda: pan_map_view(-1.0, 0.0),
        ord('J'): lambda: pan_map_view(-1.0, 0.0),
        ord('l'): lambda: pan_map_view(1.0, 0.0),
        ord('L'): lambda: pan_map_view(1.0, 0.0),
        ord('i'): lambda: pan_map_view(0.0, -1.0),
        ord('I'): lambda: pan_map_view(0.0, -1.0),
        ord('k'): lambda: pan_map_view(0.0, 1.0),
        ord('K'): lambda: pan_map_view(0.0, 1.0),
        ord('0'): reset_map_view,
        ord('q'): close_map_windows,
        27: close_map_windows,
    }
    action = key_actions.get(key)
    if action is not None:
        action()


def run_window_stage(frame, depth):
    if not SHOW_WINDOWS:
        return
    show_map_windows(frame, depth)
    if ASYNC_DEBUG_VIEWER_ENABLED and debug_viewer_client is not None:
        for key in debug_viewer_client.poll_keys():
            handle_window_key(key)
    else:
        handle_window_key(cv2.waitKey(1) & 0xFF)

def current_owner_source_label():
    """Human-readable source of the command that currently owns the wheels."""
    try:
        owner = str(control_lock.owner)
    except Exception:
        owner = ControlOwner.NONE.value
    try:
        if dock_return_active or route_commit_kind == "dock":
            return "dock"
        if auto_map_cleaning_started or known_map_coverage_eval_active():
            return "learned-k"
        if route_commit_active and route_commit_kind == "frontier":
            return "frontier-route"
        if nav_state in CONTACT_RECOVERY_STATES or str(owner).startswith("CONTACT") or str(owner).startswith("RECOVERY"):
            return "recovery"
        if nav_state in (NAV_RGBD_SNAPSHOT, NAV_SCAN_AROUND):
            return "frontier-hold"
        if planner_intent == PLANNER_INTENT_EXPAND_MAP:
            try:
                cov = float(last_coverage_percent or 0.0)
            except Exception:
                cov = 100.0
            if str(owner) == ControlOwner.ROUTE_COMMIT.value and coverage_route_kind == "frontier":
                return "frontier-route"
            if str(owner) in (ControlOwner.NONE.value, ControlOwner.PLANNER.value):
                if str(last_frontier_only_gate_debug).startswith("frontierGate=hold") or str(last_frontier_only_gate_debug).startswith("frontierGate=waitRoute"):
                    return "frontier-hold"
                if str(last_frontier_only_gate_debug).startswith("frontierGate=unsafeRecovery"):
                    return "recovery"
                if hybrid_local_mapping_active(cov):
                    return "hybrid-row"
                if frontier_direct_mapping_mode_active() and str(last_frontier_only_gate_debug).startswith("frontierGate=direct"):
                    return "frontier-direct"
            if hybrid_local_mapping_active(cov) and str(owner) == ControlOwner.ROW_FORWARD.value:
                return "hybrid-row"
            if frontier_direct_mapping_mode_active() and str(owner) == ControlOwner.ROW_FORWARD.value:
                return "frontier-direct"
            if cov < float(EXPLORATION_FRONTIER_ONLY_MIN_COVERAGE_PERCENT) and str(owner) in (ControlOwner.NONE.value, ControlOwner.PLANNER.value, ControlOwner.ROW_FORWARD.value):
                return "hybrid-row"
            if str(owner) in (ControlOwner.NONE.value, ControlOwner.PLANNER.value):
                return "frontier-hold"
            if str(owner) in (ControlOwner.ROW_FORWARD.value, ControlOwner.GRID_REALIGN.value, ControlOwner.PIVOT_90.value):
                return "fallback"
        if str(owner) == ControlOwner.ROUTE_COMMIT.value:
            return "route"
        if str(owner) == ControlOwner.NONE.value:
            return "none"
        return "fallback"
    except Exception:
        return "unknown"


def owner_source_debug():
    global last_owner_source_debug
    last_owner_source_debug = "ownerSource=" + current_owner_source_label()
    return last_owner_source_debug

def compact_debug_status_line():
    return (
        f"t={robot.getTime():.1f}s pose=({pose_x:.2f},{pose_y:.2f},{math.degrees(pose_theta):.0f}) "
        f"nav={nav_state}/{last_motion_primitive} owner={last_control_owner_debug[:20]} {owner_source_debug()[:26]} "
        f"cov={last_coverage_percent:.1f}% conf={planner_confidence} intent={planner_intent} mode={planner_mode} {known_map_eval_status[:18]} {auto_map_mission_phase[:14]} "
        f"cand={route_commit_candidate_debug()} active={route_commit_active_target_debug()} "
        f"frontier={last_frontier_cells} {last_gray_gap_debug[:26]} {last_exploration_cleanup_lock_debug[:34]} "
        f"{runtime_arena_guard_debug[:24]} {odom_arena_clamp_debug[:22]} {frontier_route_abort_hold_debug[:24]} "
        f"{last_hypothesis_obstacle_debug[:28]} {last_near_collision_hypothesis_debug[:22]} "
        f"plan={last_planning_layer_debug[:34]} "
        f"scan={last_active_scan_debug[:28]} gate={last_frontier_only_gate_debug[:30]} occ={last_rgbd_occlusion_debug[:18]} {last_debug_viewer_status[:22]} {last_render_throttle_debug[:18]} {load_shedding_debug_text()[:24]}"
    )


def verbose_debug_status_line():
    return (
        f"pose=({pose_x:.2f},{pose_y:.2f},{math.degrees(pose_theta):.1f}deg), "
        f"phase={navigation_phase}, nav={nav_state}, prim={last_motion_primitive}, "
        f"owner={last_control_owner_debug[:36]}, {owner_source_debug()}, opt={last_optional_block_reason[:28]}, "
        f"desired={math.degrees(desired_grid_heading):.0f}, coverage={coverage_status}, "
        f"depth L/F/U/C/R={last_min_left:.2f}/{last_front_narrow:.2f}/{last_front_upper:.2f}/{last_min_center:.2f}/{last_min_right:.2f}, "
        f"body={last_body_corridor_clearance:.2f}@{last_body_corridor_lateral:.2f}, "
        f"coverage={last_coverage_percent:.1f}%, conf={planner_confidence}, intent={planner_intent}, "
        f"plannerMode={planner_mode}, {known_map_eval_status}, candidate={route_commit_candidate_debug()}, "
        f"active={route_commit_active_target_debug()}, route={coverage_route_kind}/{coverage_route_len}/{coverage_route_cost:.2f}/score={coverage_route_score:.1f}, "
        f"top={coverage_route_top_debug[:80]}, frontier={last_frontier_target_debug[:44]}, "
        f"{last_exploration_cleanup_lock_debug}, plan={last_planning_layer_debug}, "
        f"scan={last_active_scan_debug}, gate={last_frontier_only_gate_debug}, "
        f"arena={runtime_arena_guard_debug}, odom={odom_arena_clamp_debug}, hold={frontier_route_abort_hold_debug}, "
        f"hyp={last_hypothesis_obstacle_debug}, near={last_near_collision_hypothesis_debug}, "
        f"occ={last_rgbd_occlusion_debug}, {last_debug_viewer_status}, {last_render_throttle_debug}, {load_shedding_debug_text()}, {last_perf_debug}"
    )


def print_debug_status_if_due():
    global last_debug_time
    if time.time() - last_debug_time <= DEBUG_PRINT_INTERVAL_SEC:
        return
    print(verbose_debug_status_line() if DEBUG_VERBOSE_CONSOLE else compact_debug_status_line())
    last_debug_time = time.time()


def invalidate_heavy_map_caches(reason="invalidate"):
    global objective_noise_filter_cache, objective_noise_filter_cache_step, objective_noise_filter_cache_debug
    global footprint_passability_cache, footprint_passability_cache_step, footprint_passability_cache_margin
    global footprint_value_cache, footprint_value_cache_step
    objective_noise_filter_cache = None
    objective_noise_filter_cache_step = -999999
    objective_noise_filter_cache_debug = str(reason or "invalidate")[:40]
    footprint_passability_cache = None
    footprint_passability_cache_step = -999999
    footprint_passability_cache_margin = None
    footprint_value_cache = None
    footprint_value_cache_step = -999999


def clamp(v, lo, hi):
    return _clamp(v, lo, hi)


def normalize_angle(a):
    return _normalize_angle(a)


def snap_to_right_angle(a):
    """Snap heading to the nearest 90-degree grid direction.

    The coverage pattern is intentionally grid/lawnmower-like. If a pivot
    finishes at 88 or 93 degrees and we keep that error forever, every next
    lane becomes skewed and the occupancy map looks like the robot turns
    diagonally. Snapping after a commanded pivot keeps the demo defendable.
    """
    return _snap_to_right_angle(a)


def strict_world_grid_heading(target_heading):
    """Return a cardinal room/grid heading for ordinary navigation.

    This is stronger than removing arcs from wheel speeds: if a recovery exits
    at 80 degrees, straight wheel speeds still make a diagonal track.  In normal
    EXPLORE/COVERAGE/FINISH_CLEANUP we therefore snap the *target heading* to
    0/90/180/270 and pivot in place before translating.
    """
    if not STRICT_WORLD_GRID_HEADING_ENABLED:
        return normalize_angle(target_heading)
    return snap_to_right_angle(target_heading)


def acquire_control(owner, min_time=0.0, min_distance=0.0, reason=""):
    """Atomically reserve wheel ownership for the current manoeuvre.

    This is the practical boundary between modules.  Optional planning is not
    allowed to pre-empt an active owner; only physical safety/recovery may do so.
    """
    global last_control_owner_debug
    if not HARD_CONTROL_ARBITER_ENABLED:
        return
    control_lock.acquire(
        str(owner.value if hasattr(owner, "value") else owner),
        now=robot.getTime(),
        x=pose_x,
        y=pose_y,
        min_time=min_time,
        min_distance=min_distance,
        reason=reason,
    )
    last_control_owner_debug = control_lock.debug(now=robot.getTime(), x=pose_x, y=pose_y)


def release_control(reason="released"):
    global last_control_owner_debug
    if not HARD_CONTROL_ARBITER_ENABLED:
        return
    control_lock.release(reason)
    last_control_owner_debug = control_lock.debug(now=robot.getTime(), x=pose_x, y=pose_y)


def control_lock_active():
    global last_control_owner_debug
    if not HARD_CONTROL_ARBITER_ENABLED:
        return False
    active = control_lock.active(now=robot.getTime(), x=pose_x, y=pose_y)
    last_control_owner_debug = control_lock.debug(now=robot.getTime(), x=pose_x, y=pose_y)
    return active


def policy_reason_text(reason):
    return str(reason() if callable(reason) else reason)


def policy_first_block(rules):
    for blocked, reason in rules:
        if bool(blocked()):
            return True, policy_reason_text(reason)
    return False, ""


def policy_allow(reason):
    return True, str(reason)


def policy_deny(reason):
    return False, str(reason)


def planner_expand_map_motion_active():
    """True when EXPAND_MAP should own the low-level exploration motion.

    made EXPAND_MAP the planner/source-of-target truth, but the hard-core
    controller still used the older time/coverage gate.  That created the visible
    failure where debug said intent=EXPAND_MAP while phase=COVERAGE and the robot
    began a lawnmower pivot before actually reaching the wall.  This predicate is
    deliberately stricter than a plain intent check: it applies only to non-mature
    maps with enough useful frontier support.
    """
    if not EXPAND_MAP_EXTENDS_BUMPER_FIRST_MAPPING:
        return False
    if auto_map_return_to_dock_active or auto_map_cleaning_started or dock_return_active or dock_return_completed or route_commit_kind == "dock":
        return False
    if map_mature or str(planner_confidence or "") == "MATURE":
        return False
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return False
    try:
        frontiers = int(last_frontier_cells or 0)
        cov = float(last_coverage_percent or 0.0)
    except Exception:
        return False
    if frontiers < EXPAND_MAP_MOTION_MIN_FRONTIER_CELLS:
        return False
    # Do not keep bumper-first probing forever on a nearly mature map unless the
    # frontier count is still very high.  This prevents reverting to endless
    # perimeter tracing late in the mission.
    if cov >= EXPAND_MAP_MOTION_MAX_COVERAGE_PERCENT and frontiers < PLANNER_INTENT_HIGH_COVERAGE_FRONTIER_CELLS:
        return False
    return True


def explore_simple_mode_active():
    """True while the robot should map first and ignore cleanup opportunism."""
    if not EXPLORE_SIMPLE_MODE_ENABLED:
        return False
    return map_building_active()


def optional_planner_intercept_policy(feature=""):
    if not HARD_CONTROL_ARBITER_ENABLED:
        return policy_allow("arbiter disabled")
    blocked, reason = policy_first_block([
        (lambda: nav_state != NAV_FORWARD, lambda: f"nav={nav_state} owns wheels"),
        (lambda: control_lock_active(), lambda: f"owner lock {last_control_owner_debug}"),
        (
            lambda: robot.getTime() < last_contact_route_kill_until + OPTIONAL_PLANNER_SUPPRESS_AFTER_CONTACT_SEC,
            "post-contact suppress",
        ),
        (lambda: explore_simple_mode_active(), lambda: f"EXPLORE map-first blocks {feature}"),
    ])
    return policy_deny(reason) if blocked else policy_allow(f"allowed {feature}")


def optional_planner_intercepts_allowed(feature=""):
    """Gate anything that can steal control from the current primitive.

    This blocks target chasing, line acquire, local pocket filling, and
    under-furniture opportunism during early map building and during atomic row
    locks.  It does not block safety/recovery or row-end handling.
    """
    global last_optional_block_reason
    allowed, reason = optional_planner_intercept_policy(feature)
    last_optional_block_reason = reason
    return bool(allowed)


def forward_owner_guard_speeds(front, center, body_clearance):
    """Keep an owned row straight; leave true row-end/recovery to later logic.

    The guard is intentionally conservative.  It only acts when the front is not
    a hard block.  Hard blocks, bumpers and recovery states continue through the
    normal safety path below.
    """
    if nav_state != NAV_FORWARD or not control_lock_active():
        return None
    if str(control_lock.owner) != ControlOwner.ROW_FORWARD.value:
        # A ROUTE_COMMIT lock is executed by route_commit_speeds() before this
        # guard.  Returning a straight row command here would recreate the bug
        # where a drawn wavefront target exists but ROW_FORWARD still drives on.
        return None
    if front < ROW_END_HARD_DISTANCE or center < SAFE_FRONT_DISTANCE * 0.92:
        return None
    if body_clearance < BODY_CORRIDOR_HARD_CLEARANCE:
        return None
    return heading_locked_wheel_speeds(CRUISE_SPEED if front > ROW_SLOW_DISTANCE else SLOW_SPEED)


def world_to_map(wx, wy):
    return _world_to_map_cell(wx, wy, MAP_ORIGIN_X, MAP_ORIGIN_Y, MAP_SCALE)


def map_inside(mx, my):
    return _map_inside_cell(mx, my, MAP_SIZE)


def debug_arena_bounds_map_rect(pad_m=0.0):
    """Return current Webots arena bounds in controller-map coordinates.

    This is strictly a visualization helper. The navigation stack must not use
    the known RectangleArena boundary as a prior, otherwise the diploma prototype
    would be using a pre-known map instead of RGB-D exploration.
    """
    if not DEBUG_ARENA_BOUNDS_ENABLED:
        return None
    x0, y0 = world_to_map(DEBUG_ARENA_X_MIN_M - pad_m, DEBUG_ARENA_Y_MAX_M + pad_m)
    x1, y1 = world_to_map(DEBUG_ARENA_X_MAX_M + pad_m, DEBUG_ARENA_Y_MIN_M - pad_m)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def draw_debug_arena_bounds(img):
    """Draw a subtle debug-only expected arena outline, not an obstacle layer."""
    rect = debug_arena_bounds_map_rect(0.0)
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    # Clip only for drawing. The crop helper may still use the unclipped bounds
    # with padding so the viewport can include the lower unknown area.
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(MAP_SIZE - 1, x1), min(MAP_SIZE - 1, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    # Dashed border so it cannot be mistaken for an observed black wall.
    dash = 18
    gap = 10
    for x in range(cx0, cx1, dash + gap):
        cv2.line(img, (x, cy0), (min(cx1, x + dash), cy0), DEBUG_ARENA_BOUNDS_COLOR, 1)
        cv2.line(img, (x, cy1), (min(cx1, x + dash), cy1), DEBUG_ARENA_BOUNDS_COLOR, 1)
    for y in range(cy0, cy1, dash + gap):
        cv2.line(img, (cx0, y), (cx0, min(cy1, y + dash)), DEBUG_ARENA_BOUNDS_COLOR, 1)
        cv2.line(img, (cx1, y), (cx1, min(cy1, y + dash)), DEBUG_ARENA_BOUNDS_COLOR, 1)
    cv2.putText(img, "arena bounds dbg", (cx0 + 8, max(18, cy0 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, DEBUG_ARENA_BOUNDS_COLOR, 1, cv2.LINE_AA)


def pose_relative_to_debug_arena(x=None, y=None, probe_heading=None, probe_m=0.0, margin_m=0.0):
    """Return (outside, reason, px, py) relative to the expected Webots room.

    This is a safety envelope, not a source of map truth.  It prevents a failed
    frontier route from falling back to straight driving beyond the finite scene.
    """
    if x is None:
        x = pose_x
    if y is None:
        y = pose_y
    px = float(x)
    py = float(y)
    if probe_heading is not None and probe_m > 0.0:
        px += math.cos(float(probe_heading)) * float(probe_m)
        py += math.sin(float(probe_heading)) * float(probe_m)
    mnx = DEBUG_ARENA_X_MIN_M + float(margin_m)
    mxx = DEBUG_ARENA_X_MAX_M - float(margin_m)
    mny = DEBUG_ARENA_Y_MIN_M + float(margin_m)
    mxy = DEBUG_ARENA_Y_MAX_M - float(margin_m)
    if px < mnx:
        return True, f"arena-x min px={px:.2f} min={mnx:.2f}", px, py
    if px > mxx:
        return True, f"arena-x max px={px:.2f} max={mxx:.2f}", px, py
    if py < mny:
        return True, f"arena-y min py={py:.2f} min={mny:.2f}", px, py
    if py > mxy:
        return True, f"arena-y max py={py:.2f} max={mxy:.2f}", px, py
    return False, "inside", px, py


def heading_to_arena_center_from_pose():
    cx = 0.5 * (DEBUG_ARENA_X_MIN_M + DEBUG_ARENA_X_MAX_M)
    cy = 0.5 * (DEBUG_ARENA_Y_MIN_M + DEBUG_ARENA_Y_MAX_M)
    return math.atan2(cy - pose_y, cx - pose_x)


def clamp_pose_to_debug_arena_if_needed():
    """Clamp impossible wheel-odometry drift outside the expected arena.

    The real Webots body cannot travel to y=-50 m in this room; that state is
    wheel-slip odometry after the controller kept commanding FORWARD against the
    lower boundary.  Clamping keeps map projection and planner windows finite so
    the recovery logic can continue instead of losing map_inside().
    """
    global pose_x, pose_y, current_linear_velocity, odom_arena_guard_until
    global odom_arena_clamp_count, odom_arena_clamp_debug
    if not (ODOM_ARENA_CLAMP_ENABLED and DEBUG_ARENA_BOUNDS_ENABLED):
        odom_arena_clamp_debug = "arenaClamp=off"
        return False
    margin = float(ODOM_ARENA_CLAMP_MARGIN_M)
    min_x = DEBUG_ARENA_X_MIN_M - margin
    max_x = DEBUG_ARENA_X_MAX_M + margin
    min_y = DEBUG_ARENA_Y_MIN_M - margin
    max_y = DEBUG_ARENA_Y_MAX_M + margin
    cx = clamp(float(pose_x), min_x, max_x)
    cy = clamp(float(pose_y), min_y, max_y)
    if abs(cx - pose_x) > 1e-6 or abs(cy - pose_y) > 1e-6:
        pose_x = cx
        pose_y = cy
        current_linear_velocity = 0.0
        odom_arena_guard_until = max(odom_arena_guard_until, robot.getTime() + float(ODOM_ARENA_CLAMP_HOLD_SEC))
        odom_arena_clamp_count += 1
        odom_arena_clamp_debug = f"arenaClamp=hit#{odom_arena_clamp_count} pose=({pose_x:.2f},{pose_y:.2f})"
        return True
    odom_arena_clamp_debug = f"arenaClamp=ok#{odom_arena_clamp_count}"
    return False

def append_debug_arena_crop_point(crop_points):
    """Keep debug auto-crop from hiding the lower part of the known arena."""
    rect = debug_arena_bounds_map_rect(DEBUG_ARENA_CROP_MARGIN_M)
    if rect is not None:
        crop_points.append(rect)


def side_wall_distance_from_map(side, heading, max_m=0.58):
    """Return centre-to-wall distance from the occupancy/contact map.

    The front RGB-D RangeFinder cannot see a perfectly parallel side wall very
    well.  Once a wall has been touched/mapped, the occupancy matrix can provide
    a weak side-distance estimate.  This is used only by the local wall-follow
    controller; it does not create a waypoint route or stop the row.
    """
    side = 1.0 if side >= 0 else -1.0
    nx = -math.sin(heading) * side
    ny = math.cos(heading) * side
    step_m = 0.018
    start_m = max(0.08, ROBOT_BODY_RADIUS * 0.72)
    d = start_m
    while d <= max_m:
        wx = pose_x + nx * d
        wy = pose_y + ny * d
        mx, my = world_to_map(wx, wy)
        if not map_inside(mx, my):
            return d
        try:
            if (
                log_odds[my, mx] > LO_OCCUPIED_EPS
                or contact_log_odds[my, mx] > CONTACT_OCCUPIED_EPS
                or visual_log_odds[my, mx] > CV_DISPLAY_DENSE_EPS
            ):
                return d
        except Exception:
            return 999.0
        d += step_m
    return 999.0


def map_building_wall_follow_speeds(front, center, upper_front, left, right, body_clearance, base_speed):
    """Small wall-follow correction used only during bumper-led map building.

    Priority is intentionally lower than bumper/recovery and higher than plain
    ROW_FORWARD.  The controller keeps a nearby wall at a small distance; it does
    not decide that a row ended and it does not choose a global target.
    """
    global wall_follow_active_side, wall_follow_last_time, wall_follow_last_reason
    global coverage_status, last_optional_block_reason, last_turn_variant
    if not (MAP_BUILD_WALL_FOLLOW_ENABLED and EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active()):
        wall_follow_active_side = 0.0
        return None
    if nav_state != NAV_FORWARD:
        wall_follow_active_side = 0.0
        return None
    bumper_active = bool(last_bumper_left or last_bumper_center or last_bumper_right)
    if bumper_active:
        wall_follow_active_side = 0.0
        return None

    # Use the actual body heading for side clearance.  The old code used the
    # desired grid heading; when the robot was already yawed into a wall the
    # side ray no longer represented the physical shell clearance and the
    # controller kept steering into the wall.
    left_map = side_wall_distance_from_map(1.0, pose_theta)
    right_map = side_wall_distance_from_map(-1.0, pose_theta)
    preferred_active_side = wall_follow_active_side
    if abs(preferred_active_side) < 0.5 and BOUNDARY_TRACE_PERIMETER_FIRST_ENABLED:
        preferred_active_side = BOUNDARY_TRACE_FOLLOW_SIDE

    # side only.  A chair/table leg that appears on the opposite side is not a new
    # boundary to acquire; otherwise the controller flips side=L/R and keeps issuing
    # map-only scrape pivots while nav_state remains FORWARD.  Do not hide true
    # emergency side evidence; that still belongs to safety/recovery.
    left_for_wall = float(left)
    right_for_wall = float(right)
    left_map_for_wall = float(left_map)
    right_map_for_wall = float(right_map)
    opposite_suppressed = "none"
    if BOUNDARY_TRACE_STRICT_SIDE_LOCK_ENABLED and BOUNDARY_TRACE_PERIMETER_FIRST_ENABLED and abs(preferred_active_side) > 0.5:
        if preferred_active_side < 0.0:
            left_emergency = bool(left_map < BOUNDARY_TRACE_OPPOSITE_EMERGENCY_CENTER_M or left < BOUNDARY_TRACE_OPPOSITE_EMERGENCY_DEPTH_M)
            if not left_emergency:
                left_for_wall = 999.0
                left_map_for_wall = 999.0
                opposite_suppressed = "L"
        else:
            right_emergency = bool(right_map < BOUNDARY_TRACE_OPPOSITE_EMERGENCY_CENTER_M or right < BOUNDARY_TRACE_OPPOSITE_EMERGENCY_DEPTH_M)
            if not right_emergency:
                right_for_wall = 999.0
                right_map_for_wall = 999.0
                opposite_suppressed = "R"

    cmd = compute_wall_follow_command(
        WallFollowInput(
            front_m=front,
            center_m=center,
            upper_front_m=upper_front,
            left_depth_m=left_for_wall,
            right_depth_m=right_for_wall,
            body_clearance_m=body_clearance,
            body_lateral_m=last_body_corridor_lateral,
            left_map_m=left_map_for_wall,
            right_map_m=right_map_for_wall,
            current_heading_rad=pose_theta,
            target_heading_rad=desired_grid_heading,
            base_speed=base_speed,
            active_side=preferred_active_side,
            bumper_active=bumper_active,
        ),
        MAP_BUILD_WALL_FOLLOW_CFG,
    )
    if cmd is None:
        # Do not instantly forget an acquired side; brief RGB-D/map flicker near a
        # wall should not switch the robot back to blind row-forward.
        if robot.getTime() - wall_follow_last_time > 0.7:
            wall_follow_active_side = 0.0
        return None

    wall_follow_active_side = cmd.side
    wall_follow_last_time = robot.getTime()
    wall_follow_last_reason = cmd.reason
    last_optional_block_reason = "wall-follow owns local parallel correction"
    last_turn_variant = "wall-follow"
    coverage_status = (
        f"wall-follow {cmd.reason} mapL/R={left_map:.2f}/{right_map:.2f} "
        f"depthL/R={left:.2f}/{right:.2f} oppSup={opposite_suppressed}"
    )
    return cmd.left_speed, cmd.right_speed


def rear_backup_clearance(max_check_m=0.35):
    """Estimate free distance behind the robot from the remembered map.

    There is no rear RangeFinder/camera in this prototype.  A blind reverse
    escape can therefore drive the back of the circular body into a wall.  This
    guard uses only already-built occupancy/contact evidence behind the rear
    shell.  Unknown space is not treated as an obstacle; confirmed black/CV/contact
    cells and map boundaries are.
    """
    global last_rear_guard_clearance, last_rear_guard_reason
    if not REVERSE_BACKUP_GUARD_ENABLED:
        last_rear_guard_clearance = float(max_check_m)
        last_rear_guard_reason = "disabled"
        return float(max_check_m)

    max_check_m = clamp(float(max_check_m), REVERSE_BACKUP_CHECK_STEP_M, REVERSE_BACKUP_LOOKAHEAD_CAP_M)
    c = math.cos(pose_theta)
    st = math.sin(pose_theta)
    lateral_offsets = (-ROBOT_BODY_RADIUS * REVERSE_BACKUP_SIDE_FRAC, 0.0, ROBOT_BODY_RADIUS * REVERSE_BACKUP_SIDE_FRAC)
    d = REVERSE_BACKUP_CHECK_STEP_M
    last_clear = 0.0
    while d <= max_check_m + 1e-6:
        shell_d = ROBOT_BODY_RADIUS + REVERSE_BACKUP_SHELL_MARGIN_M + d
        for lat in lateral_offsets:
            wx = pose_x - c * shell_d - st * lat
            wy = pose_y - st * shell_d + c * lat
            mx, my = world_to_map(wx, wy)
            if not map_inside(mx, my):
                last_rear_guard_clearance = max(0.0, last_clear)
                last_rear_guard_reason = "map-edge"
                return last_rear_guard_clearance
            blocked = (
                log_odds[my, mx] > LO_OCCUPIED_EPS
                or visual_log_odds[my, mx] > CV_DISPLAY_DENSE_EPS
                or contact_log_odds[my, mx] > CONTACT_OCCUPIED_EPS
            )
            if blocked:
                last_rear_guard_clearance = max(0.0, last_clear)
                last_rear_guard_reason = "rear-occupied"
                return last_rear_guard_clearance
        last_clear = d
        d += REVERSE_BACKUP_CHECK_STEP_M
    last_rear_guard_clearance = float(max_check_m)
    last_rear_guard_reason = "clear"
    return last_rear_guard_clearance


def reverse_backup_should_stop(remaining_m):
    """Return True when a reverse segment should stop before hitting the rear."""
    if not REVERSE_BACKUP_GUARD_ENABLED:
        return False
    check_m = min(REVERSE_BACKUP_LOOKAHEAD_CAP_M, max(REVERSE_BACKUP_STOP_CLEARANCE_M + 0.02, float(remaining_m) + 0.06))
    clearance = rear_backup_clearance(check_m)
    return clearance <= REVERSE_BACKUP_STOP_CLEARANCE_M


def read_camera_frame():
    return decode_camera_frame(camera.getImage(), CAM_H, CAM_W)


def read_depth_image():
    # Webots may return inf for no hit. Keep inf but mask later.
    return decode_depth_image(range_finder.getRangeImage(), RF_H, RF_W)


def read_imu_heading():
    """Return yaw relative to the robot start heading, or None if no IMU.

    Webots InertialUnit gives the actual body orientation. We do not use it as
    magic global position; it only prevents heading drift/slip from making the
    map and physical robot disagree.
    """
    global imu_yaw_zero
    if inertial_unit is None:
        return None
    try:
        yaw = float(inertial_unit.getRollPitchYaw()[2])
    except Exception:
        return None
    if imu_yaw_zero is None:
        imu_yaw_zero = yaw
    return normalize_angle(yaw - imu_yaw_zero)


def read_bumpers():
    """Read the three integrated bumper zones.

    Left/right pads are front-side arc bumpers only. They deliberately
    do not wrap around the rear half of the robot. Center is a straight
    frontal pad. The center contact is folded into both sides so a straight hit
    triggers a neutral backup, while a left-only or right-only hit selects an
    escape direction.
    """
    global last_bumper_left, last_bumper_center, last_bumper_right
    global last_bumper_left_corner, last_bumper_right_corner
    global last_left_side_bumper, last_right_side_bumper
    global last_bumper_raw_left, last_bumper_raw_center, last_bumper_raw_right
    global last_bumper_any_raw_active, last_contact_latch_time, last_contact_latch_x, last_contact_latch_y, last_contact_latch_theta
    global last_contact_latch_left, last_contact_latch_center, last_contact_latch_right, last_contact_latch_used
    left = center = right = False
    raw_left = raw_center = raw_right = 0.0
    try:
        if front_left_bumper is not None:
            raw_left = float(front_left_bumper.getValue())
            left = raw_left > BUMPER_ACTIVE_THRESHOLD
        if front_center_bumper is not None:
            raw_center = float(front_center_bumper.getValue())
            center = raw_center > BUMPER_ACTIVE_THRESHOLD
        if front_right_bumper is not None:
            raw_right = float(front_right_bumper.getValue())
            right = raw_right > BUMPER_ACTIVE_THRESHOLD
    except Exception:
        left = center = right = False
        raw_left = raw_center = raw_right = 0.0
    last_bumper_raw_left = raw_left
    last_bumper_raw_center = raw_center
    last_bumper_raw_right = raw_right
    raw_active = bool(left or center or right)
    # Latch the exact pose on the rising edge of a physical contact.  The
    # controller may spend the next frames backing up; if we mark the obstacle
    # after that, the map shows the obstacle behind or under the robot.
    if raw_active and not last_bumper_any_raw_active:
        last_contact_latch_time = robot.getTime()
        last_contact_latch_x = pose_x
        last_contact_latch_y = pose_y
        last_contact_latch_theta = pose_theta
        last_contact_latch_left = bool(left)
        last_contact_latch_center = bool(center)
        last_contact_latch_right = bool(right)
        last_contact_latch_used = False
    last_bumper_any_raw_active = raw_active
    last_bumper_left = bool(left or center)
    last_bumper_center = bool(center)
    last_bumper_right = bool(right or center)
    # Kept only for compatibility with older debug/logic variables. The updated
    # robot has no separate corner/side sensors.
    last_bumper_left_corner = False
    last_bumper_right_corner = False
    last_left_side_bumper = False
    last_right_side_bumper = False
    return last_bumper_left, last_bumper_right

def start_forward_row(status="row forward", post_lane_lock=False):
    """Enter forward row mode and remember where this row began.

    `post_lane_lock` is used only after completing an atomic manoeuvre sequence
    such as pivot -> shift -> pivot.  It prevents the next frame from scheduling
    another pivot before the robot has actually entered the new strip.
    """
    global nav_state, coverage_status, row_start_x, row_start_y, row_start_time
    global post_lane_forward_lock_until, post_lane_forward_lock_start_x, post_lane_forward_lock_start_y, desired_grid_heading
    global wall_follow_active_side, wall_follow_last_time, wall_follow_last_reason
    # ordinary rows are matrix/grid rows.  If a recovery/escape leaves
    # desired_grid_heading at an arbitrary angle, snap before forward movement.
    desired_grid_heading = strict_world_grid_heading(desired_grid_heading)
    # A new row/heading must reacquire the nearby wall from current RGB-D/map
    # evidence.  Keeping the previous side after a 90-degree turn is a hidden
    # controller leak and can make the robot steer back into the old perimeter.
    wall_follow_active_side = 0.0
    wall_follow_last_time = -999.0
    wall_follow_last_reason = "new forward row"
    nav_state = NAV_FORWARD
    coverage_status = status
    row_start_x = pose_x
    row_start_y = pose_y
    row_start_time = robot.getTime()
    if post_lane_lock:
        post_lane_forward_lock_until = robot.getTime() + POST_LANE_FORWARD_LOCK_SEC
        post_lane_forward_lock_start_x = pose_x
        post_lane_forward_lock_start_y = pose_y
        acquire_control(ControlOwner.ROW_FORWARD, POST_MANEUVER_ROW_OWNER_TIME_SEC, POST_MANEUVER_ROW_OWNER_DISTANCE_M, status)
    else:
        acquire_control(ControlOwner.ROW_FORWARD, ROW_OWNER_MIN_TIME_SEC, ROW_OWNER_MIN_DISTANCE_M, status)


def start_grid_realign(target_heading, reason="grid realign", after_status="row forward after grid realign"):
    """Rotate in place to a grid/tangent heading instead of curving into the row.

    GRID_REALIGN is a local-controller state, not a recovery state. It must be
    allowed to fail softly: if the IMU yaw stalls a few degrees from the target
    near a wall, continuing to pivot is worse than accepting the current tangent.
    """
    global nav_state, grid_realign_target, grid_realign_until, grid_realign_after_status
    global desired_grid_heading, coverage_status, prev_cmd_left, prev_cmd_right, map_freeze_until
    global grid_realign_start_time, grid_realign_best_abs_error, grid_realign_last_progress_time, grid_realign_soft_finish, grid_realign_retry_count
    now = robot.getTime()
    # GRID_REALIGN is part of the hard-grid controller.  Never let an
    # arbitrary escape/contact yaw such as -110 deg become the new row heading.
    # The target must be a room-cardinal heading: 0/90/180/-90.
    grid_realign_target = strict_world_grid_heading(target_heading)
    grid_realign_after_status = after_status
    desired_grid_heading = grid_realign_target
    grid_realign_start_time = now
    grid_realign_best_abs_error = abs(normalize_angle(grid_realign_target - pose_theta))
    grid_realign_last_progress_time = now
    grid_realign_soft_finish = False
    grid_realign_retry_count = 0
    hard_stop_motors()
    if grid_realign_best_abs_error <= GRID_REALIGN_TOLERANCE:
        start_forward_row(after_status)
        return
    nav_state = NAV_GRID_REALIGN
    grid_realign_until = now + GRID_REALIGN_TIMEOUT_SEC
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"grid realign: {reason} target={math.degrees(grid_realign_target):.0f}"
    acquire_control(ControlOwner.GRID_REALIGN, GRID_REALIGN_OWNER_TIME_SEC, 0.0, reason)
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)


def finish_grid_realign(accept_current_heading=False, reason=""):
    global current_linear_velocity, current_angular_velocity, map_freeze_until, desired_grid_heading, grid_realign_soft_finish, coverage_status
    global post_turn_rgbd_snapshot_pending, last_post_turn_snapshot_debug
    hard_stop_motors()
    current_linear_velocity = 0.0
    current_angular_velocity = 0.0
    if not accept_current_heading:
        desired_grid_heading = strict_world_grid_heading(grid_realign_target)
    if accept_current_heading:
        # In hard-grid mode, a soft finish must not create a new diagonal row.
        # Keep the nearest cardinal heading as the desired target; if the robot
        # cannot fully converge near a wall, the next forward guard will retry or
        # recovery will take over, instead of silently driving diagonally.
        desired_grid_heading = strict_world_grid_heading(pose_theta)
        grid_realign_soft_finish = True
        if reason:
            coverage_status = f"grid realign soft snap: {reason}"
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)
    release_control("grid realign finished")
    if POST_TURN_RGBD_SNAPSHOT_ENABLED and matrix_first_explore_active() and not known_map_coverage_eval_active():
        post_turn_rgbd_snapshot_pending = False
        last_post_turn_snapshot_debug = "snap=grid-realign"
        if start_post_turn_rgbd_snapshot("after grid realign"):
            return
    start_forward_row(grid_realign_after_status)


def invalidate_coverage_route_after_contact(reason="contact recovery"):
    """Drop stale coverage/route commitments after a physical contact.

    A bumper hit proves that the previous route/row target was not physically
    executable.  Keeping the magenta route or nav_action_queue alive after a
    release is what pulled the robot back into the same wall or made it snap from
    a recovery tangent into a curved grid re-entry.
    """
    global coverage_goal_map, coverage_goal_world, coverage_goal_kind
    global coverage_route_map, coverage_route_world, coverage_route_waypoint_map, coverage_route_waypoint_world
    global coverage_route_cost, coverage_route_score, coverage_route_len, coverage_route_kind, coverage_route_component_id, coverage_route_component_cells, coverage_route_status
    global coverage_route_commit_class, coverage_route_straight_dist, coverage_route_lateral_abs, coverage_route_turn_need, coverage_route_continuity_bonus, coverage_route_top_debug
    global coverage_route_wall_strip_bonus, coverage_route_missed_strip_bonus, coverage_route_segment_gain, coverage_route_segment_bonus, coverage_route_footprint_gain_cells
    global coverage_route_first_turn_frac, coverage_route_corner_count, coverage_route_geometry_debug
    global last_missed_strip_recovery_debug, last_rejected_components_debug, last_missed_strip_promoted_cells
    global last_coverage_footprint_debug, last_coverage_footprint_gain_cells, last_coverage_footprint_reclean_ratio
    global last_coverage_segment_debug
    global nav_action_queue, row_end_candidate_count, last_revisit_lane_change_time, last_explore_anti_revisit_turn_time
    global post_gap_commit_until, post_lane_forward_lock_until, last_line_acquire_time, last_local_pocket_fill_time
    invalidate_heavy_map_caches(f"contact:{reason[:18]}")
    coverage_goal_map = None
    coverage_goal_world = None
    coverage_goal_kind = "none"
    coverage_route_map = []
    coverage_route_world = []
    coverage_route_waypoint_map = None
    coverage_route_waypoint_world = None
    coverage_route_cost = float("inf")
    coverage_route_score = float("-inf")
    coverage_route_len = 0
    coverage_route_kind = "none"
    coverage_route_component_id = -1
    coverage_route_component_cells = 0
    coverage_route_status = f"invalidated after contact: {reason[:36]}"
    coverage_route_top_debug = "none"
    last_missed_strip_recovery_debug = "none"
    last_rejected_components_debug = "none"
    last_missed_strip_promoted_cells = 0
    last_coverage_footprint_debug = "none"
    last_coverage_footprint_gain_cells = 0
    last_coverage_footprint_reclean_ratio = 0.0
    last_coverage_segment_debug = "none"
    coverage_route_commit_class = "none"
    coverage_route_straight_dist = float("inf")
    coverage_route_lateral_abs = float("inf")
    coverage_route_turn_need = float("inf")
    coverage_route_continuity_bonus = 0.0
    coverage_route_wall_strip_bonus = 0.0
    coverage_route_missed_strip_bonus = 0.0
    coverage_route_segment_gain = 0
    coverage_route_segment_bonus = 0.0
    coverage_route_footprint_gain_cells = 0
    coverage_route_first_turn_frac = 0.0
    coverage_route_corner_count = 0
    coverage_route_geometry_debug = "geom=none"
    now = robot.getTime()
    nav_action_queue = []
    row_end_candidate_count = 0
    if route_commit_active:
        abort_route_commit(f"contact invalidated route: {reason[:42]}")
    last_revisit_lane_change_time = -999.0
    # Do not immediately anti-revisit turn after a real contact; recovery already
    # selects the next row.  Give the robot one compact forward segment first.
    last_explore_anti_revisit_turn_time = now
    post_gap_commit_until = -999.0
    post_lane_forward_lock_until = -999.0
    # Make route/pocket/edge-line helpers wait at least one decision cycle after
    # contact handoff.  The post-recovery stabilizer still owns the immediate
    # straight segment.
    last_line_acquire_time = now
    last_local_pocket_fill_time = now


def filtered_front_bumper_active(bumper_left, bumper_right):
    """Physical bumper state after the carpet false-positive filter has run."""
    return bool(bumper_left or bumper_right or last_bumper_center)


def contact_recovery_forward_timeout_for_kind(kind):
    if kind == "wall":
        return WALL_CONTACT_FORWARD_TIMEOUT_SEC
    if kind == "corner":
        return CORNER_REPEAT_FORWARD_TIMEOUT_SEC
    if kind == "gap":
        return GAP_CONTACT_NUDGE_FORWARD_TIMEOUT_SEC
    if kind == "front":
        return FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC
    return LEG_TRAP_FORWARD_TIMEOUT_SEC


def reset_contact_recovery_state():
    """Clear RecoveryManager runtime fields without touching maps."""
    global contact_recovery_kind, contact_recovery_reason, contact_recovery_clear_since
    global contact_recovery_wait_until, contact_recovery_bumper_hold_since, contact_recovery_replan_after
    contact_recovery_kind = "none"
    contact_recovery_reason = ""
    contact_recovery_clear_since = -999.0
    contact_recovery_wait_until = 0.0
    contact_recovery_bumper_hold_since = -999.0
    contact_recovery_replan_after = False


def start_contact_recovery(
    side,
    kind,
    reason,
    backup_distance,
    backup_timeout,
    turn_angle,
    forward_distance,
    forward_speed,
    forward_timeout=None,
    replan_after=True,
):
    """Start the contact RecoveryManager lifecycle.

    This replaces the old strong LEG_ESCAPE path for real bumper collisions.  It
    enforces the safety contract: while any filtered bumper is still active the
    robot may stop or reverse, but it never rotates or drives forward.
    """
    global nav_state, contact_recovery_kind, contact_recovery_reason, contact_recovery_side
    global contact_recovery_start_x, contact_recovery_start_y, contact_recovery_backup_distance, contact_recovery_backup_until
    global contact_recovery_start_time, contact_recovery_forced_side_release, last_contact_side_release_reason
    global contact_recovery_clear_since, contact_recovery_wait_until, contact_recovery_turn_angle, contact_recovery_turn_target
    global contact_recovery_forward_distance, contact_recovery_forward_speed, contact_recovery_forward_until, contact_recovery_forward_timeout
    global contact_recovery_replan_after, contact_recovery_bumper_hold_since
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until, last_leg_escape_time, last_contact_route_kill_until
    global under_furniture_until, under_furniture_active, under_furniture_suppressed_until

    if not CONTACT_RECOVERY_ENABLED:
        return False
    now = robot.getTime()
    # stop condition.  Do not retreat 20-30 cm from the wall: back up only
    # enough to physically release the thin bumper cushion, then perform a
    # square 90-degree turn.
    compact_bumper_mapping = bool(EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active())
    if compact_bumper_mapping:
        backup_distance = min(float(backup_distance), EXPLORE_CONTACT_BACKUP_GOAL_M)
        backup_timeout = min(float(backup_timeout), EXPLORE_CONTACT_BACKUP_TIMEOUT_SEC)
        forward_distance = min(float(forward_distance), EXPLORE_CONTACT_FORWARD_VERIFY_M)
        if kind in ("front", "side", "wall", "contact"):
            turn_angle = PIVOT_TURN_ANGLE
    contact_recovery_kind = kind or "contact"
    contact_recovery_reason = reason
    contact_recovery_side = 1.0 if side >= 0 else -1.0
    contact_recovery_start_x = pose_x
    contact_recovery_start_y = pose_y
    contact_recovery_start_time = now
    contact_recovery_forced_side_release = False
    last_contact_side_release_reason = ""
    contact_recovery_backup_distance = max(0.02, float(backup_distance))
    contact_recovery_backup_until = now + max(0.20, float(backup_timeout)) + CONTACT_BACKUP_EXTRA_TIMEOUT_SEC
    contact_recovery_clear_since = -999.0
    contact_recovery_wait_until = 0.0
    contact_recovery_turn_angle = abs(float(turn_angle))
    contact_recovery_turn_target = pose_theta
    contact_recovery_forward_distance = max(0.04, float(forward_distance))
    contact_recovery_forward_speed = max(0.20, float(forward_speed))
    if forward_timeout is None:
        forward_timeout = contact_recovery_forward_timeout_for_kind(contact_recovery_kind)
    contact_recovery_forward_timeout = max(0.35, float(forward_timeout))
    contact_recovery_forward_until = 0.0
    contact_recovery_replan_after = bool(replan_after)
    contact_recovery_bumper_hold_since = now

    invalidate_coverage_route_after_contact(reason)
    clear_post_recovery_stabilizer()
    under_furniture_until = 0.0
    under_furniture_active = False
    under_furniture_suppressed_until = now + UNDER_FURNITURE_TRAP_COOLDOWN_SEC
    last_contact_route_kill_until = max(last_contact_route_kill_until, now + CONTACT_RECOVERY_ROUTE_KILL_SEC)
    last_leg_escape_time = now
    nav_state = NAV_CONTACT_BACKUP
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"contact recovery backup: {contact_recovery_kind} {reason[:42]}"
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
    return True


def finish_contact_recovery_backup():
    """Transition from reverse release to a short stable clear wait."""
    global nav_state, contact_recovery_wait_until, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    hard_stop_motors()
    nav_state = NAV_CONTACT_WAIT_CLEAR
    contact_recovery_wait_until = robot.getTime() + CONTACT_WAIT_CLEAR_SEC
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"contact recovery wait-clear: {contact_recovery_kind}"
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.18)


def finish_contact_recovery_wait_clear():
    """Start the in-place tangent rotation only after the bumper is released."""
    global nav_state, contact_recovery_turn_target, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    global contact_recovery_start_time
    hard_stop_motors()
    contact_recovery_start_time = robot.getTime()
    if EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active():
        # Turn relative to the planned grid row, not to a slightly skewed physical
        # yaw at the moment of contact. This prevents 84-88 degree rows from
        # becoming the new reference direction.
        contact_recovery_turn_target = strict_world_grid_heading(
            normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
        )
    else:
        contact_recovery_turn_target = normalize_angle(pose_theta + contact_recovery_side * contact_recovery_turn_angle)
    nav_state = NAV_CONTACT_ROTATE
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"contact recovery rotate: {contact_recovery_kind} target={math.degrees(contact_recovery_turn_target):.0f}"
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)


def finish_contact_recovery_rotate():
    """Finish the contact pivot.

    during bumper-first EXPLORE the old CONTACT_FORWARD verify segment
    was harmful.  It often started as a FINE_ALIGN while the robot was still
    millimetres from the wall, so the bumper pad re-contacted and the controller
    entered CONTACT_BACKUP/CONTACT_ROTATE loops.  For map building the recovery
    contract is now atomic: compact backup -> exact cardinal pivot -> ordinary
    row forward.  No local verification nudge, no route planner, no GRID_REALIGN.
    """
    global nav_state, contact_recovery_forward_start_x, contact_recovery_forward_start_y, contact_recovery_forward_until
    global desired_grid_heading, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    global contact_recovery_forced_side_release, last_turn_variant, contact_recovery_replan_after
    hard_stop_motors()
    contact_recovery_forced_side_release = False
    if EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active():
        desired_grid_heading = strict_world_grid_heading(contact_recovery_turn_target)
        invalidate_coverage_route_after_contact(f"finish rotate {contact_recovery_kind}")
        release_control("contact rotate finished; skip verify forward")
        last_turn_variant = f"contact-rotate-grid-{contact_recovery_kind}"
        contact_recovery_replan_after = False
        map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)
        if POST_TURN_RGBD_SNAPSHOT_ENABLED and start_post_turn_rgbd_snapshot("after contact pivot"):
            return
        start_forward_row("row forward after contact pivot; verify skipped")
        return
    else:
        desired_grid_heading = pose_theta
    contact_recovery_forward_start_x = pose_x
    contact_recovery_forward_start_y = pose_y
    contact_recovery_forward_until = robot.getTime() + contact_recovery_forward_timeout
    nav_state = NAV_CONTACT_FORWARD
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"contact recovery forward: {contact_recovery_kind}"
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.25)


def finish_contact_recovery_forward():
    """Release RecoveryManager ownership by snapping back to the room grid.

    The old handoff used pose_theta as the next row heading.  That removes wheel
    arcs but still drives a straight diagonal if the robot escaped at 70-85 deg.
    For a square/lawnmower prototype, recovery may rotate away from contact, but
    normal navigation resumes only after an in-place realignment to 0/90/180/270.
    """
    global desired_grid_heading, coverage_status, last_turn_variant, contact_recovery_replan_after
    target_heading = strict_world_grid_heading(pose_theta)
    desired_grid_heading = target_heading
    invalidate_coverage_route_after_contact(f"finish {contact_recovery_kind}")
    start_post_recovery_stabilizer(
        "wall" if contact_recovery_kind == "wall" else "contact",
        target_heading,
        0.85,
        0.07,
    )
    last_turn_variant = f"contact-manager-grid-{contact_recovery_kind}"
    contact_recovery_replan_after = False
    if EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active():
        # enter GRID_REALIGN immediately after it: near a wall that state can sit
        # in place and re-contact the bumper.  Resume a short straight row with
        # the strict heading lock instead.
        release_control("contact recovery finished; map-build forward")
        start_forward_row("row forward after compact contact recovery")
    else:
        start_grid_realign(target_heading, f"after contact recovery {contact_recovery_kind}", "row forward after grid contact recovery")


def contact_recovery_manager_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right):
    """Tick the contact RecoveryManager and return a wheel command if active."""
    global nav_state, contact_recovery_clear_since, contact_recovery_bumper_hold_since, coverage_status, last_turn_variant
    global contact_recovery_start_x, contact_recovery_start_y, contact_recovery_backup_until, contact_recovery_start_time
    global contact_recovery_side, contact_recovery_turn_target, contact_recovery_forced_side_release, last_contact_side_release_reason
    global map_freeze_until

    if nav_state not in CONTACT_RECOVERY_STATES:
        return None

    now = robot.getTime()
    bumper_active = filtered_front_bumper_active(bumper_left, bumper_right)

    if bumper_active:
        if contact_recovery_bumper_hold_since < 0.0:
            contact_recovery_bumper_hold_since = now
        contact_recovery_clear_since = -999.0
    else:
        contact_recovery_bumper_hold_since = -999.0
        if contact_recovery_clear_since < 0.0:
            contact_recovery_clear_since = now

    clear_stable = (not bumper_active) and (now - contact_recovery_clear_since >= CONTACT_CLEAR_STABLE_SEC)

    if nav_state == NAV_CONTACT_BACKUP:
        moved = math.hypot(pose_x - contact_recovery_start_x, pose_y - contact_recovery_start_y)
        compact_bumper_mapping = bool(EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active())
        backup_goal = min(contact_recovery_backup_distance, EXPLORE_CONTACT_BACKUP_GOAL_M) if compact_bumper_mapping else contact_recovery_backup_distance
        remaining = max(0.0, backup_goal - moved)
        if CONTACT_SIDE_RELEASE_ENABLED and bumper_active:
            side_release = decide_side_release_after_backup(
                bumper_left=bool(bumper_left),
                bumper_center=bool(last_bumper_center),
                bumper_right=bool(bumper_right),
                moved_m=moved,
                backup_elapsed_s=max(0.0, now - contact_recovery_start_time),
                front_m=front,
                center_m=center,
                body_clearance_m=body_clearance,
                min_moved_m=CONTACT_SIDE_RELEASE_MIN_BACKUP_M,
                max_elapsed_s=CONTACT_SIDE_RELEASE_MAX_BACKUP_SEC,
                min_front_clear_m=CONTACT_SIDE_RELEASE_FRONT_CLEAR_M,
                min_center_clear_m=CONTACT_SIDE_RELEASE_CENTER_CLEAR_M,
                min_body_clear_m=CONTACT_SIDE_RELEASE_BODY_CLEAR_M,
            )
            if side_release.force_release_turn:
                contact_recovery_side = side_release.turn_side
                # the successful coverage rows.  The old non-map-build branch used
                # a local 26-degree pose-relative target; near walls this created
                # oblique recovery headings and long CONTACT_ROTATE loops.
                contact_recovery_turn_target = strict_world_grid_heading(
                    normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
                )
                if compact_bumper_mapping:
                    last_contact_side_release_reason = f"map-build-side-release: {side_release.reason}"
                else:
                    last_contact_side_release_reason = f"grid-side-release: {side_release.reason}"
                contact_recovery_forced_side_release = True
                contact_recovery_start_time = now
                nav_state = NAV_CONTACT_ROTATE
                coverage_status = f"contact side-release pivot: {last_contact_side_release_reason}"
                last_turn_variant = "contact-side-release"
                map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
                return 0.0, 0.0
        if compact_bumper_mapping and bumper_active:
            elapsed = max(0.0, now - contact_recovery_start_time)
            if moved >= EXPLORE_CONTACT_FORCE_PIVOT_AFTER_M or elapsed >= EXPLORE_CONTACT_FORCE_PIVOT_AFTER_SEC:
                # bumper before turning.  In corners the pad can remain pressed,
                # so the robot kept backing or got stuck.  This is a bounded
                # recovery manoeuvre, not normal planning: reverse-pivot toward
                # the next cardinal row heading.
                contact_recovery_turn_target = strict_world_grid_heading(
                    normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
                )
                contact_recovery_forced_side_release = True
                contact_recovery_start_time = now
                last_contact_side_release_reason = "map-build-forced-reverse-pivot"
                nav_state = NAV_CONTACT_ROTATE
                coverage_status = (
                    f"contact backup -> forced square pivot moved={moved:.2f} "
                    f"elapsed={elapsed:.2f} target={math.degrees(contact_recovery_turn_target):.0f}"
                )
                last_turn_variant = "map-build-forced-reverse-pivot"
                map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
                return 0.0, 0.0

        if clear_stable and moved >= backup_goal:
            finish_contact_recovery_backup()
            return 0.0, 0.0
        if reverse_backup_should_stop(remaining):
            # /33: the reverse guard is intentionally conservative because
            # there is no rear RGB-D sensor.  But after a real FRONT/CENTER bumper
            # hit, `map-edge` behind the robot is often a mapper boundary artifact,
            # not a physical rear obstacle.  Holding CONTACT_BACKUP in that case
            # wedges the robot into the wall (debug: prim=STOP, rear-guard hold
            # 0.00m map-edge, bump=1).  Allow a bounded slow reverse release for
            # map-edge only; never bypass a remembered occupied/contact cell.
            elapsed = max(0.0, now - contact_recovery_start_time)
            rear_reason = str(last_rear_guard_reason)
            front_bumper_contact = bool(last_bumper_center or (bumper_left and bumper_right))
            if bumper_active and front_bumper_contact and "map-edge" in rear_reason:
                release_goal = min(
                    max(min(backup_goal, CONTACT_BACKUP_FRONT_MAPEDGE_RELEASE_M), CONTACT_BACKUP_GUARD_BYPASS_M),
                    CONTACT_BACKUP_FRONT_MAPEDGE_RELEASE_M,
                )
                if moved < release_goal and elapsed < CONTACT_BACKUP_FRONT_MAPEDGE_RELEASE_TIMEOUT_SEC:
                    coverage_status = (
                        f"contact backup front-mapedge release {moved:.2f}/{release_goal:.2f} "
                        f"rear={last_rear_guard_clearance:.2f} {last_rear_guard_reason} bump={int(bumper_active)}"
                    )
                    return -CONTACT_BACKUP_FRONT_MAPEDGE_RELEASE_SPEED, -CONTACT_BACKUP_FRONT_MAPEDGE_RELEASE_SPEED
                # If the contact pad is still pressed after the bounded release,
                # do not sit forever: execute the same square escape used by the
                # perimeter/coverage row-end contract.
                contact_recovery_turn_target = strict_world_grid_heading(
                    normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
                )
                contact_recovery_forced_side_release = True
                contact_recovery_start_time = now
                last_contact_side_release_reason = CONTACT_BACKUP_FRONT_MAPEDGE_PIVOT_REASON
                nav_state = NAV_CONTACT_ROTATE
                coverage_status = f"contact backup mapedge release -> pivot moved={moved:.2f} rear={last_rear_guard_reason}"
                last_turn_variant = CONTACT_BACKUP_FRONT_MAPEDGE_PIVOT_REASON
                map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
                return 0.0, 0.0

            if bumper_active and front_bumper_contact and "rear-occupied" in rear_reason:
                #   nav=CONTACT_BACKUP prim=STOP ... rear-occupied bump=1
                # The front bumper is physically pressed, but reverse is blocked
                # by remembered occupancy behind the body.  Do not bypass the rear
                # obstacle with a blind reverse.  Switch to the same cardinal
                # in-place escape used by row-end/contact recovery.
                contact_recovery_turn_target = strict_world_grid_heading(
                    normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
                )
                contact_recovery_forced_side_release = True
                contact_recovery_start_time = now
                last_contact_side_release_reason = CONTACT_BACKUP_FRONT_REAROCC_PIVOT_REASON
                nav_state = NAV_CONTACT_ROTATE
                coverage_status = (
                    f"contact backup rear-occupied -> pivot moved={moved:.2f} "
                    f"rear={last_rear_guard_clearance:.2f}/{last_rear_guard_reason}"
                )
                last_turn_variant = CONTACT_BACKUP_FRONT_REAROCC_PIVOT_REASON
                map_freeze_until = max(map_freeze_until, now + CONTACT_BACKUP_FRONT_REAROCC_PIVOT_FREEZE_SEC)
                return 0.0, 0.0

            # during early bumper-led mapping, a conservative rear-map
            # guard can falsely report "map-edge" immediately after a wall hit.
            # If we simply hold while the centre bumper is still pressed, the robot
            # wedges forever.  First allow a tiny slow reverse release; if that is
            # still impossible, use a bounded low-speed in-place pivot as an
            # emergency unstick.  This is not normal planning; it is recovery only.
            if compact_bumper_mapping and bumper_active:
                if moved < CONTACT_BACKUP_GUARD_BYPASS_M and "map-edge" in rear_reason:
                    coverage_status = (
                        f"contact backup guarded micro-release {moved:.2f}/{CONTACT_BACKUP_GUARD_BYPASS_M:.2f} "
                        f"{last_rear_guard_reason} bump={int(bumper_active)}"
                    )
                    return -CONTACT_BACKUP_GUARD_BYPASS_SPEED, -CONTACT_BACKUP_GUARD_BYPASS_SPEED
                if elapsed >= CONTACT_BACKUP_GUARD_STUCK_PIVOT_AFTER_SEC:
                    contact_recovery_turn_target = strict_world_grid_heading(
                        normalize_angle(desired_grid_heading + contact_recovery_side * PIVOT_TURN_ANGLE)
                    )
                    contact_recovery_forced_side_release = True
                    contact_recovery_start_time = now
                    last_contact_side_release_reason = "map-build-rear-guard-stuck-pivot"
                    nav_state = NAV_CONTACT_ROTATE
                    coverage_status = f"contact backup guarded -> stuck pivot {last_rear_guard_reason}"
                    return 0.0, 0.0
            # Default safety rule: do not keep reversing into remembered obstacles.
            coverage_status = f"contact backup rear-guard hold {last_rear_guard_clearance:.2f}m {last_rear_guard_reason} bump={int(bumper_active)}"
            if clear_stable:
                finish_contact_recovery_backup()
            return 0.0, 0.0
        if (not bumper_active) and now >= contact_recovery_backup_until and moved >= min(backup_goal, 0.055):
            finish_contact_recovery_backup()
            return 0.0, 0.0
        speed = LEG_ESCAPE_BACKUP_SPEED
        if contact_recovery_kind == "wall":
            speed = WALL_CONTACT_FORWARD_SPEED * 0.66
        elif bumper_active and now >= contact_recovery_backup_until:
            speed = CONTACT_BACKUP_HOLD_SPEED
        coverage_status = (
            f"contact backup {contact_recovery_kind} {moved:.2f}/{backup_goal:.2f} "
            f"clear={max(0.0, now-contact_recovery_clear_since) if contact_recovery_clear_since > 0 else 0:.2f}s bump={int(bumper_active)} rear={last_rear_guard_clearance:.2f}"
        )
        last_turn_variant = f"contact-backup-{contact_recovery_kind}"
        return -speed, -speed

    if nav_state == NAV_CONTACT_WAIT_CLEAR:
        if bumper_active:
            # Contact returned while settling: go back to reverse release.  Do not
            # rotate into a still-pressed bumper.
            contact_recovery_start_x = pose_x
            contact_recovery_start_y = pose_y
            contact_recovery_backup_until = now + CONTACT_BACKUP_EXTRA_TIMEOUT_SEC
            nav_state = NAV_CONTACT_BACKUP
            coverage_status = f"contact wait-clear interrupted -> backup {contact_recovery_kind}"
            return -CONTACT_BACKUP_HOLD_SPEED, -CONTACT_BACKUP_HOLD_SPEED
        if now >= contact_recovery_wait_until and clear_stable:
            finish_contact_recovery_wait_clear()
            return 0.0, 0.0
        hard_stop_motors()
        coverage_status = f"contact wait-clear {contact_recovery_kind} clear={now-contact_recovery_clear_since:.2f}s"
        return 0.0, 0.0

    if nav_state == NAV_CONTACT_ROTATE:
        if bumper_active and not contact_recovery_forced_side_release:
            # Front/center contact: never rotate into a still-pressed bumper.
            contact_recovery_start_x = pose_x
            contact_recovery_start_y = pose_y
            contact_recovery_start_time = now
            contact_recovery_backup_until = now + CONTACT_BACKUP_EXTRA_TIMEOUT_SEC
            nav_state = NAV_CONTACT_BACKUP
            coverage_status = f"contact rotate blocked by bumper -> backup {contact_recovery_kind}"
            return -CONTACT_BACKUP_HOLD_SPEED, -CONTACT_BACKUP_HOLD_SPEED
        if (
            bumper_active
            and contact_recovery_forced_side_release
            and (last_bumper_center or (bumper_left and bumper_right))
            and not str(last_contact_side_release_reason).startswith("map-build")
            and str(last_contact_side_release_reason) != CONTACT_BACKUP_FRONT_MAPEDGE_PIVOT_REASON
            and str(last_contact_side_release_reason) != "rear-guard-stuck-pivot"
        ):
            # Escalated from side scrape to front contact: abandon release pivot.
            contact_recovery_forced_side_release = False
            contact_recovery_start_x = pose_x
            contact_recovery_start_y = pose_y
            contact_recovery_start_time = now
            contact_recovery_backup_until = now + CONTACT_BACKUP_EXTRA_TIMEOUT_SEC
            nav_state = NAV_CONTACT_BACKUP
            coverage_status = f"side-release hit front -> backup {contact_recovery_kind}"
            return -CONTACT_BACKUP_HOLD_SPEED, -CONTACT_BACKUP_HOLD_SPEED
        heading_remaining = normalize_angle(contact_recovery_turn_target - pose_theta)
        if abs(heading_remaining) <= CONTACT_ROTATE_TOLERANCE:
            finish_contact_recovery_rotate()
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        speed = clamp(abs(heading_remaining) * CONTACT_ROTATE_KP, CONTACT_ROTATE_MIN_SPEED, CONTACT_ROTATE_MAX_SPEED)
        if contact_recovery_forced_side_release:
            speed *= CONTACT_SIDE_RELEASE_SPEED_SCALE
            if str(last_contact_side_release_reason).startswith("map-build"):
                speed = min(max(speed, CONTACT_ROTATE_MIN_SPEED), EXPLORE_CONTACT_REVERSE_PIVOT_MAX_SPEED)
            elif last_contact_side_release_reason == CONTACT_BACKUP_FRONT_MAPEDGE_PIVOT_REASON:
                speed = min(max(speed, CONTACT_ROTATE_MIN_SPEED), EXPLORE_CONTACT_REVERSE_PIVOT_MAX_SPEED)
            elif last_contact_side_release_reason == "rear-guard-stuck-pivot":
                speed = min(speed, 0.58)
        coverage_status = f"contact rotate {contact_recovery_kind} err={math.degrees(heading_remaining):.1f}"
        if contact_recovery_forced_side_release:
            coverage_status = f"contact side-release rotate err={math.degrees(heading_remaining):.1f} {last_contact_side_release_reason[:38]}"
        last_turn_variant = f"contact-rotate-{contact_recovery_kind}"
        if contact_recovery_forced_side_release and bumper_active:
            # pressed must have a reverse component.  Otherwise the visible bumper
            # remains pinned against the wall/chair and CONTACT_ROTATE can loop
            # forever while the RGB-D camera origin clips through the wall.
            reverse_bias = CONTACT_SIDE_RELEASE_REVERSE_PIVOT_BIAS
            if (
                str(last_contact_side_release_reason).startswith("map-build")
                or last_contact_side_release_reason == CONTACT_BACKUP_FRONT_MAPEDGE_PIVOT_REASON
                or last_contact_side_release_reason == CONTACT_BACKUP_FRONT_REAROCC_PIVOT_REASON
            ):
                reverse_bias = EXPLORE_CONTACT_REVERSE_PIVOT_BIAS
            rotate_elapsed = max(0.0, now - contact_recovery_start_time)
            if rotate_elapsed > CONTACT_SIDE_RELEASE_ROTATE_WATCHDOG_SEC:
                reverse_bias = max(reverse_bias, EXPLORE_CONTACT_REVERSE_PIVOT_BIAS)
                coverage_status = (
                    f"contact side-release watchdog reverse-pivot "
                    f"err={math.degrees(heading_remaining):.1f} {last_contact_side_release_reason[:32]}"
                )
            return (-sign * speed) - reverse_bias, (sign * speed) - reverse_bias
        return -sign * speed, sign * speed

    if nav_state == NAV_CONTACT_FORWARD:
        if bumper_active:
            side = -1.0 if bumper_left and not bumper_right else (1.0 if bumper_right and not bumper_left else contact_recovery_side)
            obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_contact_recovery(
                side,
                "front" if last_bumper_center or (bumper_left and bumper_right) else "side",
                f"contact-forward bump L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
                FRONT_CONTACT_TRAP_BACKUP_DISTANCE,
                FRONT_CONTACT_TRAP_BACKUP_TIMEOUT_SEC,
                FRONT_CONTACT_TRAP_TURN_ANGLE,
                FRONT_CONTACT_TRAP_FORWARD_DISTANCE,
                CONTACT_ESCAPE_VERIFY_SPEED,
                FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
                True,
            )
            return 0.0, 0.0
        central_block = (
            (center < LOW_OBSTACLE_GUARD_CENTER_M and front < LOW_OBSTACLE_GUARD_FRONT_M)
            or (body_clearance < LOW_OBSTACLE_GUARD_BODY_M and abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M)
        )
        if central_block:
            side = choose_contact_escape_side(False, False, left, right, contact_obstacle_side_from_sensors(left, right))
            start_contact_recovery(
                side,
                "front",
                f"contact-forward low block F={front:.2f} C={center:.2f} body={body_clearance:.2f}",
                FRONT_CONTACT_TRAP_BACKUP_DISTANCE,
                FRONT_CONTACT_TRAP_BACKUP_TIMEOUT_SEC,
                FRONT_CONTACT_TRAP_TURN_ANGLE,
                FRONT_CONTACT_TRAP_FORWARD_DISTANCE,
                CONTACT_ESCAPE_VERIFY_SPEED,
                FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
                True,
            )
            return 0.0, 0.0
        moved = math.hypot(pose_x - contact_recovery_forward_start_x, pose_y - contact_recovery_forward_start_y)
        if moved >= contact_recovery_forward_distance or now >= contact_recovery_forward_until:
            finish_contact_recovery_forward()
            return 0.0, 0.0
        coverage_status = f"contact forward {contact_recovery_kind} {moved:.2f}/{contact_recovery_forward_distance:.2f}"
        return heading_locked_wheel_speeds_to(contact_recovery_forward_speed, desired_grid_heading, HEADING_LOCK_KP, POST_RECOVERY_MAX_CORRECTION)

    return None

def post_lane_forward_lock_active(front=None, center=None, body_clearance=None):
    """Keep driving straight briefly after completing a lane sequence.

    This blocks planner/pocket/under-furniture/soft row-end pivots only.  Real
    contacts and very close frontal blocks still break the lock so the robot does
    not push into furniture.
    """
    if nav_state != NAV_FORWARD:
        return False
    now = robot.getTime()
    if now >= post_lane_forward_lock_until:
        return False
    moved = math.hypot(pose_x - post_lane_forward_lock_start_x, pose_y - post_lane_forward_lock_start_y)
    if moved >= POST_LANE_FORWARD_LOCK_DISTANCE_M:
        return False

    # Do not hide a real collision.  The centre bumper is folded into left/right
    # by read_bumpers(), so check all saved bumper flags explicitly.
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return False
    if front is not None and center is not None and front < ROW_END_HARD_DISTANCE and center < SAFE_FRONT_DISTANCE:
        return False
    if (
        body_clearance is not None
        and body_clearance < BODY_CORRIDOR_HARD_CLEARANCE
        and abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M
    ):
        return False
    return True


def post_lane_forward_lock_progress():
    moved = math.hypot(pose_x - post_lane_forward_lock_start_x, pose_y - post_lane_forward_lock_start_y)
    remain_t = max(0.0, post_lane_forward_lock_until - robot.getTime())
    return moved, remain_t


def start_post_recovery_stabilizer(kind="contact", heading=None, duration=None, distance=None):
    """Start the post-recovery motion contract.

    This is intentionally a small arbitration layer, not another local-navigation
    if-chain.  Recovery decides that the robot has escaped; this layer then
    prevents coverage, pocket fill and line-acquire controllers from producing a
    curved re-entry before the robot has travelled a short straight segment.
    """
    global post_recovery_stabilize_until, post_recovery_stabilize_start_x, post_recovery_stabilize_start_y
    global post_recovery_stabilize_heading, post_recovery_stabilize_distance, post_recovery_stabilize_kind
    if not POST_RECOVERY_STABILIZER_ENABLED:
        return
    now = robot.getTime()
    post_recovery_stabilize_kind = kind
    post_recovery_stabilize_start_x = pose_x
    post_recovery_stabilize_start_y = pose_y
    raw_heading = normalize_angle(pose_theta if heading is None else heading)
    # Contact recovery may exit at an arbitrary yaw.  Do not let the post-recovery
    # lock drive a long straight segment at that yaw; snap to the room grid except
    # for a very short gap-mouth commit where a micro-heading is intentional.
    post_recovery_stabilize_heading = raw_heading if kind == "gap" else strict_world_grid_heading(raw_heading)
    if duration is None:
        duration = POST_RECOVERY_WALL_STABILIZE_SEC if kind == "wall" else POST_RECOVERY_STABILIZE_SEC
    if distance is None:
        distance = POST_RECOVERY_WALL_DISTANCE_M if kind == "wall" else POST_RECOVERY_STABILIZE_DISTANCE_M
    post_recovery_stabilize_distance = distance
    post_recovery_stabilize_until = now + duration


def post_recovery_stabilizer_progress():
    moved = math.hypot(pose_x - post_recovery_stabilize_start_x, pose_y - post_recovery_stabilize_start_y)
    remain_t = max(0.0, post_recovery_stabilize_until - robot.getTime())
    return moved, remain_t


def post_recovery_stabilizer_active(front=None, center=None, body_clearance=None):
    """Return True while the post-recovery straight/wall-trace contract owns FORWARD."""
    if not POST_RECOVERY_STABILIZER_ENABLED or nav_state != NAV_FORWARD:
        return False
    if post_recovery_stabilize_kind == "none":
        return False
    now = robot.getTime()
    moved, _ = post_recovery_stabilizer_progress()
    if moved >= post_recovery_stabilize_distance:
        return False
    if now >= post_recovery_stabilize_until:
        return False
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return False
    return True


def clear_post_recovery_stabilizer():
    global post_recovery_stabilize_until, post_recovery_stabilize_kind
    post_recovery_stabilize_until = -999.0
    post_recovery_stabilize_kind = "none"


def post_recovery_wall_trace_yaw(left, right, body_clearance):
    """Tiny wall-clearance correction used only during post-recovery wall tracing."""
    if post_recovery_stabilize_kind != "wall":
        return 0.0
    if body_clearance < BODY_CORRIDOR_PASS_CLEARANCE and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M:
        wall_side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0
        side_clear = body_clearance
    else:
        left_metric = min(left, last_cv_left_obstacle)
        right_metric = min(right, last_cv_right_obstacle)
        if left_metric <= right_metric:
            wall_side = 1.0
            side_clear = left_metric
        else:
            wall_side = -1.0
            side_clear = right_metric
    # Post-recovery must be a straight stabilisation segment.  A side wall that
    # is still far away is not a reason to bend into it; otherwise the robot
    # exits contact recovery by drawing the same diagonal arc we are trying to
    # remove.
    if side_clear > WALL_HUG_REACQUIRE_SIDE_MAX_M:
        return 0.0
    err = side_clear - WALL_HUG_TARGET_CLEARANCE_M
    if abs(err) < WALL_HUG_TARGET_DEADBAND_M:
        err = 0.0
    yaw = wall_side * clamp(err * WALL_HUG_KP * 0.30, -POST_RECOVERY_WALL_MAX_YAW, POST_RECOVERY_WALL_MAX_YAW)
    if side_clear < WALL_HUG_TOO_CLOSE_M or body_clearance < BODY_CORRIDOR_HARD_CLEARANCE + 0.012:
        yaw = -wall_side * POST_RECOVERY_WALL_MAX_YAW
    return yaw


def post_recovery_stabilizer_speeds(front, center, upper_front, left, right, body_clearance):
    """Return the only allowed command during post-recovery stabilisation."""
    global row_end_candidate_count, coverage_status, last_turn_variant
    if not post_recovery_stabilizer_active(front, center, body_clearance):
        # End the lock explicitly once it has naturally expired.  This keeps the
        # debug text truthful and prevents stale kind values from confusing logs.
        if post_recovery_stabilize_kind != "none" and nav_state == NAV_FORWARD:
            clear_post_recovery_stabilizer()
        return None

    central_body_block = (
        body_clearance < BODY_CORRIDOR_HARD_CLEARANCE + POST_RECOVERY_BODY_HARD_MARGIN_M
        and abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M
    )
    if (front < POST_RECOVERY_HARD_FRONT_M and center < POST_RECOVERY_HARD_CENTER_M) or central_body_block:
        clear_post_recovery_stabilizer()
        side = choose_contact_escape_side(False, False, left, right, contact_obstacle_side_from_sensors(left, right))
        start_recovery_backup(side, f"post-recovery hard block F={front:.2f} C={center:.2f} body={body_clearance:.2f}")
        return 0.0, 0.0

    heading_err = normalize_angle(post_recovery_stabilize_heading - pose_theta)
    if abs(heading_err) > POST_RECOVERY_REALIGN_ERR and front > FORWARD_STRAIGHT_REALIGN_MIN_FRONT_M and center > FORWARD_STRAIGHT_REALIGN_MIN_FRONT_M:
        start_grid_realign(post_recovery_stabilize_heading, f"post-recovery {post_recovery_stabilize_kind}", "post-recovery straight segment")
        row_end_candidate_count = 0
        return 0.0, 0.0

    base = POST_RECOVERY_WALL_SPEED if post_recovery_stabilize_kind == "wall" else POST_RECOVERY_SPEED
    yaw = clamp(heading_err * POST_RECOVERY_HEADING_KP, -POST_RECOVERY_MAX_CORRECTION, POST_RECOVERY_MAX_CORRECTION)
    yaw += post_recovery_wall_trace_yaw(left, right, body_clearance)
    yaw = clamp(yaw, -POST_RECOVERY_MAX_CORRECTION, POST_RECOVERY_MAX_CORRECTION)
    moved, remain_t = post_recovery_stabilizer_progress()
    row_end_candidate_count = 0
    last_turn_variant = f"post-recovery-{post_recovery_stabilize_kind}"
    coverage_status = (
        f"post-recovery {post_recovery_stabilize_kind} straight "
        f"{moved:.2f}/{post_recovery_stabilize_distance:.2f}m {remain_t:.1f}s "
        f"herr={math.degrees(heading_err):.1f}"
    )
    return base - yaw, base + yaw


def cleaned_ratio_ahead():
    """How much of the near future path is already cleaned.

    This is a simple anti-loop signal. A perfect coverage planner would use A*
    and frontier selection; here we only need to avoid repeating the same strip
    for hours in the demo.
    """
    total = 0
    cleaned = 0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta)
    ny = math.cos(pose_theta)
    for ahead_i in range(4, 15):
        ahead = 0.10 * ahead_i
        for side in (-0.12, 0.0, 0.12):
            wx = pose_x + fx * ahead + nx * side
            wy = pose_y + fy * ahead + ny * side
            mx, my = world_to_map(wx, wy)
            if not map_inside(mx, my):
                continue
            total += 1
            if cleaned_mask[my, mx] > 0:
                cleaned += 1
    return (cleaned / total) if total else 0.0


def coverage_corridor_ahead():
    """Return cleaned/uncleaned/obstacle ratios in the corridor ahead.

    This is used by the coverage objective layer. It answers a different
    question than obstacle avoidance: "is the robot about to spend time on a
    lane that has already been cleaned?"
    """
    obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
    total = 0
    cleaned_n = 0
    uncleaned_n = 0
    obstacle_n = 0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta)
    ny = math.cos(pose_theta)
    for ahead_i in range(4, 20):
        ahead = 0.10 * ahead_i
        for side in (-0.18, -0.09, 0.0, 0.09, 0.18):
            wx = pose_x + fx * ahead + nx * side
            wy = pose_y + fy * ahead + ny * side
            mx, my = world_to_map(wx, wy)
            if not map_inside(mx, my):
                continue
            total += 1
            if obstacles[my, mx]:
                obstacle_n += 1
            elif cleaned[my, mx]:
                cleaned_n += 1
            elif uncleaned[my, mx]:
                uncleaned_n += 1
    if total == 0:
        return 0.0, 0.0, 0.0
    return cleaned_n / total, uncleaned_n / total, obstacle_n / total


def coverage_guidance_world():
    """Return the immediate route waypoint, falling back to the final goal.

    The final target can be around a table or behind an already-cleaned corridor.
    Steering decisions should therefore use the next route waypoint, not the final
    target directly. This is the key difference between "crazy diagonal target"
    and a sane coverage route.
    """
    if coverage_route_waypoint_world is not None:
        return coverage_route_waypoint_world
    return coverage_goal_world


def coverage_target_side_or_default(left_dist, right_dist):
    """Choose side toward the current route waypoint when it is clearly lateral."""
    guide = coverage_guidance_world()
    if guide is not None and coverage_target_replan_allowed(for_side_bias=True):
        gx, gy = guide
        dx = gx - pose_x
        dy = gy - pose_y
        lateral = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
        forward = math.cos(pose_theta) * dx + math.sin(pose_theta) * dy
        # Do not let a waypoint behind the robot cause a random side switch. Wait
        # for the row-end/pivot logic to face a cleaner direction first.
        if forward > -0.10 and abs(lateral) > TARGET_LATERAL_MIN_M:
            side = 1.0 if lateral > 0.0 else -1.0
            # Do not choose a side that is immediately blocked. Fall back to the
            # local/map score in that case.
            if side > 0 and left_dist > SIDE_DISTANCE + 0.12:
                return side
            if side < 0 and right_dist > SIDE_DISTANCE + 0.12:
                return side
    return choose_coverage_side(left_dist, right_dist)


def coverage_goal_local_offset():
    """Return (forward, lateral, distance) to the current route waypoint/goal.

    Positive forward means the guidance point is in front of the robot. Positive
    lateral means it is to the robot's left.
    """
    guide = coverage_guidance_world()
    if guide is None:
        return None
    gx, gy = guide
    dx = gx - pose_x
    dy = gy - pose_y
    forward = math.cos(pose_theta) * dx + math.sin(pose_theta) * dy
    lateral = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
    dist = math.hypot(dx, dy)
    return forward, lateral, dist


def coverage_goal_final_local_offset():
    """Return (forward, lateral, distance) to the final selected coverage goal.

    coverage_goal_local_offset() uses the next route waypoint.  That is correct
    for steering, but for deciding whether a nearby uncleaned cluster should be
    covered now we need to know where the actual cluster/goal lies.
    """
    if coverage_goal_world is None:
        return None
    gx, gy = coverage_goal_world
    dx = gx - pose_x
    dy = gy - pose_y
    forward = math.cos(pose_theta) * dx + math.sin(pose_theta) * dy
    lateral = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
    dist = math.hypot(dx, dy)
    return forward, lateral, dist


def coverage_goal_is_ahead(lateral_tol=TARGET_AHEAD_LATERAL_TOL_M, min_forward=TARGET_AHEAD_MIN_FORWARD_M):
    """True if driving straight over the current strip can actually reach the goal."""
    local = coverage_goal_local_offset()
    if local is None:
        return False
    forward, lateral, dist = local
    # Explicit explore phase: route targets are allowed only if they lie on the
    # current strip.  The map is still being opened, so do not let detached
    # uncleaned islands pull the robot into diagonal/side-seeking behaviour.
    if navigation_phase == NavigationPhase.EXPLORE.value and coverage_goal_kind not in ("frontier", "under-surface"):
        return bool(forward > RESIDUAL_TARGET_AHEAD_FORWARD_M and abs(lateral) <= RESIDUAL_TARGET_AHEAD_LATERAL_TOL_M)
    if coverage_goal_kind == "under-surface":
        lateral_tol = min(lateral_tol, UNDER_SURFACE_DIRECT_LATERAL_TOL_M)
    return bool(forward > min_forward and abs(lateral) <= lateral_tol and dist > PLANNER_TARGET_REACHED_M)


def residual_cleanup_phase():
    return bool(last_coverage_percent >= RESIDUAL_CLEANUP_COVERAGE_PERCENT)


def known_map_residual_work_summary(cleanable=None, uncleaned=None):
    """Summarise remaining known-map work as connected residual components."""
    if cleanable is None or uncleaned is None:
        try:
            _obs, cleanable, _cleaned, uncleaned, _unknown = compute_coverage_masks()
        except Exception:
            return {"total": int(last_uncleaned_cells), "largest": int(last_uncleaned_cells), "big": 1}
    try:
        residual = (cleanable & uncleaned).astype(np.uint8)
        total = int(np.count_nonzero(residual))
        if total <= 0:
            return {"total": 0, "largest": 0, "big": 0}
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(residual, 8)
        largest = 0
        big = 0
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area > largest:
                largest = area
            if area >= int(KNOWN_MAP_EVAL_RESIDUAL_BIG_COMPONENT_CELLS):
                big += 1
        return {"total": total, "largest": int(largest), "big": int(big)}
    except Exception:
        return {"total": int(last_uncleaned_cells), "largest": int(last_uncleaned_cells), "big": 1}


def known_map_residual_cleanup_should_stop(now=None, allow_active_sweep=False, cleanable=None, uncleaned=None):
    """Return True when known-map mode should stop chasing residual islands.

    This is a mission-level policy, not a planner shortcut.  It prevents the
    robot from turning a decent full-room sweep into many tiny local cleanups.
    """
    global known_map_residual_policy_status
    if not known_map_coverage_eval_active() or not bool(KNOWN_MAP_EVAL_PRIMARY_SWEEP_ONCE):
        return False, "residual policy disabled"
    try:
        now = float(robot.getTime()) if now is None else float(now)
    except Exception:
        now = 0.0
    summary = known_map_residual_work_summary(cleanable, uncleaned)
    total = int(summary.get("total", last_uncleaned_cells))
    largest = int(summary.get("largest", total))
    big = int(summary.get("big", 0))
    base = (
        f"cov={last_coverage_percent:.1f}% primary={int(bool(known_map_primary_sweep_completed))} "
        f"mop={int(known_map_residual_cleanup_commits_started)}/{int(KNOWN_MAP_EVAL_RESIDUAL_CLEANUP_MAX_COMMITS)} "
        f"res={total} largest={largest} big={big}"
    )
    if last_coverage_percent >= float(KNOWN_MAP_EVAL_RETURN_AFTER_MOPUP_PERCENT):
        known_map_residual_policy_status = "residualPolicy=return high " + base
        return True, known_map_residual_policy_status
    primary_done = bool(known_map_primary_sweep_completed)
    if allow_active_sweep and route_commit_is_known_map_sweep() and bool(KNOWN_MAP_EVAL_EARLY_END_SWEEP_ENABLED):
        try:
            progress_frac = float(route_commit_progress_idx) / max(1.0, float(len(route_commit_route_map or [])))
        except Exception:
            progress_frac = 0.0
        small_residual = bool(
            total <= int(KNOWN_MAP_EVAL_RESIDUAL_TOTAL_FOR_RETURN_CELLS)
            and largest <= int(KNOWN_MAP_EVAL_RESIDUAL_LARGEST_COMPONENT_FOR_RETURN_CELLS)
        )
        if (
            progress_frac >= float(KNOWN_MAP_EVAL_EARLY_END_SWEEP_MIN_PROGRESS_FRAC)
            and last_coverage_percent >= float(KNOWN_MAP_EVAL_RETURN_AFTER_PRIMARY_SWEEP_PERCENT)
            and small_residual
        ):
            known_map_residual_policy_status = f"residualPolicy=cut active sweep p={progress_frac:.2f} " + base
            return True, known_map_residual_policy_status
    if not primary_done:
        known_map_residual_policy_status = "residualPolicy=primary " + base
        return False, known_map_residual_policy_status
    elapsed = max(0.0, now - float(known_map_primary_sweep_finish_time))
    small_residual = bool(
        total <= int(KNOWN_MAP_EVAL_RESIDUAL_TOTAL_FOR_RETURN_CELLS)
        and largest <= int(KNOWN_MAP_EVAL_RESIDUAL_LARGEST_COMPONENT_FOR_RETURN_CELLS)
    )
    budget_used = int(known_map_residual_cleanup_commits_started) >= int(KNOWN_MAP_EVAL_RESIDUAL_CLEANUP_MAX_COMMITS)
    timed_out = elapsed >= float(KNOWN_MAP_EVAL_RESIDUAL_CLEANUP_MAX_SEC)
    if last_coverage_percent >= float(KNOWN_MAP_EVAL_RETURN_AFTER_PRIMARY_SWEEP_PERCENT) and (small_residual or budget_used or timed_out):
        known_map_residual_policy_status = f"residualPolicy=return elapsed={elapsed:.0f}s " + base
        return True, known_map_residual_policy_status
    known_map_residual_policy_status = f"residualPolicy=mop elapsed={elapsed:.0f}s " + base
    return False, known_map_residual_policy_status


def refresh_navigation_phase():
    """Update the explicit high-level phase used by planner/debug/motion gates."""
    global navigation_phase, last_navigation_phase_reason
    try:
        sim_t = robot.getTime()
    except Exception:
        sim_t = 0.0
    navigation_phase, last_navigation_phase_reason = _update_navigation_phase(
        nav_state=nav_state,
        coverage_percent=last_coverage_percent,
        frontier_cells=last_frontier_cells,
        uncleaned_cells=last_uncleaned_cells,
        sim_time=sim_t,
        route_kind=coverage_route_kind,
        explore_percent=EARLY_EXPLORATION_COVERAGE_PERCENT,
        finish_percent=RESIDUAL_CLEANUP_COVERAGE_PERCENT,
    )
    if dock_return_active or route_commit_kind == "dock" or (dock_return_completed and not auto_map_cleaning_started):
        navigation_phase = NavigationPhase.RETURN_HOME.value
        last_navigation_phase_reason = dock_return_status[:80]
    elif auto_map_cleaning_started and known_map_coverage_eval_active():
        navigation_phase = NavigationPhase.PLANNED_COVERAGE.value
        last_navigation_phase_reason = f"learned-map K cleaning: {learned_map_sanitize_debug[:52]}"
    elif route_commit_active:
        if route_commit_kind == "frontier":
            navigation_phase = NavigationPhase.EXPLORE.value
            last_navigation_phase_reason = f"frontier route commit active: {route_commit_reason[:54]}"
        else:
            navigation_phase = NavigationPhase.PLANNED_COVERAGE.value
            last_navigation_phase_reason = f"route commit active: {route_commit_reason[:54]}"
    elif SIMPLE_SWEEP_FSM_ENABLED and not simple_sweep_completed and not known_map_coverage_eval_active():
        navigation_phase = NavigationPhase.EXPLORE.value
        last_navigation_phase_reason = f"EXPLORE_SWEEP_FSM map-discovery lane={simple_sweep_lane_index} cov={last_coverage_percent:.1f}%"
    elif planner_expand_map_motion_active():
        navigation_phase = NavigationPhase.EXPLORE.value
        last_navigation_phase_reason = f"planner-intent EXPAND_MAP: {planner_intent_reason[:54]}"
    return navigation_phase


def strict_grid_heading_lock_active():
    """True when ordinary motion must be straight-or-pivot, not curved."""
    if not STRICT_GRID_HEADING_LOCK_ENABLED:
        return False
    if nav_state != NAV_FORWARD:
        return False
    if route_commit_active:
        return False
    if under_furniture_active:
        return False
    low = str(coverage_status or "").lower()
    if any(key in low for key in STRICT_ARC_ALLOWED_STATUS_KEYWORDS):
        return False
    return navigation_phase in (NavigationPhase.EXPLORE.value, NavigationPhase.COVERAGE.value, NavigationPhase.FINISH_CLEANUP.value)


def coverage_target_replan_allowed(for_side_bias=False):
    """Gate global residual targets so they do not cause ping-pong cleanup.

    Mid-run behavior should be strip coverage.  Detached uncleaned islands are
    valid objectives, but only in the late cleanup phase or when they lie in the
    same forward corridor.  Otherwise the robot keeps crossing cleaned space and
    turns left/right around the same separated fragments.
    """
    if coverage_goal_world is None or coverage_goal_kind in ("none", "done"):
        return False
    if map_building_active():
        return False
    if coverage_goal_kind == "frontier":
        return True
    if coverage_goal_kind == "under-surface":
        return coverage_goal_is_ahead(lateral_tol=UNDER_SURFACE_DIRECT_LATERAL_TOL_M, min_forward=0.10)
    local = coverage_goal_local_offset()
    if local is None:
        return False
    forward, lateral, dist = local
    # If the target is actually in the current strip, it is not a detached island.
    if forward > RESIDUAL_TARGET_AHEAD_FORWARD_M and abs(lateral) <= RESIDUAL_TARGET_AHEAD_LATERAL_TOL_M:
        return True
    if dist <= RESIDUAL_TARGET_NEAR_DIRECT_M and forward > -0.05 and abs(lateral) <= RESIDUAL_TARGET_AHEAD_LATERAL_TOL_M:
        return True
    phase_limit = RESIDUAL_ROUTE_SIDE_BIAS_PERCENT if for_side_bias else RESIDUAL_REPLAN_COVERAGE_PERCENT
    return bool(last_coverage_percent >= phase_limit)


def edge_line_acquire_candidate():
    """Return (forward, lateral, dist) when a nearby side strip is worth merging into.

    This is deliberately narrower than global residual cleanup: it only accepts a
    strip that is already close to the robot and can be entered from the current
    pose without a full 90-degree lane-change. It prevents the top/right border
    behaviour where the robot makes small dead-end excursions beside an
    uncleaned line instead of gently joining that line.
    """
    if not EDGE_LINE_ACQUIRE_ENABLED:
        return None
    if not optional_planner_intercepts_allowed("line_acquire"):
        return None
    if coverage_goal_kind != "uncleaned" or coverage_goal_world is None:
        return None
    if last_coverage_percent < EDGE_LINE_ACQUIRE_MIN_COVERAGE_PERCENT or last_coverage_percent > EDGE_LINE_ACQUIRE_MAX_COVERAGE_PERCENT:
        return None
    local = coverage_goal_local_offset()
    if local is None:
        return None
    forward, lateral, dist = local
    if dist > EDGE_LINE_ACQUIRE_MAX_DIST_M:
        return None
    if forward < EDGE_LINE_ACQUIRE_MIN_FORWARD_M or forward > EDGE_LINE_ACQUIRE_MAX_FORWARD_M:
        return None
    if abs(lateral) < EDGE_LINE_ACQUIRE_MIN_LATERAL_M or abs(lateral) > EDGE_LINE_ACQUIRE_MAX_LATERAL_M:
        return None
    if coverage_route_kind not in ("uncleaned", "none"):
        return None
    if math.isfinite(coverage_route_cost) and coverage_route_cost > EDGE_LINE_ACQUIRE_MAX_ROUTE_COST_M:
        return None
    return forward, lateral, dist


def edge_line_acquire_speeds(front, center, upper_front, left, right, body_clearance):
    """Cautiously steer into a nearby reachable cleaning line.

    This is not the ordinary lane shift. A full lane shift must stay straight and
    grid-like. This controller is used when the planner has a close side strip
    that is reachable if the robot gently merges into it rather than treating the
    side strip as a residual island or starting another row-end turn.
    """
    global row_end_candidate_count, coverage_status, last_turn_variant
    global last_line_acquire_time, last_line_acquire_side, last_line_acquire_reason

    if nav_state != NAV_FORWARD or under_furniture_active:
        return None
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return None
    now = robot.getTime()
    if now < last_contact_route_kill_until or now - last_contact_trap_time < 1.4:
        return None
    cand = edge_line_acquire_candidate()
    if cand is None:
        return None
    forward, lateral, dist = cand

    # Do not "bravely" steer into a real front block. The merge is allowed only
    # when the front corridor and the target side are physically open.
    if front < EDGE_LINE_ACQUIRE_MIN_FRONT_M or center < SAFE_FRONT_DISTANCE:
        return None
    if body_clearance < BODY_CORRIDOR_PASS_CLEARANCE and upper_front < ROW_END_CONFIRM_DISTANCE + 0.18:
        return None

    side = 1.0 if lateral > 0.0 else -1.0
    side_clear = left if side > 0.0 else right
    if side_clear < EDGE_LINE_ACQUIRE_MIN_SIDE_CLEAR_M:
        return None

    # If a merge has just started, keep the same side briefly. This avoids a new
    # left/right decision every frame when the target point jumps along the same
    # strip.
    if now - last_line_acquire_time < EDGE_LINE_ACQUIRE_LOCK_SEC and last_line_acquire_side != 0.0:
        if side != last_line_acquire_side and abs(lateral) > EDGE_LINE_ACQUIRE_MIN_LATERAL_M * 1.4:
            return None
        side = last_line_acquire_side

    yaw = clamp(lateral * EDGE_LINE_ACQUIRE_YAW_KP, -EDGE_LINE_ACQUIRE_MAX_YAW, EDGE_LINE_ACQUIRE_MAX_YAW)
    base = EDGE_LINE_ACQUIRE_SPEED
    if front < ROW_END_CONFIRM_DISTANCE + 0.14 or side_clear < EDGE_LINE_ACQUIRE_MIN_SIDE_CLEAR_M + 0.10:
        base = EDGE_LINE_ACQUIRE_SLOW_SPEED
        yaw = clamp(yaw, -EDGE_LINE_ACQUIRE_MAX_YAW * 0.72, EDGE_LINE_ACQUIRE_MAX_YAW * 0.72)

    last_line_acquire_time = now
    last_line_acquire_side = side
    last_line_acquire_reason = f"f={forward:.2f} lat={lateral:.2f} d={dist:.2f} sideClear={side_clear:.2f}"
    last_turn_variant = "line-acquire"
    row_end_candidate_count = 0
    coverage_status = "line acquire " + last_line_acquire_reason
    return base - yaw, base + yaw


def cleaned_corridor_should_replan(cleaned_ahead, uncleaned_ahead, cleaned_threshold, uncleaned_threshold):
    """Return True when the robot is cleaning a strip that is already mostly done.

    This must not forbid transit through cleaned cells. It only asks for a
    lane/target change when there are still uncleaned cells somewhere, the
    current forward corridor is mostly cleaned, and the active target is not
    ahead in that same corridor.
    """
    if last_uncleaned_cells <= 250 and last_frontier_cells <= 150:
        return False
    if coverage_goal_is_ahead():
        return False
    return bool(cleaned_ahead >= cleaned_threshold and uncleaned_ahead <= uncleaned_threshold)

def side_shift_path_is_clear(side, distance, obstacles):
    """Check the short lateral strip used by a local pocket-fill maneuver.

    The robot will first pivot 90 degrees and then drive approximately
    `distance` sideways relative to the original row. Before doing that, make
    sure the swept strip is not already marked as occupied/contact-confirmed.
    """
    side = 1.0 if side >= 0 else -1.0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta) * side
    ny = math.cos(pose_theta) * side
    half = LOCAL_POCKET_CLEAR_HALF_WIDTH_M
    steps = max(4, int(distance / 0.07))
    for i in range(1, steps + 1):
        lateral = distance * i / steps
        for cross in (-half, 0.0, half):
            # after pivoting, cross-width lies roughly along the old row axis
            wx = pose_x + nx * lateral + fx * cross
            wy = pose_y + ny * lateral + fy * cross
            mx, my = world_to_map(wx, wy)
            if not map_inside(mx, my):
                return False
            if obstacles[my, mx]:
                return False
    return True


def shifted_footprint_corridor_score(side, offset, obstacles, cleanable, uncleaned, under_uncleaned):
    """Evaluate whether a parallel cleaning lane fits through the same opening.

    This is the practical "1:1 robot on the map" test for apertures.  A side
    offset is useful only if the full circular footprint can be swept forward
    along that offset and there are still uncleaned/under-surface cells around
    it.  It prevents the old behaviour: one centre-line pass through a wide
    chair/table opening while one side remains blue/yellow.
    """
    side = 1.0 if side >= 0 else -1.0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta) * side
    ny = math.cos(pose_theta) * side

    total = 0
    blocked = 0
    useful = 0
    under_useful = 0
    a = PASSAGE_MULTILANE_FORWARD_MIN_M
    while a <= PASSAGE_MULTILANE_FORWARD_MAX_M + 1e-6:
        wx = pose_x + fx * a + nx * offset
        wy = pose_y + fy * a + ny * offset
        mx, my = world_to_map(wx, wy)
        a += PASSAGE_MULTILANE_FORWARD_STEP_M
        if not map_inside(mx, my):
            blocked += 1
            total += 1
            continue
        total += 1
        ok, _raw = footprint_fits_at(mx, my, obstacles, cleanable, margin_m=FOOTPRINT_PASS_MARGIN_M)
        if not ok:
            blocked += 1
            continue

        # Count useful cells in a small swath around the shifted centre.  The
        # robot is circular, so a sample only on the centre line would miss the
        # actual cleaning width.
        r = max(2, int(COVERAGE_RADIUS_M * MAP_SCALE * 0.70))
        x0 = max(0, mx - r)
        x1 = min(MAP_SIZE, mx + r + 1)
        y0 = max(0, my - r)
        y1 = min(MAP_SIZE, my + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi_uncleaned = uncleaned[y0:y1, x0:x1]
        roi_under = under_uncleaned[y0:y1, x0:x1]
        # Weight under-surface cells slightly higher: the feature is meant for
        # chair/table openings, but ordinary uncleaned cells still count.
        ordinary = int(np.count_nonzero(roi_uncleaned))
        under_n = int(np.count_nonzero(roi_under))
        useful += ordinary
        under_useful += under_n

    if total <= 0:
        return None
    blocked_ratio = blocked / total
    score = useful + under_useful * 1.6 - blocked * 20.0
    return score, useful, under_useful, blocked_ratio


def score_parallel_aperture_lane(side):
    """Find the best parallel lane through the current furniture/opening gap."""
    obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
    under_uncleaned = under_surface_mask_from_obstacles(obstacles) & cleanable & uncleaned

    best = None
    offset = PASSAGE_MULTILANE_MIN_SIDE_OFFSET_M
    while offset <= PASSAGE_MULTILANE_MAX_SIDE_OFFSET_M + 1e-6:
        result = shifted_footprint_corridor_score(side, offset, obstacles, cleanable, uncleaned, under_uncleaned)
        if result is not None:
            score, useful, under_useful, blocked_ratio = result
            if blocked_ratio <= PASSAGE_MULTILANE_MAX_BLOCKED_RATIO and useful >= PASSAGE_MULTILANE_MIN_UNCLEANED_CELLS:
                if best is None or score > best[0]:
                    best = (score, offset, useful, under_useful, blocked_ratio)
        offset += PASSAGE_MULTILANE_STEP_M
    return best


def begin_parallel_aperture_lane(side, distance, reason, front, body_clearance):
    """Shift to a second parallel pass through the same opening.

    Sequence: turn 90 toward the side -> drive one offset -> turn back.  Unlike
    the normal lawnmower lane change, this does not reverse the row direction;
    it creates another line through the same passable aperture.
    """
    global nav_action_queue, last_parallel_aperture_time, last_parallel_aperture_count
    global last_parallel_aperture_side, last_parallel_aperture_x, last_parallel_aperture_y
    global coverage_status, row_end_candidate_count
    side = 1.0 if side >= 0 else -1.0
    distance = clamp(distance, PASSAGE_MULTILANE_MIN_SIDE_OFFSET_M, PASSAGE_MULTILANE_MAX_SIDE_OFFSET_M)
    nav_action_queue = [("SHIFT", distance), ("TURN", -side, "return heading after aperture second lane")]
    last_parallel_aperture_time = robot.getTime()
    last_parallel_aperture_count += 1
    last_parallel_aperture_side = side
    last_parallel_aperture_x = pose_x
    last_parallel_aperture_y = pose_y
    row_end_candidate_count = 0
    coverage_status = f"aperture parallel lane {'L' if side > 0 else 'R'} {distance:.2f}m: {reason}"
    if pivot_clearance_problem(front, body_clearance):
        start_pre_pivot_backup(side, f"aperture parallel clearance F={front:.2f} body={body_clearance:.2f}")
    else:
        begin_pivot_turn(side, coverage_status)


def maybe_start_parallel_aperture_pass(front, center, upper_front, left, right, body_clearance):
    """Opportunistically add a second line through a wide furniture opening.

    This is intentionally NOT a general side-target planner.  It is allowed only
    after early exploration, only after the current strip has become stable, and
    only when the shifted lane contains real under-surface work.  Otherwise it
    produces exactly the bad behaviour visible in the demo: repeated 90-degree
    turns around one point while ordinary uncleaned cells keep pulling left/right.
    """
    global last_parallel_aperture_score, last_parallel_aperture_blocked_ratio, last_parallel_aperture_count
    if not PASSAGE_MULTILANE_ENABLED:
        return False
    if nav_state != NAV_FORWARD or not under_furniture_active or nav_action_queue:
        return False
    now = robot.getTime()
    if last_coverage_percent < PASSAGE_MULTILANE_MIN_COVERAGE_PERCENT:
        return False
    if row_distance_from_start() < PASSAGE_MULTILANE_MIN_ROW_PROGRESS_M:
        return False
    if post_lane_forward_lock_active(front, center, body_clearance):
        return False
    if now - last_parallel_aperture_time < PASSAGE_MULTILANE_COOLDOWN_SEC:
        return False
    last_ap_dist = math.hypot(pose_x - last_parallel_aperture_x, pose_y - last_parallel_aperture_y)
    if (now - last_parallel_aperture_time < PASSAGE_MULTILANE_SPATIAL_COOLDOWN_SEC
            and last_ap_dist < PASSAGE_MULTILANE_SPATIAL_COOLDOWN_M):
        return False
    if last_parallel_aperture_count >= PASSAGE_MULTILANE_MAX_PER_OPENING:
        return False
    # Do not schedule a second line in a generic open room.  It must still look
    # like the same furniture aperture, not merely an empty corridor.
    if not detect_under_furniture_corridor(front, center, upper_front, left, right):
        return False
    if body_clearance < BODY_CORRIDOR_HARD_CLEARANCE or front < ROW_END_CONFIRM_DISTANCE:
        return False

    left_best = score_parallel_aperture_lane(1.0)
    right_best = score_parallel_aperture_lane(-1.0)
    candidates = []
    if left_best is not None and left_best[3] >= PASSAGE_MULTILANE_MIN_UNDER_USEFUL_CELLS:
        candidates.append((1.0,) + left_best)
    if right_best is not None and right_best[3] >= PASSAGE_MULTILANE_MIN_UNDER_USEFUL_CELLS:
        candidates.append((-1.0,) + right_best)
    if not candidates:
        last_parallel_aperture_score = 0
        last_parallel_aperture_blocked_ratio = 1.0
        return False
    # Prefer the lane with more useful work, but avoid reversing immediately to
    # the same side if both sides are similar.
    candidates.sort(key=lambda c: c[1], reverse=True)
    side, score, offset, useful, under_useful, blocked_ratio = candidates[0]
    last_parallel_aperture_score = int(score)
    last_parallel_aperture_blocked_ratio = float(blocked_ratio)
    begin_parallel_aperture_lane(
        side,
        offset,
        f"useful={useful} under={under_useful} block={blocked_ratio:.2f}",
        front,
        body_clearance,
    )
    return True


def side_outer_uncleaned_ratio(side, cleanable, uncleaned, obstacles):
    """Return how much uncleaned space continues beyond the pocket window.

    A real local pocket is a small bounded missed area near the robot. If the
    same side remains broadly uncleaned farther away, it is not a pocket; it is
    an open region that should be handled by the normal row/lane planner.
    """
    side = 1.0 if side >= 0 else -1.0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta) * side
    ny = math.cos(pose_theta) * side
    total = 0
    open_uncleaned = 0
    a = 0.0
    while a <= LOCAL_POCKET_SCAN_AHEAD_M + 1e-6:
        lateral = LOCAL_POCKET_OUTER_SIDE_MIN_M
        while lateral <= LOCAL_POCKET_OUTER_SIDE_MAX_M + 1e-6:
            wx = pose_x + fx * a + nx * lateral
            wy = pose_y + fy * a + ny * lateral
            mx, my = world_to_map(wx, wy)
            lateral += LOCAL_POCKET_SIDE_STEP_M
            if not map_inside(mx, my):
                continue
            if obstacles[my, mx]:
                continue
            if cleanable[my, mx]:
                total += 1
                if uncleaned[my, mx]:
                    open_uncleaned += 1
        a += LOCAL_POCKET_AHEAD_STEP_M
    return (open_uncleaned / total) if total > 0 else 0.0


def score_local_side_pocket(side):
    """Score a nearby uncleaned side pocket that is cheaper to fill now.

    This is a local correction, not a general side-target selector. It must not
    fire on a wide empty room/fan-shaped free area, because then the robot turns
    away from a productive row too early.
    """
    global last_local_pocket_score, last_local_pocket_uncleaned, last_local_pocket_obstacle_ratio
    global last_local_pocket_outer_uncleaned_ratio
    obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
    side = 1.0 if side >= 0 else -1.0
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    nx = -math.sin(pose_theta) * side
    ny = math.cos(pose_theta) * side

    total = 0
    uncleaned_count = 0
    cleaned_count = 0
    obstacle_count = 0
    best_lateral = LOCAL_POCKET_SHIFT_MIN_M
    weighted_lateral = 0.0
    weighted_cells = 0.0

    a = 0.0
    while a <= LOCAL_POCKET_SCAN_AHEAD_M + 1e-6:
        lateral = LOCAL_POCKET_SIDE_MIN_M
        while lateral <= LOCAL_POCKET_SIDE_MAX_M + 1e-6:
            sample_lateral = lateral
            wx = pose_x + fx * a + nx * sample_lateral
            wy = pose_y + fy * a + ny * sample_lateral
            mx, my = world_to_map(wx, wy)
            lateral += LOCAL_POCKET_SIDE_STEP_M
            if not map_inside(mx, my):
                continue
            total += 1
            if obstacles[my, mx]:
                obstacle_count += 1
                continue
            if uncleaned[my, mx]:
                uncleaned_count += 1
                # Prefer closer side pockets and cells near the current pose;
                # this is why the robot should paint a nearby corner before
                # continuing a long under-furniture strip.
                w = 1.0 + max(0.0, LOCAL_POCKET_SCAN_AHEAD_M - a) * 0.55 + max(0.0, LOCAL_POCKET_SIDE_MAX_M - sample_lateral) * 0.35
                weighted_lateral += sample_lateral * w
                weighted_cells += w
            elif cleaned[my, mx]:
                cleaned_count += 1
        a += LOCAL_POCKET_AHEAD_STEP_M

    if total <= 0:
        return None
    obstacle_ratio = obstacle_count / total
    if uncleaned_count < LOCAL_POCKET_MIN_UNCLEANED_CELLS or obstacle_ratio > LOCAL_POCKET_MAX_OBSTACLE_RATIO:
        return None

    # Important guard: if uncleaned cells continue outside the short side window,
    # this is not a corner/pocket. It is normal open floor and should be covered
    # by the row/lane planner, otherwise the robot makes the false early 90 deg
    # turn visible on the screenshot.
    outer_ratio = side_outer_uncleaned_ratio(side, cleanable, uncleaned, obstacles)
    last_local_pocket_outer_uncleaned_ratio = outer_ratio
    if outer_ratio > LOCAL_POCKET_MAX_OUTER_UNCLEANED_RATIO:
        return None

    if weighted_cells > 0.0:
        best_lateral = weighted_lateral / weighted_cells
    distance = clamp(best_lateral + LOCAL_POCKET_SHIFT_EXTRA_M, LOCAL_POCKET_SHIFT_MIN_M, LOCAL_POCKET_SHIFT_MAX_M)
    if not side_shift_path_is_clear(side, distance, obstacles):
        return None

    # Score favors dense local uncleaned pockets, penalizes obstacles and adds a
    # bonus when under-furniture mode would otherwise keep the robot driving past
    # the pocket.
    score = (uncleaned_count * 1.0) - (obstacle_count * 2.2) - (cleaned_count * 0.12)
    if under_furniture_active:
        score += LOCAL_POCKET_UNDER_FURNITURE_BONUS
    last_local_pocket_score = score
    last_local_pocket_uncleaned = uncleaned_count
    last_local_pocket_obstacle_ratio = obstacle_ratio
    if score < LOCAL_POCKET_MIN_SCORE:
        return None
    return score, distance, uncleaned_count, obstacle_ratio


def find_local_side_pocket(left_dist, right_dist):
    """Return (side, distance, score, cells, obstacle_ratio) or None."""
    if not LOCAL_POCKET_FILL_ENABLED:
        return None

    candidates = []
    left = score_local_side_pocket(1.0)
    if left is not None and left_dist > SIDE_DISTANCE + 0.05:
        candidates.append((1.0, left))
    right = score_local_side_pocket(-1.0)
    if right is not None and right_dist > SIDE_DISTANCE + 0.05:
        candidates.append((-1.0, right))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1][0], reverse=True)
    best_side, best = candidates[0]
    if len(candidates) > 1 and best[0] < candidates[1][1][0] + LOCAL_POCKET_SCORE_MARGIN:
        # If both sides are similar, do not add a jittery opportunistic detour.
        return None
    score, distance, cells, obstacle_ratio = best
    return best_side, distance, score, cells, obstacle_ratio


def begin_side_pocket_fill_checked(side, distance, reason, front, body_clearance):
    """Turn into a nearby uncleaned pocket, shift over it, then return to row.

    Unlike a normal lawnmower lane change, this does not flip `lane_side` and
    does not begin a new row. It is a short detour for local uncovered corners.
    """
    global nav_action_queue, coverage_status, last_local_pocket_fill_time
    global under_furniture_until, under_furniture_active, under_furniture_suppressed_until
    side = 1.0 if side >= 0 else -1.0
    now = robot.getTime()
    distance = clamp(distance, LOCAL_POCKET_SHIFT_MIN_M, LOCAL_POCKET_SHIFT_MAX_M)
    nav_action_queue = [("SHIFT", distance), ("TURN", -side, "return after local pocket fill")]
    last_local_pocket_fill_time = now
    # Do not immediately re-enter the same under-furniture straight hold; that
    # is exactly what made the robot drive past the nearby corner.
    under_furniture_until = 0.0
    under_furniture_active = False
    under_furniture_suppressed_until = max(under_furniture_suppressed_until, now + 1.2)
    coverage_status = f"local pocket fill {'L' if side > 0 else 'R'} {distance:.2f}m"
    if pivot_clearance_problem(front, body_clearance):
        start_pre_pivot_backup(side, f"{reason}; local pocket clearance F={front:.2f} body={body_clearance:.2f}")
    else:
        begin_pivot_turn(side, reason)


def maybe_start_local_pocket_fill(left_dist, right_dist, front, body_clearance):
    """Start a local side-pocket fill if it is more efficient than continuing."""
    global last_local_pocket_side, last_revisit_lane_change_time, coverage_status
    if nav_state != NAV_FORWARD:
        return False
    if not optional_planner_intercepts_allowed("local_pocket"):
        return False
    now = robot.getTime()
    if EARLY_EXPLORATION_DISABLE_TARGET_REPLAN and last_coverage_percent < EARLY_EXPLORATION_LOCAL_DETOURS_PERCENT:
        return False
    if lane_action_cooldown_active(front):
        return False
    if now - last_local_pocket_fill_time < LOCAL_POCKET_COOLDOWN_SEC:
        return False
    if now < leg_pass_grace_until:
        return False
    if front < ROW_END_CONFIRM_DISTANCE or body_clearance < BODY_CORRIDOR_HARD_CLEARANCE:
        return False
    if row_commit_active(front, body_clearance):
        coverage_status = f"row commit: keep mapping strip {row_distance_from_start():.2f}m"
        return False

    cleaned_ahead, uncleaned_ahead, _ = coverage_corridor_ahead()
    # Do not interrupt a useful open row. The previous version used
    # "target is not directly ahead" as a trigger; that was too aggressive and
    # made the robot turn into empty space. A side detour is allowed only after
    # the current strip is mostly cleaned or has little useful uncleaned floor
    # ahead. This keeps local pocket filling subordinate to efficient coverage.
    row_has_progress = row_distance_from_start() >= LOCAL_POCKET_MIN_ROW_PROGRESS_M
    forward_row_still_useful = uncleaned_ahead > LOCAL_POCKET_MAX_AHEAD_UNCLEANED_RATIO
    current_strip_mostly_done = cleaned_ahead >= LOCAL_POCKET_MIN_CLEANED_AHEAD_RATIO
    should_consider = row_has_progress and current_strip_mostly_done and not forward_row_still_useful
    if not should_consider:
        if under_furniture_active and forward_row_still_useful:
            coverage_status = f"continue useful row; postpone pocket clean={cleaned_ahead:.2f} unclean={uncleaned_ahead:.2f}"
        return False

    pocket = find_local_side_pocket(left_dist, right_dist)
    if pocket is None:
        return False
    side, distance, score, cells, obstacle_ratio = pocket
    last_local_pocket_side = side
    last_revisit_lane_change_time = now
    begin_side_pocket_fill_checked(
        side,
        distance,
        f"local bounded pocket cells={cells} score={score:.1f} obs={obstacle_ratio:.2f} outer={last_local_pocket_outer_uncleaned_ratio:.2f} ahead={cleaned_ahead:.2f}/{uncleaned_ahead:.2f}",
        front,
        body_clearance,
    )
    return True


def score_side_strip_cleanup(side):
    """Score a coherent uncleaned strip one lane to the robot side.

    This is not a global planner and not per-cell greed. It asks a local CPP
    question: "if I spend one normal lane change now, will I cover a nearby strip
    that would otherwise require a long return?"  Positive side is robot-left.
    """
    try:
        obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
    except Exception:
        return None

    side = 1.0 if side >= 0.0 else -1.0
    heading = desired_grid_heading
    fx = math.cos(heading)
    fy = math.sin(heading)
    nx = -math.sin(heading)
    ny = math.cos(heading)

    total = 0
    uncleaned_n = 0
    cleaned_n = 0
    obstacle_n = 0
    unknown_n = 0
    wall_fringe_n = 0
    best_forward = 0.0

    a = -SIDE_STRIP_CLEANUP_SCAN_BACK_M
    while a <= SIDE_STRIP_CLEANUP_SCAN_FORWARD_M + 1e-9:
        l0 = side * SIDE_STRIP_CLEANUP_LANE_OFFSET_M
        l = l0 - side * SIDE_STRIP_CLEANUP_HALF_WIDTH_M
        # Walk from lane inner edge to lane outer edge.  This makes the scan
        # robust to slight odometry/heading error and to wall-fringe width.
        while (side > 0.0 and l <= l0 + SIDE_STRIP_CLEANUP_HALF_WIDTH_M + 1e-9) or (side < 0.0 and l >= l0 - SIDE_STRIP_CLEANUP_HALF_WIDTH_M - 1e-9):
            wx = pose_x + fx * a + nx * l
            wy = pose_y + fy * a + ny * l
            mx, my = world_to_map(wx, wy)
            if map_inside(mx, my):
                total += 1
                if obstacles[my, mx]:
                    obstacle_n += 1
                elif uncleaned[my, mx] and cleanable[my, mx]:
                    uncleaned_n += 1
                    best_forward = max(best_forward, a)
                    # A small bonus for wall-adjacent residual strips.  In the
                    # map these are exactly the long thin missed areas the user
                    # points at: expensive to revisit if ignored now.
                    x0 = max(0, mx - 2); x1 = min(MAP_SIZE, mx + 3)
                    y0 = max(0, my - 2); y1 = min(MAP_SIZE, my + 3)
                    if np.any(obstacles[y0:y1, x0:x1]):
                        wall_fringe_n += 1
                elif cleaned[my, mx]:
                    cleaned_n += 1
                elif unknown[my, mx]:
                    unknown_n += 1
            l += side * SIDE_STRIP_CLEANUP_STEP_M
        a += SIDE_STRIP_CLEANUP_STEP_M

    if total <= 0:
        return None
    obstacle_ratio = obstacle_n / total
    if obstacle_ratio > SIDE_STRIP_CLEANUP_MAX_OBSTACLE_RATIO:
        return None
    if uncleaned_n < SIDE_STRIP_CLEANUP_MIN_UNCLEANED_CELLS:
        return None

    route_bonus = 0.0
    final_local = coverage_goal_final_local_offset()
    if final_local is not None and coverage_goal_kind == "uncleaned":
        forward, lateral, dist = final_local
        if (lateral * side) > 0.12 and dist < 1.35:
            route_bonus = SIDE_STRIP_CLEANUP_ROUTE_BONUS

    wall_bonus = SIDE_STRIP_CLEANUP_WALL_FRINGE_BONUS if wall_fringe_n >= max(4, uncleaned_n * 0.18) else 0.0
    score = float(uncleaned_n) + route_bonus + wall_bonus - 0.35 * cleaned_n - 2.5 * obstacle_n - 0.08 * unknown_n
    return {
        "score": score,
        "cells": int(uncleaned_n),
        "total": int(total),
        "obstacle_ratio": obstacle_ratio,
        "cleaned": int(cleaned_n),
        "unknown": int(unknown_n),
        "wall": int(wall_fringe_n),
        "best_forward": float(best_forward),
        "route_bonus": float(route_bonus),
        "wall_bonus": float(wall_bonus),
    }


def maybe_start_side_strip_cleanup(front, center, left_dist, right_dist, body_clearance):
    """Branch to a nearby uncleaned side strip at a safe decision point.

    This is the corrected version of the "clean the nearby square now" idea:
    not per-frame target chasing, but a local cost/benefit branch that still uses
    the same square-grid movement contract as the stable coverage rows.
    """
    global last_side_strip_cleanup_time, last_side_strip_cleanup_reason
    global last_revisit_lane_change_time, row_end_candidate_count, coverage_status

    if not SIDE_STRIP_CLEANUP_ENABLED:
        return False
    if nav_state != NAV_FORWARD or under_furniture_active or nav_action_queue:
        return False
    if navigation_phase != NavigationPhase.COVERAGE.value:
        return False
    if last_coverage_percent < SIDE_STRIP_CLEANUP_MIN_COVERAGE_PERCENT or last_coverage_percent > SIDE_STRIP_CLEANUP_MAX_COVERAGE_PERCENT:
        return False
    if not optional_planner_intercepts_allowed("side_strip_cleanup"):
        return False
    now = robot.getTime()
    if now - last_side_strip_cleanup_time < SIDE_STRIP_CLEANUP_COOLDOWN_SEC:
        return False
    if now - last_revisit_lane_change_time < REVISIT_FORCE_COOLDOWN_SEC:
        return False
    if lane_action_cooldown_active(front):
        return False
    if row_commit_active(front, body_clearance):
        return False
    if row_distance_from_start() < SIDE_STRIP_CLEANUP_MIN_ROW_PROGRESS_M:
        return False
    if front <= ROW_END_CONFIRM_DISTANCE or center <= SAFE_FRONT_DISTANCE * 0.88:
        return False

    cleaned_ahead, uncleaned_ahead, _ = coverage_corridor_ahead()
    current_row_still_useful = (
        uncleaned_ahead > SIDE_STRIP_CLEANUP_CURRENT_ROW_USEFUL_UNCLEANED
        and cleaned_ahead < SIDE_STRIP_CLEANUP_CURRENT_ROW_CLEANED_MAX
    )
    if current_row_still_useful:
        return False

    candidates = []
    for side, side_clear in ((1.0, left_dist), (-1.0, right_dist)):
        if side_clear < SIDE_STRIP_CLEANUP_SIDE_CLEAR_M:
            continue
        stats = score_side_strip_cleanup(side)
        if stats is None:
            continue
        if stats["score"] < SIDE_STRIP_CLEANUP_MIN_SCORE:
            continue
        candidates.append((stats["score"], side, stats))

    if not candidates:
        return False
    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, side, stats = candidates[0]
    if len(candidates) > 1 and best_score < candidates[1][0] + SIDE_STRIP_CLEANUP_SCORE_MARGIN:
        return False

    last_side_strip_cleanup_time = now
    last_revisit_lane_change_time = now
    last_side_strip_cleanup_reason = (
        f"side-strip cleanup {'L' if side > 0 else 'R'} score={best_score:.1f} "
        f"cells={stats['cells']} wall={stats['wall']} obs={stats['obstacle_ratio']:.2f} "
        f"ahead={cleaned_ahead:.2f}/{uncleaned_ahead:.2f} rb={stats['route_bonus']:.0f}"
    )
    coverage_status = last_side_strip_cleanup_reason
    begin_lawnmower_lane_change_checked(side, last_side_strip_cleanup_reason, front, body_clearance)
    row_end_candidate_count = 0
    return True

def maybe_start_nearby_route_cleanup(front, center, left_dist, right_dist, body_clearance):
    """Use the wavefront/Dijkstra target to interrupt a row only when local.

    This is the controlled version of the user's "clean the nearest square now"
    idea.  The planner is already Dijkstra-like; the missing piece was that the
    row executor treated the route as debug/side-bias and kept driving forward.

    We intentionally do NOT start arbitrary mid-row steering.  If the target is
    close and lateral, we start the same grid-safe 90 -> shift -> 90 lane-change
    used by the stable boustrophedon coverage.
    """
    global last_nearby_route_cleanup_time, last_nearby_route_cleanup_reason, last_side_strip_cleanup_time, last_side_strip_cleanup_reason
    global last_revisit_lane_change_time, row_end_candidate_count, coverage_status

    if not NEARBY_ROUTE_CLEANUP_ENABLED:
        return False
    if nav_state != NAV_FORWARD or under_furniture_active or nav_action_queue:
        return False
    if navigation_phase != NavigationPhase.COVERAGE.value:
        return False
    if coverage_goal_kind != "uncleaned" or coverage_route_kind != "uncleaned":
        return False
    if last_coverage_percent < NEARBY_ROUTE_CLEANUP_MIN_COVERAGE_PERCENT or last_coverage_percent > NEARBY_ROUTE_CLEANUP_MAX_COVERAGE_PERCENT:
        return False
    if not optional_planner_intercepts_allowed("nearby_route_cleanup"):
        return False
    now = robot.getTime()
    if now - last_nearby_route_cleanup_time < NEARBY_ROUTE_CLEANUP_COOLDOWN_SEC:
        return False
    if now - last_revisit_lane_change_time < REVISIT_FORCE_COOLDOWN_SEC:
        return False
    if lane_action_cooldown_active(front):
        return False
    if row_commit_active(front, body_clearance):
        return False
    if row_distance_from_start() < NEARBY_ROUTE_CLEANUP_MIN_ROW_PROGRESS_M:
        return False
    if front <= ROW_END_CONFIRM_DISTANCE or center <= SAFE_FRONT_DISTANCE * 0.92:
        return False

    final_local = coverage_goal_final_local_offset()
    if final_local is None:
        return False
    forward, lateral, dist = final_local
    if not math.isfinite(coverage_route_cost) or coverage_route_cost > NEARBY_ROUTE_CLEANUP_MAX_ROUTE_COST_M:
        return False
    if dist > NEARBY_ROUTE_CLEANUP_MAX_DIST_M:
        return False
    if forward < NEARBY_ROUTE_CLEANUP_MIN_FORWARD_M or forward > NEARBY_ROUTE_CLEANUP_MAX_FORWARD_M:
        return False
    if abs(lateral) < NEARBY_ROUTE_CLEANUP_MIN_LATERAL_M or abs(lateral) > NEARBY_ROUTE_CLEANUP_MAX_LATERAL_M:
        return False

    # Do not abandon a genuinely useful virgin strip immediately.  Once the row
    # has progressed enough, however, a close lateral cluster is cheaper to pick
    # up now than after a long return loop.
    cleaned_ahead, uncleaned_ahead, _ = coverage_corridor_ahead()
    current_row_still_good = (
        uncleaned_ahead > NEARBY_ROUTE_CLEANUP_USEFUL_ROW_UNCLEANED
        and cleaned_ahead < NEARBY_ROUTE_CLEANUP_USEFUL_ROW_CLEANED_MAX
    )
    if current_row_still_good and row_distance_from_start() < NEARBY_ROUTE_CLEANUP_ALLOW_USEFUL_ROW_AFTER_M:
        coverage_status = f"near-route wait: useful row clean={cleaned_ahead:.2f} unclean={uncleaned_ahead:.2f}"
        return False

    side = 1.0 if lateral > 0.0 else -1.0
    if side > 0.0 and left_dist < NEARBY_ROUTE_CLEANUP_SIDE_CLEAR_M:
        return False
    if side < 0.0 and right_dist < NEARBY_ROUTE_CLEANUP_SIDE_CLEAR_M:
        return False

    # Require a coherent local uncleaned cluster, not one noisy cell.
    try:
        obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
        if coverage_goal_map is not None:
            gx, gy = coverage_goal_map
            local_cells = local_mask_count(uncleaned & cleanable, gx, gy, RESIDUAL_TARGET_LOCAL_RADIUS_M)
        else:
            local_cells = 0
    except Exception:
        local_cells = NEARBY_ROUTE_CLEANUP_MIN_LOCAL_CELLS
    if local_cells < NEARBY_ROUTE_CLEANUP_MIN_LOCAL_CELLS:
        return False

    last_nearby_route_cleanup_time = now
    last_revisit_lane_change_time = now
    last_nearby_route_cleanup_reason = (
        f"near Dijkstra cleanup {'L' if side > 0 else 'R'} f={forward:.2f} lat={lateral:.2f} "
        f"dist={dist:.2f} route={coverage_route_cost:.2f} cells={local_cells} "
        f"ahead={cleaned_ahead:.2f}/{uncleaned_ahead:.2f}"
    )
    begin_lawnmower_lane_change_checked(side, last_nearby_route_cleanup_reason, front, body_clearance)
    row_end_candidate_count = 0
    return True


def row_distance_from_start():
    return math.hypot(pose_x - row_start_x, pose_y - row_start_y)


def row_commit_active(front=None, body_clearance=None):
    """Keep early exploration as a stable straight strip.

    The route planner is allowed to select/display a goal, but while the map is
    still sparse it must not interrupt a healthy forward row with side pockets or
    target-seeking lane changes.  Real obstacles, bumpers and row ends still win.
    """
    if nav_state != NAV_FORWARD:
        return False
    if under_furniture_active or robot.getTime() < leg_pass_grace_until:
        return False
    if last_coverage_percent >= EARLY_EXPLORATION_COVERAGE_PERCENT:
        return False
    if front is not None and front < ROW_COMMIT_FRONT_CLEARANCE_M:
        return False
    if body_clearance is not None and body_clearance < BODY_CORRIDOR_HARD_CLEARANCE:
        return False
    return (
        robot.getTime() - row_start_time < ROW_COMMIT_MIN_TIME_SEC
        or row_distance_from_start() < ROW_COMMIT_MIN_DISTANCE_M
    )


def lane_action_cooldown_active(front=None):
    """Suppress another soft 90-degree action immediately after a row transition.

    This does not hide real collisions: hard frontal blocks, bumpers and recovery
    states still override it. It only prevents planner/pocket/revisit heuristics
    from chaining pivots before the robot has actually cleaned/mapped a useful
    part of the new strip.
    """
    if nav_state != NAV_FORWARD:
        return False
    if front is not None and front < ROW_END_HARD_DISTANCE:
        return False
    if robot.getTime() - last_row_change_time >= LANE_ACTION_COOLDOWN_SEC:
        return False
    return row_distance_from_start() < LANE_ACTION_MIN_ROW_PROGRESS_M


def body_row_end_flags(front, center, upper_front, body_clearance):
    """Return (hard, soft) for body-footprint row-end decisions.

    The old logic used any body-corridor hit as a row-end signal. That is too
    aggressive for a round robot near chair/table legs: a side shell warning made
    it pivot on the spot even though the central corridor was still open.
    """
    soft_threshold = BODY_CORRIDOR_PASS_CLEARANCE
    if navigation_phase == NavigationPhase.COVERAGE.value:
        # making rows terminate before the wall fringe could be cleaned.  Keep the
        # side/leg safety elsewhere, but treat front row-end as near-contact only.
        soft_threshold = COVERAGE_BODY_ROW_END_SOFT_M
    if last_floor_front_ignore or body_clearance >= soft_threshold:
        return False, False

    central_body_hit = abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M
    front_supported = (
        front < BODY_ROW_END_FRONT_SUPPORT_M
        or center < BODY_ROW_END_CENTER_SUPPORT_M
        or upper_front < BODY_ROW_END_UPPER_SUPPORT_M
    )

    soft = bool(central_body_hit and front_supported)
    hard = bool(
        body_clearance < BODY_CORRIDOR_HARD_CLEARANCE
        and central_body_hit
        and (front < ROW_END_CONFIRM_DISTANCE or center < SAFE_FRONT_DISTANCE or upper_front < BODY_ROW_END_UPPER_SUPPORT_M)
    )
    return hard, soft


def heading_locked_wheel_speeds_to(base_speed, heading_target, kp=HEADING_LOCK_KP, max_corr=HEADING_LOCK_MAX_CORRECTION):
    """Drive straight toward an explicit heading target.

    Lane shifts use a frozen physical heading captured after the first 90-degree
    pivot.  Otherwise the robot can draw a visible arc while trying to correct a
    tiny grid-heading/yaw mismatch during the lateral translation.
    """
    global last_motion_primitive, last_motion_contract_reason
    if strict_grid_heading_lock_active():
        heading_target = strict_world_grid_heading(heading_target)
    err = normalize_angle(heading_target - pose_theta)
    if strict_grid_heading_lock_active():
        if abs(err) > STRICT_GRID_HEADING_PIVOT_TOL:
            sign = 1.0 if err > 0.0 else -1.0
            pivot = clamp(abs(err) * STRICT_GRID_PIVOT_KP, STRICT_GRID_PIVOT_MIN_SPEED, STRICT_GRID_PIVOT_MAX_SPEED)
            last_motion_primitive = MotionPrimitive.FINE_ALIGN.value
            last_motion_contract_reason = f"strict heading pivot err={math.degrees(err):.1f}"
            return -sign * pivot, sign * pivot
        corr = clamp(kp * err, -STRICT_GRID_TINY_CORRECTION, STRICT_GRID_TINY_CORRECTION) if abs(err) > STRICT_GRID_HEADING_TINY_TOL else 0.0
        last_motion_primitive = MotionPrimitive.MOVE_FORWARD_CELL.value
        last_motion_contract_reason = f"strict straight heading err={math.degrees(err):.1f}"
        return base_speed - corr, base_speed + corr
    corr = clamp(kp * err, -max_corr, max_corr)
    return base_speed - corr, base_speed + corr


def heading_locked_wheel_speeds(base_speed):
    """Drive straight while correcting yaw toward desired_grid_heading."""
    return heading_locked_wheel_speeds_to(base_speed, desired_grid_heading)

def update_odometry():
    global pose_x, pose_y, pose_theta, prev_left, prev_right
    global current_linear_velocity, current_angular_velocity, prev_imu_theta

    left = left_sensor.getValue()
    right = right_sensor.getValue()
    imu_theta = read_imu_heading()

    if prev_left is None or prev_right is None:
        prev_left, prev_right = left, right
        if imu_theta is not None:
            pose_theta = imu_theta
            prev_imu_theta = imu_theta
        return pose_x, pose_y, pose_theta

    d_left = (left - prev_left) * WHEEL_RADIUS * ODOM_LEFT_SIGN
    d_right = (right - prev_right) * WHEEL_RADIUS * ODOM_RIGHT_SIGN
    prev_left, prev_right = left, right

    dc = (d_left + d_right) / 2.0
    encoder_dtheta = (d_right - d_left) / WHEEL_BASE

    if imu_theta is not None:
        dtheta = normalize_angle(imu_theta - pose_theta)
        next_theta = imu_theta
    else:
        dtheta = encoder_dtheta
        next_theta = normalize_angle(pose_theta + dtheta)

    # During explicit pivot/settle we do not trust encoder translational drift.
    # If the robot really slides a few millimeters in Webots, that is less harmful
    # than drawing a large fake arc during a commanded in-place turn.
    if globals().get("nav_state") in (NAV_TURN_90, NAV_SETTLE, NAV_LEG_ESCAPE_TURN):
        dc = 0.0

    dt = max(timestep / 1000.0, 1e-3)
    current_linear_velocity = dc / dt
    current_angular_velocity = dtheta / dt

    mid = pose_theta + dtheta / 2.0
    pose_x += dc * math.cos(mid)
    pose_y += dc * math.sin(mid)
    pose_theta = normalize_angle(next_theta)
    prev_imu_theta = imu_theta
    clamp_pose_to_debug_arena_if_needed()
    return pose_x, pose_y, pose_theta

def bresenham(x0, y0, x1, y1):
    """Integer line cells from start to end."""
    yield from _bresenham_cells(x0, y0, x1, y1)


def update_log_odds_cell(mx, my, delta):
    if map_inside(mx, my):
        log_odds[my, mx] = _update_log_odds_value(log_odds[my, mx], delta, LO_MIN, LO_MAX)


def clear_contact_evidence_cell(mx, my, delta=CONTACT_FREE_UPDATE):
    """Reduce bumper-confirmed obstacle evidence when RGB-D/depth sees free space.

    This implements the rule: contact creates a confident local obstacle, but it
    is not permanent. It stays until later observations repeatedly see that same
    place as empty.
    """
    if map_inside(mx, my) and contact_log_odds[my, mx] > 0.0:
        contact_log_odds[my, mx] = clamp(contact_log_odds[my, mx] + delta, CONTACT_MIN, CONTACT_MAX)


def clear_contact_evidence_along_ray(x0, y0, x1, y1, delta=CONTACT_FREE_UPDATE):
    for cx, cy in bresenham(x0, y0, x1, y1):
        clear_contact_evidence_cell(cx, cy, delta)


def update_structural_obstacle_memory():
    """Maintain a slow obstacle layer for planning/debug persistence.

    Late in coverage, green cleaned cells can visually dominate the objective map
    while weak log-odds evidence is gradually cleared by visibility rays.  That is
    acceptable for transient free-space evidence, but not for furniture/walls that
    were repeatedly confirmed by RGB-D/OpenCV or bumper contact.  This layer keeps
    such geometry stable without changing the conceptual sensor model: RGB camera
    finds visual structures, RangeFinder provides depth, bumpers confirm contact.
    """
    global structural_log_odds
    if not STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
        return 0
    promote_visual = visual_log_odds > STRUCTURAL_VISUAL_PROMOTE_EPS
    promote_log = log_odds > STRUCTURAL_LOG_PROMOTE_EPS
    promote_contact = contact_log_odds > STRUCTURAL_CONTACT_PROMOTE_EPS
    promote = promote_visual | promote_log | promote_contact
    if np.any(promote):
        structural_log_odds[promote] = np.minimum(STRUCTURAL_MAX, structural_log_odds[promote] + STRUCTURAL_OCC_UPDATE)
    if np.any(promote_contact):
        structural_log_odds[promote_contact] = np.minimum(STRUCTURAL_MAX, structural_log_odds[promote_contact] + STRUCTURAL_CONTACT_UPDATE)

    # Decay only where the ordinary map has accumulated strong free evidence and
    # RGB/contact layers no longer support an obstacle.  A single visibility ray
    # must not delete a remembered chair/table/wall edge.
    strong_free = (
        (log_odds < STRUCTURAL_STRONG_FREE_LO)
        & (visual_log_odds <= CV_DISPLAY_LIGHT_EPS)
        & (contact_log_odds <= CONTACT_OCCUPIED_EPS)
        & (~promote)
    )
    if np.any(strong_free):
        structural_log_odds[strong_free] = np.maximum(
            STRUCTURAL_MIN,
            structural_log_odds[strong_free] + STRUCTURAL_FREE_DECAY,
        )
    return int(np.count_nonzero(structural_log_odds > STRUCTURAL_OCCUPIED_EPS))


def base_physical_obstacle_mask():
    """Obstacle evidence before furniture-zone inflation.

    keeps the sensor model honest: RGB-D/RangeFinder/bumper evidence
    creates obstacles; furniture zones are a derived planning/debug layer over
    that evidence, not a new sensor and not semantic recognition.
    """
    obstacles = (
        (log_odds > LO_OCCUPIED_EPS)
        | (visual_log_odds > CV_DISPLAY_DENSE_EPS)
        | (contact_log_odds > CONTACT_OCCUPIED_EPS)
    )
    if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
        obstacles = obstacles | (structural_log_odds > STRUCTURAL_OCCUPIED_EPS)
    return obstacles


def observed_evidence_mask_for_zones():
    return (
        (np.abs(log_odds) > LO_UNKNOWN_EPS)
        | (visual_log_odds > CV_DISPLAY_LIGHT_EPS)
        | (contact_log_odds > CONTACT_OCCUPIED_EPS)
        | ((structural_log_odds > STRUCTURAL_OCCUPIED_EPS) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(log_odds, dtype=np.bool_))
        | (cleaned_mask > 0)
    )


def furniture_zone_candidate_obstacle_mask():
    """Stable obstacle evidence allowed to seed a furniture/complex zone.

    The display/logical furniture layer must not be created from one bad RGB-D
    viewing angle.  Raw visual/log-odds speckles are still drawn as ordinary map
    evidence, but they become a purple furniture island only when they are either
    bumper-confirmed or supported by repeated structural/occupancy evidence.
    """
    contact = contact_log_odds > CONTACT_OCCUPIED_EPS
    repeated_contact = contact_log_odds > FURNITURE_ZONE_CONTACT_ISLAND_EPS
    strong_log = log_odds > (LO_OCCUPIED_EPS + 0.75)
    strong_visual = visual_log_odds > (CV_DISPLAY_DENSE_EPS + 0.90)
    structural_strong = (
        (structural_log_odds > (STRUCTURAL_OCCUPIED_EPS + 0.85))
        if STRUCTURAL_OBSTACLE_MEMORY_ENABLED
        else np.zeros_like(log_odds, dtype=np.bool_)
    )
    # Visual-only pixels are too unstable for furniture zones.  Structural memory
    # that came only from a single bumper dot is also too weak: it remains a
    # compact contact obstacle, not a purple furniture island.  Repeated contact
    # may seed a small island only after contact_log_odds has accumulated.
    rgbd_structural = structural_strong & (strong_log | strong_visual)
    contact_island = repeated_contact & structural_strong
    return rgbd_structural | (strong_log & (strong_visual | rgbd_structural)) | contact_island


def arena_wall_touch_mask():
    """Approximate known wall/arena border in map cells for island rejection."""
    wall = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    if not DEBUG_ARENA_BOUNDS_ENABLED:
        return wall > 0
    try:
        x0, y_top = world_to_map(DEBUG_ARENA_X_MIN_M, DEBUG_ARENA_Y_MAX_M)
        x1, y_bottom = world_to_map(DEBUG_ARENA_X_MAX_M, DEBUG_ARENA_Y_MIN_M)
        x_min = max(0, min(x0, x1))
        x_max = min(MAP_SIZE - 1, max(x0, x1))
        y_min = max(0, min(y_top, y_bottom))
        y_max = min(MAP_SIZE - 1, max(y_top, y_bottom))
        pad = max(2, int(round(float(FURNITURE_ZONE_WALL_TOUCH_PAD_M) * MAP_SCALE)))
        wall[max(0, y_min - pad):min(MAP_SIZE, y_min + pad + 1), x_min:x_max + 1] = 1
        wall[max(0, y_max - pad):min(MAP_SIZE, y_max + pad + 1), x_min:x_max + 1] = 1
        wall[y_min:y_max + 1, max(0, x_min - pad):min(MAP_SIZE, x_min + pad + 1)] = 1
        wall[y_min:y_max + 1, max(0, x_max - pad):min(MAP_SIZE, x_max + pad + 1)] = 1
    except Exception:
        return wall > 0
    return wall > 0


def update_furniture_zone_cache(force=False):
    """Detect internal obstacle islands such as chair/table leg clusters.

    The output has two masks: a core mask (stable internal obstacle cluster) and
    an inflated mask (planning/debug safety zone around it).  Wall-connected and
    extremely long components are rejected so room walls are not labelled as
    furniture.
    """
    global furniture_zone_core_cache, furniture_zone_inflated_cache, furniture_zone_cache_step, last_furniture_zone_debug
    if not FURNITURE_ZONE_DETECTION_ENABLED:
        furniture_zone_core_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        furniture_zone_inflated_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        last_furniture_zone_debug = "furnitureZone=off"
        return furniture_zone_core_cache, furniture_zone_inflated_cache
    if (
        (not force)
        and furniture_zone_core_cache is not None
        and furniture_zone_inflated_cache is not None
        and int(step_id) - int(furniture_zone_cache_step) < max(1, int(FURNITURE_ZONE_UPDATE_STEPS))
    ):
        return furniture_zone_core_cache, furniture_zone_inflated_cache
    try:
        # Furniture/complex zones are derived only from stable obstacle seeds.
        # The ordinary map may contain raw one-view RGB-D noise; do not promote
        # that noise into magenta planning zones.
        obs = furniture_zone_candidate_obstacle_mask()
        support_contact = contact_log_odds > CONTACT_OCCUPIED_EPS
        evidence = observed_evidence_mask_for_zones()
        if not np.any(obs & evidence):
            furniture_zone_core_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
            furniture_zone_inflated_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
            furniture_zone_cache_step = int(step_id)
            last_furniture_zone_debug = "furnitureZone=none"
            return furniture_zone_core_cache, furniture_zone_inflated_cache

        # Work inside the observed envelope only; this prevents unknown map padding
        # and arena borders from becoming artificial furniture components.
        ys, xs = np.nonzero(evidence)
        pad = max(3, int(round(0.35 * MAP_SCALE)))
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(MAP_SIZE, int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(MAP_SIZE, int(ys.max()) + pad + 1)

        raw = obs[y0:y1, x0:x1].astype(np.uint8)
        contact_roi = support_contact[y0:y1, x0:x1]
        log_roi = (log_odds[y0:y1, x0:x1] > (LO_OCCUPIED_EPS + 0.75))
        visual_roi = (visual_log_odds[y0:y1, x0:x1] > (CV_DISPLAY_DENSE_EPS + 0.90))
        structural_roi = (
            (structural_log_odds[y0:y1, x0:x1] > (STRUCTURAL_OCCUPIED_EPS + 0.85))
            if STRUCTURAL_OBSTACLE_MEMORY_ENABLED
            else np.zeros_like(contact_roi, dtype=np.bool_)
        )
        repeated_contact_roi = contact_log_odds[y0:y1, x0:x1] > FURNITURE_ZONE_CONTACT_ISLAND_EPS
        if raw.size == 0:
            raise ValueError("empty furniture ROI")
        # Remove single-pixel depth/visual glitches before joining nearby legs.
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contact_raw = contact_roi.astype(np.uint8)
        # c: contact cells stay in the physical obstacle layer, but they are
        # no longer injected into the furniture-zone seed.  Otherwise a single
        # bumper touch creates a large magenta inflated island and blocks nearby
        # sweep lanes.
        close_r = max(1, int(round(float(FURNITURE_ZONE_CLOSE_M) * MAP_SCALE)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        grouped = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=1) > 0

        wall_full = arena_wall_touch_mask()
        wall = wall_full[y0:y1, x0:x1]
        n, labels, stats, _cent = cv2.connectedComponentsWithStats(grouped.astype(np.uint8), 8)
        core_roi = np.zeros_like(grouped, dtype=np.bool_)
        comps = 0
        cells = 0
        rejected_wall = 0
        rejected_sparse = 0
        rejected_fan = 0
        rejected_source = 0
        max_span_px = max(6, int(round(float(FURNITURE_ZONE_MAX_SPAN_M) * MAP_SCALE)))
        raw_dilated = cv2.dilate(raw, np.ones((3, 3), np.uint8), iterations=1) > 0
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < int(FURNITURE_ZONE_MIN_AREA_PX) or area > int(FURNITURE_ZONE_MAX_AREA_PX):
                rejected_sparse += 1
                continue
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            if span > max_span_px:
                rejected_fan += 1
                continue
            comp = labels == cid
            # Reject walls/borders: components touching the arena border or ROI
            # border are not internal obstacle islands.
            if np.any(comp & wall):
                rejected_wall += 1
                continue
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            if x <= 1 or y <= 1 or x + width >= grouped.shape[1] - 2 or y + height >= grouped.shape[0] - 2:
                rejected_wall += 1
                continue
            # Preserve only components backed by real raw obstacle cells, not
            # dilation-only bridges.  One-view visual/depth fans tend to be sparse
            # diagonal components with high aspect ratio and low fill; those must
            # not become purple furniture zones.
            raw_support = int(np.count_nonzero(comp & (raw > 0)))
            contact_support = int(np.count_nonzero(comp & contact_roi))
            repeated_contact_support = int(np.count_nonzero(comp & repeated_contact_roi))
            log_support = int(np.count_nonzero(comp & log_roi))
            visual_support = int(np.count_nonzero(comp & visual_roi))
            structural_support = int(np.count_nonzero(comp & structural_roi))
            density = raw_support / max(1, area)
            fill = raw_support / max(1, width * height)
            aspect = span / max(1, min(width, height))
            source_types = int(log_support >= FURNITURE_ZONE_MIN_STRONG_SUPPORT) + int(visual_support >= FURNITURE_ZONE_MIN_STRONG_SUPPORT) + int(structural_support >= FURNITURE_ZONE_MIN_STRONG_SUPPORT)
            repeated_contact_island = repeated_contact_support >= max(3, int(FURNITURE_ZONE_MIN_STRONG_SUPPORT // 2))
            if raw_support <= 0:
                rejected_sparse += 1
                continue
            if (not repeated_contact_island) and source_types < int(FURNITURE_ZONE_MIN_SOURCE_TYPES):
                rejected_source += 1
                continue
            if (not repeated_contact_island) and raw_support < int(FURNITURE_ZONE_MIN_STRONG_SUPPORT):
                rejected_sparse += 1
                continue
            if (not repeated_contact_island) and density < float(FURNITURE_ZONE_MIN_DENSITY):
                rejected_sparse += 1
                continue
            if aspect > float(FURNITURE_ZONE_HARD_MAX_ASPECT):
                rejected_fan += 1
                continue
            if aspect > float(FURNITURE_ZONE_MAX_ASPECT) and fill < 0.34:
                rejected_fan += 1
                continue
            if aspect > float(FURNITURE_ZONE_DIAGONAL_FAN_MAX_ASPECT) and fill < float(FURNITURE_ZONE_DIAGONAL_FAN_MIN_FILL):
                rejected_fan += 1
                continue
            core_roi |= (comp & raw_dilated)
            comps += 1
            cells += raw_support

        core = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        core[y0:y1, x0:x1] = core_roi
        if np.any(core):
            inflate_r = max(1, int(round(float(FURNITURE_ZONE_INFLATE_M) * MAP_SCALE)))
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflate_r + 1, 2 * inflate_r + 1))
            inflated = cv2.dilate(core.astype(np.uint8), kk, iterations=1) > 0
            # Do not fill the obstacle itself into unknown-only outside-room space.
            near_known = cv2.dilate(evidence.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            inflated &= near_known
        else:
            inflated = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        furniture_zone_core_cache = core
        furniture_zone_inflated_cache = inflated
        furniture_zone_cache_step = int(step_id)
        last_furniture_zone_debug = (
            f"furnitureZone={comps} raw={cells} core={int(np.count_nonzero(core))} "
            f"infl={int(np.count_nonzero(inflated))} rejW/S/F/src={rejected_wall}/{rejected_sparse}/{rejected_fan}/{rejected_source}"
        )
        return core, inflated
    except Exception as exc:
        furniture_zone_core_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        furniture_zone_inflated_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        furniture_zone_cache_step = int(step_id)
        last_furniture_zone_debug = f"furnitureZone=err {type(exc).__name__}"
        return furniture_zone_core_cache, furniture_zone_inflated_cache


def furniture_zone_masks():
    return update_furniture_zone_cache(force=False)


def _hypothesis_clear_on_free_observations():
    """Decay inferred obstacles where the RGB-D map has later seen strong free floor."""
    global hypothesis_obstacle_log_odds
    if not OBSTACLE_HYPOTHESIS_ENABLED:
        return 0
    try:
        free = log_odds < float(OBSTACLE_HYPOTHESIS_FREE_CLEAR_LO)
        # Never let free carving erase bumper/structural-confirmed obstacles; it
        # only clears the orange hypothesis layer.
        keep_confirmed = (
            (contact_log_odds > CONTACT_OCCUPIED_EPS)
            | (visual_log_odds > CV_DISPLAY_DENSE_EPS)
            | ((structural_log_odds > STRUCTURAL_OCCUPIED_EPS) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(free, dtype=np.bool_))
        )
        clear = free & (~keep_confirmed) & (hypothesis_obstacle_log_odds > 0.0)
        if np.any(clear):
            hypothesis_obstacle_log_odds[clear] = np.maximum(
                0.0,
                hypothesis_obstacle_log_odds[clear] + float(OBSTACLE_HYPOTHESIS_DECAY) * 6.0,
            )
        decay = (~clear) & (hypothesis_obstacle_log_odds > 0.0) & (log_odds < -LO_UNKNOWN_EPS)
        if np.any(decay):
            hypothesis_obstacle_log_odds[decay] = np.maximum(
                0.0,
                hypothesis_obstacle_log_odds[decay] + float(OBSTACLE_HYPOTHESIS_DECAY),
            )
        return int(np.count_nonzero(clear | decay))
    except Exception:
        return 0


def _obstacle_hypothesis_side_support(comp, x, y, w, h):
    """Count which sides of a compact obstacle bbox have observed obstacle pixels."""
    try:
        min_side = int(OBSTACLE_HYPOTHESIS_MIN_SIDE_SUPPORT)
        # Work in a 3-pixel band because RGB-D/OpenCV edges are rarely perfectly
        # aligned with the connected-component bounding rectangle.
        left = int(np.count_nonzero(comp[y:y + h, x:min(comp.shape[1], x + 3)]))
        right = int(np.count_nonzero(comp[y:y + h, max(0, x + w - 3):x + w]))
        top = int(np.count_nonzero(comp[y:min(comp.shape[0], y + 3), x:x + w]))
        bottom = int(np.count_nonzero(comp[max(0, y + h - 3):y + h, x:x + w]))
        sides = int(left >= min_side) + int(right >= min_side) + int(top >= min_side) + int(bottom >= min_side)
        return sides, left, right, top, bottom
    except Exception:
        return 0, 0, 0, 0, 0


def update_obstacle_hypothesis_cache(force=False):
    """Build conservative inferred-obstacle cells from compact RGB-D geometry.

    This does not recognize object classes.  It only says: if a compact obstacle
    island has several stable visible sides and the interior is unknown/not free,
    treat that interior/occluded side as an orange hypothesis no-go.  This stops
    frontier from trying to drive 'inside' a box/table just because the front
    camera never saw its far wall.
    """
    global hypothesis_obstacle_cache, hypothesis_obstacle_cache_step, hypothesis_obstacle_log_odds
    global last_hypothesis_obstacle_debug
    if not OBSTACLE_HYPOTHESIS_ENABLED:
        hypothesis_obstacle_cache = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.bool_)
        hypothesis_obstacle_cache_step = int(step_id)
        last_hypothesis_obstacle_debug = "hypObs=off"
        return hypothesis_obstacle_cache
    if (
        (not force)
        and hypothesis_obstacle_cache is not None
        and int(step_id) - int(hypothesis_obstacle_cache_step) < max(1, int(OBSTACLE_HYPOTHESIS_UPDATE_STEPS))
    ):
        return hypothesis_obstacle_cache
    try:
        cleared = _hypothesis_clear_on_free_observations()
        confirmed = base_physical_obstacle_mask().astype(np.bool_)
        if not np.any(confirmed):
            hypothesis_obstacle_cache = hypothesis_obstacle_log_odds > float(OBSTACLE_HYPOTHESIS_OCC_EPS)
            hypothesis_obstacle_cache_step = int(step_id)
            last_hypothesis_obstacle_debug = f"hypObs=none cleared={cleared}"
            return hypothesis_obstacle_cache

        # Smooth tiny cracks but do not connect the whole wall perimeter.
        close_px = max(1, int(round(float(OBSTACLE_HYPOTHESIS_BOX_PAD_M) * MAP_SCALE)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        grouped = cv2.morphologyEx(confirmed.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0
        wall = arena_wall_touch_mask()
        n, labels, stats, _cent = cv2.connectedComponentsWithStats(grouped.astype(np.uint8), 8)
        additions = np.zeros_like(confirmed, dtype=np.bool_)
        comps = 0
        fill_cells = 0
        rej_wall = 0
        rej_shape = 0
        rej_free = 0
        max_span_px = max(6, int(round(float(OBSTACLE_HYPOTHESIS_MAX_SPAN_M) * MAP_SCALE)))
        free_strong = log_odds < float(OBSTACLE_HYPOTHESIS_FREE_CLEAR_LO)
        observed_free = (log_odds < -LO_UNKNOWN_EPS) | (cleaned_mask > 0)
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < int(OBSTACLE_HYPOTHESIS_MIN_COMPONENT_CELLS) or area > int(OBSTACLE_HYPOTHESIS_MAX_COMPONENT_CELLS):
                rej_shape += 1
                continue
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            w = int(stats[cid, cv2.CC_STAT_WIDTH])
            h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            if max(w, h) > max_span_px or w <= 3 or h <= 3:
                rej_shape += 1
                continue
            comp = labels == cid
            if np.any(comp & wall) or x <= 2 or y <= 2 or x + w >= MAP_SIZE - 3 or y + h >= MAP_SIZE - 3:
                rej_wall += 1
                continue
            side_count, _l, _r, _t, _b = _obstacle_hypothesis_side_support(comp, x, y, w, h)
            if side_count < int(OBSTACLE_HYPOTHESIS_MIN_SIDE_COUNT):
                rej_shape += 1
                continue
            pad = max(1, int(round(float(OBSTACLE_HYPOTHESIS_BOX_PAD_M) * MAP_SCALE)))
            x0 = max(0, x - pad)
            x1 = min(MAP_SIZE, x + w + pad)
            y0 = max(0, y - pad)
            y1 = min(MAP_SIZE, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            roi_confirmed = confirmed[y0:y1, x0:x1]
            roi_free = free_strong[y0:y1, x0:x1]
            roi_observed_free = observed_free[y0:y1, x0:x1]
            roi_unknownish = (~roi_confirmed) & (~roi_free)
            fill = roi_unknownish.copy()
            # Keep a 1 px contact/edge contour visible; fill mostly the unknown
            # interior/shadow, not the already-black confirmed obstacle pixels.
            if np.count_nonzero(fill) < int(OBSTACLE_HYPOTHESIS_MIN_FILL_CELLS):
                rej_shape += 1
                continue
            free_ratio = float(np.count_nonzero(roi_observed_free & (~roi_confirmed))) / float(max(1, np.count_nonzero(~roi_confirmed)))
            if free_ratio > float(OBSTACLE_HYPOTHESIS_MAX_FREE_RATIO):
                rej_free += 1
                continue
            local_add = np.zeros_like(additions[y0:y1, x0:x1], dtype=np.bool_)
            local_add[fill] = True
            # Restrict to the bbox; this is deliberately conservative shape
            # completion, not rectangle hallucination outside the observed island.
            additions[y0:y1, x0:x1] |= local_add
            comps += 1
            fill_cells += int(np.count_nonzero(local_add))

        if np.any(additions):
            hypothesis_obstacle_log_odds[additions] = np.minimum(
                float(OBSTACLE_HYPOTHESIS_MAX),
                hypothesis_obstacle_log_odds[additions] + float(OBSTACLE_HYPOTHESIS_SHAPE_UPDATE),
            )
        cache = hypothesis_obstacle_log_odds > float(OBSTACLE_HYPOTHESIS_OCC_EPS)
        # Confirmed free always wins over hypothesis.
        cache &= ~(log_odds < float(OBSTACLE_HYPOTHESIS_FREE_CLEAR_LO))
        hypothesis_obstacle_cache = cache.astype(np.bool_)
        hypothesis_obstacle_cache_step = int(step_id)
        last_hypothesis_obstacle_debug = (
            f"hypObs={int(np.count_nonzero(cache))} add={fill_cells} comps={comps} "
            f"clr={cleared} rejW/S/F={rej_wall}/{rej_shape}/{rej_free}"
        )
        return hypothesis_obstacle_cache
    except Exception as exc:
        hypothesis_obstacle_cache = hypothesis_obstacle_log_odds > float(OBSTACLE_HYPOTHESIS_OCC_EPS)
        hypothesis_obstacle_cache_step = int(step_id)
        last_hypothesis_obstacle_debug = f"hypObs=err {type(exc).__name__}"
        return hypothesis_obstacle_cache.astype(np.bool_)


def hypothesis_obstacle_mask(force=False):
    return update_obstacle_hypothesis_cache(force=force).astype(np.bool_)


def mark_hypothesis_obstacle_at_world(wx, wy, radius_m=None, update=None, reason="hypothesis"):
    """Add a compact orange inferred obstacle, without claiming bumper contact."""
    global hypothesis_obstacle_log_odds, hypothesis_obstacle_cache_step, last_near_collision_hypothesis_debug
    if not OBSTACLE_HYPOTHESIS_ENABLED:
        return False
    try:
        mx, my = world_to_map(float(wx), float(wy))
        if not map_inside(mx, my):
            return False
        r = max(1, int(round(float(radius_m if radius_m is not None else NEAR_COLLISION_HYPOTHESIS_RADIUS_M) * MAP_SCALE)))
        val = float(update if update is not None else OBSTACLE_HYPOTHESIS_SHADOW_UPDATE)
        cv2.circle(hypothesis_obstacle_log_odds, (int(mx), int(my)), r, val, -1)
        hypothesis_obstacle_log_odds[:, :] = np.minimum(hypothesis_obstacle_log_odds, float(OBSTACLE_HYPOTHESIS_MAX))
        hypothesis_obstacle_cache_step = -999999
        try:
            invalidate_heavy_map_caches("hypothesis obstacle")
        except Exception:
            pass
        last_near_collision_hypothesis_debug = f"nearHyp=mark ({int(mx)},{int(my)}) r={r} {str(reason)[:18]}"
        return True
    except Exception as exc:
        last_near_collision_hypothesis_debug = f"nearHyp=err {type(exc).__name__}"
        return False


def maybe_mark_near_collision_hypothesis(reason="unsafe stop"):
    """Turn repeated non-contact body/front blocking into a temporary map hypothesis."""
    global last_near_collision_hypothesis_time, last_near_collision_hypothesis_debug
    if not (OBSTACLE_HYPOTHESIS_ENABLED and NEAR_COLLISION_HYPOTHESIS_ENABLED):
        return False
    try:
        now = float(robot.getTime())
        if now - float(last_near_collision_hypothesis_time) < float(NEAR_COLLISION_HYPOTHESIS_COOLDOWN_SEC):
            return False
        if float(last_coverage_percent or 0.0) < float(NEAR_COLLISION_HYPOTHESIS_MIN_COV_PERCENT):
            return False
        # Only infer from real local clearance pressure.  This must not mark a
        # random failed far-away route as an obstacle.
        front_close = float(last_front_narrow) < float(NEAR_COLLISION_HYPOTHESIS_FRONT_M)
        body_close = float(last_body_corridor_clearance) < float(NEAR_COLLISION_HYPOTHESIS_BODY_M)
        if not (front_close or body_close):
            return False
        d = min(float(NEAR_COLLISION_HYPOTHESIS_FRONT_M), max(0.16, float(last_front_narrow if np.isfinite(last_front_narrow) else 0.22)))
        lateral = 0.0
        if body_close:
            lateral = float(last_body_corridor_lateral)
            lateral = max(-ROBOT_BODY_RADIUS * 0.85, min(ROBOT_BODY_RADIUS * 0.85, lateral))
        wx = pose_x + math.cos(pose_theta) * d - math.sin(pose_theta) * lateral
        wy = pose_y + math.sin(pose_theta) * d + math.cos(pose_theta) * lateral
        if mark_hypothesis_obstacle_at_world(wx, wy, reason=reason):
            last_near_collision_hypothesis_time = now
            return True
        return False
    except Exception as exc:
        last_near_collision_hypothesis_debug = f"nearHyp=err {type(exc).__name__}"
        return False


def actual_physical_obstacle_mask():
    """Physical/core obstacle evidence only.

    This mask answers "is this cell occupied by a wall/object/contact point?".
    It deliberately does not include the inflated purple furniture safety margin,
    because that margin is a robot-centre no-go layer, not physical geometry.
    """
    obstacles = base_physical_obstacle_mask()
    if FURNITURE_ZONE_DETECTION_ENABLED:
        try:
            core, _inflated = furniture_zone_masks()
            obstacles = obstacles | core
        except Exception:
            pass
    if OBSTACLE_HYPOTHESIS_ENABLED and OBSTACLE_HYPOTHESIS_NO_GO_IN_PLANNING:
        try:
            # Planning-grade physical/core mask includes orange occlusion/shape
            # hypotheses so frontier/coverage do not drive into object interiors.
            # The raw layer is still visually distinct and can decay when seen free.
            obstacles = obstacles | hypothesis_obstacle_mask(force=False)
        except Exception:
            pass
    return obstacles.astype(np.bool_)


def planning_no_go_obstacle_mask(actual_obstacles=None):
    """Obstacle mask for robot-centre path planning.

    actual_obstacles are physical geometry; center_no_go additionally includes
    inflated furniture margin so the centreline planner does not graze table/chair
    clusters.  Coverage masks should normally use actual_obstacles instead.
    """
    if actual_obstacles is None:
        actual_obstacles = actual_physical_obstacle_mask()
    center_no_go = actual_obstacles.astype(np.bool_).copy()
    if PLANNING_CENTER_NO_GO_USES_FURNITURE_INFLATION and FURNITURE_ZONE_DETECTION_ENABLED:
        try:
            _core, inflated = furniture_zone_masks()
            center_no_go |= inflated
        except Exception:
            pass
    try:
        close_r = int(round(float(PLANNING_CENTER_NO_GO_CLOSE_GAPS_M) * MAP_SCALE))
        if close_r > 0 and np.any(center_no_go):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
            center_no_go = cv2.morphologyEx(center_no_go.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0
    except Exception:
        pass
    return center_no_go.astype(np.bool_)


def build_planning_layers(force=False):
    """Return actual_obstacles, center_no_go and cleanable_floor masks.

    This is the map-polishing layer between raw RGB-D occupancy and K-mode route
    planning.  It keeps cleanable floor independent from inflated no-go margins.
    """
    global last_planning_layer_debug
    if not PLANNING_LAYER_ENABLED:
        actual = base_physical_obstacle_mask().astype(np.bool_)
        under = under_surface_mask_from_obstacles(actual)
        cleanable_floor = (((log_odds < -LO_UNKNOWN_EPS) | under) | (cleaned_mask > 0)) & (~actual)
        last_planning_layer_debug = f"planningMap=off actual={int(np.count_nonzero(actual))}"
        return actual, actual.copy(), cleanable_floor.astype(np.bool_)
    actual = actual_physical_obstacle_mask()
    center_no_go = planning_no_go_obstacle_mask(actual)
    under = under_surface_mask_from_obstacles(actual)
    cleanable_floor = (((log_odds < -LO_UNKNOWN_EPS) | under) | (cleaned_mask > 0)) & (~actual)
    try:
        core, inflated = furniture_zone_masks()
        fcore = int(np.count_nonzero(core))
        finfl = int(np.count_nonzero(inflated))
    except Exception:
        fcore = 0
        finfl = 0
    last_planning_layer_debug = (
        f"rawObs={int(np.count_nonzero(base_physical_obstacle_mask()))} "
        f"actual={int(np.count_nonzero(actual))} noGo={int(np.count_nonzero(center_no_go))} "
        f"furn={fcore}/{finfl} hyp={int(np.count_nonzero(hypothesis_obstacle_mask(False))) if OBSTACLE_HYPOTHESIS_ENABLED else 0} "
        f"clean={int(np.count_nonzero(cleanable_floor))}"
    )
    return actual.astype(np.bool_), center_no_go.astype(np.bool_), cleanable_floor.astype(np.bool_)


def physical_obstacle_mask():
    """Legacy obstacle mask for passability/safety callers.

    It keeps the old conservative behavior by returning center_no_go.  New
    coverage-objective code should use actual_physical_obstacle_mask() or
    build_planning_layers() to avoid treating inflated margins as physical walls.
    """
    actual, center_no_go, _cleanable_floor = build_planning_layers(force=False)
    return center_no_go


def mark_edge_trim_coverage(mx, my):
    """Mark the near-wall strip that a side brush would cover.

    This solves a coverage-planning artifact, not an obstacle-avoidance problem.
    The robot should not keep chasing uncleaned cells that are too close to a
    wall/leg for the robot center to occupy. We mark only cells that are:
    - just outside the normal circular cleaning footprint;
    - near a known obstacle/contact boundary;
    - not themselves obstacles;
    - already observed as free/cleanable, unless explicitly configured otherwise.
    """
    global last_edge_trim_cells
    last_edge_trim_cells = 0
    if not EDGE_TRIM_COVERAGE_ENABLED:
        return 0

    core_r = max(2, int(COVERAGE_RADIUS_M * MAP_SCALE))
    edge_r = max(core_r + 1, int((COVERAGE_RADIUS_M + EDGE_TRIM_EXTRA_M) * MAP_SCALE))
    near_r = max(2, int(EDGE_TRIM_NEAR_OBSTACLE_M * MAP_SCALE))
    roi_r = edge_r + near_r + 3

    x0 = max(0, mx - roi_r)
    x1 = min(MAP_SIZE, mx + roi_r + 1)
    y0 = max(0, my - roi_r)
    y1 = min(MAP_SIZE, my + roi_r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0

    cx = mx - x0
    cy = my - y0
    h = y1 - y0
    w = x1 - x0

    outer = np.zeros((h, w), dtype=np.uint8)
    core = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(outer, (cx, cy), edge_r, 1, -1)
    cv2.circle(core, (cx, cy), core_r, 1, -1)
    annulus = (outer > 0) & (core == 0)
    if not np.any(annulus):
        return 0

    lo = log_odds[y0:y1, x0:x1]
    vio = visual_log_odds[y0:y1, x0:x1]
    contact = contact_log_odds[y0:y1, x0:x1]
    structural = structural_log_odds[y0:y1, x0:x1] if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros((h, w), dtype=np.float32)
    under = under_surface_log_odds[y0:y1, x0:x1] > UNDER_SURFACE_EPS

    obstacles = (lo > LO_OCCUPIED_EPS) | (vio > CV_DISPLAY_DENSE_EPS) | (contact > CONTACT_OCCUPIED_EPS) | (structural > STRUCTURAL_OCCUPIED_EPS)
    if not np.any(obstacles):
        return 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near_r + 1, 2 * near_r + 1))
    near_obstacle = cv2.dilate(obstacles.astype(np.uint8), k, iterations=1) > 0

    if EDGE_TRIM_REQUIRE_KNOWN_FREE:
        known_free = (lo < EDGE_TRIM_MIN_FREE_LO) | (cleaned_mask[y0:y1, x0:x1] > 0)
    else:
        known_free = np.ones((h, w), dtype=np.bool_)

    candidate = annulus & near_obstacle & (~obstacles) & known_free
    if not EDGE_TRIM_CLEAN_UNDER_SURFACE:
        candidate = candidate & (~under)
    if not np.any(candidate):
        return 0

    roi_cleaned = cleaned_mask[y0:y1, x0:x1]
    before = int(np.count_nonzero(roi_cleaned[candidate] > 0))
    roi_cleaned[candidate] = 255
    after = int(np.count_nonzero(candidate))
    last_edge_trim_cells = max(0, after - before)
    return last_edge_trim_cells


def trim_wall_edge_fringes():
    """Convert unreachable near-wall free fringes into cleaned edge coverage.

    This is a coverage-objective fix, not an obstacle mapper. The rule is:
    if a cell is already observed as free/cleanable, is directly beside a known
    obstacle/wall, and is also adjacent to an already-cleaned lane, then it is a
    wall-edge fringe that a round robot should not chase with its center. Mark it
    as covered by the edge/side-brush model.
    """
    global last_wall_fringe_trim_cells
    last_wall_fringe_trim_cells = 0
    if not WALL_FRINGE_TRIM_ENABLED:
        return 0

    obstacles = actual_physical_obstacle_mask()
    cleaned = (cleaned_mask > 0) & (~obstacles)
    if not np.any(cleaned) or not np.any(obstacles):
        return 0

    under = under_surface_log_odds > UNDER_SURFACE_EPS
    # Under-surface is only a future objective. Do not mark it as cleaned from a
    # nearby pass; it becomes cleaned only when the real footprint overlaps it.
    known_free = ((log_odds < EDGE_TRIM_MIN_FREE_LO) | cleaned) & (~obstacles) & (~under)

    obs_r = max(2, int(WALL_FRINGE_NEAR_OBSTACLE_M * MAP_SCALE))
    clean_r = max(2, int(WALL_FRINGE_NEAR_CLEANED_M * MAP_SCALE))
    obs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * obs_r + 1, 2 * obs_r + 1))
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * clean_r + 1, 2 * clean_r + 1))

    near_obstacle = cv2.dilate(obstacles.astype(np.uint8), obs_kernel, iterations=1) > 0
    near_cleaned = cv2.dilate(cleaned.astype(np.uint8), clean_kernel, iterations=1) > 0

    # Do not erase reachable strips. The old version trimmed any known-free cell
    # near a wall if a cleaned lane was nearby; that made the robot ignore areas
    # it could physically still enter. Keep only the very thin dead band next to
    # obstacles, where placing the robot center is not possible.
    free_from_obstacle_px = cv2.distanceTransform((~obstacles).astype(np.uint8), cv2.DIST_L2, 3)
    unreachable_band = free_from_obstacle_px <= max(1.0, WALL_FRINGE_UNREACHABLE_BAND_M * MAP_SCALE)

    candidate = known_free & (~cleaned) & near_obstacle & near_cleaned & unreachable_band
    if not np.any(candidate):
        return 0

    ys, xs = np.where(candidate)
    count = len(xs)
    if count > WALL_FRINGE_MAX_MARK_PER_UPDATE:
        # Mark the cells closest to the current robot first, so each update is
        # bounded and deterministic enough for realtime Webots.
        mx, my = world_to_map(pose_x, pose_y)
        order = np.argsort((xs - mx) * (xs - mx) + (ys - my) * (ys - my))[:WALL_FRINGE_MAX_MARK_PER_UPDATE]
        xs = xs[order]
        ys = ys[order]
        count = len(xs)

    cleaned_mask[ys, xs] = 255
    last_wall_fringe_trim_cells = int(count)
    return last_wall_fringe_trim_cells

def under_surface_cleaning_allowed(mx, my):
    """Return True when yellow under-furniture cells may become cleaned.

    A normal pass next to furniture should not mark the yellow under-surface
    layer as cleaned.  We allow cleaning that layer only when the robot centre or
    the front part of the body is actually inside the remembered under-surface
    corridor, or while the controller is actively driving through a detected
    under-furniture opening.
    """
    if not UNDER_SURFACE_STRICT_CLEANING:
        return True
    if under_furniture_active or coverage_goal_kind == "under-surface":
        # Still require the centre/front probe to be near the yellow corridor;
        # otherwise a stale target label can clean a chair from a neighbouring row.
        pass
    under = under_surface_log_odds > UNDER_SURFACE_EPS
    if map_inside(mx, my) and under[my, mx]:
        return True
    fx = int(round(mx + UNDER_SURFACE_CLEAN_FRONT_PROBE_M * MAP_SCALE * math.cos(pose_theta)))
    fy = int(round(my - UNDER_SURFACE_CLEAN_FRONT_PROBE_M * MAP_SCALE * math.sin(pose_theta)))
    if map_inside(fx, fy) and under[fy, fx]:
        return True
    if under_furniture_active:
        # A little tolerance for the first frames after entering a passage.
        probe_r = max(2, int(0.055 * MAP_SCALE))
        x0 = max(0, fx - probe_r)
        x1 = min(MAP_SIZE, fx + probe_r + 1)
        y0 = max(0, fy - probe_r)
        y1 = min(MAP_SIZE, fy + probe_r + 1)
        return bool(np.any(under[y0:y1, x0:x1]))
    return False


def mark_recent_visit_footprint(mx, my):
    """Mark where the robot has recently driven.

    This layer prevents visible route loops without making old tracks forbidden.
    A freshly visited corridor receives a planning penalty and then fades out.
    """
    global last_recent_visit_cells
    if not RECENT_VISIT_ROUTE_MEMORY_ENABLED or not map_inside(mx, my):
        return
    if step_id % RECENT_VISIT_DECAY_STEPS == 0:
        recent_visit_log_odds[:] *= RECENT_VISIT_DECAY
        recent_visit_log_odds[recent_visit_log_odds < RECENT_VISIT_LOW_VALUE_EPS] = 0.0
    r = max(2, int(RECENT_VISIT_RADIUS_M * MAP_SCALE))
    mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    cv2.circle(mask, (mx, my), r, 255, -1)
    recent_visit_log_odds[mask > 0] = np.minimum(RECENT_VISIT_MAX, recent_visit_log_odds[mask > 0] + RECENT_VISIT_UPDATE)
    if step_id % max(1, WINDOW_UPDATE_STEPS) == 0:
        last_recent_visit_cells = int(np.count_nonzero(recent_visit_log_odds > 0.5))


def mark_cleaned_footprint(x, y):
    """Remember cells actually covered by the cleaning footprint.

    change: do not paint the whole geometric circle blindly.  The previous
    footprint could leak green cells through a table/chair edge because the circle
    overlapped map cells that were not known free from the RGB-D occupancy layer.
    The cleaned layer is now constrained to known-free/non-obstacle cells; only a
    separately confirmed under-surface traversal may mark the yellow under-furniture
    layer.  This keeps the coverage map honest without changing the raw map.
    """
    global last_edge_trim_cells
    mx, my = world_to_map(x, y)
    if not map_inside(mx, my):
        last_edge_trim_cells = 0
        return

    mark_recent_visit_footprint(mx, my)

    radius_px = max(2, int(COVERAGE_RADIUS_M * MAP_SCALE))
    disk = np.zeros_like(cleaned_mask, dtype=np.uint8)
    cv2.circle(disk, (mx, my), radius_px, 255, -1)

    obstacles = actual_physical_obstacle_mask()
    known_free = (log_odds < -LO_UNKNOWN_EPS) | (cleaned_mask > 0)

    if UNDER_SURFACE_STRICT_CLEANING:
        under = under_surface_log_odds > UNDER_SURFACE_EPS
        ordinary_disk = (disk > 0) & known_free & (~obstacles) & (~under)
        cleaned_mask[ordinary_disk] = 255
        if under_surface_cleaning_allowed(mx, my):
            # When actually under the furniture, mark a smaller central swath.
            # This is intentionally conservative; a later parallel aperture pass
            # can cover the neighbouring swath if the opening is wide enough.
            under_disk = np.zeros_like(cleaned_mask, dtype=np.uint8)
            under_r = max(2, int(UNDER_SURFACE_CLEAN_RADIUS_M * MAP_SCALE))
            cv2.circle(under_disk, (mx, my), under_r, 255, -1)
            cleaned_mask[(under_disk > 0) & under & (~obstacles)] = 255
    else:
        cleaned_mask[(disk > 0) & known_free & (~obstacles)] = 255

    mark_edge_trim_coverage(mx, my)
    if step_id % WALL_FRINGE_UPDATE_STEPS == 0:
        trim_wall_edge_fringes()
    if step_id % UNREACHABLE_TARGET_TRIM_UPDATE_STEPS == 0:
        trim_unreachable_coverage_targets()


def under_surface_mask_from_obstacles(obstacles=None):
    """Cells remembered as passable floor under furniture.

    This is intentionally a separate coverage layer. It must not become a wall,
    and it must not pretend that the system recognizes a chair/table class. The
    only claim is: RGB-D saw nearby furniture-like geometry, the robot footprint
    corridor was clear, therefore this floor corridor is worth cleaning later
    when the robot is close to it.
    """
    mask = under_surface_log_odds > UNDER_SURFACE_EPS
    if obstacles is not None:
        mask = mask & (~obstacles)
    return mask


def cleanup_raw_map_speckles():
    """Decay tiny weak obstacle speckles in the persistent debug/raw map.

    makes this less naive: isolated strong-looking one-frame dots are
    decayed, but clusters that participate in an internal furniture zone are
    protected.  This reduces black salt-and-pepper noise without deleting stable
    chair/table legs.
    """
    global last_raw_map_speckle_debug, log_odds, thin_obstacle_log_odds, visual_log_odds, structural_log_odds
    if not RAW_MAP_SPECKLE_CLEANUP_ENABLED:
        last_raw_map_speckle_debug = "rawNoise=off"
        return 0
    try:
        if THIN_OBSTACLE_CONFIRM_ENABLED:
            thin_obstacle_log_odds[:, :] *= float(THIN_OBSTACLE_CONFIRM_DECAY)
        weak_obs = log_odds > LO_OCCUPIED_EPS
        if not bool(np.any(weak_obs)):
            last_raw_map_speckle_debug = "rawNoise=none"
            return 0

        furniture_core = np.zeros_like(weak_obs, dtype=np.bool_)
        if FURNITURE_ZONE_DETECTION_ENABLED:
            try:
                furniture_core, _inflated = update_furniture_zone_cache(force=True)
            except Exception:
                furniture_core = np.zeros_like(weak_obs, dtype=np.bool_)
        contact_occ = contact_log_odds > CONTACT_OCCUPIED_EPS
        structural_strong = (structural_log_odds > (STRUCTURAL_OCCUPIED_EPS + 0.75)) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(weak_obs, dtype=np.bool_)
        visual_strong = visual_log_odds > (CV_DISPLAY_DENSE_EPS + 1.10)
        log_strong = log_odds > (LO_OCCUPIED_EPS + RAW_MAP_SPECKLE_KEEP_STRONG_MARGIN + 0.35)
        hyp_strong = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(weak_obs, dtype=np.bool_)
        thin_confirmed = thin_obstacle_log_odds > THIN_OBSTACLE_CONFIRM_EPS if THIN_OBSTACLE_CONFIRM_ENABLED else np.zeros_like(weak_obs, dtype=np.bool_)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(weak_obs.astype(np.uint8), 8)
        remove = np.zeros_like(weak_obs, dtype=np.bool_)
        comps = 0
        cells = 0
        kept_furniture = 0
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            comp = labels == cid
            if bool(np.any(comp & furniture_core)):
                kept_furniture += 1
                continue
            # Contact and coherent structural memory are hard evidence.  A single
            # isolated visual/log spike is not; it must either be a larger component
            # or part of a furniture cluster to survive.
            hard_protected = bool(np.any(comp & (contact_occ | structural_strong | hyp_strong | thin_confirmed)))
            bbox_area = max(1, int(width) * int(height))
            density = float(area) / float(bbox_area)
            narrow = min(int(width), int(height))
            touches_outer = bool(
                int(stats[cid, cv2.CC_STAT_LEFT]) <= 2
                or int(stats[cid, cv2.CC_STAT_TOP]) <= 2
                or int(stats[cid, cv2.CC_STAT_LEFT]) + int(width) >= MAP_SIZE - 3
                or int(stats[cid, cv2.CC_STAT_TOP]) + int(height) >= MAP_SIZE - 3
            )
            wall_like = bool(touches_outer or span >= RAW_MAP_RAY_STREAK_WALL_PROTECT_SPAN_PX or area >= RAW_MAP_RAY_STREAK_MAX_AREA_PX)
            strong_pixels = int(np.count_nonzero(comp & (visual_strong | log_strong)))
            ray_streak = bool(
                RAW_MAP_RAY_STREAK_CLEANUP_ENABLED
                and (not hard_protected)
                and strong_pixels < 4
                and (not wall_like)
                and area <= RAW_MAP_RAY_STREAK_MAX_AREA_PX
                and span >= RAW_MAP_RAY_STREAK_MIN_SPAN_PX
                and (density <= RAW_MAP_RAY_STREAK_MAX_DENSITY or narrow <= RAW_MAP_RAY_STREAK_MAX_NARROW_PX)
            )
            soft_protected = bool(area > RAW_MAP_SPECKLE_MIN_AREA_PX or span > RAW_MAP_SPECKLE_MIN_SPAN_PX) and bool(strong_pixels > 0)
            if ray_streak:
                remove[comp] = True
                comps += 1
                cells += area
                continue
            if hard_protected or soft_protected:
                continue
            if area <= RAW_MAP_SPECKLE_MIN_AREA_PX and span <= RAW_MAP_SPECKLE_MIN_SPAN_PX:
                remove[comp] = True
                comps += 1
                cells += area
        if cells > 0:
            log_odds[remove] = np.minimum(log_odds[remove], RAW_MAP_SPECKLE_DECAY_TO)
            visual_log_odds[remove] = np.minimum(visual_log_odds[remove], CV_DISPLAY_LIGHT_EPS * 0.25)
            if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
                structural_log_odds[remove] = np.minimum(structural_log_odds[remove], STRUCTURAL_MIN + 0.05)
            if THIN_OBSTACLE_CONFIRM_ENABLED:
                thin_obstacle_log_odds[remove] = 0.0
        last_raw_map_speckle_debug = f"rawNoise-{cells}/{comps} keepFurn={kept_furniture}"
        return cells
    except Exception as exc:
        last_raw_map_speckle_debug = f"rawNoise=err {type(exc).__name__}"
        return 0


def filter_objective_noise_masks(obstacles, cleanable, cleaned, uncleaned, unknown):
    """Filter objective-layer speckles without rewriting the raw RGB-D map.

    The raw log-odds/visual layers may contain small isolated black points from
    RGB edge/depth noise.  Treating those points as real geometry makes the CPP
    planner draw small squares and reject otherwise clean wall lanes.  This
    function only filters the planning/debug objective masks; contact-confirmed
    obstacles, coherent wall/furniture components, and strong structural memory
    remain obstacles.
    """
    global last_objective_noise_filter_debug, objective_noise_filter_cache, objective_noise_filter_cache_step, objective_noise_filter_cache_debug
    if not OBJECTIVE_NOISE_FILTER_ENABLED:
        last_objective_noise_filter_debug = "off"
        return obstacles, cleanable, cleaned, uncleaned, unknown

    if OBJECTIVE_NOISE_FILTER_CACHE_ENABLED and objective_noise_filter_cache is not None:
        try:
            age = int(step_id) - int(objective_noise_filter_cache_step)
            if 0 <= age < max(1, int(OBJECTIVE_NOISE_FILTER_UPDATE_STEPS)):
                last_objective_noise_filter_debug = f"cache{age}:{objective_noise_filter_cache_debug}"[:48]
                return tuple(arr.copy() for arr in objective_noise_filter_cache)
        except Exception:
            pass

    t_perf_noise = perf_start()
    try:
        obs = obstacles.astype(np.bool_).copy()
        contact_occ = contact_log_odds > CONTACT_OCCUPIED_EPS
        structural_strong = (structural_log_odds > (STRUCTURAL_OCCUPIED_EPS + 1.15)) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(obs, dtype=np.bool_)
        log_strong = log_odds > (LO_OCCUPIED_EPS + 1.10)
        visual_strong = visual_log_odds > (CV_DISPLAY_DENSE_EPS + 0.65)
        hyp_strong = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(obs, dtype=np.bool_)
        thin_confirmed = thin_obstacle_log_odds > THIN_OBSTACLE_CONFIRM_EPS if THIN_OBSTACLE_CONFIRM_ENABLED else np.zeros_like(obs, dtype=np.bool_)

        keep_obs = np.zeros_like(obs, dtype=np.bool_)
        removed_obs_cells = 0
        components_seen = 0
        n_obs, labels, stats, _centroids = cv2.connectedComponentsWithStats(obs.astype(np.uint8), 8)
        for cid in range(1, int(n_obs)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            components_seen += 1
            left = int(stats[cid, cv2.CC_STAT_LEFT])
            top = int(stats[cid, cv2.CC_STAT_TOP])
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            comp = labels == cid
            contact_count = int(np.count_nonzero(comp & contact_occ))
            protected_count = int(np.count_nonzero(comp & (contact_occ | structural_strong | hyp_strong | thin_confirmed)))
            strong_count = int(np.count_nonzero(comp & (structural_strong | log_strong | visual_strong | hyp_strong | thin_confirmed)))
            bbox_area = max(1, int(width) * int(height))
            density = float(area) / float(bbox_area)
            narrow = min(int(width), int(height))
            touches_outer = bool(left <= 2 or top <= 2 or left + width >= MAP_SIZE - 3 or top + height >= MAP_SIZE - 3)
            wall_like = bool(touches_outer or span >= RAW_MAP_RAY_STREAK_WALL_PROTECT_SPAN_PX or area >= OBJECTIVE_RAY_STREAK_MAX_AREA_PX)
            ray_streak = bool(
                OBJECTIVE_RAY_STREAK_FILTER_ENABLED
                and protected_count <= 0
                and strong_count < max(2, OBJECTIVE_OBSTACLE_KEEP_STRONG_COUNT)
                and (not wall_like)
                and area <= OBJECTIVE_RAY_STREAK_MAX_AREA_PX
                and span >= OBJECTIVE_RAY_STREAK_MIN_SPAN_PX
                and (density <= OBJECTIVE_RAY_STREAK_MAX_DENSITY or narrow <= OBJECTIVE_RAY_STREAK_MAX_NARROW_PX)
            )
            keep = bool(
                (not ray_streak)
                and (
                    area >= OBJECTIVE_OBSTACLE_SPECKLE_MIN_AREA_PX
                    or span >= OBJECTIVE_OBSTACLE_SPECKLE_MIN_SPAN_PX
                    or protected_count > 0
                    or strong_count >= min(area, OBJECTIVE_OBSTACLE_KEEP_STRONG_COUNT)
                )
            )
            if keep:
                keep_obs[comp] = True
            else:
                removed_obs_cells += area

        removed_obs = obs & (~keep_obs)
        obstacles2 = keep_obs

        # If an obstacle speckle was sitting inside known floor, expose that floor
        # again for objective planning.  Unknown gray regions are not opened just
        # because a black dot was removed.
        k = np.ones((2 * OBJECTIVE_NOISE_NEAR_FLOOR_DILATE_PX + 1, 2 * OBJECTIVE_NOISE_NEAR_FLOOR_DILATE_PX + 1), np.uint8)
        near_cleanable = cv2.dilate(cleanable.astype(np.uint8), k, iterations=1) > 0
        near_cleaned = cv2.dilate(cleaned.astype(np.uint8), k, iterations=1) > 0
        cleanable2 = (cleanable | (removed_obs & near_cleanable)) & (~obstacles2)
        cleaned2 = (cleaned | (removed_obs & near_cleaned)) & cleanable2
        uncleaned2 = cleanable2 & (~cleaned2)

        # Remove tiny isolated dirty islands inside already explored/cleaned floor.
        # Wall-adjacent bands are preserved, because those are exactly the edge
        # strips a robot vacuum must deliberately clean.
        removed_dirty_cells = 0
        if OBJECTIVE_UNCLEANED_SPECKLE_MIN_AREA_PX > 0:
            wall_near = cv2.dilate(obstacles2.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            cleaned_near = cv2.dilate(cleaned2.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            n_dirty, dlabels, dstats, _ = cv2.connectedComponentsWithStats(uncleaned2.astype(np.uint8), 8)
            dirty_noise = np.zeros_like(uncleaned2, dtype=np.bool_)
            for cid in range(1, int(n_dirty)):
                area = int(dstats[cid, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue
                width = int(dstats[cid, cv2.CC_STAT_WIDTH])
                height = int(dstats[cid, cv2.CC_STAT_HEIGHT])
                span = max(width, height)
                if area >= OBJECTIVE_UNCLEANED_SPECKLE_MIN_AREA_PX or span >= OBJECTIVE_UNCLEANED_SPECKLE_MIN_SPAN_PX:
                    continue
                comp = dlabels == cid
                if OBJECTIVE_UNCLEANED_KEEP_WALL_ADJACENT and bool(np.any(comp & wall_near)):
                    continue
                # Only suppress speckles embedded in known floor/cleaned corridors.
                if bool(np.any(comp & cleaned_near)):
                    dirty_noise[comp] = True
                    removed_dirty_cells += area
            if removed_dirty_cells > 0:
                cleaned2[dirty_noise] = True
                uncleaned2 = cleanable2 & (~cleaned2)

        unknown2 = unknown & (~cleanable2) & (~obstacles2)
        last_objective_noise_filter_debug = (
            f"obs-{removed_obs_cells}/{components_seen} dirty-{removed_dirty_cells}"
        )
        result = (obstacles2, cleanable2, cleaned2, uncleaned2, unknown2)
        if OBJECTIVE_NOISE_FILTER_CACHE_ENABLED:
            objective_noise_filter_cache = tuple(arr.copy() for arr in result)
            objective_noise_filter_cache_step = int(step_id)
            objective_noise_filter_cache_debug = last_objective_noise_filter_debug
        perf_end("noise", t_perf_noise)
        return result
    except Exception as exc:
        perf_end("noise", t_perf_noise)
        last_objective_noise_filter_debug = f"err {type(exc).__name__}"
        return obstacles, cleanable, cleaned, uncleaned, unknown


def known_map_coverage_eval_active():
    return bool(KNOWN_MAP_COVERAGE_EVAL_DEFAULT or known_map_eval_runtime_enabled)


def wbt_to_controller_xy(x_wbt, y_wbt):
    """Convert WBT world coordinates into this controller's odometry frame."""
    return float(x_wbt) - KNOWN_MAP_WBT_ROBOT_START_X, float(y_wbt) - KNOWN_MAP_WBT_ROBOT_START_Y


def known_map_eval_arena_bounds_px():
    """Return (x_left, x_right, y_top, y_bottom) for the seeded arena in map px."""
    try:
        xl, yt = world_to_map(KNOWN_MAP_ARENA_X_MIN_M, KNOWN_MAP_ARENA_Y_MAX_M)
        xr, yb = world_to_map(KNOWN_MAP_ARENA_X_MAX_M, KNOWN_MAP_ARENA_Y_MIN_M)
        x_left = max(0, min(int(xl), int(xr)))
        x_right = min(MAP_SIZE - 1, max(int(xl), int(xr)))
        y_top = max(0, min(int(yt), int(yb)))
        y_bottom = min(MAP_SIZE - 1, max(int(yt), int(yb)))
        return x_left, x_right, y_top, y_bottom
    except Exception:
        return 0, MAP_SIZE - 1, 0, MAP_SIZE - 1


def known_map_eval_sweep_priority_score(cx, cy, route_cost_m=0.0):
    """Deterministic sweep-order bonus for KNOWN_MAP_COVERAGE_EVAL.

    The normal coverage scorer is opportunistic: it picks high-gain residuals.
    For evaluation with a known static map we need a reproducible coverage order:
    top-to-bottom horizontal lanes, alternating left-to-right / right-to-left.
    This makes the test answerable: if an upper strip is still uncleaned, it wins
    over a nearer lower leftover unless the upper strip is not reachable.
    """
    if not (KNOWN_MAP_EVAL_SWEEP_PRIORITY_ENABLED and known_map_coverage_eval_active()):
        return 0.0, "knownSweep=off"
    try:
        x_left, x_right, y_top, y_bottom = known_map_eval_arena_bounds_px()
        lane_px = max(3, int(round(KNOWN_MAP_EVAL_SWEEP_LANE_SPACING_M * MAP_SCALE)))
        cy_i = int(clamp(int(cy), y_top, y_bottom))
        cx_i = int(clamp(int(cx), x_left, x_right))
        lane = max(0, int((cy_i - y_top) // lane_px))
        max_lane = max(1, int((y_bottom - y_top) // lane_px))
        lane_priority = max(0, max_lane - lane)
        x_span = max(1, int(x_right - x_left))
        x_norm = (cx_i - x_left) / float(x_span)
        # Even lanes: left -> right. Odd lanes: right -> left.  Use a small
        # within-lane term; lane order remains the dominant criterion.
        dir_progress = (1.0 - x_norm) if (lane % 2 == 0) else x_norm
        score = (
            KNOWN_MAP_EVAL_SWEEP_LANE_PRIORITY_BONUS * float(lane_priority)
            + KNOWN_MAP_EVAL_SWEEP_DIRECTION_BONUS * float(dir_progress)
            - KNOWN_MAP_EVAL_SWEEP_ROUTE_COST_WEIGHT * float(max(0.0, route_cost_m))
        )
        return float(score), f"knownSweep=lane{lane}/{max_lane} dir={dir_progress:.2f}"
    except Exception as exc:
        return 0.0, f"knownSweep=err:{type(exc).__name__}"


def known_map_eval_strip_route_from_grid(
    x0, y0, step, gw, gh, passable, dist_grid, parent, start,
    coarse_footprint_gain_cells, coarse_footprint_gain_ratio, coarse_footprint_reclean_ratio,
):
    """Return the known-map coverage route that ROUTE_COMMIT should execute.

    still planned one strip at a time.  That made the robot look as if it
    was driving to a far uncleaned point instead of executing a defendable room
    coverage pattern.  builds a bounded boustrophedon sweep over the known
    occupancy map: horizontal lane segments are ordered top-to-bottom, alternating
    direction, and Dijkstra connectors are inserted between disconnected pieces.

    This is still a diploma-level coverage planner, not industrial optimal CPP.
    The important contract is stronger: in known-map evaluation, the orange route
    is the actual committed route, and the executor is not allowed to jump to a
    later parallel lane just because it is geometrically closer.
    """
    if not (known_map_coverage_eval_active() and KNOWN_MAP_EVAL_STRIP_ROUTE_ENABLED):
        return None
    if (
        bool(KNOWN_MAP_EVAL_PRIMARY_SWEEP_ONCE)
        and bool(KNOWN_MAP_EVAL_FULL_SWEEP_ROUTE_ENABLED)
        and bool(known_map_primary_sweep_completed)
    ):
        # The deterministic full-room sweep is a primary mission phase.  After it
        # completes, do not rebuild another enormous sweep over tiny residuals;
        # let the ordinary component scorer produce at most a few bounded mop-up
        # commits, then return home.
        return None
    try:
        x_left, x_right, y_top, y_bottom = known_map_eval_arena_bounds_px()
        lane_px = max(3, int(round(KNOWN_MAP_EVAL_SWEEP_LANE_SPACING_M * MAP_SCALE)))
        useful = passable & (coarse_footprint_gain_cells >= int(KNOWN_MAP_EVAL_STRIP_MIN_COARSE_GAIN))
        if not np.any(useful):
            return None

        def coarse_to_map_cell(cell):
            gx, gy = cell
            return coarse_route_center(x0, y0, step, int(gx), int(gy))

        def coarse_cells_to_map(cells):
            return [coarse_to_map_cell(c) for c in cells]

        def append_map_points(dst, pts):
            for p in pts:
                pp = (int(p[0]), int(p[1]))
                if dst and dst[-1] == pp:
                    continue
                dst.append(pp)

        def route_length_m(route):
            if not route or len(route) < 2:
                return 0.0
            total = 0.0
            for a, b in zip(route[:-1], route[1:]):
                total += math.hypot(float(b[0] - a[0]), float(b[1] - a[1])) / float(MAP_SCALE)
            return float(total)

        def grid_connector(src, dst):
            """Cardinal Dijkstra connector on the coarse passable grid."""
            sx, sy = int(src[0]), int(src[1])
            gx, gy = int(dst[0]), int(dst[1])
            if sx == gx and sy == gy:
                return [(sx, sy)]
            if sx < 0 or sy < 0 or gx < 0 or gy < 0 or sx >= gw or gx >= gw or sy >= gh or gy >= gh:
                return None
            if not bool(passable[sy, sx]) or not bool(passable[gy, gx]):
                return None
            inf = 1e18
            local_dist = np.full((int(gh), int(gw)), inf, dtype=np.float32)
            local_parent = np.full((int(gh), int(gw)), -1, dtype=np.int32)
            local_dist[sy, sx] = 0.0
            heap = [(0.0, sx, sy)]
            expanded = 0
            nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while heap:
                d, cx, cy = heapq.heappop(heap)
                if d > float(local_dist[cy, cx]) + 1e-6:
                    continue
                expanded += 1
                if expanded > int(KNOWN_MAP_EVAL_FULL_SWEEP_CONNECTOR_MAX_NODES):
                    break
                if cx == gx and cy == gy:
                    break
                for dx, dy in nbrs:
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or ny < 0 or nx >= gw or ny >= gh or not bool(passable[ny, nx]):
                        continue
                    # Prefer transit through useful cells and slightly avoid fully
                    # recleaned cells, but keep the connector stable and cardinal.
                    cost_factor = 1.0
                    try:
                        cost_factor += 0.18 * float(coarse_footprint_reclean_ratio[ny, nx])
                        cost_factor -= 0.10 * float(coarse_footprint_gain_ratio[ny, nx])
                        cost_factor = max(0.72, cost_factor)
                    except Exception:
                        cost_factor = 1.0
                    nd = d + cost_factor
                    if nd < float(local_dist[ny, nx]):
                        local_dist[ny, nx] = nd
                        local_parent[ny, nx] = cy * int(gw) + cx
                        heapq.heappush(heap, (float(nd), nx, ny))
            if not math.isfinite(float(local_dist[gy, gx])) or float(local_dist[gy, gx]) >= inf * 0.5:
                return None
            cells = []
            cur = (gx, gy)
            guard = 0
            while guard < int(gw) * int(gh) + 2:
                cells.append(cur)
                if cur == (sx, sy):
                    break
                flat = int(local_parent[cur[1], cur[0]])
                if flat < 0:
                    return None
                cur = (flat % int(gw), flat // int(gw))
                guard += 1
            cells.reverse()
            return cells

        # Build candidate rows per physical sweep lane.  The coarse grid is
        # denser than the cleaning lane spacing, so using every gy would create
        # gain-driven representative for interior lanes, but explicitly anchors
        # the first and last physical lanes to the nearest reachable wall-side row.
        # Without this, a high-gain row in the same bottom lane can be selected a
        # little too high, leaving the exact blue band highlighted near the lower
        # wall / under the low obstacle.
        lane_row_candidates = {}
        for gy in range(int(gh)):
            cy = y0 + gy * step + step // 2
            if cy < y_top or cy > y_bottom:
                continue
            lane_id = max(0, int((int(cy) - int(y_top)) // lane_px))
            edge_lane_guess = bool(
                int(cy) <= int(y_top) + int(lane_px)
                or int(cy) >= int(y_bottom) - int(lane_px)
            )
            segments = []
            row_gain = 0
            gx = 0
            while gx < int(gw):
                if not bool(passable[gy, gx]):
                    gx += 1
                    continue
                seg_start = gx
                while gx < int(gw) and bool(passable[gy, gx]):
                    gx += 1
                seg_end = gx - 1
                work_xs = [ix for ix in range(seg_start, seg_end + 1) if bool(useful[gy, ix])]
                if not work_xs:
                    continue
                total_gain = int(np.sum(coarse_footprint_gain_cells[gy, work_xs]))
                min_total_gain = int(KNOWN_MAP_EVAL_STRIP_MIN_TOTAL_GAIN)
                if bool(KNOWN_MAP_EVAL_EDGE_LANE_BIAS_ENABLED) and edge_lane_guess:
                    min_total_gain = min(min_total_gain, int(KNOWN_MAP_EVAL_EDGE_LANE_MIN_TOTAL_GAIN))
                if total_gain < int(min_total_gain):
                    continue
                left_work = max(seg_start, min(work_xs) - int(KNOWN_MAP_EVAL_STRIP_EDGE_MARGIN_CELLS))
                right_work = min(seg_end, max(work_xs) + int(KNOWN_MAP_EVAL_STRIP_EDGE_MARGIN_CELLS))
                if right_work <= left_work:
                    continue
                # Keep endpoints inside passable cells.  This matters near legs and
                # inflated obstacles where the useful footprint may touch the edge.
                while left_work <= right_work and not bool(passable[gy, left_work]):
                    left_work += 1
                while right_work >= left_work and not bool(passable[gy, right_work]):
                    right_work -= 1
                if right_work <= left_work:
                    continue
                mean_gain_ratio = float(np.mean(coarse_footprint_gain_ratio[gy, work_xs])) if work_xs else 0.0
                mean_reclean = float(np.mean(coarse_footprint_reclean_ratio[gy, work_xs])) if work_xs else 0.0
                segments.append({
                    "lane_id": int(lane_id),
                    "gy": int(gy),
                    "left": int(left_work),
                    "right": int(right_work),
                    "gain": int(total_gain),
                    "gain_ratio": float(mean_gain_ratio),
                    "reclean": float(mean_reclean),
                    "edge": bool(edge_lane_guess),
                })
                row_gain += int(total_gain)
            if segments:
                lane_row_candidates.setdefault(int(lane_id), []).append({
                    "gy": int(gy),
                    "row_gain": int(row_gain),
                    "segments": segments,
                })

        rows_by_lane = {}
        if lane_row_candidates:
            min_lane_id = min(lane_row_candidates.keys())
            max_lane_id = max(lane_row_candidates.keys())
            for lane_id, candidates in lane_row_candidates.items():
                cand_list = list(candidates)
                if not cand_list:
                    continue
                if bool(KNOWN_MAP_EVAL_EDGE_LANE_BIAS_ENABLED) and int(lane_id) == int(min_lane_id):
                    chosen_row = min(cand_list, key=lambda r: int(r.get("gy", 0)))
                elif bool(KNOWN_MAP_EVAL_EDGE_LANE_BIAS_ENABLED) and int(lane_id) == int(max_lane_id):
                    chosen_row = max(cand_list, key=lambda r: int(r.get("gy", 0)))
                else:
                    # Interior lanes stay gain-driven; tie-break toward the lane
                    # centre to avoid unnecessary overlap/re-cleaning.
                    ideal_cy = int(y_top) + int(lane_id) * int(lane_px) + int(lane_px) // 2
                    ideal_gy = int(round((ideal_cy - int(y0) - int(step) // 2) / max(1, int(step))))
                    chosen_row = max(
                        cand_list,
                        key=lambda r: (int(r.get("row_gain", 0)), -abs(int(r.get("gy", 0)) - int(ideal_gy))),
                    )
                rows_by_lane[int(lane_id)] = chosen_row

        ordered_segments = []
        sweep_order_debug = "order=legacy"

        def orient_segment(seg, eastward):
            ss = dict(seg)
            if eastward:
                ss["entry"] = (int(ss["left"]), int(ss["gy"]))
                ss["exit"] = (int(ss["right"]), int(ss["gy"]))
                ss["dir"] = "E"
            else:
                ss["entry"] = (int(ss["right"]), int(ss["gy"]))
                ss["exit"] = (int(ss["left"]), int(ss["gy"]))
                ss["dir"] = "W"
            return ss

        def oriented_lane_segments(segs, eastward):
            """Return lane segments with concrete entry/exit direction.

            The old strict parity rule made the drawing deterministic, but it was
            not always a good path: after an obstacle split a lane, the next lane
            could start from the far side even when the robot was already near the
            other endpoint.  Keep top-to-bottom sweep order, but choose the
            direction of each lane from the current route end when the optimizer is
            disabled.
            """
            src = sorted(list(segs), key=lambda s: int(s["left"])) if eastward else sorted(list(segs), key=lambda s: int(s["right"]), reverse=True)
            return [orient_segment(seg, eastward) for seg in src]

        def rough_variant_cost(current_cell, variant):
            if not variant:
                return float("inf")
            cx, cy = int(current_cell[0]), int(current_cell[1])
            total = 0.0
            for seg in variant:
                ex, ey = seg["entry"]
                lx, ly = seg["exit"]
                total += abs(int(ex) - cx) + abs(int(ey) - cy)
                total += abs(int(lx) - int(ex)) + abs(int(ly) - int(ey))
                cx, cy = int(lx), int(ly)
            return float(total)

        dock_goal_grid = None
        try:
            dmx, dmy = world_to_map(DOCK_TARGET_X, DOCK_TARGET_Y)
            dgx = int((int(dmx) - int(x0)) // max(1, int(step)))
            dgy = int((int(dmy) - int(y0)) // max(1, int(step)))
            if 0 <= dgx < int(gw) and 0 <= dgy < int(gh):
                dock_goal_grid = (int(dgx), int(dgy))
        except Exception:
            dock_goal_grid = None

        def legacy_top_down_order():
            out = []
            current_for_order = (int(start[0]), int(start[1]))
            lane_order_local = sorted(rows_by_lane.keys())
            for lane_pos, lane_id in enumerate(lane_order_local):
                row = rows_by_lane[lane_id]
                base_segs = list(row["segments"])
                east_variant = oriented_lane_segments(base_segs, True)
                west_variant = oriented_lane_segments(base_segs, False)
                if bool(KNOWN_MAP_EVAL_SWEEP_ADAPTIVE_LANE_DIRECTION):
                    east_cost = rough_variant_cost(current_for_order, east_variant)
                    west_cost = rough_variant_cost(current_for_order, west_variant)
                    if (
                        bool(KNOWN_MAP_EVAL_DOCK_END_BIAS_ENABLED)
                        and dock_goal_grid is not None
                        and int(lane_pos) == len(lane_order_local) - 1
                    ):
                        for variant_name, variant in (("east", east_variant), ("west", west_variant)):
                            if not variant:
                                continue
                            ex, ey = variant[-1]["exit"]
                            terminal_cost = abs(int(ex) - int(dock_goal_grid[0])) + abs(int(ey) - int(dock_goal_grid[1]))
                            if variant_name == "east":
                                east_cost += float(KNOWN_MAP_EVAL_DOCK_END_BIAS_WEIGHT) * float(terminal_cost)
                            else:
                                west_cost += float(KNOWN_MAP_EVAL_DOCK_END_BIAS_WEIGHT) * float(terminal_cost)
                    segs = east_variant if east_cost <= west_cost else west_variant
                else:
                    segs = east_variant if (int(lane_id) % 2 == 0) else west_variant
                out.extend(segs)
                if segs:
                    current_for_order = (int(segs[-1]["exit"][0]), int(segs[-1]["exit"][1]))
            return out

        def segment_len_cells(seg):
            return float(abs(int(seg["right"]) - int(seg["left"])))

        def segment_pose(seg, eastward):
            ss = orient_segment(seg, bool(eastward))
            return ss["entry"], ss["exit"], segment_len_cells(ss)

        connector_len_cache = {}

        def approx_connector_cells(a, b):
            ax, ay = int(a[0]), int(a[1])
            bx, by = int(b[0]), int(b[1])
            base = float(abs(bx - ax) + abs(by - ay))
            # A small row-jump term discourages visually noisy jumps across many
            # lanes when the Manhattan distance is tied, but it is deliberately
            # small so obstacles/short connectors remain dominant.
            base += float(KNOWN_MAP_EVAL_OPTIMIZER_ROW_JUMP_PENALTY) * float(abs(by - ay))
            return float(base)

        def exact_connector_cells(a, b):
            key = (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            if key in connector_len_cache:
                return connector_len_cache[key]
            rev = (key[2], key[3], key[0], key[1])
            if rev in connector_len_cache:
                return connector_len_cache[rev]
            path = grid_connector((key[0], key[1]), (key[2], key[3]))
            if not path:
                val = float("inf")
            else:
                val = 0.0
                for aa, bb in zip(path[:-1], path[1:]):
                    val += abs(int(bb[0]) - int(aa[0])) + abs(int(bb[1]) - int(aa[1]))
            connector_len_cache[key] = float(val)
            return float(val)

        def seq_cost(seq, exact=False):
            if not seq:
                return float("inf")
            conn = exact_connector_cells if exact else approx_connector_cells
            cur = (int(start[0]), int(start[1]))
            total = 0.0
            last_lane = None
            for idx, eastward in seq:
                base_seg = opt_segments[int(idx)]
                entry, exit_, seg_len = segment_pose(base_seg, bool(eastward))
                c = conn(cur, entry)
                if not math.isfinite(float(c)):
                    return float("inf")
                total += float(c) + float(seg_len)
                if last_lane is not None and int(base_seg.get("lane_id", 0)) != int(last_lane):
                    total += 0.05 * abs(int(base_seg.get("lane_id", 0)) - int(last_lane))
                last_lane = int(base_seg.get("lane_id", 0))
                cur = exit_
            if bool(KNOWN_MAP_EVAL_DOCK_END_BIAS_ENABLED) and dock_goal_grid is not None:
                total += float(KNOWN_MAP_EVAL_OPTIMIZER_CONNECTOR_DOCK_BIAS) * float(conn(cur, dock_goal_grid))
            return float(total)

        def held_karp_oriented_order(items):
            n = len(items)
            if n <= 0:
                return []
            if n > int(KNOWN_MAP_EVAL_OPTIMIZER_EXACT_MAX_SEGMENTS):
                return None
            # State: (mask, last_segment_index, last_orientation_bit).  The cost
            # includes the connector from the previous exit to this entry and the
            # cost of traversing this strip segment once.  Keep states grouped by
            # mask; scanning a single global dict for every mask made the exact
            # solver needlessly expensive.
            full_mask = (1 << n) - 1
            dp_by_mask = [dict() for _ in range(full_mask + 1)]
            parent_state = {}
            for i in range(n):
                for d in (0, 1):
                    entry, exit_, seg_len = segment_pose(items[i], bool(d))
                    c = exact_connector_cells(start, entry)
                    if not math.isfinite(float(c)):
                        continue
                    mask = 1 << i
                    key = (i, d)
                    dp_by_mask[mask][key] = float(c) + float(seg_len)
                    parent_state[(mask, i, d)] = None
            for mask in range(1, full_mask + 1):
                if not dp_by_mask[mask]:
                    continue
                for (last, last_d), base_cost in list(dp_by_mask[mask].items()):
                    last_exit = segment_pose(items[int(last)], bool(last_d))[1]
                    for nxt in range(n):
                        if mask & (1 << nxt):
                            continue
                        for nd in (0, 1):
                            entry, exit_, seg_len = segment_pose(items[nxt], bool(nd))
                            c = exact_connector_cells(last_exit, entry)
                            if not math.isfinite(float(c)):
                                continue
                            nmask = mask | (1 << nxt)
                            nkey = (nxt, nd)
                            val = float(base_cost) + float(c) + float(seg_len)
                            old = dp_by_mask[nmask].get(nkey)
                            if old is None or val < float(old):
                                dp_by_mask[nmask][nkey] = float(val)
                                parent_state[(nmask, nxt, nd)] = (mask, last, last_d)
            best_state = None
            best_cost = float("inf")
            for (last, last_d), val in dp_by_mask[full_mask].items():
                last_exit = segment_pose(items[int(last)], bool(last_d))[1]
                term = 0.0
                if bool(KNOWN_MAP_EVAL_DOCK_END_BIAS_ENABLED) and dock_goal_grid is not None:
                    term = float(KNOWN_MAP_EVAL_OPTIMIZER_CONNECTOR_DOCK_BIAS) * float(exact_connector_cells(last_exit, dock_goal_grid))
                total = float(val) + float(term)
                if total < best_cost:
                    best_cost = total
                    best_state = (full_mask, int(last), int(last_d))
            if best_state is None:
                return None
            seq = []
            cur_state = best_state
            guard = 0
            while cur_state is not None and guard <= n + 2:
                mask, last, d = cur_state
                seq.append((int(last), bool(d)))
                cur_state = parent_state.get(cur_state)
                guard += 1
            seq.reverse()
            if len(seq) != n:
                return None
            return seq


        def greedy_2opt_order(items):
            n = len(items)
            if n <= 0:
                return []
            remaining = set(range(n))
            cur = (int(start[0]), int(start[1]))
            seq = []
            while remaining:
                best = None
                best_val = float("inf")
                for i in list(remaining):
                    for d in (False, True):
                        entry, exit_, seg_len = segment_pose(items[i], d)
                        val = approx_connector_cells(cur, entry) + 0.35 * float(seg_len)
                        # Prefer high-gain strips when connector distances tie;
                        # every segment is still visited exactly once.
                        val -= 0.002 * float(items[i].get("gain", 0))
                        if val < best_val:
                            best_val = float(val)
                            best = (i, d, exit_)
                if best is None:
                    break
                i, d, exit_ = best
                seq.append((int(i), bool(d)))
                remaining.remove(int(i))
                cur = exit_
            if len(seq) != n:
                return seq

            best_seq = list(seq)
            best_score = seq_cost(best_seq, exact=False)
            passes = max(0, int(KNOWN_MAP_EVAL_OPTIMIZER_2OPT_PASSES))
            for _ in range(passes):
                improved = False
                for i in range(0, max(0, n - 1)):
                    for j in range(i + 1, n):
                        candidate = (
                            best_seq[:i]
                            + [(idx, not d) for idx, d in reversed(best_seq[i:j + 1])]
                            + best_seq[j + 1:]
                        )
                        c = seq_cost(candidate, exact=False)
                        if c + 1e-6 < best_score:
                            best_seq = candidate
                            best_score = float(c)
                            improved = True
                if not improved:
                    break
            return best_seq

        def optimized_order():
            work = []
            for lane_id in sorted(rows_by_lane.keys()):
                row = rows_by_lane[lane_id]
                for seg in sorted(list(row.get("segments", [])), key=lambda s: int(s.get("left", 0))):
                    work.append(dict(seg))
            if not work:
                return []
            if len(work) > int(KNOWN_MAP_EVAL_FULL_SWEEP_MAX_SEGMENTS):
                work = work[:int(KNOWN_MAP_EVAL_FULL_SWEEP_MAX_SEGMENTS)]
            return work

        opt_segments = optimized_order()
        if bool(KNOWN_MAP_EVAL_OPTIMIZED_SWEEP_ORDER_ENABLED) and opt_segments:
            seq = held_karp_oriented_order(opt_segments)
            if seq is not None:
                ordered_segments = [orient_segment(opt_segments[idx], eastward) for idx, eastward in seq]
                sweep_order_debug = f"order=dp n={len(seq)}"
            else:
                seq = greedy_2opt_order(opt_segments)
                ordered_segments = [orient_segment(opt_segments[idx], eastward) for idx, eastward in seq]
                sweep_order_debug = f"order=2opt n={len(seq)}"
        else:
            ordered_segments = legacy_top_down_order()
            sweep_order_debug = f"order=legacy n={len(ordered_segments)}"

        if not ordered_segments:
            ordered_segments = legacy_top_down_order()
            sweep_order_debug = f"order=fallback n={len(ordered_segments)}"

        if not ordered_segments:
            return None
        ordered_segments = ordered_segments[:int(KNOWN_MAP_EVAL_FULL_SWEEP_MAX_SEGMENTS)]

        route = []
        included = []
        current = (int(start[0]), int(start[1]))
        total_gain = 0
        weighted_gain_ratio = 0.0
        weighted_reclean = 0.0
        skipped = 0

        # it because the completed route was 180.5m while the commit gate was
        # 180.0m.  That is not a navigation decision; it is a planner/executor
        # contract bug.  Build a route chunk that is guaranteed to fit the same
        # budget used by ROUTE_COMMIT, with a small margin for map/world rounding.
        try:
            route_budget_m = min(
                float(KNOWN_MAP_EVAL_FULL_SWEEP_MAX_ROUTE_COST_M),
                float(KNOWN_MAP_EVAL_ROUTE_COMMIT_MAX_COST_M),
            ) - float(KNOWN_MAP_EVAL_FULL_SWEEP_BUDGET_MARGIN_M)
        except Exception:
            route_budget_m = float(KNOWN_MAP_EVAL_FULL_SWEEP_MAX_ROUTE_COST_M) - 1.0
        route_budget_m = max(6.0, float(route_budget_m))

        for seg in ordered_segments:
            entry = (int(seg["entry"][0]), int(seg["entry"][1]))
            exit_ = (int(seg["exit"][0]), int(seg["exit"][1]))
            connector = grid_connector(current, entry)
            if not connector:
                skipped += 1
                continue
            sign = 1 if exit_[0] >= entry[0] else -1
            lane_cells = []
            for ix in range(entry[0], exit_[0] + sign, sign):
                if 0 <= ix < int(gw) and bool(passable[entry[1], ix]):
                    lane_cells.append((int(ix), int(entry[1])))
            if len(lane_cells) < 2:
                skipped += 1
                continue

            old_route_len = len(route)
            old_current = current
            old_total_gain = total_gain
            old_weighted_gain_ratio = weighted_gain_ratio
            old_weighted_reclean = weighted_reclean

            append_map_points(route, coarse_cells_to_map(connector))
            append_map_points(route, coarse_cells_to_map(lane_cells))
            tentative_cost = route_length_m(route)

            if tentative_cost > route_budget_m and included:
                # Do not append a segment that makes the candidate uncommittable.
                # Replanning after this chunk will pick up the remaining lower
                # lanes from the updated cleaned mask.
                del route[old_route_len:]
                current = old_current
                total_gain = old_total_gain
                weighted_gain_ratio = old_weighted_gain_ratio
                weighted_reclean = old_weighted_reclean
                skipped += 1
                break

            current = exit_
            included.append(seg)
            g = int(seg["gain"])
            total_gain += g
            weighted_gain_ratio += float(seg["gain_ratio"]) * float(max(1, g))
            weighted_reclean += float(seg["reclean"]) * float(max(1, g))

            if tentative_cost > route_budget_m:
                # First reachable lane alone exceeded the budget.  Keep it rather
                # path rare.
                break

        if len(route) < 2 or not included or total_gain <= 0:
            return None

        route_cost_m = route_length_m(route)
        if route_cost_m > float(KNOWN_MAP_EVAL_ROUTE_COMMIT_MAX_COST_M):
            return None
        goal_map = (int(route[-1][0]), int(route[-1][1]))
        goal_world = ((goal_map[0] - MAP_ORIGIN_X) / MAP_SCALE, (MAP_ORIGIN_Y - goal_map[1]) / MAP_SCALE)
        wp_map, wp_world = route_waypoint_from_path(route)
        route_world = [((mx - MAP_ORIGIN_X) / MAP_SCALE, (MAP_ORIGIN_Y - my) / MAP_SCALE) for mx, my in route]
        geo = route_path_geometry_stats(route)
        first_turn = float(geo.get("first_turn_frac", 0.0))
        corners = int(geo.get("corner_count", 0))
        mean_gain_ratio = float(weighted_gain_ratio / max(1.0, float(total_gain)))
        mean_reclean = float(weighted_reclean / max(1.0, float(total_gain)))
        first_lane = int(included[0]["lane_id"])
        last_lane = int(included[-1]["lane_id"])
        score = (
            100000.0
            + KNOWN_MAP_EVAL_STRIP_GAIN_BONUS * float(total_gain)
            - KNOWN_MAP_EVAL_STRIP_ROUTE_COST_WEIGHT * float(route_cost_m)
            - KNOWN_MAP_EVAL_STRIP_TURN_PENALTY * float(first_turn)
            - KNOWN_MAP_EVAL_STRIP_CORNER_PENALTY * float(corners)
        )
        top_dbg = " | ".join(
            f"lane{s['lane_id']} g={s['gain']} x={s['entry'][0]}->{s['exit'][0]}"
            for s in included[:int(KNOWN_MAP_EVAL_STRIP_TOP_DEBUG_COUNT)]
        )
        route_kind_label = "known-full-sweep" if bool(KNOWN_MAP_EVAL_FULL_SWEEP_ROUTE_ENABLED) else "known-strip"
        return {
            "goal_map": goal_map,
            "goal_world": (float(goal_world[0]), float(goal_world[1])),
            "kind": "uncleaned",
            "route_map": route,
            "route_world": route_world,
            "waypoint_map": wp_map,
            "waypoint_world": wp_world,
            "cost": float(route_cost_m),
            "score": float(score),
            "length": int(len(route)),
            "component_id": int(first_lane),
            "component_cells": int(total_gain),
            "component_gain": float(total_gain * 0.10),
            "wall_strip_bonus": 0.0,
            "straight_dist": float(math.hypot(float(goal_world[0]) - pose_x, float(goal_world[1]) - pose_y)),
            "lateral_abs": float(abs(-math.sin(pose_theta) * (float(goal_world[0]) - pose_x) + math.cos(pose_theta) * (float(goal_world[1]) - pose_y))),
            "turn_need": float(abs(normalize_angle(math.atan2(float(goal_world[1]) - pose_y, float(goal_world[0]) - pose_x) - pose_theta)) / math.pi),
            "continuity_bonus": 0.0,
            "missed_strip_bonus": 0.0,
            "large_component_bonus": 0.0,
            "stale_uncleaned_bonus": 0.0,
            "footprint_gain_cells": int(total_gain),
            "footprint_gain_ratio": float(mean_gain_ratio),
            "footprint_reclean_ratio": float(mean_reclean),
            "footprint_gain_bonus": float(total_gain * 0.10),
            "footprint_reclean_penalty": 0.0,
            "coverage_segment_debug": (
                f"{route_kind_label} lanes={first_lane}-{last_lane} seg={len(included)} "
                f"skip={skipped} edgeBias={int(bool(KNOWN_MAP_EVAL_EDGE_LANE_BIAS_ENABLED))} "
                f"dockBias={int(bool(KNOWN_MAP_EVAL_DOCK_END_BIAS_ENABLED))} "
                f"laneSp={float(KNOWN_MAP_EVAL_SWEEP_LANE_SPACING_M):.2f} "
                f"margin={float(KNOWN_MAP_EVAL_ROUTE_MARGIN_M):.2f} "
                f"{sweep_order_debug} "
                f"gain={total_gain} cost={route_cost_m:.1f}"
            ),
            "coverage_segment_bonus": float(total_gain * 0.05),
            "coverage_segment_gain": int(total_gain),
            "coverage_segment_reclean": float(mean_reclean),
            "first_turn_frac": float(first_turn),
            "corner_count": int(corners),
            "geometry_debug": f"geom={route_kind_label} len={len(route)} corners={corners} first={first_turn:.2f}",
            "known_sweep_top_debug": top_dbg,
        }
    except Exception as exc:
        try:
            global last_coverage_segment_debug
            last_coverage_segment_debug = f"known-sweep err:{type(exc).__name__}"
        except Exception:
            pass
        return None

def map_rect_slices_from_world(cx, cy, sx, sy, pad_m=0.0):
    half_x = max(0.0, float(sx) * 0.5 + float(pad_m))
    half_y = max(0.0, float(sy) * 0.5 + float(pad_m))
    x_a, y_top = world_to_map(float(cx) - half_x, float(cy) + half_y)
    x_b, y_bot = world_to_map(float(cx) + half_x, float(cy) - half_y)
    x0 = max(0, min(int(x_a), int(x_b)))
    x1 = min(MAP_SIZE, max(int(x_a), int(x_b)) + 1)
    y0 = max(0, min(int(y_top), int(y_bot)))
    y1 = min(MAP_SIZE, max(int(y_top), int(y_bot)) + 1)
    return x0, x1, y0, y1


def known_map_mark_rect(cx, cy, sx, sy, pad_m=0.0, occ=True):
    x0, x1, y0, y1 = map_rect_slices_from_world(cx, cy, sx, sy, pad_m=pad_m)
    if x1 <= x0 or y1 <= y0:
        return 0
    if occ:
        log_odds[y0:y1, x0:x1] = KNOWN_MAP_EVAL_OCC_LO
        structural_log_odds[y0:y1, x0:x1] = KNOWN_MAP_EVAL_STRUCTURAL_OCC
        visual_log_odds[y0:y1, x0:x1] = np.maximum(visual_log_odds[y0:y1, x0:x1], KNOWN_MAP_EVAL_VISUAL_OCC)
    else:
        log_odds[y0:y1, x0:x1] = KNOWN_MAP_EVAL_FREE_LO
    return int((x1 - x0) * (y1 - y0))


def known_map_mark_circle(cx, cy, radius_m, pad_m=0.0):
    mx, my = world_to_map(float(cx), float(cy))
    r = max(1, int(round((float(radius_m) + float(pad_m)) * MAP_SCALE)))
    mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    if map_inside(mx, my):
        cv2.circle(mask, (int(mx), int(my)), r, 255, -1)
    occ = mask > 0
    count = int(np.count_nonzero(occ))
    if count > 0:
        log_odds[occ] = KNOWN_MAP_EVAL_OCC_LO
        structural_log_odds[occ] = KNOWN_MAP_EVAL_STRUCTURAL_OCC
        visual_log_odds[occ] = np.maximum(visual_log_odds[occ], KNOWN_MAP_EVAL_VISUAL_OCC)
    return count


def known_map_controller_rect_from_wbt(x_wbt, y_wbt, sx, sy, pad_m=None):
    if pad_m is None:
        pad_m = KNOWN_MAP_EVAL_OBSTACLE_PADDING_M
    cx, cy = wbt_to_controller_xy(x_wbt, y_wbt)
    return known_map_mark_rect(cx, cy, sx, sy, pad_m=pad_m, occ=True)


def known_map_controller_circle_from_wbt(x_wbt, y_wbt, radius, pad_m=None):
    if pad_m is None:
        pad_m = KNOWN_MAP_EVAL_OBSTACLE_PADDING_M
    cx, cy = wbt_to_controller_xy(x_wbt, y_wbt)
    return known_map_mark_circle(cx, cy, radius, pad_m=pad_m)


def seed_known_map_coverage_eval(reason="manual"):
    """Initialize the map from known WBT geometry for coverage-only evaluation.

    This is intentionally an evaluation shortcut, not the normal RGB-D mapping
    mode.  The obstacle list mirrors worlds/cleaning_world.wbt at floor/collision
    height: high open-under tabletops/seats are not solid blocks for the vacuum;
    their legs are marked instead.
    """
    global known_map_eval_status, known_map_eval_last_seed_time
    global known_map_primary_sweep_completed, known_map_primary_sweep_finish_time, known_map_primary_sweep_finish_coverage
    global known_map_residual_cleanup_commits_started, known_map_residual_cleanup_started_at, known_map_residual_policy_status
    log_odds[:, :] = 0.0
    visual_log_odds[:, :] = 0.0
    thin_obstacle_log_odds[:, :] = 0.0
    contact_log_odds[:, :] = 0.0
    structural_log_odds[:, :] = 0.0
    under_surface_log_odds[:, :] = 0.0
    try:
        hypothesis_obstacle_log_odds[:, :] = 0.0
    except Exception:
        pass

    # Entire room interior is known free; a thick wall band prevents artificial
    # frontiers just outside the arena boundary.
    x0, x1, y0, y1 = map_rect_slices_from_world(
        (KNOWN_MAP_ARENA_X_MIN_M + KNOWN_MAP_ARENA_X_MAX_M) * 0.5,
        (KNOWN_MAP_ARENA_Y_MIN_M + KNOWN_MAP_ARENA_Y_MAX_M) * 0.5,
        KNOWN_MAP_ARENA_X_MAX_M - KNOWN_MAP_ARENA_X_MIN_M,
        KNOWN_MAP_ARENA_Y_MAX_M - KNOWN_MAP_ARENA_Y_MIN_M,
        pad_m=0.0,
    )
    if x1 > x0 and y1 > y0:
        log_odds[y0:y1, x0:x1] = KNOWN_MAP_EVAL_FREE_LO

    wall = max(2, int(round(KNOWN_MAP_EVAL_WALL_THICKNESS_M * MAP_SCALE)))
    if x1 > x0 and y1 > y0:
        log_odds[y0:min(y1, y0 + wall), x0:x1] = KNOWN_MAP_EVAL_OCC_LO
        log_odds[max(y0, y1 - wall):y1, x0:x1] = KNOWN_MAP_EVAL_OCC_LO
        log_odds[y0:y1, x0:min(x1, x0 + wall)] = KNOWN_MAP_EVAL_OCC_LO
        log_odds[y0:y1, max(x0, x1 - wall):x1] = KNOWN_MAP_EVAL_OCC_LO
        structural_log_odds[log_odds > LO_OCCUPIED_EPS] = KNOWN_MAP_EVAL_STRUCTURAL_OCC
        visual_log_odds[log_odds > LO_OCCUPIED_EPS] = KNOWN_MAP_EVAL_VISUAL_OCC

    cells = 0
    # Low/solid obstacles from worlds/cleaning_world.wbt.
    cells += known_map_controller_rect_from_wbt(-3.0, -2.0, 1.90, 0.75)       # low_sofa
    cells += known_map_controller_rect_from_wbt(3.35, 1.25, 0.65, 1.35)       # white_cabinet
    cells += known_map_controller_rect_from_wbt(-0.15, 1.35, 0.65, 0.55)      # cardboard_box
    cells += known_map_controller_rect_from_wbt(0.90, 0.15, 0.38, 0.38)       # black_low_obstacle
    cells += known_map_controller_circle_from_wbt(-2.40, 1.35, 0.28)          # round_bin
    cells += known_map_controller_rect_from_wbt(2.60, 2.35, 0.28, 0.18)       # small_floor_object

    # Open-under table and chair: mark legs/back-supports that collide near the
    # floor.  High tabletops/seats are intentionally left cleanable.
    for xw, yw in ((1.35, -2.35), (2.45, -2.35), (1.35, -1.65), (2.45, -1.65)):
        cells += known_map_controller_rect_from_wbt(xw, yw, 0.07, 0.07, pad_m=0.035)
    cells += known_map_controller_rect_from_wbt(2.94, -0.70, 0.06, 0.58, pad_m=0.030)  # chair backrest footprint edge
    for xw, yw in ((2.44, -0.91), (2.86, -0.91), (2.44, -0.49), (2.86, -0.49)):
        cells += known_map_controller_rect_from_wbt(xw, yw, 0.055, 0.055, pad_m=0.035)

    # Keep the robot start/dock area free even after wall seeding.
    sx, sy = world_to_map(0.0, 0.0)
    if map_inside(sx, sy):
        r = max(2, int(round(0.28 * MAP_SCALE)))
        cv2.circle(log_odds, (int(sx), int(sy)), r, KNOWN_MAP_EVAL_FREE_LO, -1)
        cv2.circle(structural_log_odds, (int(sx), int(sy)), r, 0.0, -1)
        cv2.circle(visual_log_odds, (int(sx), int(sy)), r, 0.0, -1)

    # The known-map experiment evaluates coverage from scratch.
    cleaned_mask[:, :] = 0
    recent_visit_log_odds[:, :] = 0.0
    invalidate_heavy_map_caches("known-map eval seed")
    try:
        known_map_eval_last_seed_time = float(robot.getTime())
    except Exception:
        known_map_eval_last_seed_time = 0.0
    known_map_primary_sweep_completed = False
    known_map_primary_sweep_finish_time = -999.0
    known_map_primary_sweep_finish_coverage = 0.0
    known_map_residual_cleanup_commits_started = 0
    known_map_residual_cleanup_started_at = -999.0
    known_map_residual_policy_status = "residualPolicy=primary"
    known_map_eval_status = f"knownMap=on seed {reason} obs={int(cells)}"
    return cells


def enable_known_map_coverage_eval(reason="manual"):
    global known_map_eval_runtime_enabled, known_map_eval_status, nav_action_queue
    known_map_eval_runtime_enabled = True
    # K mode is the known-map optimizer.  Clear explore-sweep atomic queues and
    # release its short control lock so route_commit can own motion cleanly.
    nav_action_queue = []
    try:
        if str(control_lock.owner) == SIMPLE_SWEEP_OWNER:
            release_control("K known-map planner enabled")
    except Exception:
        pass
    abort_route_commit("known map eval enable")
    seed_known_map_coverage_eval(reason=reason)
    update_coverage_objective()
    refresh_navigation_phase()
    known_map_eval_status = f"knownMap=on {reason}"
    print("Known-map coverage evaluation enabled")


def disable_known_map_coverage_eval(reason="manual"):
    global known_map_eval_runtime_enabled, known_map_eval_status
    known_map_eval_runtime_enabled = False
    known_map_eval_status = f"knownMap=off {reason}"
    reset_map()
    print("Known-map coverage evaluation disabled")


def toggle_known_map_coverage_eval():
    if known_map_coverage_eval_active():
        disable_known_map_coverage_eval("key")
    else:
        enable_known_map_coverage_eval("key")


def filter_frontier_noise_mask(frontier_mask, cleanable, obstacles, unknown=None):
    """Suppress tiny/ray-like or obstacle-shadow frontier components.

    This does not rewrite the map. It only keeps the objective frontier list from
    targeting black-dot noise, diagonal whiskers, or gray pockets inside/behind a
    furniture obstacle. Real frontier components remain if they have a cleanable
    floor boundary or enough area to be worth an RGB-D viewpoint.
    """
    if not FRONTIER_NOISE_FILTER_ENABLED:
        return frontier_mask.astype(np.bool_), "frNoise=off"
    try:
        fr = frontier_mask.astype(np.bool_).copy()
        if not bool(np.any(fr)):
            return fr, "frNoise=0"
        clean_dil = cv2.dilate(cleanable.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        obs_dil = cv2.dilate(obstacles.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fr.astype(np.uint8), 8)
        keep = np.zeros_like(fr, dtype=np.bool_)
        rejected_small = 0
        rejected_ray = 0
        rejected_shadow = 0
        kept = 0
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            left = int(stats[cid, cv2.CC_STAT_LEFT])
            top = int(stats[cid, cv2.CC_STAT_TOP])
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            narrow = min(width, height)
            bbox_area = max(1, width * height)
            density = float(area) / float(bbox_area)
            comp = labels == cid
            boundary = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            ring = boundary & (~comp)
            clean_edge = int(np.count_nonzero(ring & clean_dil))
            obs_edge = int(np.count_nonzero(ring & obs_dil))
            large_keep = area >= int(FRONTIER_NOISE_LARGE_KEEP_CELLS)
            if area < int(FRONTIER_NOISE_MIN_COMPONENT_CELLS) and not large_keep:
                rejected_small += 1
                continue
            ray_like = bool(
                area <= int(FRONTIER_NOISE_RAY_MAX_AREA_PX)
                and span >= int(FRONTIER_NOISE_RAY_MIN_SPAN_PX)
                and (density <= float(FRONTIER_NOISE_RAY_MAX_DENSITY) or narrow <= int(FRONTIER_NOISE_RAY_MAX_NARROW_PX))
                and clean_edge < int(FRONTIER_NOISE_MIN_CLEANABLE_EDGE_CELLS)
            )
            if ray_like and not large_keep:
                rejected_ray += 1
                continue
            shadow_like = bool(
                (obs_edge > clean_edge * float(FRONTIER_NOISE_OBS_SHADOW_RATIO))
                and clean_edge < max(int(FRONTIER_NOISE_MIN_CLEANABLE_EDGE_CELLS), area // 9)
                and not large_keep
            )
            if shadow_like:
                rejected_shadow += 1
                continue
            keep[comp] = True
            kept += 1
        return keep.astype(np.bool_), f"frNoise=keep{kept} rejS/R/O={rejected_small}/{rejected_ray}/{rejected_shadow}"
    except Exception as exc:
        return frontier_mask.astype(np.bool_), f"frNoise=err {type(exc).__name__}"

def build_frontier_revisit_mask(unknown, cleanable, obstacles):
    """Return frontier cells biased toward reachable interior gray gaps.

    The old frontier mask was only "unknown next to cleanable". That catches open
    map boundaries, but it does not explicitly distinguish useful gray pockets
    inside the already mapped room from outside-room unknown background.
    """
    global last_gray_gap_cells, last_gray_gap_components, last_gray_gap_debug
    try:
        hyp_shadow = hypothesis_obstacle_mask(force=False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(unknown, dtype=np.bool_)
        frontier_unknown = unknown & (~hyp_shadow)
        base_frontier = frontier_unknown & (cv2.dilate(cleanable.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0)
        base_frontier, frontier_noise_debug = filter_frontier_noise_mask(base_frontier, cleanable, obstacles, unknown)
        last_gray_gap_cells = 0
        last_gray_gap_components = 0
        if not GRAY_REVISIT_ENABLED:
            last_gray_gap_debug = "grayGap=off " + frontier_noise_debug
            return base_frontier.astype(np.bool_)
        known = cleanable | obstacles
        if not bool(np.any(known)):
            last_gray_gap_debug = "grayGap=noKnown " + frontier_noise_debug
            return base_frontier.astype(np.bool_)
        ys, xs = np.where(known)
        pad = max(2, int(round(float(GRAY_REVISIT_BBOX_PAD_M) * MAP_SCALE)))
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(MAP_SIZE, int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(MAP_SIZE, int(ys.max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            last_gray_gap_debug = "grayGap=badBbox " + frontier_noise_debug
            return base_frontier.astype(np.bool_)
        inside = np.zeros_like(unknown, dtype=np.bool_)
        inside[y0:y1, x0:x1] = True
        near_r = max(2, int(round(float(GRAY_REVISIT_NEAR_CLEANABLE_M) * MAP_SCALE)))
        near_cleanable = cv2.dilate(cleanable.astype(np.uint8), np.ones((2 * near_r + 1, 2 * near_r + 1), np.uint8), iterations=1) > 0
        candidates = frontier_unknown & inside & near_cleanable
        if not bool(np.any(candidates)):
            last_gray_gap_debug = "grayGap=0/0 " + frontier_noise_debug
            return base_frontier.astype(np.bool_)
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
        gap_unknown = np.zeros_like(unknown, dtype=np.bool_)
        comps = 0
        cells = 0
        edge_pad = max(1, int(GRAY_REVISIT_REJECT_BBOX_EDGE_PAD_PX))
        rejected_edge = 0
        rejected_size = 0
        rejected_shadow = 0
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < int(GRAY_REVISIT_MIN_COMPONENT_CELLS) or area > int(GRAY_REVISIT_MAX_COMPONENT_CELLS):
                rejected_size += 1
                continue
            left = int(stats[cid, cv2.CC_STAT_LEFT])
            top = int(stats[cid, cv2.CC_STAT_TOP])
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            touches_bbox_edge = bool(
                left <= x0 + edge_pad
                or top <= y0 + edge_pad
                or left + width >= x1 - edge_pad
                or top + height >= y1 - edge_pad
            )
            touches_map_edge = bool(left <= 2 or top <= 2 or left + width >= MAP_SIZE - 3 or top + height >= MAP_SIZE - 3)
            if touches_bbox_edge or touches_map_edge:
                rejected_edge += 1
                continue
            comp = labels == cid
            boundary = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
            ring = boundary & (~comp)
            clean_edge = int(np.count_nonzero(ring & cleanable))
            if clean_edge < int(GRAY_REVISIT_MIN_CLEANABLE_EDGE_CELLS):
                rejected_edge += 1
                continue
            # Obstacle-shadow rejection: unknown pixels immediately behind/inside
            # a furniture/obstacle component are not useful map-revisit goals.
            # A real missed floor pocket has a sizeable cleanable/free boundary;
            # an occluded object interior is dominated by obstacle boundary.
            try:
                obs_r = max(1, int(round(float(GRAY_REVISIT_OBSTACLE_EDGE_DILATE_M) * MAP_SCALE)))
                obs_near = cv2.dilate(obstacles.astype(np.uint8), np.ones((2 * obs_r + 1, 2 * obs_r + 1), np.uint8), iterations=1) > 0
                obs_edge = int(np.count_nonzero(ring & obs_near))
                edge_total = max(1, int(np.count_nonzero(ring)))
                clean_ratio = float(clean_edge) / float(edge_total)
                obs_ratio = float(obs_edge) / float(edge_total)
                shadow_like = bool(
                    clean_ratio < float(GRAY_REVISIT_MIN_CLEANABLE_EDGE_RATIO)
                    or obs_ratio > float(GRAY_REVISIT_MAX_OBSTACLE_EDGE_RATIO)
                    or (obs_edge > clean_edge * float(GRAY_REVISIT_OBSTACLE_SHADOW_RATIO) and area < int(GRAY_REVISIT_MAX_COMPONENT_CELLS) * 0.75)
                )
                if shadow_like:
                    # This gray component is likely the unseen interior/occlusion
                    # side of an obstacle.  Do not make it a frontier target; mark
                    # it as an orange hypothesis so it is visible and suppressed.
                    try:
                        if OBSTACLE_HYPOTHESIS_ENABLED and area >= int(OBSTACLE_HYPOTHESIS_MIN_FILL_CELLS):
                            hypothesis_obstacle_log_odds[comp] = np.minimum(
                                float(OBSTACLE_HYPOTHESIS_MAX),
                                hypothesis_obstacle_log_odds[comp] + float(OBSTACLE_HYPOTHESIS_SHADOW_UPDATE),
                            )
                            globals()["hypothesis_obstacle_cache_step"] = -999999
                    except Exception:
                        pass
                    rejected_shadow += 1
                    rejected_edge += 1
                    continue
            except Exception:
                pass
            gap_unknown[comp] = True
            comps += 1
            cells += area
        dil_r = max(1, int(round(float(GRAY_REVISIT_FRONTIER_DILATE_M) * MAP_SCALE)))
        around_cleanable = cv2.dilate(cleanable.astype(np.uint8), np.ones((2 * dil_r + 1, 2 * dil_r + 1), np.uint8), iterations=1) > 0
        gap_frontier = gap_unknown & around_cleanable
        frontier = base_frontier | gap_frontier
        frontier, frontier_noise_debug = filter_frontier_noise_mask(frontier, cleanable, obstacles, unknown)
        last_gray_gap_cells = int(cells)
        last_gray_gap_components = int(comps)
        last_gray_gap_debug = f"grayGap={int(cells)}/{int(comps)} rejE/S/O={rejected_edge}/{rejected_size}/{rejected_shadow} {frontier_noise_debug}"
        return frontier.astype(np.bool_)
    except Exception as exc:
        last_gray_gap_cells = 0
        last_gray_gap_components = 0
        last_gray_gap_debug = f"grayGap=err {type(exc).__name__}"
        try:
            hyp_shadow = hypothesis_obstacle_mask(force=False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(unknown, dtype=np.bool_)
            return ((unknown & (~hyp_shadow)) & (cv2.dilate(cleanable.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0)).astype(np.bool_)
        except Exception:
            return np.zeros_like(unknown, dtype=np.bool_)


def compute_coverage_masks():
    """Build a cleanable/covered/obstacle view from the persistent map.

    This is the second map the diploma can show: not raw sensor data, but the
    navigation objective layer: where obstacles are, what has been cleaned, and
    what is still worth visiting. The passable-under-furniture layer is allowed
    to add cleanable floor cells, but never to override obstacles/contact hits.
    """
    # floor is computed against actual obstacles only; inflated furniture margins
    # are applied later inside the robot-centre planner.
    obstacles, _center_no_go, _cleanable_floor = build_planning_layers(force=False)
    under_surface = under_surface_mask_from_obstacles(obstacles)
    free = ((log_odds < -LO_UNKNOWN_EPS) | under_surface) & (~obstacles)
    cleaned = (cleaned_mask > 0) & (~obstacles)
    cleanable = (free | cleaned) & (~obstacles)
    uncleaned = cleanable & (~cleaned)
    structural_occ = (structural_log_odds > STRUCTURAL_OCCUPIED_EPS) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(log_odds, dtype=np.bool_)
    if known_map_coverage_eval_active():
        # In known-map evaluation the arena has been seeded as fully known.  Do
        # not synthesize frontiers from the unknown space outside walls or from
        # still-unobserved RGB-D pixels; coverage is the only objective.
        unknown = np.zeros_like(log_odds, dtype=np.bool_)
    else:
        unknown = (np.abs(log_odds) <= LO_UNKNOWN_EPS) & (visual_log_odds <= CV_DISPLAY_LIGHT_EPS) & (~under_surface) & (~structural_occ) & (~obstacles)
    obstacles, cleanable, cleaned, uncleaned, unknown = filter_objective_noise_masks(obstacles, cleanable, cleaned, uncleaned, unknown)
    return obstacles, cleanable, cleaned, uncleaned, unknown


def update_under_surface_from_depth(depth):
    """Remember passable floor corridors under/through furniture.

    The controller already has a local under-furniture test. This function turns
    that momentary test into map memory: if the body footprint can pass and
    furniture-like geometry is nearby, mark the free corridor as a future
    coverage objective. Later, if contact/occupancy contradicts it, the mark is
    erased.
    """
    global last_under_surface_cells, last_under_surface_target_cells, last_under_surface_marked

    if not UNDER_SURFACE_ENABLED or depth is None:
        last_under_surface_marked = 0
        return 0

    # Under-surface memory also writes persistent map/free hints; keep it under
    # the same no-rotation contract as RGB-D mapping.
    mapping_frozen, _freeze_reason = update_mapping_freeze_state(robot.getTime())
    if mapping_frozen or nav_state not in MAPPING_ALLOWED_STATES:
        last_under_surface_marked = 0
        return 0

    # Use actual physical obstacles here; an inflated no-go margin around furniture
    # must not erase remembered cleanable under-surface floor.
    obstacles = actual_physical_obstacle_mask()
    if np.any(obstacles):
        under_surface_log_odds[obstacles] = np.maximum(
            0.0, under_surface_log_odds[obstacles] + UNDER_SURFACE_DECAY_ON_OBSTACLE
        )

    left, center, right = depth_sectors(depth)
    front = depth_front_narrow(depth)
    upper_front = depth_front_upper_corridor(depth)
    overhead = depth_overhead_structure_distance(depth)
    body_clearance = depth_body_corridor_clearance(depth)

    overhead_structure = (
        UNDER_SURFACE_OVERHEAD_MIN < overhead < UNDER_SURFACE_OVERHEAD_MAX
        and (last_cv_map_hits >= UNDER_SURFACE_MIN_CV_CONTEXT or last_cv_depth_confirmed >= UNDER_SURFACE_MIN_DEPTH_CONTEXT)
    )
    local_corridor = detect_under_furniture_corridor(front, center, upper_front, left, right)

    if body_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
        last_under_surface_marked = 0
    elif overhead_structure and local_corridor and front > UNDER_SURFACE_MIN_FRONT_OPEN and center > UNDER_FURNITURE_CENTER_OPEN:
        # Do not paint a huge distant yellow strip. Mark only the part that the
        # robot can realistically clean while already near the furniture. Empty
        # open space must not become a yellow under-furniture target.
        max_forward = clamp(min(front, center, overhead, UNDER_SURFACE_MARK_AHEAD_M), UNDER_SURFACE_MARK_MIN_M, UNDER_SURFACE_MARK_AHEAD_M)
        fx = math.cos(pose_theta)
        fy = math.sin(pose_theta)
        nx = -math.sin(pose_theta)
        ny = math.cos(pose_theta)
        marked = 0
        a = UNDER_SURFACE_MARK_MIN_M
        while a <= max_forward + 1e-6:
            lat = -UNDER_SURFACE_HALF_WIDTH_M
            while lat <= UNDER_SURFACE_HALF_WIDTH_M + 1e-6:
                wx = pose_x + fx * a + nx * lat
                wy = pose_y + fy * a + ny * lat
                mx, my = world_to_map(wx, wy)
                lat += UNDER_SURFACE_CELL_STEP_M
                if not map_inside(mx, my) or obstacles[my, mx]:
                    continue
                under_surface_log_odds[my, mx] = clamp(
                    under_surface_log_odds[my, mx] + UNDER_SURFACE_UPDATE, 0.0, CONTACT_MAX
                )
                # Also give a weak free-space hint. The under-surface layer is
                # the main evidence, but this helps the ordinary coverage mask
                # not lose the corridor if raw depth was conservative.
                log_odds[my, mx] = clamp(log_odds[my, mx] + LO_FREE_UPDATE * 0.25, LO_MIN, LO_MAX)
                marked += 1
            a += UNDER_SURFACE_CELL_STEP_M
        last_under_surface_marked = marked
    else:
        last_under_surface_marked = 0

    under = under_surface_mask_from_obstacles(obstacles)
    last_under_surface_cells = int(np.count_nonzero(under))
    last_under_surface_target_cells = int(np.count_nonzero(under & (cleaned_mask == 0)))
    return last_under_surface_marked


def build_footprint_passability_map(obstacles, cleanable, route_margin_m=FOOTPRINT_ROUTE_MARGIN_M):
    """Return a full-resolution map of places where the robot centre can stand.

    This is the requested "1:1" prediction layer: for each candidate centre cell
    it checks the real circular footprint radius, not just the centre line.  Raw
    RGB-D maps contain many isolated black speckles, so the decision combines a
    metric clearance field with a small disk collision-ratio test.  Real contact
    obstacles are treated much more strictly.
    """
    global last_footprint_passable_cells, footprint_passability_cache, footprint_passability_cache_step, footprint_passability_cache_margin
    t_perf_fp = perf_start()
    if FOOTPRINT_MAP_CACHE_ENABLED and footprint_passability_cache is not None:
        try:
            age = int(step_id) - int(footprint_passability_cache_step)
            margin_same = footprint_passability_cache_margin is not None and abs(float(footprint_passability_cache_margin) - float(route_margin_m)) < 1e-6
            if margin_same and 0 <= age < max(1, int(FOOTPRINT_MAP_CACHE_STEPS)):
                passable_cached, clearance_cached = footprint_passability_cache
                last_footprint_passable_cells = int(np.count_nonzero(passable_cached))
                perf_end("fp", t_perf_fp)
                return passable_cached.copy(), None if clearance_cached is None else clearance_cached.copy()
        except Exception:
            pass
    if not FOOTPRINT_PREDICTION_ENABLED:
        last_footprint_passable_cells = int(np.count_nonzero(cleanable))
        perf_end("fp", t_perf_fp)
        return cleanable.copy(), None

    hard_obstacles = obstacles.astype(np.uint8)
    # Light cleanup: remove single-pixel speckles from visual/depth noise, but do
    # not erase contact-confirmed obstacles.  This prevents the 1:1 disk check
    # from treating every isolated RGB edge as an impassable wall.
    hard_obstacles = cv2.morphologyEx(hard_obstacles, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    hard_obstacles = np.maximum(hard_obstacles, (contact_log_odds > CONTACT_OCCUPIED_EPS).astype(np.uint8))

    free_for_distance = (hard_obstacles == 0).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_for_distance, cv2.DIST_L2, 3)
    need_px = max(1.0, (FOOTPRINT_RADIUS_M + route_margin_m) * MAP_SCALE)
    metric_ok = clearance_px >= need_px

    # Known free ratio prevents route planning through unexplored gray space.  We
    # allow already-cleaned and under-surface cells because both are intentional
    # coverage layers, but unknown cells are not treated as passable centres.
    r_px = max(2, int(round(FOOTPRINT_RADIUS_M * MAP_SCALE)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1)).astype(np.float32)
    area = max(1.0, float(np.count_nonzero(kernel)))
    raw_ratio = cv2.filter2D(obstacles.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT) / area
    contact_ratio = cv2.filter2D((contact_log_odds > CONTACT_OCCUPIED_EPS).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT) / area
    known_ratio = cv2.filter2D(cleanable.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT) / area

    speckle_ok = (raw_ratio <= FOOTPRINT_MAX_RAW_OBS_RATIO) & (contact_ratio <= FOOTPRINT_MAX_CONTACT_RATIO)
    passable = cleanable & ((metric_ok | speckle_ok) & (known_ratio >= FOOTPRINT_MIN_KNOWN_FREE_RATIO))

    # The robot already occupies its current position; keep a small island around
    # it passable so the planner can recover even after a contact mark was just
    # written near the bumper.
    rx, ry = world_to_map(pose_x, pose_y)
    passable_u8 = passable.astype(np.uint8)
    if map_inside(rx, ry):
        cv2.circle(passable_u8, (rx, ry), max(2, int(FOOTPRINT_RADIUS_M * MAP_SCALE * 0.55)), 1, -1)
    passable = passable_u8 > 0

    last_footprint_passable_cells = int(np.count_nonzero(passable))
    result_passable = passable.astype(np.bool_)
    result_clearance = clearance_px / MAP_SCALE
    if FOOTPRINT_MAP_CACHE_ENABLED:
        footprint_passability_cache = (result_passable.copy(), result_clearance.copy())
        footprint_passability_cache_step = int(step_id)
        footprint_passability_cache_margin = float(route_margin_m)
    perf_end("fp", t_perf_fp)
    return result_passable, result_clearance


def cleanable_reachable_by_body_center(footprint_passable, cleanable, obstacles):
    """Cells that can be cleaned from at least one physically valid centre pose."""
    r_px = max(2, int(round(COVERAGE_RADIUS_M * MAP_SCALE)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    reachable_from_center = cv2.dilate(footprint_passable.astype(np.uint8), kernel, iterations=1) > 0
    under = under_surface_log_odds > UNDER_SURFACE_EPS
    return reachable_from_center & cleanable & (~obstacles) & (~under)


def build_cleaning_footprint_value_maps(cleanable, cleaned, reachable_uncleaned, footprint_passable=None):
    """Pre-compute what the real cleaning disk covers from every centre pose.

    Earlier route scoring treated a route/target cell almost as a point.  That is
    the wrong abstraction for a vacuum: the centre may drive through a cleaned
    lane while the brush disk covers a wall strip, or it may waste time re-driving
    a fully cleaned patch.  These maps let the goal selector score a candidate
    centre by the footprint swath it would actually clean.
    """
    global footprint_value_cache, footprint_value_cache_step
    t_perf_fp_value = perf_start()
    if FOOTPRINT_VALUE_CACHE_ENABLED and footprint_value_cache is not None:
        try:
            age = int(step_id) - int(footprint_value_cache_step)
            if 0 <= age < max(1, int(FOOTPRINT_VALUE_CACHE_STEPS)):
                perf_end("fp", t_perf_fp_value)
                return tuple(arr.copy() for arr in footprint_value_cache)
        except Exception:
            pass
    r_px = max(2, int(round(COVERAGE_RADIUS_M * MAP_SCALE)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1)).astype(np.float32)
    area = max(1.0, float(np.count_nonzero(kernel)))
    cleanable_f = cleanable.astype(np.float32)
    uncleaned_f = (reachable_uncleaned & cleanable).astype(np.float32)
    cleaned_f = (cleaned & cleanable).astype(np.float32)
    gain_count = cv2.filter2D(uncleaned_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    cleaned_count = cv2.filter2D(cleaned_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    cleanable_count = cv2.filter2D(cleanable_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    denom = np.maximum(cleanable_count, 1.0)
    gain_ratio = gain_count / denom
    reclean_ratio = cleaned_count / denom
    known_ratio = cleanable_count / area
    if footprint_passable is not None:
        invalid = ~footprint_passable
        gain_count = gain_count.copy(); gain_ratio = gain_ratio.copy(); reclean_ratio = reclean_ratio.copy(); known_ratio = known_ratio.copy()
        gain_count[invalid] = 0.0
        gain_ratio[invalid] = 0.0
        reclean_ratio[invalid] = 1.0
        known_ratio[invalid] = 0.0
    result = (gain_count.astype(np.float32), gain_ratio.astype(np.float32), reclean_ratio.astype(np.float32), known_ratio.astype(np.float32))
    if FOOTPRINT_VALUE_CACHE_ENABLED:
        footprint_value_cache = tuple(arr.copy() for arr in result)
        footprint_value_cache_step = int(step_id)
    perf_end("fp", t_perf_fp_value)
    return result


def trim_unreachable_coverage_targets():
    """Mark nearby non-under-furniture objective cells that no body pose can clean.

    This prevents the planner from repeatedly turning toward edge slivers next
    to walls/legs.  It is deliberately gated by near_cleaned: the robot must have
    already cleaned the reachable adjacent lane before the unreachable fringe is
    treated as covered by the edge/side-brush model.
    """
    global last_unreachable_target_trim_cells
    last_unreachable_target_trim_cells = 0
    if not UNREACHABLE_TARGET_TRIM_ENABLED:
        return 0
    obstacles, cleanable, cleaned, uncleaned, _unknown = compute_coverage_masks()
    if not np.any(uncleaned) or not np.any(cleaned):
        return 0
    route_obstacles = planning_no_go_obstacle_mask(obstacles)
    footprint_passable, _clearance = build_footprint_passability_map(route_obstacles, cleanable, FOOTPRINT_ROUTE_MARGIN_M)
    reachable_cleanable = cleanable_reachable_by_body_center(footprint_passable, cleanable, obstacles)
    under = under_surface_log_odds > UNDER_SURFACE_EPS
    clean_r = max(2, int(round(UNREACHABLE_TARGET_TRIM_NEAR_CLEANED_M * MAP_SCALE)))
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * clean_r + 1, 2 * clean_r + 1))
    near_cleaned = cv2.dilate(cleaned.astype(np.uint8), clean_kernel, iterations=1) > 0
    candidate = uncleaned & cleanable & (~reachable_cleanable) & near_cleaned & (~under) & (~obstacles)
    if not np.any(candidate):
        return 0
    ys, xs = np.where(candidate)
    count = len(xs)
    if count > UNREACHABLE_TARGET_TRIM_MAX_MARK:
        mx, my = world_to_map(pose_x, pose_y)
        order = np.argsort((xs - mx) * (xs - mx) + (ys - my) * (ys - my))[:UNREACHABLE_TARGET_TRIM_MAX_MARK]
        xs = xs[order]
        ys = ys[order]
        count = len(xs)
    cleaned_mask[ys, xs] = 255
    last_unreachable_target_trim_cells = int(count)
    return last_unreachable_target_trim_cells


def footprint_fits_at(mx, my, obstacles=None, cleanable=None, margin_m=FOOTPRINT_PASS_MARGIN_M):
    """Check whether the full circular robot footprint fits at a map position."""
    if obstacles is None or cleanable is None:
        obstacles, cleanable, _cleaned, _uncleaned, _unknown = compute_coverage_masks()
    if not map_inside(int(mx), int(my)) or not cleanable[int(my), int(mx)]:
        return False, 0.0
    r_px = max(2, int(round((FOOTPRINT_RADIUS_M + margin_m) * MAP_SCALE)))
    x0 = max(0, int(mx) - r_px)
    x1 = min(MAP_SIZE, int(mx) + r_px + 1)
    y0 = max(0, int(my) - r_px)
    y1 = min(MAP_SIZE, int(my) + r_px + 1)
    if x1 <= x0 or y1 <= y0:
        return False, 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - int(mx)) * (xx - int(mx)) + (yy - int(my)) * (yy - int(my)) <= r_px * r_px
    area = max(1, int(np.count_nonzero(disk)))
    raw = obstacles[y0:y1, x0:x1]
    contact = contact_log_odds[y0:y1, x0:x1] > CONTACT_OCCUPIED_EPS
    known = cleanable[y0:y1, x0:x1]
    raw_ratio = float(np.count_nonzero(raw & disk)) / area
    contact_ratio = float(np.count_nonzero(contact & disk)) / area
    known_ratio = float(np.count_nonzero(known & disk)) / area
    ok = (raw_ratio <= FOOTPRINT_MAX_RAW_OBS_RATIO and contact_ratio <= FOOTPRINT_MAX_CONTACT_RATIO and known_ratio >= FOOTPRINT_MIN_KNOWN_FREE_RATIO)
    return bool(ok), raw_ratio


def map_line_footprint_blocked_ratio(mx0, my0, mx1, my1, obstacles, cleanable, step_m=FOOTPRINT_SWEEP_STEP_M):
    """Sweep the real robot disk along a line and return the blocked fraction."""
    dx = int(mx1) - int(mx0)
    dy = int(my1) - int(my0)
    length_px = math.hypot(dx, dy)
    if length_px <= 1.0:
        ok, _ = footprint_fits_at(mx0, my0, obstacles, cleanable)
        return 0.0 if ok else 1.0
    steps = max(1, int((length_px / MAP_SCALE) / max(0.02, step_m)))
    bad = 0
    total = 0
    skip_px = int(max(2, FOOTPRINT_RADIUS_M * MAP_SCALE * 0.35))
    for i in range(1, steps + 1):
        t = i / steps
        mx = int(round(mx0 + dx * t))
        my = int(round(my0 + dy * t))
        if math.hypot(mx - mx0, my - my0) < skip_px:
            continue
        total += 1
        ok, _ = footprint_fits_at(mx, my, obstacles, cleanable, margin_m=FOOTPRINT_PASS_MARGIN_M)
        if not ok:
            bad += 1
    return bad / max(1, total)


def map_line_obstacle_ratio(mx0, my0, mx1, my1, obstacles, step_px=UNDER_SURFACE_LINE_CLEAR_STEP_PX):
    """Return obstacle/contact density along a coarse map segment.

    This is not A*. It is a cheap route sanity check used before selecting a
    soft coverage target. A target behind a table leg or across a furniture edge
    may look attractive by distance, but following it bends the row into a crash.
    """
    dx = int(mx1) - int(mx0)
    dy = int(my1) - int(my0)
    length = max(abs(dx), abs(dy))
    if length <= 0:
        return 0.0
    steps = max(1, length // max(1, int(step_px)))
    bad = 0
    total = 0
    skip_r = int(max(2, ROBOT_BODY_RADIUS * MAP_SCALE * 0.55))
    for i in range(1, steps + 1):
        t = i / steps
        mx = int(round(mx0 + dx * t))
        my = int(round(my0 + dy * t))
        if not map_inside(mx, my):
            bad += 1
            total += 1
            continue
        if math.hypot(mx - mx0, my - my0) < skip_r:
            continue
        total += 1
        if obstacles[my, mx] or contact_log_odds[my, mx] > CONTACT_OCCUPIED_EPS or visual_log_odds[my, mx] > CV_DISPLAY_DENSE_EPS:
            bad += 1
    return bad / max(1, total)


def under_surface_target_is_direct_route(mx, my, longitudinal, lateral, dist, obstacles, cleanable=None):
    """Gate yellow under-furniture cells so they remain opportunistic.

    The previous planner let any nearby under-surface patch become the target.
    That created diagonal paths into table legs. Yellow cells should be cleaned
    when the robot is already lined up with a passable opening, not by steering a
    whole row sideways through cleaned space.
    """
    if dist > UNDER_SURFACE_TARGET_MAX_DIST:
        return False
    if longitudinal < 0.10 or longitudinal > UNDER_SURFACE_TARGET_FORWARD_LIMIT_M:
        return False
    lateral_limit = min(
        UNDER_SURFACE_TARGET_LATERAL_LIMIT_M,
        max(UNDER_SURFACE_DIRECT_LATERAL_TOL_M, longitudinal * UNDER_SURFACE_MAX_LATERAL_PER_FORWARD + 0.04),
    )
    if abs(lateral) > lateral_limit:
        return False
    robot_mx, robot_my = world_to_map(pose_x, pose_y)
    if cleanable is None:
        _obs2, cleanable2, _cleaned2, _uncleaned2, _unknown2 = compute_coverage_masks()
    else:
        cleanable2 = cleanable
    obstacle_ok = map_line_obstacle_ratio(robot_mx, robot_my, mx, my, obstacles) <= UNDER_SURFACE_LINE_CLEAR_OBSTACLE_RATIO
    footprint_ok = map_line_footprint_blocked_ratio(robot_mx, robot_my, mx, my, obstacles, cleanable2) <= 0.18
    return bool(obstacle_ok and footprint_ok)


def coarse_route_center(x0, y0, step, gx, gy):
    return int(x0 + gx * step + step * 0.5), int(y0 + gy * step + step * 0.5)


def nearest_passable_route_cell(passable, sx, sy, max_r=4):
    gh, gw = passable.shape
    if 0 <= sx < gw and 0 <= sy < gh and passable[sy, sx]:
        return sx, sy
    best = None
    best_d = 1e9
    for r in range(1, max_r + 1):
        for yy in range(max(0, sy - r), min(gh, sy + r + 1)):
            for xx in range(max(0, sx - r), min(gw, sx + r + 1)):
                if not passable[yy, xx]:
                    continue
                d = (xx - sx) * (xx - sx) + (yy - sy) * (yy - sy)
                if d < best_d:
                    best_d = d
                    best = (xx, yy)
        if best is not None:
            return best
    return None


def route_start_cell_is_locally_reachable(start, x0, y0, step, obstacles, cleanable):
    """Reject a coarse route start that was snapped through a wall/obstacle.

    When the robot centre is next to an inflated wall, the exact coarse cell can be
    non-passable. A blind nearest-passable search may snap the wavefront start to a
    cell behind/beside the obstacle, producing the L-shaped route visible on the
    map and making GRID_REALIGN point at nonsense. The snapped start is accepted
    only if the robot can reach its centre by a short footprint-safe local segment.
    """
    if start is None:
        return False
    sx, sy = start
    cx, cy = coarse_route_center(x0, y0, step, sx, sy)
    robot_mx, robot_my = world_to_map(pose_x, pose_y)
    jump_m = math.hypot(cx - robot_mx, cy - robot_my) / MAP_SCALE
    if jump_m > 0.42:
        return False
    blocked = map_line_footprint_blocked_ratio(robot_mx, robot_my, cx, cy, obstacles, cleanable)
    return bool(blocked <= 0.20)


def reconstruct_coarse_route(parent, start, goal, x0, y0, step):
    route = []
    cur = goal
    guard = 0
    while cur is not None and guard < parent.size + 2:
        gx, gy = cur
        route.append(coarse_route_center(x0, y0, step, gx, gy))
        if cur == start:
            break
        flat_parent = int(parent[gy, gx])
        if flat_parent < 0:
            break
        gw = parent.shape[1]
        cur = (flat_parent % gw, flat_parent // gw)
        guard += 1
    route.reverse()
    return route


def route_waypoint_from_path(route):
    if not route:
        return None, None
    rx, ry = world_to_map(pose_x, pose_y)
    min_px = GLOBAL_ROUTE_WAYPOINT_LOOKAHEAD_M * MAP_SCALE
    waypoint = route[-1]
    for p in route[1:]:
        if math.hypot(p[0] - rx, p[1] - ry) >= min_px:
            waypoint = p
            break
    wx = (waypoint[0] - MAP_ORIGIN_X) / MAP_SCALE
    wy = (MAP_ORIGIN_Y - waypoint[1]) / MAP_SCALE
    return waypoint, (float(wx), float(wy))


def route_path_geometry_stats(route):
    """Return simple motion-shape metrics for a planned route.

    Coverage may tolerate a Manhattan strip route.  Exploration should not pivot
    through many short grid segments just to look at a frontier.  These metrics
    let the frontier scorer/commit policy prefer routes whose first movement is
    close to the robot's current heading and whose path has few corners.
    """
    try:
        pts = [(int(x), int(y)) for x, y in (route or []) if map_inside(int(x), int(y))]
    except Exception:
        pts = []
    if len(pts) < 2:
        return {
            "first_turn_frac": 0.0,
            "first_turn_deg": 0.0,
            "corner_count": 0,
            "debug": "geom=short",
        }

    # First movement is measured from the physical robot pose to the first route
    # point ahead, not from the snapped coarse start cell.  Snapping can be beside
    # the robot near walls and would otherwise overstate a 90-degree turn.
    first = pts[1] if len(pts) > 1 else pts[0]
    fwx = (int(first[0]) - MAP_ORIGIN_X) / MAP_SCALE
    fwy = (MAP_ORIGIN_Y - int(first[1])) / MAP_SCALE
    first_heading = math.atan2(float(fwy) - pose_y, float(fwx) - pose_x)
    first_err = abs(normalize_angle(first_heading - pose_theta))

    def step_dir(a, b):
        dx = int(b[0]) - int(a[0])
        dy = int(b[1]) - int(a[1])
        if abs(dx) >= abs(dy):
            return (1 if dx > 0 else -1 if dx < 0 else 0, 0)
        return (0, 1 if dy > 0 else -1 if dy < 0 else 0)

    dirs = []
    for a, b in zip(pts, pts[1:]):
        d = step_dir(a, b)
        if d != (0, 0):
            dirs.append(d)
    corners = 0
    for a, b in zip(dirs, dirs[1:]):
        if a != b:
            corners += 1
    first_frac = float(first_err / math.pi)
    return {
        "first_turn_frac": first_frac,
        "first_turn_deg": math.degrees(first_err),
        "corner_count": int(corners),
        "debug": f"geom first={math.degrees(first_err):.0f}deg corners={int(corners)}",
    }


def reconstruct_route_and_geometry(parent, start, goal, x0, y0, step):
    route = reconstruct_coarse_route(parent, start, goal, x0, y0, step)
    return route, route_path_geometry_stats(route)


def build_component_coverage_segment_suffix(component_labels, comp_id, entry_gx, entry_gy, passable,
                                            coarse_gain_cells, coarse_reclean_ratio):
    """Return a coarse-grid suffix that sweeps a coherent uncleaned component.

    A single target cell is the wrong abstraction for a wall strip.  This helper
    converts long/thin or coherent components into a sweep segment: once Dijkstra
    brings the robot to an entry cell, the committed route continues along the
    component's main axis.  The executor already follows route grid segments, so
    this gives us a lightweight CPP segment without introducing a new wheel owner.
    """
    t_perf_segment = perf_start()
    if not COVERAGE_SEGMENT_PLANNER_ENABLED or comp_id <= 0:
        perf_end("segment", t_perf_segment)
        return None
    try:
        comp_mask = component_labels == int(comp_id)
        ys, xs = np.where(comp_mask)
        cells = int(len(xs))
        if cells < COVERAGE_SEGMENT_MIN_COMP_CELLS:
            return None
        min_x = int(xs.min()); max_x = int(xs.max())
        min_y = int(ys.min()); max_y = int(ys.max())
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        span = max(width, height)
        if span < COVERAGE_SEGMENT_MIN_SPAN_CELLS:
            return None
        horizontal = width >= height
        sequence = []
        if horizontal:
            # One representative cell per coarse column; choose the row with the
            # largest cleaning-footprint gain, then prefer the median component row
            # for visual smoothness.
            med_y = float(np.median(ys))
            for x in range(min_x, max_x + 1):
                cand_y = ys[xs == x]
                if cand_y.size <= 0:
                    continue
                best_y = None
                best_key = None
                for yy in cand_y.tolist():
                    if yy < 0 or yy >= passable.shape[0] or x < 0 or x >= passable.shape[1] or not passable[int(yy), int(x)]:
                        continue
                    key = (int(coarse_gain_cells[int(yy), int(x)]), -abs(float(yy) - med_y))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_y = int(yy)
                if best_y is not None:
                    sequence.append((int(x), int(best_y)))
        else:
            med_x = float(np.median(xs))
            for y in range(min_y, max_y + 1):
                cand_x = xs[ys == y]
                if cand_x.size <= 0:
                    continue
                best_x = None
                best_key = None
                for xx in cand_x.tolist():
                    if y < 0 or y >= passable.shape[0] or xx < 0 or xx >= passable.shape[1] or not passable[int(y), int(xx)]:
                        continue
                    key = (int(coarse_gain_cells[int(y), int(xx)]), -abs(float(xx) - med_x))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_x = int(xx)
                if best_x is not None:
                    sequence.append((int(best_x), int(y)))

        # Remove duplicate consecutive cells and split on large gaps.  We keep the
        # segment containing the entry candidate; otherwise route execution would
        # jump across an obstacle hole.
        cleaned_seq = []
        for cell in sequence:
            if not cleaned_seq or cleaned_seq[-1] != cell:
                cleaned_seq.append(cell)
        if len(cleaned_seq) < COVERAGE_SEGMENT_MIN_SPAN_CELLS:
            return None

        chunks = []
        cur = [cleaned_seq[0]]
        for cell in cleaned_seq[1:]:
            px, py = cur[-1]
            if abs(cell[0] - px) + abs(cell[1] - py) <= 2:
                cur.append(cell)
            else:
                chunks.append(cur)
                cur = [cell]
        chunks.append(cur)
        entry = (int(entry_gx), int(entry_gy))
        best_chunk = None
        best_entry_idx = -1
        best_entry_dist = 1e9
        for chunk in chunks:
            if len(chunk) < COVERAGE_SEGMENT_MIN_SPAN_CELLS:
                continue
            for idx, cell in enumerate(chunk):
                d = abs(cell[0] - entry[0]) + abs(cell[1] - entry[1])
                if d < best_entry_dist:
                    best_entry_dist = d
                    best_entry_idx = idx
                    best_chunk = chunk
        if best_chunk is None or best_entry_dist > 2:
            return None

        # Sweep toward the farther end of the segment so that a route commit cleans
        # a lane, not only the entry point.  Limit length to keep partial-map commits
        # stable when the component is long/noisy.
        left_len = best_entry_idx + 1
        right_len = len(best_chunk) - best_entry_idx
        if right_len >= left_len:
            suffix = best_chunk[best_entry_idx:]
            direction = "E/S" if horizontal else "S"
        else:
            suffix = list(reversed(best_chunk[:best_entry_idx + 1]))
            direction = "W/N" if horizontal else "N"
        if len(suffix) > COVERAGE_SEGMENT_MAX_CELLS:
            suffix = suffix[:COVERAGE_SEGMENT_MAX_CELLS]
        if len(suffix) < COVERAGE_SEGMENT_MIN_SPAN_CELLS:
            return None

        gain_sum = 0
        reclean_sum = 0.0
        for gx, gy in suffix:
            if 0 <= gy < coarse_gain_cells.shape[0] and 0 <= gx < coarse_gain_cells.shape[1]:
                gain_sum += int(coarse_gain_cells[gy, gx])
                reclean_sum += float(coarse_reclean_ratio[gy, gx])
        reclean_avg = reclean_sum / max(1, len(suffix))
        if gain_sum < COVERAGE_SEGMENT_MIN_GAIN_CELLS and len(suffix) < COVERAGE_SEGMENT_MIN_SPAN_CELLS + 2:
            return None
        length_bonus = COVERAGE_SEGMENT_LENGTH_BONUS * float(len(suffix))
        gain_bonus = min(COVERAGE_SEGMENT_GAIN_MAX, COVERAGE_SEGMENT_GAIN_SCALE * float(gain_sum))
        reclean_penalty = COVERAGE_SEGMENT_RECLEAN_PENALTY * float(reclean_avg)
        bonus = gain_bonus + length_bonus - reclean_penalty
        debug = f"seg {'H' if horizontal else 'V'} n={len(suffix)} gain={gain_sum} rc={reclean_avg:.2f} {direction}"
        result = {
            "suffix": [(int(x), int(y)) for x, y in suffix],
            "gain": int(gain_sum),
            "reclean": float(reclean_avg),
            "bonus": float(bonus),
            "debug": debug,
        }
        perf_end("segment", t_perf_segment)
        return result
    except Exception:
        perf_end("segment", t_perf_segment)
        return None


def append_segment_suffix_to_route(route_map, segment_suffix, x0, y0, step):
    """Append coarse segment suffix to a reconstructed map-coordinate route."""
    if not segment_suffix:
        return route_map
    result = list(route_map or [])
    for cgx, cgy in segment_suffix:
        p = coarse_route_center(x0, y0, step, int(cgx), int(cgy))
        if result and result[-1] == p:
            continue
        result.append(p)
    return result

def plan_route_to_world_goal(goal_world, obstacles=None, cleanable=None, padding_m=DOCK_ROUTE_GRID_PADDING_M):
    """Plan a footprint-aware cardinal route to an explicit world goal.

    Coverage planning chooses the best target.  Dock return is different: the
    target is fixed, so this function only solves path-to-goal.  It reuses the
    same 1:1 footprint passability map and 4-connected Dijkstra/wavefront style
    as planned coverage, keeping the architecture consistent.
    """
    if goal_world is None:
        return None
    if obstacles is None or cleanable is None:
        obstacles, cleanable, _cleaned, _uncleaned, _unknown = compute_coverage_masks()
    actual_obstacles = obstacles.astype(np.bool_)
    route_obstacles = planning_no_go_obstacle_mask(actual_obstacles)

    robot_mx, robot_my = world_to_map(pose_x, pose_y)
    goal_mx, goal_my = world_to_map(float(goal_world[0]), float(goal_world[1]))
    if not map_inside(robot_mx, robot_my):
        return None
    if not map_inside(goal_mx, goal_my):
        return None

    step = int(GLOBAL_ROUTE_GRID_STEP_PX)
    pad = int(max(step * 3, padding_m * MAP_SCALE))
    x0 = max(0, min(robot_mx, goal_mx) - pad)
    x1 = min(MAP_SIZE, max(robot_mx, goal_mx) + pad)
    y0 = max(0, min(robot_my, goal_my) - pad)
    y1 = min(MAP_SIZE, max(robot_my, goal_my) + pad)
    if x1 - x0 < step * 4 or y1 - y0 < step * 4:
        x0 = max(0, robot_mx - int(1.3 * MAP_SCALE))
        x1 = min(MAP_SIZE, robot_mx + int(1.3 * MAP_SCALE))
        y0 = max(0, robot_my - int(1.3 * MAP_SCALE))
        y1 = min(MAP_SIZE, robot_my + int(1.3 * MAP_SCALE))

    gw = int(math.ceil((x1 - x0) / step))
    gh = int(math.ceil((y1 - y0) / step))
    if gw <= 1 or gh <= 1:
        return None
    if gw * gh > GLOBAL_ROUTE_MAX_NODES:
        factor = math.sqrt((gw * gh) / GLOBAL_ROUTE_MAX_NODES)
        step = int(math.ceil(step * factor))
        gw = int(math.ceil((x1 - x0) / step))
        gh = int(math.ceil((y1 - y0) / step))

    footprint_passable, _clearance = build_footprint_passability_map(obstacles, cleanable, FOOTPRINT_ROUTE_MARGIN_M)
    passable = np.zeros((gh, gw), dtype=np.bool_)
    cleaned_ratio = np.zeros((gh, gw), dtype=np.float32)
    recent_ratio = np.zeros((gh, gw), dtype=np.float32)
    for gy in range(gh):
        py0 = y0 + gy * step
        py1 = min(y1, py0 + step)
        if py1 <= py0:
            continue
        for gx in range(gw):
            px0 = x0 + gx * step
            px1 = min(x1, px0 + step)
            if px1 <= px0:
                continue
            area = max(1, (py1 - py0) * (px1 - px0))
            obs_ratio = float(np.count_nonzero(route_obstacles[py0:py1, px0:px1])) / area
            cleanable_ratio = float(np.count_nonzero(cleanable[py0:py1, px0:px1])) / area
            footprint_ratio = float(np.count_nonzero(footprint_passable[py0:py1, px0:px1])) / area
            # The dock cell itself can be at the boundary of known space.  Prefer
            # footprint-passable known/cleaned cells, but allow a coarse cell near
            # the odometry origin if it contains enough cleaned/free support.
            near_dock = math.hypot((px0 + px1) * 0.5 - goal_mx, (py0 + py1) * 0.5 - goal_my) <= max(step * DOCK_ROUTE_NEAREST_GOAL_RADIUS, 0.30 * MAP_SCALE)
            min_cleanable = 0.04 if near_dock else GLOBAL_ROUTE_MIN_CLEANABLE_RATIO
            min_footprint = 0.03 if near_dock else 0.06
            if obs_ratio > GLOBAL_ROUTE_OBSTACLE_RATIO_BLOCK or cleanable_ratio < min_cleanable or footprint_ratio < min_footprint:
                continue
            passable[gy, gx] = True
            cleaned_ratio[gy, gx] = float(np.count_nonzero((cleaned_mask[py0:py1, px0:px1] > 0))) / area
            if RECENT_VISIT_ROUTE_MEMORY_ENABLED:
                recent_ratio[gy, gx] = min(1.0, float(np.mean(recent_visit_log_odds[py0:py1, px0:px1])) / max(1e-6, RECENT_VISIT_MAX))

    sx = int((robot_mx - x0) // step)
    sy = int((robot_my - y0) // step)
    gx0 = int((goal_mx - x0) // step)
    gy0 = int((goal_my - y0) // step)
    start = nearest_passable_route_cell(passable, sx, sy, max_r=5)
    goal = nearest_passable_route_cell(passable, gx0, gy0, max_r=DOCK_ROUTE_NEAREST_GOAL_RADIUS)
    if start is None or goal is None:
        return None
    if not route_start_cell_is_locally_reachable(start, x0, y0, step, route_obstacles, cleanable):
        return None

    inf = 1e18
    dist_grid = np.full((gh, gw), inf, dtype=np.float32)
    parent = np.full((gh, gw), -1, dtype=np.int32)
    sx, sy = start
    gx_goal, gy_goal = goal
    dist_grid[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    expanded = 0
    while heap:
        d, gx, gy = heapq.heappop(heap)
        expanded += 1
        if expanded > GLOBAL_ROUTE_MAX_NODES:
            break
        if (gx, gy) == goal:
            break
        if d > float(dist_grid[gy, gx]) + 1e-6:
            continue
        for dx, dy, base_cost in nbrs:
            nx = gx + dx
            ny = gy + dy
            if nx < 0 or ny < 0 or nx >= gw or ny >= gh or not passable[ny, nx]:
                continue
            step_cost = base_cost * (
                1.0
                + GLOBAL_ROUTE_CLEANED_TRANSIT_PENALTY * 0.25 * float(cleaned_ratio[ny, nx])
                + GLOBAL_ROUTE_RECENT_TRANSIT_PENALTY * 0.15 * float(recent_ratio[ny, nx])
            )
            nd = d + step_cost
            if nd < float(dist_grid[ny, nx]):
                dist_grid[ny, nx] = nd
                parent[ny, nx] = gy * gw + gx
                heapq.heappush(heap, (nd, nx, ny))

    if not np.isfinite(dist_grid[gy_goal, gx_goal]) or dist_grid[gy_goal, gx_goal] >= inf * 0.5:
        return None
    route = reconstruct_coarse_route(parent, start, goal, x0, y0, step)
    if not route:
        return None
    cost_m = float(dist_grid[gy_goal, gx_goal]) * step / MAP_SCALE
    route_world = [((mx - MAP_ORIGIN_X) / MAP_SCALE, (MAP_ORIGIN_Y - my) / MAP_SCALE) for mx, my in route]
    wp_map, wp_world = route_waypoint_from_path(route)
    goal_map = route[-1]
    goal_world2 = ((goal_map[0] - MAP_ORIGIN_X) / MAP_SCALE, (MAP_ORIGIN_Y - goal_map[1]) / MAP_SCALE)
    return {
        "goal_map": (int(goal_map[0]), int(goal_map[1])),
        "goal_world": (float(goal_world2[0]), float(goal_world2[1])),
        "route_map": route,
        "route_world": route_world,
        "waypoint_map": wp_map,
        "waypoint_world": wp_world,
        "cost": float(cost_m),
        "length": int(len(route)),
        "expanded": int(expanded),
    }


def local_mask_count(mask, cx, cy, radius_m):
    r = max(2, int(round(radius_m * MAP_SCALE)))
    x0 = max(0, int(cx) - r)
    x1 = min(MAP_SIZE, int(cx) + r + 1)
    y0 = max(0, int(cy) - r)
    y1 = min(MAP_SIZE, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1]))


def compute_planner_intent():
    """Return (intent, reason) for the high-level planner arbitration.

    The intent is the single source of truth for EXPAND_MAP vs CLEAN_KNOWN.  A
    PARTIAL map with significant frontier cells must not behave like mature
    cleanup just because a residual uncleaned component has a high local score.
    """
    if not EXPLORATION_COVERAGE_ARBITRATION_ENABLED:
        return PLANNER_INTENT_CLEAN_KNOWN, "arbitration disabled"
    if dock_return_active or dock_return_completed:
        return PLANNER_INTENT_FINISH_CLEANUP, "dock/mission ending"
    if known_map_coverage_eval_active():
        cov_now = float(last_coverage_percent or 0.0)
        if cov_now >= RESIDUAL_CLEANUP_COVERAGE_PERCENT:
            return PLANNER_INTENT_FINISH_CLEANUP, f"known-map finish cov={cov_now:.1f}"
        return PLANNER_INTENT_CLEAN_KNOWN, f"known-map planned coverage cov={cov_now:.1f}"
    try:
        frontiers = int(last_frontier_cells or 0)
        cov = float(last_coverage_percent or 0.0)
        conf = str(planner_confidence or "EARLY")
    except Exception:
        return PLANNER_INTENT_CLEAN_KNOWN, "bad counters"

    if map_mature or conf == "MATURE":
        if frontiers >= PLANNER_INTENT_HIGH_COVERAGE_FRONTIER_CELLS and cov < MAP_MATURE_COVERAGE_PERCENT:
            return PLANNER_INTENT_EXPAND_MAP, f"mature override: frontier={frontiers} cov={cov:.1f}"
        return PLANNER_INTENT_FINISH_CLEANUP, f"mature cov={cov:.1f} frontier={frontiers}"

    # EARLY/PARTIAL map: a meaningful frontier is a map-expansion task.  This is
    # intentionally stricter than the old coverage-only threshold: it does not
    # let PARTIAL confidence silently turn into cleanup while the room is still
    # being discovered.
    if frontiers >= PLANNER_INTENT_EXPAND_FRONTIER_CELLS:
        return PLANNER_INTENT_EXPAND_MAP, f"partial frontier={frontiers} cov={cov:.1f} conf={conf}"
    if navigation_phase == NavigationPhase.EXPLORE.value and frontiers >= PLANNER_INTENT_MIN_FRONTIER_CELLS_EARLY:
        return PLANNER_INTENT_EXPAND_MAP, f"explore phase frontier={frontiers} cov={cov:.1f}"

    if cov >= MAP_MATURE_TIME_COVERAGE_PERCENT and frontiers < EXPLORE_PRIORITY_FORCE_FRONTIER_CELLS:
        return PLANNER_INTENT_CLEAN_KNOWN, f"frontier low={frontiers} cov={cov:.1f}"
    return PLANNER_INTENT_CLEAN_KNOWN, f"no strong frontier={frontiers} cov={cov:.1f} conf={conf}"


def update_planner_intent():
    """Update global planner_intent after map maturity/frontier counters change."""
    global planner_intent, planner_intent_reason, last_explore_arbitration_debug
    intent, reason = compute_planner_intent()
    old_intent = planner_intent
    planner_intent = intent
    planner_intent_reason = reason
    if intent == PLANNER_INTENT_EXPAND_MAP and old_intent != PLANNER_INTENT_EXPAND_MAP:
        cancel_coverage_lane_queue_for_expand_map("intent switched to EXPAND_MAP")
    last_explore_arbitration_debug = f"intent={intent} {reason}"
    return planner_intent


def exploration_priority_active():
    """Compatibility wrapper: true only when the central intent is EXPAND_MAP."""
    if not EXPLORATION_COVERAGE_ARBITRATION_ENABLED:
        return False
    return planner_intent == PLANNER_INTENT_EXPAND_MAP


def exploration_mapping_snapshot_allowed():
    """True when stationary RGB-D mapping snapshots are allowed.

    originally tied post-turn snapshots to matrix_first_explore_active(),
    which becomes false in late MAP_STABILIZATION even while intent=EXPAND_MAP and
    conf=PARTIAL.  That is exactly when the robot needs stationary Camera +
    RangeFinder fusion instead of falling back to coverage rows.
    """
    if known_map_coverage_eval_active() or dock_return_active or dock_return_completed:
        return False
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return False
    if map_mature or str(planner_confidence or "") == "MATURE":
        return False
    return True


def frontier_cells_near_map(mx, my, unknown=None, cleanable=None):
    """Count useful frontier cells around a map target."""
    try:
        if unknown is None or cleanable is None:
            _obs, cleanable, _cleaned, _uncleaned, unknown = compute_coverage_masks()
        hyp_shadow = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(unknown, dtype=np.bool_)
        frontier_mask = (unknown & (~hyp_shadow)) & (cv2.dilate(cleanable.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0)
        return local_mask_count(frontier_mask, int(mx), int(my), EXPLORE_FRONTIER_LOCAL_RADIUS_M)
    except Exception:
        return 0

def exploration_cleanup_unlocked():
    """True when known-floor cleanup planning may compete with frontiers.

    This is an occupancy-map closure proxy, not ORB/RTAB loop closure.  While the
    room still has a large frontier mass, EXPAND_MAP should keep the robot
    discovering boundaries instead of route-optimizing uncleaned islands.
    """
    global last_exploration_cleanup_lock_debug
    if known_map_coverage_eval_active():
        last_exploration_cleanup_lock_debug = "cleanupLock=unlocked known-map"
        return True
    if not EXPLORE_LOCK_CLEANUP_TARGETS_ENABLED:
        last_exploration_cleanup_lock_debug = "cleanupLock=off"
        return True
    if map_mature or str(planner_confidence or "") == "MATURE":
        last_exploration_cleanup_lock_debug = "cleanupLock=unlocked mature"
        return True
    try:
        t = float(robot.getTime())
    except Exception:
        t = 0.0
    total = max(1, int(last_coverage_total_cells or 0))
    frontiers = int(last_frontier_cells or 0)
    ratio = float(frontiers) / float(total)
    cov = float(last_coverage_percent or 0.0)
    unlocked = bool(
        cov >= EXPLORE_CLEANUP_UNLOCK_MIN_COVERAGE_PERCENT
        and t >= EXPLORE_CLEANUP_UNLOCK_MIN_TIME_SEC
        and (frontiers <= EXPLORE_CLEANUP_UNLOCK_MAX_FRONTIER_CELLS or ratio <= EXPLORE_CLEANUP_UNLOCK_MAX_FRONTIER_RATIO)
    )
    state = "unlocked" if unlocked else "locked"
    last_exploration_cleanup_lock_debug = f"cleanupLock={state} cov={cov:.1f} fr={frontiers} r={ratio:.3f}"
    return unlocked


def exploration_cleanup_locked():
    # Keep the debug label honest.  Known-map and CLEAN_KNOWN modes must never
    # show a stale "cleanupLock=locked" line from a previous EXPAND_MAP cycle.
    global last_exploration_cleanup_lock_debug
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        last_exploration_cleanup_lock_debug = "cleanupLock=unlocked intent"
        return False
    return bool(
        EXPLORE_LOCK_CLEANUP_TARGETS_ENABLED
        and planner_intent == PLANNER_INTENT_EXPAND_MAP
        and not exploration_cleanup_unlocked()
    )


def update_planner_mode_label():
    """Make the debug/control mode match the active intent, not legacy coverage names."""
    global planner_mode
    if auto_map_cleaning_started and known_map_coverage_eval_active():
        planner_mode = "K_LEARNED_MAP_PLANNER"
    elif known_map_coverage_eval_active() and planner_intent != PLANNER_INTENT_FINISH_CLEANUP:
        planner_mode = "K_KNOWN_MAP_PLANNER"
    elif SIMPLE_SWEEP_FSM_ENABLED and not simple_sweep_completed and not (dock_return_active or dock_return_completed):
        planner_mode = "EXPLORE_SWEEP_FSM"
    elif planner_intent == PLANNER_INTENT_EXPAND_MAP:
        planner_mode = "EXPLORATION_FRONTIER" if exploration_cleanup_locked() else "MAP_STABILIZATION"
    elif planner_intent == PLANNER_INTENT_FINISH_CLEANUP:
        planner_mode = "FINISH_CLEANUP"
    elif map_mature:
        planner_mode = "PLANNED_COVERAGE"
    elif map_mature_soft:
        planner_mode = "HYBRID_COVERAGE"
    else:
        planner_mode = "ROW_COVERAGE"
    return planner_mode


def active_scan_tile_key(x=None, y=None):
    try:
        if x is None:
            x = pose_x
        if y is None:
            y = pose_y
        scale = max(0.10, float(ACTIVE_SCAN_AREA_TILE_M))
        return (int(math.floor(float(x) / scale)), int(math.floor(float(y) / scale)))
    except Exception:
        return (0, 0)


def active_scan_area_suppressed(now, tile_key=None):
    """Suppress repeated scan-around in the same approximate map area."""
    global active_scan_area_memory, last_active_scan_debug
    if not ACTIVE_SCAN_AREA_MEMORY_ENABLED:
        return False
    if tile_key is None:
        tile_key = active_scan_tile_key()
    keep = []
    suppressed = False
    age_dbg = 0.0
    for item in active_scan_area_memory:
        try:
            age = float(now) - float(item.get("time", -999.0))
            if age <= ACTIVE_SCAN_AREA_REPEAT_SUPPRESS_SEC:
                keep.append(item)
                if tuple(item.get("tile", (None, None))) == tuple(tile_key):
                    suppressed = True
                    age_dbg = age
        except Exception:
            continue
    active_scan_area_memory = keep[-ACTIVE_SCAN_AREA_MEMORY_MAX:]
    if suppressed:
        last_active_scan_debug = f"scan=area-cooldown {age_dbg:.0f}s tile={tile_key}"
    return suppressed


def remember_active_scan_area(reason):
    global active_scan_area_memory
    if not ACTIVE_SCAN_AREA_MEMORY_ENABLED:
        return
    try:
        now = float(robot.getTime())
    except Exception:
        now = 0.0
    tile = active_scan_tile_key(active_scan_started_x, active_scan_started_y)
    entry = {
        "tile": tile,
        "time": now,
        "x": float(active_scan_started_x),
        "y": float(active_scan_started_y),
        "reason": str(reason or "done")[:40],
        "start_frontier": int(active_scan_start_frontier_local or 0),
        "start_unknown": int(active_scan_start_unknown_local or 0),
    }
    active_scan_area_memory = [e for e in active_scan_area_memory if tuple(e.get("tile", (None, None))) != tile]
    active_scan_area_memory.append(entry)
    if len(active_scan_area_memory) > ACTIVE_SCAN_AREA_MEMORY_MAX:
        active_scan_area_memory = active_scan_area_memory[-ACTIVE_SCAN_AREA_MEMORY_MAX:]


def post_turn_rgbd_snapshot_phase(now=None):
    """Return idle/settle/capture for the bounded post-turn RGB-D snapshot."""
    if now is None:
        now = robot.getTime()
    if nav_state != NAV_RGBD_SNAPSHOT or now >= post_turn_rgbd_snapshot_started_at + float(POST_TURN_RGBD_SNAPSHOT_MAX_SEC):
        return "idle"
    elapsed = max(0.0, now - post_turn_rgbd_snapshot_started_at)
    if elapsed < float(POST_TURN_RGBD_SNAPSHOT_SETTLE_SEC):
        return "settle"
    return "capture"


def post_turn_rgbd_snapshot_capture_ready(now=None):
    return post_turn_rgbd_snapshot_phase(now) == "capture"


def note_post_turn_rgbd_snapshot_mapping(depth_hits=0, cv_hits=0):
    """Count actual capture-frame mapping opportunities, not settle/turn ticks."""
    global post_turn_rgbd_snapshot_frames, post_turn_rgbd_snapshot_write_frames
    global last_post_turn_snapshot_debug
    if nav_state != NAV_RGBD_SNAPSHOT:
        return
    now = robot.getTime()
    phase = post_turn_rgbd_snapshot_phase(now)
    elapsed = max(0.0, now - post_turn_rgbd_snapshot_started_at)
    if phase == "settle":
        last_post_turn_snapshot_debug = f"snap=settle {elapsed:.2f}s"
        return
    if phase != "capture":
        return
    post_turn_rgbd_snapshot_frames += 1
    if int(depth_hits or 0) > 0 or int(cv_hits or 0) > 0:
        post_turn_rgbd_snapshot_write_frames += 1
    last_post_turn_snapshot_debug = (
        f"snap=capture f={post_turn_rgbd_snapshot_frames} "
        f"w={post_turn_rgbd_snapshot_write_frames} d/c={int(depth_hits or 0)}/{int(cv_hits or 0)}"
    )


def update_mapping_write_debug(depth_hits=0, cv_hits=0, forced=False):
    """Compact HUD/debug text that explains whether persistent mapping may write."""
    global last_mapping_write_debug, last_mapping_reason
    global last_mapping_depth_hits_debug, last_mapping_cv_hits_debug
    now = robot.getTime()
    frozen, freeze_reason = update_mapping_freeze_state(now)
    last_mapping_depth_hits_debug = int(depth_hits or 0)
    last_mapping_cv_hits_debug = int(cv_hits or 0)
    if nav_state == NAV_RGBD_SNAPSHOT:
        phase = post_turn_rgbd_snapshot_phase(now)
        allowed = (phase == "capture") and (not frozen)
        last_mapping_reason = f"snapshot-{phase if phase != 'idle' else 'done'}"
    elif nav_state == NAV_SCAN_AROUND:
        allowed = (not frozen) and active_scan_mapping_dwell_ready(now)
        last_mapping_reason = "scan-dwell" if allowed else f"scan-blocked:{freeze_reason}"
    elif frozen:
        allowed = False
        last_mapping_reason = f"turn-blocked:{freeze_reason}"
    elif nav_state in MAPPING_ALLOWED_STATES:
        allowed = True
        last_mapping_reason = "move" if nav_state == NAV_FORWARD else str(nav_state)
    else:
        allowed = False
        last_mapping_reason = f"nav-blocked:{nav_state}"
    state = "on" if allowed else "off"
    force_tag = "!" if forced else ""
    last_mapping_write_debug = f"mapWrite={state}{force_tag} {last_mapping_reason[:24]} d/c={last_mapping_depth_hits_debug}/{last_mapping_cv_hits_debug}"


def mark_thin_obstacle_candidate(mx, my):
    """Temporally confirm close lower-band depth clusters before raw occupancy."""
    global thin_obstacle_log_odds
    if not THIN_OBSTACLE_CONFIRM_ENABLED:
        cv2.circle(log_odds, (int(mx), int(my)), 2, THIN_OBSTACLE_OCC_UPDATE, -1)
        return True
    if not map_inside(mx, my):
        return False
    r = int(THIN_OBSTACLE_CONFIRM_RADIUS)
    cv2.circle(thin_obstacle_log_odds, (int(mx), int(my)), r, float(THIN_OBSTACLE_CONFIRM_UPDATE), -1)
    x0 = max(0, int(mx) - r)
    x1 = min(MAP_SIZE, int(mx) + r + 1)
    y0 = max(0, int(my) - r)
    y1 = min(MAP_SIZE, int(my) + r + 1)
    if float(np.max(thin_obstacle_log_odds[y0:y1, x0:x1])) < float(THIN_OBSTACLE_CONFIRM_EPS):
        return False
    # Confirmed thin obstacle: small radius, moderate update. Repeated snapshot
    # frames strengthen it; one bad lower-band cluster will not.
    cv2.circle(log_odds, (int(mx), int(my)), 1, THIN_OBSTACLE_OCC_UPDATE, -1)
    return True


def note_route_commit_inplace_turn(heading_err, reason):
    """Arm a post-turn snapshot when an exploration route really spins in place."""
    global route_commit_turn_snapshot_armed, route_commit_turn_snapshot_started_at
    global route_commit_turn_snapshot_peak_err, route_commit_turn_snapshot_reason
    if known_map_coverage_eval_active():
        return
    if not (POST_TURN_RGBD_SNAPSHOT_ENABLED and exploration_mapping_snapshot_allowed()):
        return
    if route_commit_kind != "frontier" or planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return
    now = robot.getTime()
    err = abs(float(heading_err))
    if not route_commit_turn_snapshot_armed:
        route_commit_turn_snapshot_started_at = now
        route_commit_turn_snapshot_peak_err = err
        route_commit_turn_snapshot_reason = str(reason or "route pivot")[:70]
    else:
        route_commit_turn_snapshot_peak_err = max(route_commit_turn_snapshot_peak_err, err)
    route_commit_turn_snapshot_armed = True


def maybe_start_route_commit_post_turn_snapshot(reason="route pivot aligned"):
    """Start RGB-D snapshot after ROUTE_COMMIT/FINE_ALIGN-like exploration turns."""
    global route_commit_turn_snapshot_armed, route_commit_turn_snapshot_peak_err
    global route_commit_turn_snapshot_started_at, route_commit_turn_snapshot_reason
    if not route_commit_turn_snapshot_armed:
        return False
    now = robot.getTime()
    peak_deg = math.degrees(abs(route_commit_turn_snapshot_peak_err))
    duration = max(0.0, now - route_commit_turn_snapshot_started_at)
    armed_reason = route_commit_turn_snapshot_reason
    route_commit_turn_snapshot_armed = False
    route_commit_turn_snapshot_peak_err = 0.0
    route_commit_turn_snapshot_started_at = -999.0
    route_commit_turn_snapshot_reason = "none"
    if peak_deg < float(POST_TURN_ROUTE_PIVOT_MIN_ERR_DEG) or duration < float(POST_TURN_ROUTE_PIVOT_MIN_SEC):
        return False
    return start_post_turn_rgbd_snapshot(f"{reason}: {armed_reason} peak={peak_deg:.0f}deg")

def active_rgbd_scan_need(front, body_clearance):
    """Return (should_scan, reason) for active-perception sweep.

    This is deliberately a planner-intent primitive: it is allowed only when the
    high-level intent is EXPAND_MAP and the robot is at a safe vantage pose near
    a sizeable frontier/unknown area.  It prevents the robot from committing to
    a strange long route while the local RGB-D map is still one-view incomplete.
    """
    global last_active_scan_debug
    if not ACTIVE_RGBD_SCAN_ENABLED:
        last_active_scan_debug = "scan=off"
        return False, "disabled"
    now = robot.getTime()
    if planner_intent != PLANNER_INTENT_EXPAND_MAP or map_mature:
        last_active_scan_debug = f"scan=skip intent={planner_intent}"
        return False, "not expand-map"
    if nav_state != NAV_FORWARD:
        last_active_scan_debug = f"scan=skip nav={nav_state}"
        return False, "not forward"
    if route_commit_active or dock_return_active or dock_return_completed:
        last_active_scan_debug = "scan=skip route/dock"
        return False, "route/dock active"
    if nav_action_queue:
        last_active_scan_debug = "scan=skip queue"
        return False, "action queue"
    if frontier_direct_mapping_safe(front, front, body_clearance):
        last_active_scan_debug = "scan=skip frontier-direct"
        return False, "frontier direct mapping"

    if ACTIVE_SCAN_RESPECT_ROUTE_CANDIDATE_ENABLED:
        try:
            route_cost = float(coverage_route_cost)
            relaxed_cost_limit = max(float(ACTIVE_SCAN_CANDIDATE_ROUTE_MAX_COST_M), 3.25)
            good_frontier_route = bool(
                coverage_route_kind == "frontier"
                and coverage_goal_map is not None
                and coverage_route_map
                and math.isfinite(route_cost)
                and route_cost <= relaxed_cost_limit
                and coverage_route_first_turn_frac <= 0.72
                and coverage_route_corner_count <= 5
            )
            if good_frontier_route:
                # A displayed frontier path is not the same thing as a real motion
                # owner.  If policy/safety would not let ROUTE_COMMIT start, do
                # not let this stale candidate suppress the perception scan; that
                # was one way ROW_FORWARD became a hidden coverage fallback.
                route_ok, route_reason = coverage_candidate_is_committable(front, None, body_clearance)
                if route_ok:
                    last_active_scan_debug = f"scan=skip committable frontier-route {route_cost:.2f}m"
                    return False, "frontier route committable"
                last_active_scan_debug = f"scan=allow noncommit-route {str(route_reason)[:32]}"
        except Exception:
            pass
    tile_key = active_scan_tile_key()
    if active_scan_area_suppressed(now, tile_key):
        return False, "area cooldown"
    if last_bumper_left or last_bumper_center or last_bumper_right:
        last_active_scan_debug = "scan=skip bumper"
        return False, "bumper"
    if front < ACTIVE_SCAN_MIN_FRONT_CLEAR_M or body_clearance < ACTIVE_SCAN_MIN_BODY_CLEAR_M:
        last_active_scan_debug = f"scan=skip unsafe F={front:.2f} body={body_clearance:.2f}"
        return False, "unsafe clearance"
    moved_since = math.hypot(pose_x - active_scan_last_x, pose_y - active_scan_last_y)
    if now - active_scan_last_completed_at < ACTIVE_SCAN_COOLDOWN_SEC and moved_since < ACTIVE_SCAN_SPATIAL_COOLDOWN_M:
        last_active_scan_debug = f"scan=cooldown {now-active_scan_last_completed_at:.1f}s d={moved_since:.2f}"
        return False, "cooldown"
    try:
        _obstacles, cleanable, _cleaned, _uncleaned, unknown = compute_coverage_masks()
        cleanable_u8 = cleanable.astype(np.uint8)
        frontier = unknown & (cv2.dilate(cleanable_u8, np.ones((7, 7), np.uint8), iterations=1) > 0)
        mx, my = world_to_map(pose_x, pose_y)
        if not map_inside(mx, my):
            last_active_scan_debug = "scan=skip outside-map"
            return False, "outside map"
        local_frontier = local_mask_count(frontier, mx, my, ACTIVE_SCAN_TRIGGER_RADIUS_M)
        local_unknown = local_mask_count(unknown, mx, my, ACTIVE_SCAN_TRIGGER_RADIUS_M)
        should = bool(
            local_frontier >= ACTIVE_SCAN_TRIGGER_FRONTIER_CELLS
            and local_unknown >= ACTIVE_SCAN_TRIGGER_UNKNOWN_CELLS
            and int(last_frontier_cells or 0) >= PLANNER_INTENT_EXPAND_FRONTIER_CELLS
        )
        last_active_scan_debug = f"scanNeed={int(should)} frLocal={local_frontier} unkLocal={local_unknown} fr={last_frontier_cells}"
        if should:
            return True, f"large local frontier frLocal={local_frontier} unkLocal={local_unknown}"
        return False, "frontier not large enough"
    except Exception as exc:
        last_active_scan_debug = f"scan=err {str(exc)[:28]}"
        return False, "scan metric error"


def frontier_only_stall_watchdog_update(route_present, route_reason, safe_snapshot, cov):
    """Return an action string for frontier-only deadlock recovery."""
    global frontier_only_stall_since, frontier_only_stall_x, frontier_only_stall_y, frontier_only_stall_theta
    global frontier_only_stall_blacklists, frontier_only_last_blacklist_time, frontier_only_last_stall_debug
    if not EXPLORATION_FRONTIER_ONLY_STALL_WATCHDOG_ENABLED:
        frontier_only_last_stall_debug = "stall=off"
        return "none"
    try:
        now = float(robot.getTime())
    except Exception:
        now = 0.0
    moved = math.hypot(float(pose_x) - float(frontier_only_stall_x), float(pose_y) - float(frontier_only_stall_y))
    turned = abs(normalize_angle(float(pose_theta) - float(frontier_only_stall_theta)))
    if (
        frontier_only_stall_since < -100.0
        or moved > float(EXPLORATION_FRONTIER_ONLY_STALL_POSE_EPS_M)
        or turned > float(EXPLORATION_FRONTIER_ONLY_STALL_HEADING_EPS_RAD)
    ):
        frontier_only_stall_since = now
        frontier_only_stall_x = float(pose_x)
        frontier_only_stall_y = float(pose_y)
        frontier_only_stall_theta = float(pose_theta)
        frontier_only_last_stall_debug = "stall=reset"
        return "none"
    stall_sec = max(0.0, now - float(frontier_only_stall_since))
    frontier_only_last_stall_debug = f"stall={stall_sec:.0f}s bl={frontier_only_stall_blacklists} safe={int(bool(safe_snapshot))}"
    if stall_sec < float(EXPLORATION_FRONTIER_ONLY_STALL_TIMEOUT_SEC):
        return "none"
    # If a long sequence of unreachable local frontiers produces no map progress,
    # stop chasing gray/obstacle-shadow pockets and go to dock for learned K-clean.
    if (
        float(cov) >= float(EXPLORATION_FRONTIER_ONLY_STALL_RETURN_MIN_COVERAGE_PERCENT)
        and (
            stall_sec >= float(EXPLORATION_FRONTIER_ONLY_STALL_RETURN_TIMEOUT_SEC)
            or frontier_only_stall_blacklists >= int(EXPLORATION_FRONTIER_ONLY_STALL_MAX_BLACKLISTS_BEFORE_DOCK)
        )
    ):
        frontier_only_last_stall_debug = f"stall=dock {stall_sec:.0f}s cov={cov:.1f} bl={frontier_only_stall_blacklists}"
        return "dock"
    # Otherwise suppress the current frontier viewpoint/component and force the
    # planner to search a different vantage point.
    if now - float(frontier_only_last_blacklist_time) >= max(2.0, float(EXPLORATION_FRONTIER_ONLY_STALL_TIMEOUT_SEC) * 0.65):
        try:
            if coverage_goal_map is not None:
                gx, gy = coverage_goal_map
                register_map_target_blacklist(
                    int(gx), int(gy), "frontier",
                    f"frontier stall {str(route_reason)[:24]}",
                    ttl_sec=EXPLORATION_FRONTIER_ONLY_STALL_BLACKLIST_SEC,
                    radius_m=EXPLORATION_FRONTIER_ONLY_STALL_BLACKLIST_RADIUS_M,
                )
                try:
                    maybe_mark_near_collision_hypothesis("frontier stall: " + str(route_reason)[:24])
                except Exception:
                    pass
                frontier_only_stall_blacklists += 1
                frontier_only_last_blacklist_time = now
                frontier_only_stall_since = now
                frontier_only_stall_x = float(pose_x)
                frontier_only_stall_y = float(pose_y)
                frontier_only_stall_theta = float(pose_theta)
                frontier_only_last_stall_debug = f"stall=blacklist {int(gx)},{int(gy)} bl={frontier_only_stall_blacklists}"
                return "blacklist"
        except Exception as exc:
            frontier_only_last_stall_debug = f"stall=blacklistErr {type(exc).__name__}"
    return "none"


def exploration_frontier_only_owner_gate(front, center, body_clearance, left=None, right=None):
    """Block ROW_FORWARD/coverage fallback during late partial EXPAND_MAP.

    This is intentionally placed after ROUTE_COMMIT and active-scan attempts in
    choose_motion_from_depth().  It does not choose a route; it only prevents the
    old row/coverage controller from owning motion while the planner still says
    the map is partial and useful frontiers remain.
    """
    global last_frontier_only_gate_debug, frontier_only_last_snapshot_at
    global coverage_status, last_optional_block_reason, last_planner_update_step
    global frontier_only_stall_since, frontier_only_stall_blacklists

    if not EXPLORATION_FRONTIER_ONLY_OWNER_GATE_ENABLED:
        last_frontier_only_gate_debug = "frontierGate=off"
        return False
    if known_map_coverage_eval_active() or dock_return_active or dock_return_completed:
        last_frontier_only_gate_debug = "frontierGate=skip known/dock"
        return False
    if nav_state != NAV_FORWARD:
        last_frontier_only_gate_debug = f"frontierGate=skip nav={nav_state}"
        return False
    if route_commit_active:
        last_frontier_only_gate_debug = "frontierGate=skip activeRoute"
        return False
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        last_frontier_only_gate_debug = f"frontierGate=skip intent={planner_intent}"
        return False
    if map_mature or str(planner_confidence or "") == "MATURE":
        last_frontier_only_gate_debug = "frontierGate=skip mature"
        return False

    try:
        cov = float(last_coverage_percent or 0.0)
        frontiers = int(last_frontier_cells or 0)
    except Exception:
        cov = 0.0
        frontiers = 0

    if frontiers < int(EXPLORATION_FRONTIER_ONLY_MIN_FRONTIER_CELLS):
        last_frontier_only_gate_debug = f"frontierGate=skip low-fr={frontiers}"
        return False

    # Hybrid local-map pass: before the map has a reliable connected free-space
    # base, a frontier candidate is only advice.  ROUTE_COMMIT already had the
    # first chance above; if it did not accept the route, let the local depth/bumper
    # row controller keep moving and collect more RGB-D evidence instead of holding
    # owner=NONE on a noisy/shadow frontier.
    if hybrid_local_mapping_active(cov):
        last_frontier_only_gate_debug = f"frontierGate=hybrid-row cov={cov:.1f}/{EXPLORATION_FRONTIER_ONLY_MIN_COVERAGE_PERCENT:.1f}"
        return False

    route_present = bool(
        coverage_route_kind == "frontier"
        and coverage_goal_map is not None
        and coverage_route_map
        and math.isfinite(float(coverage_route_cost))
    )
    route_ok = False
    route_reason = "no frontier route"
    if coverage_goal_kind == "frontier" and coverage_goal_map is not None and not route_present:
        route_reason = f"candidate frontier has no active route len={len(coverage_route_map or [])} kind={coverage_route_kind}"
    if route_present:
        try:
            route_ok, route_reason = coverage_candidate_is_committable(front, center, body_clearance)
        except Exception as exc:
            route_ok, route_reason = False, f"committable err {type(exc).__name__}"
    if route_ok and FRONTIER_DIRECT_MAPPING_HOLD_IF_ROUTE_OK:
        # route_commit_speeds() should take this on the next control cycle; do not
        # start ROW_FORWARD in between if a stale ROW lock exists.
        try:
            if str(control_lock.owner) == ControlOwner.ROW_FORWARD.value:
                release_control("frontier route will own; clear row fallback")
        except Exception:
            pass
        coverage_status = f"frontier-only wait route: {str(route_reason)[:54]}"
        last_optional_block_reason = "frontier-only waiting for ROUTE_COMMIT"
        last_frontier_only_gate_debug = f"frontierGate=waitRoute cost={coverage_route_cost:.2f}"
        last_planner_update_step = -999999
        return True

    if frontier_direct_mapping_safe(front, center, body_clearance):
        # No active ROUTE_COMMIT means the frontier candidate is only an advisor.
        # Let the deterministic forward/perimeter controller collect more RGB-D
        # evidence; the heavy planner is throttled by frontier_direct_mapping_mode_active().
        coverage_status = f"frontier-direct: defer {str(route_reason)[:44]} fr={frontiers} cov={cov:.1f}"
        last_optional_block_reason = "frontier candidate not committed -> direct mapping"
        last_frontier_only_gate_debug = (
            f"frontierGate=direct route={int(route_present)} "
            f"cost={float(coverage_route_cost) if math.isfinite(float(coverage_route_cost)) else -1.0:.2f} "
            f"fr={frontiers} cov={cov:.1f}"
        )
        return False

    if frontier_direct_mapping_unsafe(front, center, body_clearance):
        if start_frontier_direct_unsafe_recovery(front, center, left, right, body_clearance, route_reason):
            last_frontier_only_gate_debug = (
                f"frontierGate=unsafeRecovery route={int(route_present)} "
                f"{frontier_direct_last_unsafe_recovery_debug}"
            )
            return True

    # Clear hidden row/coverage ownership before it reaches forward_owner_guard().
    try:
        if str(control_lock.owner) in (ControlOwner.ROW_FORWARD.value, ControlOwner.PLANNER.value):
            release_control("frontier-only gate blocks row fallback")
    except Exception:
        pass

    now = robot.getTime()
    safe_snapshot = bool(
        front is not None
        and body_clearance is not None
        and float(front) >= float(EXPLORATION_FRONTIER_ONLY_SNAPSHOT_MIN_FRONT_CLEAR_M)
        and float(body_clearance) >= float(EXPLORATION_FRONTIER_ONLY_SNAPSHOT_MIN_BODY_CLEAR_M)
    )
    stall_action = frontier_only_stall_watchdog_update(route_present, route_reason, safe_snapshot, cov)
    if stall_action == "dock":
        coverage_status = f"frontier-only stalled; return dock {frontier_only_last_stall_debug}"
        last_optional_block_reason = "frontier-only stalled -> dock"
        last_frontier_only_gate_debug = f"frontierGate=stallDock {frontier_only_last_stall_debug}"
        last_planner_update_step = -999999
        try:
            start_map_complete_return_to_dock("frontier-only stalled after partial map")
        except Exception:
            try:
                start_return_to_dock("frontier-only stalled after partial map")
            except Exception:
                pass
        return True
    if stall_action == "blacklist":
        coverage_status = f"frontier-only stalled; blacklist target {frontier_only_last_stall_debug}"
        last_optional_block_reason = "frontier-only stalled -> blacklist"
        last_frontier_only_gate_debug = f"frontierGate=stallBlacklist {frontier_only_last_stall_debug}"
        last_planner_update_step = -999999
        abort_route_commit("frontier stalled/blacklisted")
        return True
    if (
        safe_snapshot
        and now - float(frontier_only_last_snapshot_at) >= float(EXPLORATION_FRONTIER_ONLY_SNAPSHOT_COOLDOWN_SEC)
        and start_post_turn_rgbd_snapshot(
            f"frontier-only stationary RGB-D: route={str(route_reason)[:36]} fr={frontiers} cov={cov:.1f}"
        )
    ):
        frontier_only_last_snapshot_at = now
        coverage_status = "frontier-only -> stationary RGB-D snapshot"
        last_optional_block_reason = "frontier-only starts snapshot"
        last_frontier_only_gate_debug = f"frontierGate=snapshot fr={frontiers} cov={cov:.1f}"
        last_planner_update_step = -999999
        return True

    coverage_status = f"frontier-only hold: {str(route_reason)[:52]} fr={frontiers} cov={cov:.1f}"
    last_optional_block_reason = "frontier-only blocks ROW_FORWARD"
    last_frontier_only_gate_debug = f"frontierGate=hold route={int(route_present)} {str(route_reason)[:28]} {frontier_only_last_stall_debug}"
    return True


def start_active_rgbd_scan(reason):
    """Acquire wheel ownership for a bounded RGB-D look-around sweep."""
    global active_scan_target_yaws, active_scan_index, active_scan_dwell_until
    global active_scan_started_at, active_scan_started_x, active_scan_started_y
    global active_scan_reason, last_active_scan_debug, nav_state, nav_action_queue
    global coverage_status, map_freeze_until, active_scan_start_frontier_local, active_scan_start_unknown_local
    now = robot.getTime()
    nav_action_queue = []
    base = pose_theta
    active_scan_target_yaws = [normalize_angle(base + math.radians(float(a))) for a in ACTIVE_SCAN_YAW_OFFSETS_DEG]
    active_scan_index = 0
    active_scan_dwell_until = -999.0
    active_scan_started_at = now
    active_scan_started_x = pose_x
    active_scan_started_y = pose_y
    try:
        _obs, cleanable, _cleaned, _uncleaned, unknown = compute_coverage_masks()
        hyp_shadow = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(unknown, dtype=np.bool_)
        frontier = (unknown & (~hyp_shadow)) & (cv2.dilate(cleanable.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0)
        smx, smy = world_to_map(pose_x, pose_y)
        active_scan_start_frontier_local = local_mask_count(frontier, smx, smy, ACTIVE_SCAN_TRIGGER_RADIUS_M)
        active_scan_start_unknown_local = local_mask_count(unknown, smx, smy, ACTIVE_SCAN_TRIGGER_RADIUS_M)
    except Exception:
        active_scan_start_frontier_local = 0
        active_scan_start_unknown_local = 0
    active_scan_reason = str(reason or "expand-map scan")
    nav_state = NAV_SCAN_AROUND
    coverage_status = f"active RGB-D scan: {active_scan_reason[:54]}"
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
    acquire_control(ControlOwner.SCAN_AROUND, ACTIVE_SCAN_TIMEOUT_SEC, 0.0, coverage_status)
    last_active_scan_debug = f"scan=start n={len(active_scan_target_yaws)} {active_scan_reason[:32]}"


def finish_active_rgbd_scan(reason="done"):
    global nav_state, active_scan_target_yaws, active_scan_index, active_scan_dwell_until
    global active_scan_last_completed_at, active_scan_last_x, active_scan_last_y
    global last_active_scan_debug, coverage_status, last_planner_update_step
    active_scan_last_completed_at = robot.getTime()
    active_scan_last_x = pose_x
    active_scan_last_y = pose_y
    active_scan_target_yaws = []
    active_scan_index = 0
    active_scan_dwell_until = -999.0
    nav_state = NAV_FORWARD
    coverage_status = f"active RGB-D scan finished: {reason}"
    remember_active_scan_area(reason)
    last_active_scan_debug = f"scan=done {reason[:42]}"
    release_control(f"scan finished: {reason}")
    last_planner_update_step = -999999


def active_rgbd_scan_speeds(front, center, upper_front, left, right, body_clearance):
    """Execute the bounded scan state, returning wheel speeds or None."""
    global active_scan_index, active_scan_dwell_until, coverage_status, map_freeze_until, last_active_scan_debug
    if nav_state != NAV_SCAN_AROUND:
        return None
    now = robot.getTime()
    if not active_scan_target_yaws or active_scan_index >= len(active_scan_target_yaws):
        finish_active_rgbd_scan("sequence complete")
        return 0.0, 0.0
    if now - active_scan_started_at > ACTIVE_SCAN_TIMEOUT_SEC:
        finish_active_rgbd_scan("timeout")
        return 0.0, 0.0
    if front < ROW_END_HARD_DISTANCE or body_clearance < BODY_CORRIDOR_HARD_CLEARANCE:
        finish_active_rgbd_scan(f"clearance stop F={front:.2f} body={body_clearance:.2f}")
        return 0.0, 0.0

    target = active_scan_target_yaws[active_scan_index]
    err = normalize_angle(target - pose_theta)
    abs_err = abs(err)
    tol = math.radians(ACTIVE_SCAN_TOLERANCE_DEG)
    if abs_err <= tol:
        if active_scan_dwell_until < 0.0:
            active_scan_dwell_until = now + ACTIVE_SCAN_DWELL_SEC
            # Reached the scan yaw and command zero wheel speed: allow mapping
            # during the dwell instead of carrying the turn-freeze through it.
            map_freeze_until = min(map_freeze_until, now)
        if now < active_scan_dwell_until:
            coverage_status = f"active RGB-D scan dwell {active_scan_index+1}/{len(active_scan_target_yaws)} err={math.degrees(err):.1f}"
            last_active_scan_debug = f"scan=dwell {active_scan_index+1}/{len(active_scan_target_yaws)} yaw={math.degrees(target):.0f}"
            return 0.0, 0.0
        active_scan_index += 1
        active_scan_dwell_until = -999.0
        if active_scan_index >= len(active_scan_target_yaws):
            finish_active_rgbd_scan("sequence complete")
            return 0.0, 0.0
        return 0.0, 0.0

    sign = 1.0 if err > 0.0 else -1.0
    speed = clamp(abs_err * ACTIVE_SCAN_TURN_KP, ACTIVE_SCAN_TURN_MIN_SPEED, ACTIVE_SCAN_TURN_MAX_SPEED)
    coverage_status = f"active RGB-D scan turn {active_scan_index+1}/{len(active_scan_target_yaws)} err={math.degrees(err):.1f}"
    last_active_scan_debug = f"scan=turn {active_scan_index+1}/{len(active_scan_target_yaws)} err={math.degrees(err):.1f}"
    return -sign * speed, sign * speed


def post_turn_rgbd_snapshot_active(now=None):
    if now is None:
        now = robot.getTime()
    return bool(nav_state == NAV_RGBD_SNAPSHOT and now < post_turn_rgbd_snapshot_until)


def start_post_turn_rgbd_snapshot(reason="post-turn map snapshot"):
    """Hold still after a turn and fuse fresh RGB-D frames into the map.

    Mapping while rotating is correctly frozen. The missing step was a bounded
    stationary observation after the heading changed: settle first, then capture.
    """
    global nav_state, post_turn_rgbd_snapshot_until, post_turn_rgbd_snapshot_started_at
    global post_turn_rgbd_snapshot_frames, post_turn_rgbd_snapshot_write_frames
    global post_turn_rgbd_snapshot_reason, post_turn_rgbd_snapshot_pending
    global last_post_turn_snapshot_debug, coverage_status, map_freeze_until
    if (not POST_TURN_RGBD_SNAPSHOT_ENABLED) or known_map_coverage_eval_active():
        post_turn_rgbd_snapshot_pending = False
        return False
    if not exploration_mapping_snapshot_allowed():
        return False
    now = robot.getTime()
    hard_stop_motors()
    nav_state = NAV_RGBD_SNAPSHOT
    post_turn_rgbd_snapshot_started_at = now
    post_turn_rgbd_snapshot_until = now + float(POST_TURN_RGBD_SNAPSHOT_SEC)
    post_turn_rgbd_snapshot_frames = 0
    post_turn_rgbd_snapshot_write_frames = 0
    post_turn_rgbd_snapshot_reason = str(reason or "post-turn map snapshot")[:70]
    post_turn_rgbd_snapshot_pending = False
    # force-cleared the freeze immediately; that let the first dirty post-turn
    # frame write into the persistent map.
    map_freeze_until = max(map_freeze_until, now + float(POST_TURN_RGBD_SNAPSHOT_SETTLE_SEC))
    coverage_status = f"RGB-D snapshot settle: {post_turn_rgbd_snapshot_reason[:48]}"
    last_post_turn_snapshot_debug = f"snap=settle {post_turn_rgbd_snapshot_reason[:38]}"
    acquire_control(ControlOwner.RGBD_SNAPSHOT, POST_TURN_RGBD_SNAPSHOT_MAX_SEC + 0.15, 0.0, coverage_status)
    return True


def finish_post_turn_rgbd_snapshot(reason="done", front_for_pivot=None, body_clearance_for_pivot=None):
    global nav_state, post_turn_rgbd_snapshot_until, post_turn_rgbd_snapshot_frames
    global post_turn_rgbd_snapshot_write_frames
    global last_post_turn_snapshot_debug, coverage_status, last_planner_update_step
    hard_stop_motors()
    nav_state = NAV_FORWARD
    post_turn_rgbd_snapshot_until = -999.0
    coverage_status = f"RGB-D snapshot finished: {reason}"
    last_post_turn_snapshot_debug = f"snap=done {reason[:30]} f/w={post_turn_rgbd_snapshot_frames}/{post_turn_rgbd_snapshot_write_frames}"
    release_control(f"post-turn RGB-D snapshot finished: {reason}")
    last_planner_update_step = -999999
    run_next_nav_action(front_for_pivot, body_clearance_for_pivot)


def post_turn_rgbd_snapshot_speeds(front, body_clearance):
    """Return a stop command while a post-turn RGB-D snapshot is active."""
    global last_post_turn_snapshot_debug, coverage_status
    if nav_state != NAV_RGBD_SNAPSHOT:
        return None
    now = robot.getTime()
    elapsed = max(0.0, now - post_turn_rgbd_snapshot_started_at)
    phase = post_turn_rgbd_snapshot_phase(now)
    enough_frames = post_turn_rgbd_snapshot_frames >= int(POST_TURN_RGBD_SNAPSHOT_MIN_FRAMES)
    enough_writes = post_turn_rgbd_snapshot_write_frames >= int(POST_TURN_RGBD_SNAPSHOT_MIN_WRITES)
    timed_out = elapsed >= float(POST_TURN_RGBD_SNAPSHOT_MAX_SEC)
    duration_done = now >= post_turn_rgbd_snapshot_until
    if phase == "settle":
        coverage_status = f"RGB-D snapshot settle {elapsed:.2f}s: {post_turn_rgbd_snapshot_reason[:34]}"
        last_post_turn_snapshot_debug = f"snap=settle {elapsed:.2f}s"
    else:
        coverage_status = (
            f"RGB-D snapshot capture f/w={post_turn_rgbd_snapshot_frames}/{post_turn_rgbd_snapshot_write_frames} "
            f"{elapsed:.2f}s"
        )
        if not last_post_turn_snapshot_debug.startswith("snap=capture"):
            last_post_turn_snapshot_debug = f"snap=capture f/w={post_turn_rgbd_snapshot_frames}/{post_turn_rgbd_snapshot_write_frames}"
    if timed_out or (duration_done and enough_frames and enough_writes):
        finish_post_turn_rgbd_snapshot("complete" if not timed_out else "timeout", front, body_clearance)
        return 0.0, 0.0
    return 0.0, 0.0


def integral_rect_count(integral_img, cx, cy, radius_px):
    """Fast rectangular count around a map cell using a cv2.integral image."""
    try:
        r = max(0, int(radius_px))
        x0 = max(0, int(cx) - r)
        y0 = max(0, int(cy) - r)
        x1 = min(MAP_SIZE, int(cx) + r + 1)
        y1 = min(MAP_SIZE, int(cy) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return 0
        return int(integral_img[y1, x1] - integral_img[y0, x1] - integral_img[y1, x0] + integral_img[y0, x0])
    except Exception:
        return 0


def uncleaned_exploration_backtrack_decision(route_cost_m, straight_dist, longitudinal_m, lateral, turn_need,
                                             comp_cells, fp_gain_cells, coverage_segment_gain, missed_strip_bonus,
                                             wall_strip_bonus):
    """Return (reject, penalty, reason) for cleanup targets during map expansion."""
    if not exploration_priority_active():
        return False, 0.0, "clean-known"
    try:
        comp_cells = int(comp_cells or 0)
        fp_gain_cells = int(fp_gain_cells or 0)
        coverage_segment_gain = int(coverage_segment_gain or 0)
        missed_strip_bonus = float(missed_strip_bonus or 0.0)
        wall_strip_bonus = float(wall_strip_bonus or 0.0)
        route_cost_m = float(route_cost_m)
        straight_dist = float(straight_dist)
        longitudinal_m = float(longitudinal_m)
        lateral = float(lateral)
        turn_need = float(turn_need)
    except Exception:
        return False, 0.0, "bad-values"

    local_inline = bool(
        straight_dist <= EXPLORE_UNCLEANED_LOCAL_MAX_DIST_M
        and route_cost_m <= EXPLORE_UNCLEANED_LOCAL_MAX_ROUTE_COST_M
        and longitudinal_m >= EXPLORE_UNCLEANED_BACKTRACK_FORWARD_M
        and lateral <= EXPLORE_UNCLEANED_LOCAL_LATERAL_M
    )
    important = bool(
        comp_cells >= EXPLORE_UNCLEANED_IMPORTANT_COMP_CELLS
        or fp_gain_cells >= EXPLORE_UNCLEANED_IMPORTANT_FOOTPRINT_GAIN
        or coverage_segment_gain >= EXPLORE_UNCLEANED_IMPORTANT_SEGMENT_GAIN
        or missed_strip_bonus >= EXPLORE_UNCLEANED_IMPORTANT_MISSED_BONUS
        or (wall_strip_bonus >= ROUTE_COMMIT_WALL_STRIP_BONUS * 0.75 and straight_dist <= 1.25)
    )
    if local_inline or important:
        return False, 0.0, "allowed-local" if local_inline else "allowed-important"

    backtrack = bool(
        longitudinal_m < EXPLORE_UNCLEANED_BACKTRACK_FORWARD_M
        or turn_need > EXPLORE_UNCLEANED_BACKTRACK_TURN_FRAC
        or route_cost_m > EXPLORE_UNCLEANED_BACKTRACK_MIN_COST_M
    )
    if backtrack:
        penalty = EXPLORE_UNCLEANED_BACKTRACK_PENALTY
        penalty += max(0.0, -longitudinal_m) * 5.0
        penalty += max(0.0, route_cost_m - EXPLORE_UNCLEANED_BACKTRACK_MIN_COST_M) * 3.0
        penalty += max(0.0, turn_need - EXPLORE_UNCLEANED_BACKTRACK_TURN_FRAC) * 8.0
        hard_reject = bool(route_cost_m > 0.95 and longitudinal_m < -0.15 and comp_cells < EXPLORE_UNCLEANED_IMPORTANT_COMP_CELLS)
        return hard_reject, penalty, f"backtrack p={penalty:.1f}"
    if lateral > EXPLORE_UNCLEANED_LOCAL_LATERAL_M:
        return False, EXPLORE_UNCLEANED_SIDE_PENALTY, f"side p={EXPLORE_UNCLEANED_SIDE_PENALTY:.1f}"
    return False, 0.0, "neutral"


def cleanup_target_allowed_under_intent(route_cost_m, straight_dist, longitudinal_m, lateral, turn_need,
                                        comp_cells=0, fp_gain_cells=0, coverage_segment_gain=0,
                                        missed_strip_bonus=0.0, wall_strip_bonus=0.0):
    """Gate known-cleanup targets while the map-expansion intent is active.

    Returns (allowed, penalty, reason).  During EXPAND_MAP, cleanup is only
    allowed when it is local/inline or objectively important; backtracking to a
    small residual island is blocked before it can become a target/commit.
    """
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return True, 0.0, "intent-clean-known"
    if exploration_cleanup_locked():
        return False, 99.0, "cleanup-locked-until-map-closed"
    reject, penalty, reason = uncleaned_exploration_backtrack_decision(
        route_cost_m, straight_dist, longitudinal_m, lateral, turn_need,
        comp_cells, fp_gain_cells, coverage_segment_gain, missed_strip_bonus, wall_strip_bonus,
    )
    if reject:
        return False, penalty, reason
    return True, penalty, reason


def route_goal_relative_to_pose(goal_world):
    """Return straight distance, longitudinal, lateral_abs, turn_frac to a world target."""
    try:
        wx, wy = goal_world
        dx = float(wx) - float(pose_x)
        dy = float(wy) - float(pose_y)
        straight = math.hypot(dx, dy)
        longitudinal = math.cos(pose_theta) * dx + math.sin(pose_theta) * dy
        lateral = abs(-math.sin(pose_theta) * dx + math.cos(pose_theta) * dy)
        turn_need = abs(normalize_angle(math.atan2(dy, dx) - pose_theta)) / math.pi if straight > 1e-6 else 0.0
        return straight, longitudinal, lateral, turn_need
    except Exception:
        return float("inf"), -float("inf"), float("inf"), float("inf")

def uncleaned_route_target_allowed_for_phase(cx, cy, longitudinal, lateral, dist, local_cells):
    if exploration_cleanup_locked():
        return False
    if residual_cleanup_phase():
        return bool(local_cells >= RESIDUAL_TARGET_MIN_LOCAL_CELLS_LATE)
    if longitudinal > RESIDUAL_TARGET_AHEAD_FORWARD_M and abs(lateral) <= RESIDUAL_TARGET_AHEAD_LATERAL_TOL_M:
        return True
    if dist <= RESIDUAL_TARGET_NEAR_DIRECT_M and longitudinal > -0.05 and abs(lateral) <= RESIDUAL_TARGET_AHEAD_LATERAL_TOL_M:
        return True
    # Nearby adjacent strip: this is not a random residual island. If there are
    # enough uncleaned cells in the local cluster and the target is close enough
    # to be acquired by gentle steering, keep it selectable before the late
    # cleanup phase.
    if (
        dist <= RESIDUAL_TARGET_ADJACENT_LINE_MAX_DIST_M
        and longitudinal >= RESIDUAL_TARGET_ADJACENT_LINE_MIN_FORWARD_M
        and abs(lateral) <= RESIDUAL_TARGET_ADJACENT_LINE_LATERAL_TOL_M
        and local_cells >= RESIDUAL_TARGET_ADJACENT_LINE_MIN_CELLS
    ):
        return True
    # leftovers through already-cleaned corridors.
    if dist <= RESIDUAL_TARGET_MAX_MID_PHASE_ROUTE_M and local_cells >= RESIDUAL_TARGET_MIN_LOCAL_CELLS_EARLY:
        return True
    return False


def _now_seconds_safe():
    try:
        return float(robot.getTime())
    except Exception:
        return 0.0


def _dock_distance_to_map_cell(mx, my):
    """Distance from a map cell to the odometry-origin dock/start pose."""
    try:
        dwx, dwy = DOCK_WORLD_X, DOCK_WORLD_Y
    except Exception:
        dwx, dwy = 0.0, 0.0
    wx = (int(mx) - MAP_ORIGIN_X) / MAP_SCALE
    wy = (MAP_ORIGIN_Y - int(my)) / MAP_SCALE
    return math.hypot(wx - float(dwx), wy - float(dwy))


def route_target_blacklist_prune(now=None):
    """Remove expired failed-target suppressions."""
    global route_commit_target_blacklist
    if now is None:
        now = _now_seconds_safe()
    if not ROUTE_COMMIT_TARGET_BLACKLIST_ENABLED:
        route_commit_target_blacklist = []
        return
    keep = []
    for item in route_commit_target_blacklist:
        try:
            if float(item.get("until", 0.0)) > now:
                keep.append(item)
        except Exception:
            continue
    if len(keep) > ROUTE_COMMIT_TARGET_BLACKLIST_MAX_ENTRIES:
        keep = keep[-ROUTE_COMMIT_TARGET_BLACKLIST_MAX_ENTRIES:]
    route_commit_target_blacklist = keep


def route_target_is_blacklisted(mx, my, kind="uncleaned"):
    """True if a recently failed active target should remain candidate-only."""
    global last_route_target_blacklist_debug
    if not ROUTE_COMMIT_TARGET_BLACKLIST_ENABLED:
        last_route_target_blacklist_debug = "disabled"
        return False
    now = _now_seconds_safe()
    route_target_blacklist_prune(now)
    for item in route_commit_target_blacklist:
        try:
            item_kind = str(item.get("kind", kind))
            # "uncleaned" blacklist is historical and should not suppress frontier
            # revisit targets.  "any" suppresses both.
            if item_kind not in (str(kind), "any"):
                continue
            rad_m = float(item.get("radius_m", ROUTE_COMMIT_TARGET_BLACKLIST_RADIUS_M))
            rad_px = max(2.0, rad_m * MAP_SCALE)
            dx = float(mx) - float(item.get("mx", -99999.0))
            dy = float(my) - float(item.get("my", -99999.0))
            if math.hypot(dx, dy) <= rad_px:
                last_route_target_blacklist_debug = f"hit {kind}@({int(mx)},{int(my)}) {item.get('reason','')}"
                return True
        except Exception:
            continue
    last_route_target_blacklist_debug = f"clear n={len(route_commit_target_blacklist)}"
    return False


def register_map_target_blacklist(mx, my, kind="any", reason="failed route", ttl_sec=None, radius_m=None):
    """Suppress an arbitrary map target after a failed viewpoint/route attempt."""
    global route_commit_target_blacklist, last_route_target_blacklist_debug
    if not ROUTE_COMMIT_TARGET_BLACKLIST_ENABLED:
        return False
    try:
        item = {
            "mx": int(mx),
            "my": int(my),
            "kind": str(kind or "any"),
            "until": _now_seconds_safe() + float(ttl_sec if ttl_sec is not None else ROUTE_COMMIT_TARGET_BLACKLIST_SEC),
            "reason": str(reason or "failed route")[:40],
        }
        if radius_m is not None:
            item["radius_m"] = float(radius_m)
        route_target_blacklist_prune()
        route_commit_target_blacklist.append(item)
        if len(route_commit_target_blacklist) > ROUTE_COMMIT_TARGET_BLACKLIST_MAX_ENTRIES:
            route_commit_target_blacklist = route_commit_target_blacklist[-ROUTE_COMMIT_TARGET_BLACKLIST_MAX_ENTRIES:]
        last_route_target_blacklist_debug = f"add {item['kind']}@({item['mx']},{item['my']}) {item['reason']}"
        return True
    except Exception as exc:
        last_route_target_blacklist_debug = f"add map failed {type(exc).__name__}"
        return False


def register_route_target_blacklist(reason="failed route"):
    """Suppress the current active target after a physical/path failure.

    This is not marking the map as cleaned. It only prevents the same tiny
    residual island from immediately becoming activeTarget again; ordinary row
    coverage can still clean it later if the robot reaches it naturally.
    """
    global route_commit_target_blacklist, last_route_target_blacklist_debug
    if not ROUTE_COMMIT_TARGET_BLACKLIST_ENABLED:
        return False
    if route_commit_target_map is None or route_commit_kind in ("dock", "none"):
        return False
    r = str(reason or "failed route")
    rl = r.lower()
    if not any(tok in rl for tok in ROUTE_COMMIT_BLACKLIST_ON_REASONS):
        return False
    try:
        mx, my = route_commit_target_map
        return register_map_target_blacklist(mx, my, str(route_commit_kind or "any"), r, ROUTE_COMMIT_TARGET_BLACKLIST_SEC)
    except Exception as exc:
        last_route_target_blacklist_debug = f"add failed {type(exc).__name__}"
        return False


def route_target_is_edge_residual_trap(mx, my, comp_cells=0):
    """Reject small early edge/dock residuals as ROUTE_COMMIT targets.

    A visible wall strip is important, but a small component pinned to the map
    border or the start/dock pocket during PARTIAL coverage is usually a bad
    active route target: the robot can keep re-entering recovery and repainting
    the same little square. Large components are still allowed.
    """
    if not ROUTE_COMMIT_EDGE_RESIDUAL_GUARD_ENABLED or map_mature:
        return False
    if last_coverage_percent >= ROUTE_COMMIT_EDGE_RESIDUAL_COVERAGE_PERCENT:
        return False
    try:
        mx = int(mx); my = int(my); comp_cells = int(comp_cells)
        edge_px = max(2, int(ROUTE_COMMIT_EDGE_RESIDUAL_BAND_M * MAP_SCALE))
        near_edge = mx <= edge_px or my <= edge_px or mx >= MAP_SIZE - 1 - edge_px or my >= MAP_SIZE - 1 - edge_px
        if near_edge and comp_cells < ROUTE_COMMIT_EDGE_RESIDUAL_MIN_COMP_CELLS:
            return True
        near_dock = _dock_distance_to_map_cell(mx, my) <= ROUTE_COMMIT_DOCK_RESIDUAL_GUARD_RADIUS_M
        if near_dock and comp_cells < ROUTE_COMMIT_DOCK_RESIDUAL_MIN_COMP_CELLS:
            return True
    except Exception:
        return False
    return False


def route_target_is_dock_loiter_trap(mx, my, kind="uncleaned", comp_cells=0):
    """True when a candidate is a tiny dock/start-pocket residual.

    The dock marker and the wall beside it create a pocket where the map often
    has a few uncleaned/frontier pixels even after the reachable floor around it
    was already traversed.  Letting those pixels compete as normal global
    targets creates the square loops visible near the dock.  This only blocks
    coverage candidates; return-to-dock plans still go directly to DOCK_TARGET.
    """
    global last_route_dock_loiter_guard_debug
    if not DOCK_LOITER_GUARD_ENABLED or dock_return_active or route_commit_kind == "dock":
        return False
    try:
        if last_coverage_percent < DOCK_LOITER_GUARD_COVERAGE_PERCENT:
            return False
        mx = int(mx); my = int(my); comp_cells = int(comp_cells or 0)
        dist = _dock_distance_to_map_cell(mx, my)
        if dist > DOCK_LOITER_GUARD_RADIUS_M:
            return False
        if str(kind) == "frontier" and DOCK_LOITER_GUARD_FRONTIER_ALWAYS:
            last_route_dock_loiter_guard_debug = f"dock-pocket frontier ({mx},{my}) d={dist:.2f}"
            return True
        if comp_cells < DOCK_LOITER_GUARD_MIN_COMP_CELLS:
            last_route_dock_loiter_guard_debug = f"dock-pocket {kind} ({mx},{my}) comp={comp_cells} d={dist:.2f}"
            return True
    except Exception as exc:
        last_route_dock_loiter_guard_debug = f"dock guard err {type(exc).__name__}"
        return False
    return False


def clear_coverage_target_sticky(reason="clear"):
    global coverage_target_sticky_map, coverage_target_sticky_kind, coverage_target_sticky_component_id
    global coverage_target_sticky_score, coverage_target_sticky_until, coverage_target_sticky_debug
    coverage_target_sticky_map = None
    coverage_target_sticky_kind = "none"
    coverage_target_sticky_component_id = -1
    coverage_target_sticky_score = float("-inf")
    coverage_target_sticky_until = -999.0
    coverage_target_sticky_debug = str(reason or "clear")[:80]


def coverage_target_record_matches_sticky(record):
    """Return True if a scored candidate belongs to the current sticky target."""
    if not COVERAGE_TARGET_STICKY_ENABLED or coverage_target_sticky_map is None:
        return False
    try:
        now = robot.getTime()
    except Exception:
        now = 0.0
    if now > coverage_target_sticky_until:
        return False
    try:
        _score, best_tuple, _kind_name = record
        _gx, _gy, cx, cy, _wx, _wy, kind_code, _route_cost_m, _score2, _recent_penalty, comp_id, comp_cells, *_rest = best_tuple
        kind_name = {1: "uncleaned", 2: "frontier", 3: "under-surface"}.get(int(kind_code), "uncleaned")
        if kind_name != coverage_target_sticky_kind:
            return False
        smx, smy = coverage_target_sticky_map
        # Component id is the best match for coherent uncleaned islands.  Fall
        # back to a metric radius for frontier/under-surface or relabelled cells.
        if int(comp_id) > 0 and int(comp_id) == int(coverage_target_sticky_component_id):
            return True
        radius_px = int(COVERAGE_TARGET_STICKY_RADIUS_M * MAP_SCALE)
        return (int(cx) - int(smx)) ** 2 + (int(cy) - int(smy)) ** 2 <= radius_px ** 2
    except Exception:
        return False


def choose_sticky_coverage_record(candidate_records):
    """Choose a target with hysteresis so candidateTarget does not flicker.

    Dijkstra/wavefront still scores every candidate each cycle.  Hysteresis only
    prevents a near-tie from replacing the current target while the robot is
    executing pivot/lane-shift/recovery around the same component.
    """
    global coverage_target_sticky_map, coverage_target_sticky_kind, coverage_target_sticky_component_id
    global coverage_target_sticky_score, coverage_target_sticky_until, coverage_target_sticky_debug
    if not candidate_records:
        clear_coverage_target_sticky("none")
        return None
    candidate_records.sort(key=lambda item: item[0], reverse=True)
    best_record = candidate_records[0]
    best_score = float(best_record[0])
    chosen = best_record
    sticky_used = False
    try:
        best_kind_name = str(best_record[2])
    except Exception:
        best_kind_name = "none"
    if COVERAGE_TARGET_STICKY_ENABLED and coverage_target_sticky_map is not None:
        sticky_record = None
        for rec in candidate_records:
            if coverage_target_record_matches_sticky(rec):
                sticky_record = rec
                break
        if sticky_record is not None:
            sticky_score = float(sticky_record[0])
            best_tuple = best_record[1]
            sticky_tuple = sticky_record[1]
            try:
                best_comp_cells = int(best_tuple[11])
                best_missed_bonus = float(best_tuple[18]) if len(best_tuple) > 18 else 0.0
                sticky_missed_bonus = float(sticky_tuple[18]) if len(sticky_tuple) > 18 else 0.0
            except Exception:
                best_comp_cells = 0
                best_missed_bonus = 0.0
                sticky_missed_bonus = 0.0
            # Only switch away if the new best is clearly better.  Exception:
            # a newly detected missed wall strip/large component may break sticky
            # even with a small score advantage, otherwise visible wall bands stay
            # ignored for several seconds while a local residual remains sticky.
            missed_strip_break = (
                best_missed_bonus >= COVERAGE_TARGET_STICKY_BREAK_MISSED_STRIP_BONUS
                and best_comp_cells >= COVERAGE_TARGET_STICKY_MIN_COMP_CELLS
                and best_missed_bonus > sticky_missed_bonus + 0.5
                and best_score >= sticky_score - COVERAGE_TARGET_STICKY_BREAK_MARGIN
            )
            frontier_explore_break = (
                exploration_priority_active()
                and best_kind_name == "frontier"
                and coverage_target_sticky_kind != "frontier"
                and best_score >= sticky_score - EXPLORE_FRONTIER_STICKY_BREAK_MARGIN
            )
            if (not missed_strip_break) and (not frontier_explore_break) and sticky_score >= best_score - COVERAGE_TARGET_STICKY_SCORE_MARGIN:
                chosen = sticky_record
                sticky_used = True
    score, best_tuple, kind_name = chosen
    _gx, _gy, cx, cy, _wx, _wy, _kind_code, _route_cost_m, _score2, _recent_penalty, comp_id, comp_cells, *_rest = best_tuple
    try:
        now = robot.getTime()
    except Exception:
        now = 0.0
    coverage_target_sticky_map = (int(cx), int(cy))
    coverage_target_sticky_kind = str(kind_name)
    coverage_target_sticky_component_id = int(comp_id)
    coverage_target_sticky_score = float(score)
    coverage_target_sticky_until = now + COVERAGE_TARGET_STICKY_SEC
    if sticky_used:
        coverage_target_sticky_debug = f"hold {kind_name}@({int(cx)},{int(cy)}) sc={float(score):.1f}/{best_score:.1f}"
    else:
        coverage_target_sticky_debug = f"select {kind_name}@({int(cx)},{int(cy)}) sc={float(score):.1f}"
    return chosen


def plan_best_coverage_route(obstacles, cleanable, cleaned, uncleaned, unknown, frontier, under_surface_uncleaned):
    """Choose a reachable coverage target using a coarse wavefront route search.

    This replaces the previous straight-line target choice. A cell can be close in
    Euclidean distance but effectively bad if a table leg/wall/contact patch sits
    between the robot and that cell. Wavefront search makes the target selection
    path-aware while remaining light enough for the Webots controller.
    """
    global last_route_planner_ms, last_route_planner_nodes, last_recent_target_penalty, last_residual_route_deferred_cells
    global coverage_route_top_debug, last_route_target_blacklist_debug, last_route_dock_loiter_guard_debug
    global last_missed_strip_recovery_debug, last_rejected_components_debug, last_missed_strip_promoted_cells
    global last_coverage_footprint_debug, last_coverage_footprint_gain_cells, last_coverage_footprint_reclean_ratio
    global last_coverage_segment_debug
    global last_explore_arbitration_debug, last_exploration_route_debug, last_frontier_target_debug, last_known_backtrack_debug, last_explore_gain_debug, last_commit_type_debug, planner_mode
    planner_t0 = time.perf_counter()
    route_target_blacklist_prune()
    last_route_dock_loiter_guard_debug = "clear"
    last_missed_strip_recovery_debug = "none"
    last_rejected_components_debug = "none"
    last_missed_strip_promoted_cells = 0
    last_coverage_footprint_debug = "none"
    last_coverage_footprint_gain_cells = 0
    last_coverage_footprint_reclean_ratio = 0.0
    last_coverage_segment_debug = "none"
    update_planner_intent()
    explore_priority = exploration_priority_active()
    cleanup_locked = exploration_cleanup_locked()
    exploration_route_only = bool(EXPLORATION_FRONTIER_ROUTING_ENABLED and planner_intent == PLANNER_INTENT_EXPAND_MAP and cleanup_locked)
    update_planner_mode_label()
    last_explore_arbitration_debug = (
        f"intent={planner_intent} fr={last_frontier_cells} cov={last_coverage_percent:.1f} "
        f"{planner_intent_reason[:24]} {last_exploration_cleanup_lock_debug[:32]}"
    )
    last_exploration_route_debug = (
        "exploreRoute=frontier-only" if exploration_route_only else "exploreRoute=coverage-enabled"
    )
    last_frontier_target_debug = "frontierTarget=none"
    last_known_backtrack_debug = "knownBacktrackPenalty=0"
    last_explore_gain_debug = "exploreGain=0 coverageGain=0"
    last_commit_type_debug = "commitType=none"
    last_recent_target_penalty = 0.0
    last_residual_route_deferred_cells = 0
    last_route_planner_nodes = 0
    coverage_route_top_debug = "none"
    if not GLOBAL_ROUTE_PLANNER_ENABLED:
        last_route_planner_ms = 0.0
        return None

    robot_mx, robot_my = world_to_map(pose_x, pose_y)
    # uses center_no_go so robot centre does not graze furniture safety margins.
    actual_obstacles = obstacles.astype(np.bool_)
    route_obstacles = planning_no_go_obstacle_mask(actual_obstacles)
    search_r = int(PLANNER_TARGET_SEARCH_RADIUS_M * MAP_SCALE)
    step = int(GLOBAL_ROUTE_GRID_STEP_PX)
    if known_map_coverage_eval_active() and not exploration_route_only:
        # few more grid nodes to align strips with reachable wall lanes.  Using
        # the live 14 px graph could miss the first drivable row next to the top
        # wall and made the route appear to skip the upper strip.
        try:
            step = max(4, int(KNOWN_MAP_EVAL_ROUTE_GRID_STEP_PX))
        except Exception:
            step = int(GLOBAL_ROUTE_GRID_STEP_PX)
    if (
        known_map_coverage_eval_active()
        and KNOWN_MAP_EVAL_GLOBAL_ROUTE_WINDOW
        and not exploration_route_only
    ):
        # Known-map coverage is not a local exploration problem.  Build the
        # coarse graph over the known arena, but include the robot pose as well
        # because the spawn/odometry frame may be slightly outside the rectangular
        # arena bounds used to seed the evaluation map.
        try:
            ax0, ax1, ay0, ay1 = known_map_eval_arena_bounds_px()
            pad = int(max(float(KNOWN_MAP_EVAL_GLOBAL_ROUTE_PADDING_M) * MAP_SCALE, step * 3))
            x0 = max(0, min(int(ax0), int(robot_mx)) - pad)
            x1 = min(MAP_SIZE, max(int(ax1), int(robot_mx)) + pad)
            y0 = max(0, min(int(ay0), int(robot_my)) - pad)
            y1 = min(MAP_SIZE, max(int(ay1), int(robot_my)) + pad)
        except Exception:
            x0 = max(0, robot_mx - search_r)
            x1 = min(MAP_SIZE, robot_mx + search_r)
            y0 = max(0, robot_my - search_r)
            y1 = min(MAP_SIZE, robot_my + search_r)
    else:
        x0 = max(0, robot_mx - search_r)
        x1 = min(MAP_SIZE, robot_mx + search_r)
        y0 = max(0, robot_my - search_r)
        y1 = min(MAP_SIZE, robot_my + search_r)
    if x1 - x0 < step * 4 or y1 - y0 < step * 4:
        return None

    gw = int(math.ceil((x1 - x0) / step))
    gh = int(math.ceil((y1 - y0) / step))
    if gw * gh > GLOBAL_ROUTE_MAX_NODES:
        # Keep runtime predictable in the optimized Webots build.
        factor = math.sqrt((gw * gh) / GLOBAL_ROUTE_MAX_NODES)
        step = int(math.ceil(step * factor))
        gw = int(math.ceil((x1 - x0) / step))
        gh = int(math.ceil((y1 - y0) / step))

    # Full-body passability replaces the old centreline-only obstacle inflation.
    # A coarse grid cell is passable only if it contains enough locations where
    # the 1:1 circular robot footprint can stand without overlapping obstacles.
    route_margin_m = float(FOOTPRINT_ROUTE_MARGIN_M)
    if known_map_coverage_eval_active() and not exploration_route_only:
        # are almost tangent to furniture.  The old zero-margin graph maximized
        # coverage, but it produced visually redundant/unsafe passes around every
        # obstacle.  Do not change the online exploration margin; only the known
        # full-sweep graph gets this extra clearance.
        try:
            route_margin_m = max(route_margin_m, float(KNOWN_MAP_EVAL_ROUTE_MARGIN_M))
        except Exception:
            route_margin_m = max(route_margin_m, 0.030)
    footprint_passable_full, footprint_clearance_m = build_footprint_passability_map(route_obstacles, cleanable, route_margin_m)
    # A coarse cell may contain uncleaned pixels along a wall/leg that are floor,
    # but no valid robot-centre pose can bring the cleaning disk over them.  Do
    # not let those pixels become global targets.  They are handled by edge trim
    # once the adjacent reachable lane has been cleaned.
    reachable_uncleaned = uncleaned & cleanable_reachable_by_body_center(footprint_passable_full, cleanable, actual_obstacles)
    footprint_gain_count, footprint_gain_ratio, footprint_reclean_ratio, footprint_known_ratio = build_cleaning_footprint_value_maps(
        cleanable, cleaned, reachable_uncleaned, footprint_passable_full if COVERAGE_FOOTPRINT_TARGETING_ENABLED else None
    )

    frontier_radius_px = max(1, int(EXPLORE_FRONTIER_VANTAGE_RADIUS_M * MAP_SCALE))
    frontier_integral = cv2.integral(frontier.astype(np.uint8))
    unknown_integral = cv2.integral(unknown.astype(np.uint8))
    frontier_component_labels = np.zeros_like(frontier, dtype=np.int32)
    frontier_component_sizes = np.zeros(1, dtype=np.int32)
    try:
        if FRONTIER_COMPONENT_TARGETING_ENABLED and np.any(frontier):
            _nf, frontier_component_labels, frontier_component_stats, _frontier_centroids = cv2.connectedComponentsWithStats(
                frontier.astype(np.uint8), 8
            )
            frontier_component_sizes = frontier_component_stats[:, cv2.CC_STAT_AREA].astype(np.int32)
    except Exception:
        frontier_component_labels = np.zeros_like(frontier, dtype=np.int32)
        frontier_component_sizes = np.zeros(1, dtype=np.int32)

    passable = np.zeros((gh, gw), dtype=np.bool_)
    cleaned_ratio = np.zeros((gh, gw), dtype=np.float32)
    recent_ratio = np.zeros((gh, gw), dtype=np.float32)
    target_kind = np.zeros((gh, gw), dtype=np.int8)  # 0 none, 1 uncleaned, 2 frontier, 3 under-surface
    target_reward = np.zeros((gh, gw), dtype=np.float32)
    # this to recover long wall strips that are visually obvious but were not
    # promoted to target_kind because the early/partial phase was too cautious.
    raw_uncleaned_ratio = np.zeros((gh, gw), dtype=np.float32)
    raw_uncleaned_cells = np.zeros((gh, gw), dtype=np.int32)
    coarse_footprint_gain_cells = np.zeros((gh, gw), dtype=np.int32)
    coarse_footprint_gain_ratio = np.zeros((gh, gw), dtype=np.float32)
    coarse_footprint_reclean_ratio = np.zeros((gh, gw), dtype=np.float32)
    coarse_frontier_local_cells = np.zeros((gh, gw), dtype=np.int32)
    coarse_frontier_component_cells = np.zeros((gh, gw), dtype=np.int32)
    coarse_frontier_unknown_cells = np.zeros((gh, gw), dtype=np.int32)

    for gy in range(gh):
        py0 = y0 + gy * step
        py1 = min(y1, py0 + step)
        if py1 <= py0:
            continue
        for gx in range(gw):
            px0 = x0 + gx * step
            px1 = min(x1, px0 + step)
            if px1 <= px0:
                continue
            area = max(1, (py1 - py0) * (px1 - px0))
            obs_ratio = float(np.count_nonzero(route_obstacles[py0:py1, px0:px1])) / area
            cleanable_ratio = float(np.count_nonzero(cleanable[py0:py1, px0:px1])) / area
            footprint_ratio = float(np.count_nonzero(footprint_passable_full[py0:py1, px0:px1])) / area
            cx, cy = coarse_route_center(x0, y0, step, gx, gy)
            center_footprint_ok = bool(map_inside(cx, cy) and footprint_passable_full[cy, cx])
            if obs_ratio > GLOBAL_ROUTE_OBSTACLE_RATIO_BLOCK or cleanable_ratio < GLOBAL_ROUTE_MIN_CLEANABLE_RATIO:
                continue
            if GLOBAL_ROUTE_REQUIRE_CENTER_FOOTPRINT:
                if not center_footprint_ok:
                    continue
            elif footprint_ratio < 0.06:
                continue
            passable[gy, gx] = True
            cleaned_ratio[gy, gx] = float(np.count_nonzero(cleaned[py0:py1, px0:px1])) / area
            if RECENT_VISIT_ROUTE_MEMORY_ENABLED:
                recent_ratio[gy, gx] = min(1.0, float(np.mean(recent_visit_log_odds[py0:py1, px0:px1])) / max(1e-6, RECENT_VISIT_MAX))
            un_cells_here = int(np.count_nonzero(reachable_uncleaned[py0:py1, px0:px1]))
            un_ratio = float(un_cells_here) / area
            raw_uncleaned_ratio[gy, gx] = float(un_ratio)
            raw_uncleaned_cells[gy, gx] = int(un_cells_here)
            under_ratio = float(np.count_nonzero(under_surface_uncleaned[py0:py1, px0:px1])) / area
            front_cells_here = int(np.count_nonzero(frontier[py0:py1, px0:px1]))
            front_ratio = float(front_cells_here) / area
            front_local_cells = integral_rect_count(frontier_integral, cx, cy, frontier_radius_px)
            frontier_component_cells = 0
            if FRONTIER_COMPONENT_TARGETING_ENABLED and front_local_cells > 0:
                fx0 = max(0, int(cx) - frontier_radius_px)
                fx1 = min(MAP_SIZE, int(cx) + frontier_radius_px + 1)
                fy0 = max(0, int(cy) - frontier_radius_px)
                fy1 = min(MAP_SIZE, int(cy) + frontier_radius_px + 1)
                if fx1 > fx0 and fy1 > fy0:
                    labels_crop = frontier_component_labels[fy0:fy1, fx0:fx1]
                    ids = np.unique(labels_crop[labels_crop > 0])
                    if ids.size > 0:
                        frontier_component_cells = int(np.max(frontier_component_sizes[ids]))
            unknown_local_cells = integral_rect_count(unknown_integral, cx, cy, frontier_radius_px)
            coarse_frontier_local_cells[gy, gx] = int(front_local_cells)
            coarse_frontier_component_cells[gy, gx] = int(frontier_component_cells)
            coarse_frontier_unknown_cells[gy, gx] = int(unknown_local_cells)
            fp_gain_cells = int(footprint_gain_count[cy, cx]) if map_inside(cx, cy) else 0
            fp_gain_ratio = float(footprint_gain_ratio[cy, cx]) if map_inside(cx, cy) else 0.0
            fp_reclean_ratio = float(footprint_reclean_ratio[cy, cx]) if map_inside(cx, cy) else 1.0
            coarse_footprint_gain_cells[gy, gx] = int(fp_gain_cells)
            coarse_footprint_gain_ratio[gy, gx] = float(fp_gain_ratio)
            coarse_footprint_reclean_ratio[gy, gx] = float(fp_reclean_ratio)
            wx = (cx - MAP_ORIGIN_X) / MAP_SCALE
            wy = (MAP_ORIGIN_Y - cy) / MAP_SCALE
            dist = math.hypot(wx - pose_x, wy - pose_y)
            if dist < GLOBAL_ROUTE_MIN_TARGET_DIST_M:
                continue
            longitudinal = math.cos(pose_theta) * (wx - pose_x) + math.sin(pose_theta) * (wy - pose_y)
            lateral_to_target = -math.sin(pose_theta) * (wx - pose_x) + math.cos(pose_theta) * (wy - pose_y)
            under_direct = under_surface_target_is_direct_route(cx, cy, longitudinal, lateral_to_target, dist, route_obstacles, cleanable)
            frontier_vantage = bool(
                explore_priority
                and (
                    front_ratio >= EXPLORE_FRONTIER_TARGET_RATIO_MIN
                    or front_local_cells >= EXPLORE_FRONTIER_VANTAGE_MIN_LOCAL_CELLS
                )
            )
            if frontier_vantage:
                target_kind[gy, gx] = 2
                fwd = longitudinal / max(dist, 1e-6)
                local_frontier_bonus = min(EXPLORE_FRONTIER_LOCAL_BONUS_MAX, EXPLORE_FRONTIER_LOCAL_BONUS_SCALE * float(front_local_cells))
                component_frontier_bonus = min(EXPLORE_FRONTIER_COMPONENT_BONUS_MAX, EXPLORE_FRONTIER_COMPONENT_BONUS_SCALE * float(frontier_component_cells))
                open_area_bonus = min(EXPLORE_FRONTIER_OPEN_AREA_BONUS_MAX, EXPLORE_FRONTIER_OPEN_AREA_BONUS_SCALE * float(unknown_local_cells))
                target_reward[gy, gx] = (
                    EXPLORE_FRONTIER_REWARD
                    + EXPLORE_FRONTIER_RATIO_REWARD * max(front_ratio, min(0.60, float(front_local_cells) / max(1.0, float(area))))
                    + local_frontier_bonus
                    + component_frontier_bonus
                    + open_area_bonus
                    + 2.0 * max(0.0, fwd)
                )
            elif (not cleanup_locked) and under_ratio >= GLOBAL_ROUTE_TARGET_UNDER_RATIO and dist <= GLOBAL_ROUTE_UNDER_MAX_DIST_M and under_direct:
                # Keep under-furniture cleanup opportunistic. It may win only when
                # the robot is already near and lined up with the opening. Without
                # this gate, yellow cells become a magnet and create diagonal routes
                # into furniture instead of continuing a sane coverage strip.
                target_kind[gy, gx] = 3
                target_reward[gy, gx] = 8.0 + 10.0 * under_ratio
            elif (not cleanup_locked) and (
                un_ratio >= GLOBAL_ROUTE_TARGET_UNCLEANED_RATIO
                or (
                    COVERAGE_FOOTPRINT_TARGETING_ENABLED
                    and fp_gain_cells >= COVERAGE_FOOTPRINT_MIN_GAIN_CELLS
                    and fp_gain_ratio >= COVERAGE_FOOTPRINT_MIN_GAIN_RATIO
                )
            ):
                local_un_cells = max(
                    local_mask_count(reachable_uncleaned, cx, cy, RESIDUAL_TARGET_LOCAL_RADIUS_M),
                    int(fp_gain_cells),
                )
                if uncleaned_route_target_allowed_for_phase(cx, cy, longitudinal, lateral_to_target, dist, local_un_cells):
                    target_kind[gy, gx] = 1
                    # Large coherent patches beat isolated edge leftovers.  The
                    # footprint gain term means the centre may target a cleaned
                    # lane when the real cleaning disk will cover an adjacent
                    # uncleaned wall strip.
                    local_bonus = min(4.0, local_un_cells / max(1.0, RESIDUAL_TARGET_MIN_LOCAL_CELLS_EARLY))
                    footprint_bonus = min(COVERAGE_FOOTPRINT_TARGET_GAIN_MAX, COVERAGE_FOOTPRINT_TARGET_GAIN_SCALE * float(fp_gain_cells))
                    target_reward[gy, gx] = 14.0 + 12.0 * max(un_ratio, fp_gain_ratio) + local_bonus + footprint_bonus
                else:
                    last_residual_route_deferred_cells += max(1, int(fp_gain_cells), int(np.count_nonzero(reachable_uncleaned[py0:py1, px0:px1])))
            elif front_ratio >= GLOBAL_ROUTE_TARGET_FRONTIER_RATIO or front_local_cells >= EXPLORE_FRONTIER_VANTAGE_MIN_LOCAL_CELLS * 2:
                target_kind[gy, gx] = 2
                local_frontier_bonus = min(EXPLORE_FRONTIER_LOCAL_BONUS_MAX * 0.55, EXPLORE_FRONTIER_LOCAL_BONUS_SCALE * float(front_local_cells))
                component_frontier_bonus = min(EXPLORE_FRONTIER_COMPONENT_BONUS_MAX * 0.45, EXPLORE_FRONTIER_COMPONENT_BONUS_SCALE * float(frontier_component_cells))
                target_reward[gy, gx] = 7.0 + 6.0 * front_ratio + local_frontier_bonus + component_frontier_bonus

    # present near a wall/obstacle boundary, allow it into the component scorer
    # even when normal partial-map gates would have treated it as residual noise.
    # This fixes the visually bad case where a long right/left wall strip remains
    # blue/orange while the robot keeps working a smaller sticky target elsewhere.
    if (not cleanup_locked) and MISSED_STRIP_RECOVERY_ENABLED and last_coverage_percent >= MISSED_STRIP_MIN_COVERAGE_PERCENT:
        try:
            raw_mask = ((raw_uncleaned_ratio >= GLOBAL_ROUTE_TARGET_UNCLEANED_RATIO) | (coarse_footprint_gain_ratio >= COVERAGE_FOOTPRINT_MIN_GAIN_RATIO)) & passable
            n_raw, raw_labels, raw_stats, _raw_centroids = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), 8)
            promoted_debug = []
            min_cells = MISSED_STRIP_MIN_COMPONENT_CELLS_MATURE if map_mature else MISSED_STRIP_MIN_COMPONENT_CELLS_PARTIAL
            for rcid in range(1, int(n_raw)):
                cells = int(raw_stats[rcid, cv2.CC_STAT_AREA])
                if cells < int(min_cells):
                    continue
                left = int(raw_stats[rcid, cv2.CC_STAT_LEFT])
                top = int(raw_stats[rcid, cv2.CC_STAT_TOP])
                width = int(raw_stats[rcid, cv2.CC_STAT_WIDTH])
                height = int(raw_stats[rcid, cv2.CC_STAT_HEIGHT])
                span = max(width, height)
                if span < MISSED_STRIP_MIN_SPAN_CELLS:
                    continue
                comp_mask = raw_labels == rcid
                dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
                wall_contacts = int(np.count_nonzero(dilated & (~passable)))
                strip_like = (wall_contacts >= MISSED_STRIP_MIN_WALL_CONTACTS) or (min(width, height) <= 2 and span >= MISSED_STRIP_MIN_SPAN_CELLS + 1)
                if not strip_like:
                    continue
                promote_mask = comp_mask & (target_kind != 3)
                promoted = int(np.count_nonzero(promote_mask & (target_kind != 1)))
                if promoted <= 0:
                    continue
                target_kind[promote_mask] = 1
                # Keep the original per-cell reward where it was higher, but make
                # the strip visible to the scorer as a valuable coverage target.
                target_reward[promote_mask] = np.maximum(
                    target_reward[promote_mask],
                    np.float32(14.0 + MISSED_STRIP_PROMOTE_REWARD),
                )
                last_missed_strip_promoted_cells += promoted
                cxg = left + width // 2
                cyg = top + height // 2
                cmx, cmy = coarse_route_center(x0, y0, step, cxg, cyg)
                promoted_debug.append(f"strip@({int(cmx)},{int(cmy)}) cells={cells} span={span} wall={wall_contacts} +{promoted}")
            if promoted_debug:
                last_missed_strip_recovery_debug = " | ".join(promoted_debug[:REJECTED_COMPONENT_DEBUG_COUNT])
            else:
                last_missed_strip_recovery_debug = "no strip"
        except Exception as exc:
            last_missed_strip_recovery_debug = f"err {type(exc).__name__}"

    sx = int((robot_mx - x0) // step)
    sy = int((robot_my - y0) // step)
    start = nearest_passable_route_cell(passable, sx, sy, max_r=5)
    if start is None or not route_start_cell_is_locally_reachable(start, x0, y0, step, route_obstacles, cleanable):
        # The robot is probably next to an inflated/contact wall. Returning a route
        # from a snapped cell would lie to the local controller and draw the
        # characteristic right-angle/L-shaped path through the wall side.
        last_route_planner_ms = (time.perf_counter() - planner_t0) * 1000.0
        return None
    sx, sy = start

    inf = 1e18
    dist_grid = np.full((gh, gw), inf, dtype=np.float32)
    parent = np.full((gh, gw), -1, dtype=np.int32)
    dist_grid[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    if GLOBAL_ROUTE_CARDINAL_ONLY:
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    else:
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (1, 1, 1.4142)]
    expanded_nodes = 0
    while heap:
        d, gx, gy = heapq.heappop(heap)
        expanded_nodes += 1
        if expanded_nodes > GLOBAL_ROUTE_MAX_NODES:
            break
        if d > float(dist_grid[gy, gx]) + 1e-6:
            continue
        for dx, dy, base_cost in nbrs:
            nx = gx + dx
            ny = gy + dy
            if nx < 0 or ny < 0 or nx >= gw or ny >= gh or not passable[ny, nx]:
                continue
            # Avoid diagonal corner cutting through blocked cells.
            if dx != 0 and dy != 0 and (not passable[gy, nx] or not passable[ny, gx]):
                continue
            if exploration_route_only:
                # Frontier exploration should find a short/safe vantage path.
                # Do not make the path prefer dirty strips or avoid already
                # cleaned cells: that is planned coverage logic, not mapping.
                route_cost_factor = (
                    1.0
                    + EXPLORATION_ROUTE_CLEANED_TRANSIT_PENALTY * float(cleaned_ratio[ny, nx])
                    + EXPLORATION_ROUTE_RECENT_TRANSIT_PENALTY * float(recent_ratio[ny, nx])
                )
            else:
                route_cost_factor = (
                    1.0
                    + GLOBAL_ROUTE_CLEANED_TRANSIT_PENALTY * float(cleaned_ratio[ny, nx])
                    + GLOBAL_ROUTE_RECENT_TRANSIT_PENALTY * float(recent_ratio[ny, nx])
                )
                if COVERAGE_FOOTPRINT_TARGETING_ENABLED:
                    route_cost_factor += COVERAGE_FOOTPRINT_ROUTE_RECLEAN_PENALTY * float(coarse_footprint_reclean_ratio[ny, nx])
                    route_cost_factor -= COVERAGE_FOOTPRINT_ROUTE_GAIN_DISCOUNT * float(coarse_footprint_gain_ratio[ny, nx])
                    route_cost_factor = max(COVERAGE_FOOTPRINT_ROUTE_MIN_COST_FACTOR, route_cost_factor)
            step_cost = base_cost * route_cost_factor
            nd = d + step_cost
            if nd < float(dist_grid[ny, nx]):
                dist_grid[ny, nx] = nd
                parent[ny, nx] = gy * gw + gx
                heapq.heappush(heap, (nd, nx, ny))

    # target-cell score.  The orange route now represents the actual coverage
    # path that ROUTE_COMMIT will execute.
    if known_map_coverage_eval_active() and KNOWN_MAP_EVAL_STRIP_ROUTE_ENABLED and not exploration_route_only:
        known_strip = known_map_eval_strip_route_from_grid(
            x0, y0, step, gw, gh, passable, dist_grid, parent, start,
            coarse_footprint_gain_cells, coarse_footprint_gain_ratio, coarse_footprint_reclean_ratio,
        )
        if known_strip is not None:
            coverage_route_top_debug = str(known_strip.get("known_sweep_top_debug", "knownStrip=route"))
            last_coverage_segment_debug = str(known_strip.get("coverage_segment_debug", "known-strip"))
            last_coverage_footprint_gain_cells = int(known_strip.get("footprint_gain_cells", 0))
            last_coverage_footprint_reclean_ratio = float(known_strip.get("footprint_reclean_ratio", 0.0))
            last_coverage_footprint_debug = (
                f"fpGain={last_coverage_footprint_gain_cells} "
                f"fpRatio={float(known_strip.get('footprint_gain_ratio', 0.0)):.2f} "
                f"fpReclean={last_coverage_footprint_reclean_ratio:.2f}"
            )
            last_commit_type_debug = "commitType=known-full-sweep"
            last_explore_gain_debug = f"exploreGain=0 coverageGain={last_coverage_footprint_gain_cells}"
            last_route_planner_ms = (time.perf_counter() - planner_t0) * 1000.0
            return known_strip

    # cells.  This keeps planned coverage from committing to one-pixel leftovers
    # when a larger nearby island would be more useful.
    component_labels = np.zeros((gh, gw), dtype=np.int32)
    component_stats = None
    component_wall_bonus = {}
    component_gain_bonus = {}
    component_missed_strip_bonus = {}
    component_large_bonus = {}
    try:
        n_components, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats(
            (target_kind == 1).astype(np.uint8), 8
        )
        for cid in range(1, int(n_components)):
            comp_mask = component_labels == cid
            comp_cells = int(component_stats[cid, cv2.CC_STAT_AREA])
            if comp_cells <= 0:
                continue
            dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            wall_contacts = int(np.count_nonzero(dilated & (~passable)))
            component_gain_bonus[cid] = min(8.0, 0.18 * comp_cells)
            # vacuum must deliberately run along walls; otherwise the map reaches
            # a high global coverage percentage while a visible border band remains
            # uncleaned.  Keep this as a score term, not a hard if: blocked or far
            # wall strips can still lose to a better component.
            if ROUTE_COMMIT_WALL_STRIP_PRIORITY_ENABLED:
                component_wall_bonus[cid] = min(ROUTE_COMMIT_WALL_STRIP_BONUS, 0.22 * wall_contacts)
            else:
                component_wall_bonus[cid] = min(3.5, 0.08 * wall_contacts)
            left = int(component_stats[cid, cv2.CC_STAT_LEFT])
            width = int(component_stats[cid, cv2.CC_STAT_WIDTH])
            height = int(component_stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            thin_band = min(width, height) <= 2 and span >= MISSED_STRIP_MIN_SPAN_CELLS
            wall_band = wall_contacts >= MISSED_STRIP_MIN_WALL_CONTACTS and span >= MISSED_STRIP_MIN_SPAN_CELLS
            if MISSED_STRIP_RECOVERY_ENABLED and (thin_band or wall_band):
                component_missed_strip_bonus[cid] = min(
                    MISSED_STRIP_MAX_BONUS,
                    MISSED_STRIP_SCORE_BONUS + MISSED_STRIP_WALL_CONTACT_SCALE * float(wall_contacts),
                )
            else:
                component_missed_strip_bonus[cid] = 0.0
            component_large_bonus[cid] = min(LARGE_UNCLEANED_COMPONENT_BONUS_MAX, LARGE_UNCLEANED_COMPONENT_BONUS_SCALE * max(0, comp_cells - 6))
    except Exception:
        component_labels = np.zeros((gh, gw), dtype=np.int32)
        component_stats = None
        component_wall_bonus = {}
        component_gain_bonus = {}
        component_missed_strip_bonus = {}
        component_large_bonus = {}

    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    rxw = pose_x
    ryw = pose_y
    best = None
    best_score = -1e18
    top_candidates = []
    candidate_records = []
    rejected_component_reason = {}
    accepted_component_ids = set()
    local_first_frontier_available = False
    local_first_candidate_count = 0
    if LOCAL_FIRST_FRONTIER_ENABLED and exploration_route_only:
        try:
            fys, fxs = np.where((target_kind == 2) & passable & np.isfinite(dist_grid) & (dist_grid < inf * 0.5))
            for lgy, lgx in zip(fys.tolist(), fxs.tolist()):
                lcx, lcy = coarse_route_center(x0, y0, step, int(lgx), int(lgy))
                lwx = (lcx - MAP_ORIGIN_X) / MAP_SCALE
                lwy = (MAP_ORIGIN_Y - lcy) / MAP_SCALE
                ldist = math.hypot(lwx - rxw, lwy - ryw)
                lroute_cost = float(dist_grid[int(lgy), int(lgx)]) * step / MAP_SCALE
                lmetric = lroute_cost if LOCAL_FIRST_FRONTIER_USE_ROUTE_COST_RADIUS else ldist
                if lmetric <= float(LOCAL_FIRST_FRONTIER_NEAR_RADIUS_M):
                    try:
                        lcells = int(coarse_frontier_local_cells[int(lgy), int(lgx)])
                    except Exception:
                        lcells = 0
                    if lcells >= int(LOCAL_FIRST_FRONTIER_MIN_LOCAL_CELLS) and not route_target_is_blacklisted(lcx, lcy, "frontier"):
                        local_first_candidate_count += 1
                        local_first_frontier_available = True
            if local_first_frontier_available and LOCAL_FIRST_FRONTIER_DEBUG_ENABLED:
                last_exploration_route_debug = f"exploreRoute=local-first candidates={local_first_candidate_count}"
        except Exception:
            local_first_frontier_available = False
            local_first_candidate_count = 0
    ys, xs = np.where((target_kind > 0) & np.isfinite(dist_grid) & (dist_grid < inf * 0.5))
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        cx, cy = coarse_route_center(x0, y0, step, gx, gy)
        wx = (cx - MAP_ORIGIN_X) / MAP_SCALE
        wy = (MAP_ORIGIN_Y - cy) / MAP_SCALE
        dx = wx - rxw
        dy = wy - ryw
        straight_dist = math.hypot(dx, dy)
        if straight_dist < PLANNER_TARGET_REACHED_M or straight_dist > PLANNER_TARGET_SEARCH_RADIUS_M:
            continue
        route_cost_m = float(dist_grid[gy, gx]) * step / MAP_SCALE
        footprint_penalty = 0.0
        if footprint_clearance_m is not None and map_inside(cx, cy):
            clear_m = float(footprint_clearance_m[cy, cx])
            if clear_m < FOOTPRINT_RADIUS_M + FOOTPRINT_ROUTE_MARGIN_M:
                footprint_penalty = 4.0
            elif clear_m < FOOTPRINT_RADIUS_M + 0.04:
                footprint_penalty = 1.2
        longitudinal_m = dx * fx + dy * fy
        forward = longitudinal_m / max(straight_dist, 1e-6)
        lateral_signed = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
        lateral = abs(lateral_signed)
        # A target that demands an immediate 90-degree detour is not necessarily
        # bad, but it must beat nearby forward/continuation work by route utility.
        angle_to_target = math.atan2(dy, dx)
        turn_need = abs(normalize_angle(angle_to_target - pose_theta)) / math.pi
        reward = float(target_reward[gy, gx])
        kind_code = int(target_kind[gy, gx])
        if exploration_route_only and kind_code != 2:
            continue
        if kind_code == 3 and straight_dist > GLOBAL_ROUTE_UNDER_MAX_DIST_M:
            continue
        if exploration_route_only:
            max_detour = straight_dist * EXPLORATION_ROUTE_MAX_DETOUR_RATIO + EXPLORATION_ROUTE_MAX_EXTRA_DETOUR_M
            if route_cost_m > max_detour:
                last_exploration_route_debug = f"exploreRoute=reject detour cost={route_cost_m:.2f} straight={straight_dist:.2f}"
                continue
        recent_target = float(recent_ratio[gy, gx])
        # Recently driven cells can still be valid transit, but they are poor
        # new objectives. This directly reduces repeated loops around the same
        # table/chair opening. Do not hard-block every recent target, because in
        # a narrow corridor all reachable routes may temporarily pass through
        # recent cells.
        recent_penalty = RECENT_VISIT_TARGET_PENALTY * recent_target
        if recent_target > RECENT_VISIT_TARGET_SOFT_BLOCK and straight_dist > 0.35:
            recent_penalty += 3.0
        fp_gain_cells = int(coarse_footprint_gain_cells[gy, gx])
        fp_gain_ratio = float(coarse_footprint_gain_ratio[gy, gx])
        fp_reclean_ratio = float(coarse_footprint_reclean_ratio[gy, gx])
        fp_gain_bonus = min(COVERAGE_FOOTPRINT_TARGET_GAIN_MAX, COVERAGE_FOOTPRINT_TARGET_GAIN_SCALE * float(fp_gain_cells)) if kind_code == 1 else 0.0
        fp_reclean_penalty = COVERAGE_FOOTPRINT_TARGET_RECLEAN_PENALTY * fp_reclean_ratio if kind_code == 1 else 0.0
        known_sweep_bonus = 0.0
        known_sweep_debug = "knownSweep=off"
        if kind_code == 1 and known_map_coverage_eval_active():
            known_sweep_bonus, known_sweep_debug = known_map_eval_sweep_priority_score(cx, cy, route_cost_m)
            # In known-map evaluation, very low-gain centre points are often just
            # accidental residuals on a path, not a meaningful coverage objective.
            if fp_gain_cells < KNOWN_MAP_EVAL_SWEEP_MIN_FP_GAIN and raw_uncleaned_cells[gy, gx] <= 0:
                rejected_component_reason.setdefault(int(component_labels[gy, gx]) if component_labels is not None else -1, "known_low_gain")
                continue

        comp_id = int(component_labels[gy, gx]) if kind_code == 1 else -1
        comp_cells = 0
        comp_gain = 0.0
        wall_strip_bonus = 0.0
        missed_strip_bonus = 0.0
        large_component_bonus = 0.0
        stale_uncleaned_bonus = 0.0
        coverage_segment_suffix = []
        coverage_segment_bonus = 0.0
        coverage_segment_gain = 0
        coverage_segment_reclean = 0.0
        coverage_segment_debug = "point"
        coverage_segment_extra_cost_m = 0.0
        if comp_id > 0 and component_stats is not None:
            comp_cells = int(component_stats[comp_id, cv2.CC_STAT_AREA])
            comp_gain = float(component_gain_bonus.get(comp_id, 0.0))
            wall_strip_bonus = float(component_wall_bonus.get(comp_id, 0.0))
            missed_strip_bonus = float(component_missed_strip_bonus.get(comp_id, 0.0))
            large_component_bonus = float(component_large_bonus.get(comp_id, 0.0))
            if last_coverage_percent >= MISSED_STRIP_MIN_COVERAGE_PERCENT and comp_cells >= MISSED_STRIP_MIN_COMPONENT_CELLS_PARTIAL:
                stale_uncleaned_bonus = float(STALE_UNCLEANED_COMPONENT_BONUS)
            seg = build_component_coverage_segment_suffix(
                component_labels, comp_id, gx, gy, passable,
                coarse_footprint_gain_cells, coarse_footprint_reclean_ratio,
            )
            if seg is not None:
                coverage_segment_suffix = list(seg.get("suffix", []))
                coverage_segment_bonus = float(seg.get("bonus", 0.0))
                coverage_segment_gain = int(seg.get("gain", 0))
                coverage_segment_reclean = float(seg.get("reclean", 0.0))
                coverage_segment_debug = str(seg.get("debug", "seg"))
                coverage_segment_extra_cost_m = max(0.0, (len(coverage_segment_suffix) - 1) * step / MAP_SCALE * COVERAGE_SEGMENT_INTERNAL_COST_FACTOR)

        if kind_code == 1 and route_target_is_blacklisted(cx, cy, "uncleaned"):
            last_residual_route_deferred_cells += max(1, int(comp_cells or 1))
            if comp_id > 0:
                rejected_component_reason.setdefault(comp_id, "blacklist")
            continue
        if kind_code == 1 and route_target_is_edge_residual_trap(cx, cy, comp_cells):
            last_residual_route_deferred_cells += max(1, int(comp_cells or 1))
            last_route_target_blacklist_debug = f"edge-residual defer ({int(cx)},{int(cy)}) comp={int(comp_cells)}"
            if comp_id > 0:
                rejected_component_reason.setdefault(comp_id, "edge_residual")
            continue
        kind_name_loop = {1: "uncleaned", 2: "frontier", 3: "under-surface"}.get(kind_code, "uncleaned")
        if kind_code == 2 and route_target_is_blacklisted(cx, cy, "frontier"):
            last_residual_route_deferred_cells += 1
            if FRONTIER_COMPONENT_TARGETING_ENABLED:
                try:
                    fcid = int(frontier_component_labels[cy, cx])
                    if fcid > 0:
                        rejected_component_reason.setdefault(fcid, "frontier_blacklist")
                except Exception:
                    pass
            continue
        if route_target_is_dock_loiter_trap(cx, cy, kind_name_loop, comp_cells):
            last_residual_route_deferred_cells += max(1, int(comp_cells or 1))
            if comp_id > 0:
                rejected_component_reason.setdefault(comp_id, "dock_guard")
            continue

        continuity_bonus = 0.0
        if kind_code == 1:
            # Prefer finishing the strip/nearby wall band the robot is already working
            # on instead of jumping to a small detached island.  This is a scoring
            # term, not a hard-coded exception: a bad/blocked continuation can still
            # lose to a genuinely better component.
            if straight_dist <= 1.65 and longitudinal_m >= -0.08:
                lateral_gate = 0.52
                if lateral <= lateral_gate:
                    continuity_bonus += GLOBAL_ROUTE_CONTINUITY_BONUS * (1.0 - lateral / max(lateral_gate, 1e-6))
                if turn_need <= 0.23:
                    continuity_bonus += 1.25 * (1.0 - turn_need / 0.23)
            if wall_strip_bonus > 0.0 and straight_dist <= 1.70:
                continuity_bonus += GLOBAL_ROUTE_WALL_CONTINUITY_BONUS
                if ROUTE_COMMIT_WALL_STRIP_PRIORITY_ENABLED:
                    continuity_bonus += ROUTE_COMMIT_WALL_STRIP_CONTINUITY_BONUS

        exploration_penalty = 0.0
        if kind_code == 1:
            reject_cleanup, exploration_penalty, explore_reason = uncleaned_exploration_backtrack_decision(
                route_cost_m + coverage_segment_extra_cost_m, straight_dist, longitudinal_m, lateral, turn_need,
                comp_cells, fp_gain_cells, coverage_segment_gain, missed_strip_bonus, wall_strip_bonus,
            )
            if reject_cleanup:
                last_residual_route_deferred_cells += max(1, int(comp_cells or fp_gain_cells or 1))
                if comp_id > 0:
                    rejected_component_reason.setdefault(comp_id, "explore_backtrack")
                last_known_backtrack_debug = f"knownBacktrackPenalty=reject {explore_reason}"
                continue
            if exploration_penalty > 0.0:
                last_known_backtrack_debug = f"knownBacktrackPenalty={exploration_penalty:.1f} {explore_reason}"
        elif kind_code == 2:
            try:
                front_cells_dbg = int(coarse_frontier_local_cells[gy, gx])
                front_comp_dbg = int(coarse_frontier_component_cells[gy, gx])
                front_unknown_dbg = int(coarse_frontier_unknown_cells[gy, gx])
                if front_cells_dbg <= 0:
                    front_cells_dbg = local_mask_count(frontier, int(cx), int(cy), EXPLORE_FRONTIER_LOCAL_RADIUS_M)
            except Exception:
                front_cells_dbg = 0
                front_comp_dbg = 0
                front_unknown_dbg = 0
            last_frontier_target_debug = (
                f"frontierTarget=({int(cx)},{int(cy)}) local={front_cells_dbg} "
                f"comp={front_comp_dbg} unk={front_unknown_dbg} cost={route_cost_m:.2f}"
            )

        route_geometry_debug = "geom=none"
        route_first_turn_frac = 0.0
        route_corner_count = 0
        route_for_candidate = None

        # In frontier-only exploration, use a clean route score: utility of the
        # frontier/vantage minus route cost and motion complexity. Coverage
        # route-shape penalties: a huge side frontier must not beat a simple
        # forward frontier just because its component has more unknown cells.
        if exploration_route_only and kind_code == 2:
            if EXPLORATION_ROUTE_GEOMETRY_STABILITY_ENABLED:
                route_for_candidate, route_geo = reconstruct_route_and_geometry(parent, (sx, sy), (gx, gy), x0, y0, step)
                route_first_turn_frac = float(route_geo.get("first_turn_frac", 0.0))
                route_corner_count = int(route_geo.get("corner_count", 0))
                route_geometry_debug = str(route_geo.get("debug", "geom=none"))
            extra_corners = max(0, int(route_corner_count) - int(EXPLORATION_ROUTE_MAX_SOFT_CORNERS))
            local_first_bonus = 0.0
            local_first_penalty = 0.0
            local_first_debug = "localFirst=off"
            if LOCAL_FIRST_FRONTIER_ENABLED:
                local_cells = int(front_cells_dbg) if 'front_cells_dbg' in locals() else 0
                local_unknown = int(front_unknown_dbg) if 'front_unknown_dbg' in locals() else 0
                useful_local = max(local_cells, min(local_unknown, local_cells * int(LOCAL_FIRST_FRONTIER_USEFUL_UNKNOWN_MULT) if local_cells > 0 else 0))
                if local_cells >= int(LOCAL_FIRST_FRONTIER_MIN_LOCAL_CELLS):
                    near_t = max(0.0, 1.0 - straight_dist / max(1e-6, float(LOCAL_FIRST_FRONTIER_NEAR_RADIUS_M)))
                    mid_t = max(0.0, 1.0 - straight_dist / max(1e-6, float(LOCAL_FIRST_FRONTIER_MID_RADIUS_M)))
                    local_first_bonus += float(LOCAL_FIRST_FRONTIER_CLOSE_BONUS) * near_t
                    local_first_bonus += float(LOCAL_FIRST_FRONTIER_MID_BONUS) * mid_t
                    local_first_bonus += min(float(LOCAL_FIRST_FRONTIER_GRAY_BONUS_MAX), float(LOCAL_FIRST_FRONTIER_GRAY_BONUS_SCALE) * float(local_unknown))
                    local_first_penalty += float(LOCAL_FIRST_FRONTIER_ROUTE_COST_WEIGHT) * max(0.0, route_cost_m - straight_dist)
                    if straight_dist > float(LOCAL_FIRST_FRONTIER_MID_RADIUS_M):
                        local_first_penalty += float(LOCAL_FIRST_FRONTIER_FAR_PENALTY) * min(2.0, (straight_dist - float(LOCAL_FIRST_FRONTIER_MID_RADIUS_M)) / max(0.35, float(LOCAL_FIRST_FRONTIER_NEAR_RADIUS_M)))
                    if local_first_frontier_available and route_cost_m > float(LOCAL_FIRST_FRONTIER_MID_RADIUS_M):
                        # Nearby reachable frontier exists; do not let a large far
                        # component pull the robot across the room before the local
                        # gray/floor pocket is mapped.
                        local_first_penalty += float(LOCAL_FIRST_FRONTIER_FAR_PENALTY) * float(LOCAL_FIRST_FRONTIER_NEAR_EXISTS_FAR_EXTRA_PENALTY)
                    local_first_penalty += float(LOCAL_FIRST_FRONTIER_RECENT_PENALTY_SCALE) * max(0.0, recent_target - 0.35)
                    local_first_debug = (
                        f"localFirst=near d={straight_dist:.2f} bonus={local_first_bonus:.1f} "
                        f"pen={local_first_penalty:.1f} loc={local_cells} unk={local_unknown} "
                        f"nearN={local_first_candidate_count}"
                    )
                else:
                    local_first_penalty += 2.0
                    local_first_debug = f"localFirst=weak loc={local_cells} unk={local_unknown}"
            score = (
                reward
                + local_first_bonus
                - local_first_penalty
                - EXPLORATION_ROUTE_COST_PENALTY * route_cost_m
                - EXPLORATION_ROUTE_TURN_PENALTY * turn_need
                - EXPLORATION_ROUTE_LATERAL_PENALTY * lateral
                - EXPLORATION_ROUTE_FIRST_TURN_PENALTY * float(route_first_turn_frac)
                - EXPLORATION_ROUTE_CORNER_PENALTY * float(route_corner_count)
                - EXPLORATION_ROUTE_EXTRA_CORNER_PENALTY * float(extra_corners)
                - footprint_penalty
                - 0.25 * recent_penalty
                + EXPLORATION_ROUTE_FORWARD_BONUS * max(0.0, forward)
            )
            last_exploration_route_debug = (
                f"exploreRoute=frontier-only cost={route_cost_m:.2f} d={straight_dist:.2f} "
                f"turn={turn_need:.2f} {route_geometry_debug} {local_first_debug}"
            )
        else:
            # Equivalent to the requested cost expression, now using the
            # actual cleaning footprint:
            # score = -(path_cost + turn_penalty + reclean_penalty + obstacle_risk)
            #         + uncleaned_gain + footprint_gain + wall_strip_bonus.
            score = (
                reward
                + comp_gain
                + wall_strip_bonus
                + missed_strip_bonus
                + large_component_bonus
                + stale_uncleaned_bonus
                + continuity_bonus
                + fp_gain_bonus
                + coverage_segment_bonus
                + known_sweep_bonus
                - GLOBAL_ROUTE_COST_PENALTY * (route_cost_m + coverage_segment_extra_cost_m)
                - GLOBAL_ROUTE_TURN_PENALTY * turn_need
                - GLOBAL_ROUTE_LATERAL_PENALTY * lateral
                - footprint_penalty
                - fp_reclean_penalty
                - recent_penalty
                - exploration_penalty
                + GLOBAL_ROUTE_FORWARD_BONUS * max(0.0, forward)
            )
        best_tuple = (gx, gy, cx, cy, wx, wy, kind_code, route_cost_m + coverage_segment_extra_cost_m, score, recent_penalty, comp_id, comp_cells, comp_gain, wall_strip_bonus, straight_dist, lateral, turn_need, continuity_bonus, missed_strip_bonus, large_component_bonus, stale_uncleaned_bonus, fp_gain_cells, fp_gain_ratio, fp_reclean_ratio, fp_gain_bonus, fp_reclean_penalty, coverage_segment_suffix, coverage_segment_debug, coverage_segment_bonus, coverage_segment_gain, coverage_segment_reclean, float(route_first_turn_frac), int(route_corner_count), str(route_geometry_debug), route_for_candidate, float(known_sweep_bonus), str(known_sweep_debug))
        debug_gain_cells = comp_gain
        if kind_code == 2:
            debug_gain_cells = max(comp_gain, front_cells_dbg)
        top_candidates.append((
            float(score), kind_name_loop, int(cx), int(cy), float(route_cost_m), float(straight_dist),
            float(lateral), float(turn_need), int(comp_cells), float(debug_gain_cells),
            float(wall_strip_bonus), float(continuity_bonus), float(missed_strip_bonus), float(large_component_bonus),
            int(fp_gain_cells), float(fp_reclean_ratio), str(coverage_segment_debug), float(coverage_segment_bonus),
            float(route_first_turn_frac), int(route_corner_count), float(known_sweep_bonus), str(known_sweep_debug)
        ))
        if comp_id > 0:
            accepted_component_ids.add(int(comp_id))
        candidate_records.append((float(score), best_tuple, kind_name_loop))
        if score > best_score:
            best_score = score
            best = best_tuple

    if top_candidates:
        top_candidates.sort(key=lambda item: item[0], reverse=True)
        if exploration_route_only:
            coverage_route_top_debug = "frontierTop: " + " | ".join(
                f"F{i+1} score={score:.1f} p=({cx},{cy}) d={dist:.2f} cost={cost:.2f} "
                f"gray={gain:.0f} route=ok fail={last_route_target_blacklist_debug[:12]} "
                f"noise={last_gray_gap_debug[-18:]} recent={fp_rc:.2f} turn={turn:.2f} "
                f"reason={'local_gray' if cost <= LOCAL_FIRST_FRONTIER_NEAR_RADIUS_M else ('mid_frontier' if cost <= LOCAL_FIRST_FRONTIER_MID_RADIUS_M else 'far_frontier')}"
                for i, (score, kind, cx, cy, cost, dist, lat, turn, comp, gain, wall, cont, miss, large, fp_gain, fp_rc, seg_dbg, seg_bonus, first_turn, corners, ks, ks_dbg)
                in enumerate(top_candidates[:GLOBAL_ROUTE_TOP_DEBUG_COUNT])
                if kind == "frontier"
            )
            if coverage_route_top_debug == "frontierTop: ":
                coverage_route_top_debug = "frontierTop: none"
        else:
            coverage_route_top_debug = " | ".join(
                f"#{i+1}:{kind}@({cx},{cy}) sc={score:.1f} cost={cost:.2f} d={dist:.2f} "
                f"lat={lat:.2f} turn={turn:.2f} first={first_turn:.2f} crn={corners} comp={comp} fp={fp_gain} rc={fp_rc:.2f} cont={cont:.1f} wall={wall:.1f} miss={miss:.1f} sweep={ks:.0f} seg={seg_dbg[:12]}+{seg_bonus:.1f}"
                for i, (score, kind, cx, cy, cost, dist, lat, turn, comp, gain, wall, cont, miss, large, fp_gain, fp_rc, seg_dbg, seg_bonus, first_turn, corners, ks, ks_dbg)
                in enumerate(top_candidates[:GLOBAL_ROUTE_TOP_DEBUG_COUNT])
            )
    else:
        coverage_route_top_debug = "none"

    # Explain why visible components were not in topComponents.  This is the
    # diagnostic needed for highlighted missed zones: no_path means the route
    # grid cannot reach the component; filtered means a guard/gate suppressed it;
    # low_score means it existed but lost normally.
    try:
        rejected_debug = []
        if component_stats is not None:
            for cid in range(1, int(component_stats.shape[0])):
                if cid in accepted_component_ids:
                    continue
                cells = int(component_stats[cid, cv2.CC_STAT_AREA])
                if cells <= 0:
                    continue
                comp_mask = component_labels == cid
                finite_any = bool(np.any(np.isfinite(dist_grid[comp_mask]) & (dist_grid[comp_mask] < inf * 0.5)))
                reason = rejected_component_reason.get(cid, "no_path" if not finite_any else "filtered/low")
                left = int(component_stats[cid, cv2.CC_STAT_LEFT])
                top = int(component_stats[cid, cv2.CC_STAT_TOP])
                width = int(component_stats[cid, cv2.CC_STAT_WIDTH])
                height = int(component_stats[cid, cv2.CC_STAT_HEIGHT])
                cmx, cmy = coarse_route_center(x0, y0, step, left + width // 2, top + height // 2)
                rejected_debug.append((cells, f"#{cid}@({int(cmx)},{int(cmy)}) {reason} c={cells} miss={float(component_missed_strip_bonus.get(cid, 0.0)):.1f}"))
        if rejected_debug:
            rejected_debug.sort(key=lambda item: item[0], reverse=True)
            last_rejected_components_debug = " | ".join(txt for _cells, txt in rejected_debug[:REJECTED_COMPONENT_DEBUG_COUNT])
        else:
            last_rejected_components_debug = "none"
    except Exception as exc:
        last_rejected_components_debug = f"err {type(exc).__name__}"

    selected_record = None
    if KNOWN_MAP_EVAL_DISABLE_STICKY_TARGETS and known_map_coverage_eval_active():
        clear_coverage_target_sticky("known-map sweep planner")
    else:
        selected_record = choose_sticky_coverage_record(candidate_records)
    if selected_record is not None:
        best_score = float(selected_record[0])
        best = selected_record[1]

    if best is None:
        clear_coverage_target_sticky("no candidate")
        last_route_planner_ms = (time.perf_counter() - planner_t0) * 1000.0
        return None

    (gx, gy, cx, cy, wx, wy, kind_code, route_cost_m, score, recent_penalty, comp_id, comp_cells,
     comp_gain, wall_strip_bonus, straight_dist, lateral, turn_need, continuity_bonus, missed_strip_bonus,
     large_component_bonus, stale_uncleaned_bonus, fp_gain_cells, fp_gain_ratio, fp_reclean_ratio,
     fp_gain_bonus, fp_reclean_penalty, coverage_segment_suffix, coverage_segment_debug,
     coverage_segment_bonus, coverage_segment_gain, coverage_segment_reclean,
     route_first_turn_frac, route_corner_count, route_geometry_debug, route_for_candidate,
     known_sweep_bonus, known_sweep_debug) = best
    last_recent_target_penalty = float(recent_penalty)
    if int(kind_code) == 2:
        try:
            selected_gray = int(local_mask_count(frontier, int(cx), int(cy), EXPLORE_FRONTIER_LOCAL_RADIUS_M))
        except Exception:
            selected_gray = 0
        selected_reason = "local_gray" if float(route_cost_m) <= float(LOCAL_FIRST_FRONTIER_NEAR_RADIUS_M) else ("mid_frontier" if float(route_cost_m) <= float(LOCAL_FIRST_FRONTIER_MID_RADIUS_M) else "far_frontier")
        last_frontier_target_debug = (
            f"frontierChoice selected=({int(cx)},{int(cy)}) score={float(score):.1f} "
            f"d={float(straight_dist):.2f} cost={float(route_cost_m):.2f} gray={selected_gray} "
            f"route=ok fail={last_route_target_blacklist_debug[:18]} reason={selected_reason}"
        )
    last_explore_gain_debug = f"exploreGain={int(local_mask_count(frontier, int(cx), int(cy), EXPLORE_FRONTIER_LOCAL_RADIUS_M)) if kind_code == 2 else 0} coverageGain={int(fp_gain_cells) if kind_code == 1 else int(coverage_segment_gain)}"
    last_commit_type_debug = "commitType=" + {1: "uncleaned", 2: "frontier", 3: "under-surface"}.get(int(kind_code), "unknown")
    last_coverage_footprint_gain_cells = int(fp_gain_cells)
    last_coverage_footprint_reclean_ratio = float(fp_reclean_ratio)
    last_coverage_footprint_debug = f"fpGain={int(fp_gain_cells)} fpRatio={float(fp_gain_ratio):.2f} fpReclean={float(fp_reclean_ratio):.2f}"
    last_coverage_segment_debug = (str(coverage_segment_debug or "point") + (" " + str(known_sweep_debug) if known_map_coverage_eval_active() else ""))
    if route_for_candidate is not None:
        route = list(route_for_candidate)
    else:
        route = reconstruct_coarse_route(parent, (sx, sy), (gx, gy), x0, y0, step)
    route = append_segment_suffix_to_route(route, coverage_segment_suffix, x0, y0, step)
    if not exploration_route_only and route:
        geo = route_path_geometry_stats(route)
        route_first_turn_frac = float(geo.get("first_turn_frac", route_first_turn_frac))
        route_corner_count = int(geo.get("corner_count", route_corner_count))
        route_geometry_debug = str(geo.get("debug", route_geometry_debug))
    if route:
        cx, cy = int(route[-1][0]), int(route[-1][1])
        wx = (cx - MAP_ORIGIN_X) / MAP_SCALE
        wy = (MAP_ORIGIN_Y - cy) / MAP_SCALE
    wp_map, wp_world = route_waypoint_from_path(route)
    kind_name = {1: "uncleaned", 2: "frontier", 3: "under-surface"}.get(kind_code, "uncleaned")
    route_world = [((mx - MAP_ORIGIN_X) / MAP_SCALE, (MAP_ORIGIN_Y - my) / MAP_SCALE) for mx, my in route]
    last_route_planner_ms = (time.perf_counter() - planner_t0) * 1000.0
    last_route_planner_nodes = int(expanded_nodes)
    return {
        "goal_map": (int(cx), int(cy)),
        "goal_world": (float(wx), float(wy)),
        "kind": kind_name,
        "route_map": route,
        "route_world": route_world,
        "waypoint_map": wp_map,
        "waypoint_world": wp_world,
        "cost": float(route_cost_m),
        "score": float(score),
        "length": len(route),
        "component_id": int(comp_id),
        "component_cells": int(comp_cells),
        "component_gain": float(comp_gain),
        "wall_strip_bonus": float(wall_strip_bonus),
        "straight_dist": float(straight_dist),
        "lateral_abs": float(lateral),
        "turn_need": float(turn_need),
        "continuity_bonus": float(continuity_bonus),
        "missed_strip_bonus": float(missed_strip_bonus),
        "large_component_bonus": float(large_component_bonus),
        "stale_uncleaned_bonus": float(stale_uncleaned_bonus),
        "footprint_gain_cells": int(fp_gain_cells),
        "footprint_gain_ratio": float(fp_gain_ratio),
        "footprint_reclean_ratio": float(fp_reclean_ratio),
        "footprint_gain_bonus": float(fp_gain_bonus),
        "footprint_reclean_penalty": float(fp_reclean_penalty),
        "coverage_segment_debug": str(coverage_segment_debug or "point"),
        "coverage_segment_bonus": float(coverage_segment_bonus),
        "coverage_segment_gain": int(coverage_segment_gain),
        "coverage_segment_reclean": float(coverage_segment_reclean),
        "first_turn_frac": float(route_first_turn_frac),
        "corner_count": int(route_corner_count),
        "geometry_debug": str(route_geometry_debug),
    }

def update_coverage_objective():
    """Select a soft next coverage target from map memory.

    The previous selector picked the best-looking cell by straight-line distance.
    That is exactly why the map sometimes showed a strange diagonal path: the
    target was not checked for reachable route cost. This version first builds a
    coarse traversability grid and chooses a reachable uncleaned/frontier/under-
    surface target by wavefront cost. The row cleaner remains simple, but its
    decisions are now biased by a path-aware waypoint instead of a raw Euclidean
    target.
    """
    global coverage_goal_map, coverage_goal_world, coverage_goal_kind
    global coverage_route_map, coverage_route_world, coverage_route_waypoint_map, coverage_route_waypoint_world
    global coverage_route_cost, coverage_route_score, coverage_route_len, coverage_route_kind, coverage_route_component_id, coverage_route_component_cells, coverage_route_status
    global coverage_route_commit_class, coverage_route_straight_dist, coverage_route_lateral_abs, coverage_route_turn_need, coverage_route_continuity_bonus, coverage_route_top_debug
    global coverage_route_wall_strip_bonus, coverage_route_missed_strip_bonus, coverage_route_segment_gain, coverage_route_segment_bonus, coverage_route_footprint_gain_cells
    global coverage_route_first_turn_frac, coverage_route_corner_count, coverage_route_geometry_debug
    global last_coverage_total_cells, last_coverage_cleaned_cells, last_coverage_percent
    global last_frontier_cells, last_gray_gap_cells, last_gray_gap_components, last_gray_gap_debug, last_uncleaned_cells, last_under_surface_target_cells
    global last_residual_route_deferred_cells, last_route_planner_ms, last_route_planner_nodes

    t_perf_planner = perf_start()
    obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
    under_surface = under_surface_mask_from_obstacles(obstacles) & cleanable
    under_surface_uncleaned = under_surface & uncleaned
    last_under_surface_target_cells = int(np.count_nonzero(under_surface_uncleaned))

    total = int(np.count_nonzero(cleanable))
    done = int(np.count_nonzero(cleaned & cleanable))
    last_coverage_total_cells = total
    last_coverage_cleaned_cells = done
    last_coverage_percent = (100.0 * done / total) if total > 0 else 0.0
    last_uncleaned_cells = int(np.count_nonzero(uncleaned))

    frontier = build_frontier_revisit_mask(unknown, cleanable, obstacles)
    last_frontier_cells = int(np.count_nonzero(frontier))
    update_map_maturity()
    update_planner_intent()
    cleanup_locked = exploration_cleanup_locked()
    if planner_intent == PLANNER_INTENT_EXPAND_MAP and cleanup_locked and coverage_target_sticky_kind in ("uncleaned", "under-surface"):
        clear_coverage_target_sticky("intent expand-map cleanup locked")

    if (
        known_map_coverage_eval_active()
        and bool(KNOWN_MAP_EVAL_FREEZE_PLANNER_DURING_COMMIT)
        and route_commit_active
        and route_commit_kind == "uncleaned"
    ):
        # Re-running plan_best_coverage_route() here creates a second candidate
        # route that changes under the robot and makes the HUD look as if the
        # robot is still searching for an optimum instead of following the
        # committed orange route. Freeze the candidate/visualization to the
        # active route and only update coverage metrics above. When the route
        # finishes or aborts, route_commit_active becomes False and the next call
        # computes the next residual chunk normally.
        coverage_goal_map = route_commit_target_map
        coverage_goal_world = route_commit_target_world
        coverage_goal_kind = route_commit_kind
        coverage_route_map = list(route_commit_route_map or [])
        coverage_route_world = list(route_commit_route_world or [])
        coverage_route_waypoint_map = route_commit_waypoint_map
        coverage_route_waypoint_world = route_commit_waypoint_world
        coverage_route_cost = float(route_commit_cost)
        coverage_route_len = int(len(coverage_route_map))
        coverage_route_kind = route_commit_kind
        coverage_route_score = float(route_commit_score)
        coverage_route_component_id = int(route_commit_component_id)
        coverage_route_component_cells = int(route_commit_component_cells)
        coverage_route_straight_dist = 0.0
        coverage_route_lateral_abs = 0.0
        coverage_route_turn_need = 0.0
        coverage_route_continuity_bonus = 0.0
        coverage_route_wall_strip_bonus = float(route_commit_wall_strip_bonus)
        coverage_route_missed_strip_bonus = float(route_commit_missed_strip_bonus)
        coverage_route_segment_gain = int(route_commit_segment_gain)
        coverage_route_segment_bonus = 0.0
        coverage_route_footprint_gain_cells = int(route_commit_footprint_gain_cells)
        coverage_route_first_turn_frac = float(route_commit_first_turn_frac)
        coverage_route_corner_count = int(route_commit_corner_count)
        coverage_route_geometry_debug = str(route_commit_geometry_debug)
        coverage_route_commit_class = "active-frozen"
        coverage_route_status = (
            f"active known-sweep frozen progress={int(route_commit_progress_idx)}/{max(1, len(coverage_route_map))} "
            f"wp={int(route_commit_waypoint_idx)} cost={float(route_commit_cost):.1f} "
            f"cov={last_coverage_percent:.1f}% no-new-plan {known_map_residual_policy_status[:46]}"
        )
        last_route_planner_ms = 0.0
        perf_end("planner", t_perf_planner)
        return coverage_route_waypoint_world or coverage_goal_world

    route = plan_best_coverage_route(obstacles, cleanable, cleaned, uncleaned, unknown, frontier, under_surface_uncleaned)
    if route is not None:
        coverage_goal_map = route["goal_map"]
        coverage_goal_world = route["goal_world"]
        coverage_goal_kind = route["kind"]
        coverage_route_map = route["route_map"]
        coverage_route_world = route["route_world"]
        coverage_route_waypoint_map = route["waypoint_map"]
        coverage_route_waypoint_world = route["waypoint_world"]
        coverage_route_cost = route["cost"]
        coverage_route_len = route["length"]
        coverage_route_kind = route["kind"]
        coverage_route_score = float(route.get("score", float("-inf")))
        coverage_route_component_id = int(route.get("component_id", -1))
        coverage_route_component_cells = int(route.get("component_cells", 0))
        coverage_route_straight_dist = float(route.get("straight_dist", float("inf")))
        coverage_route_lateral_abs = float(route.get("lateral_abs", float("inf")))
        coverage_route_turn_need = float(route.get("turn_need", float("inf")))
        coverage_route_continuity_bonus = float(route.get("continuity_bonus", 0.0))
        coverage_route_wall_strip_bonus = float(route.get("wall_strip_bonus", 0.0))
        coverage_route_missed_strip_bonus = float(route.get("missed_strip_bonus", 0.0))
        coverage_route_segment_gain = int(route.get("coverage_segment_gain", 0))
        coverage_route_segment_bonus = float(route.get("coverage_segment_bonus", 0.0))
        coverage_route_footprint_gain_cells = int(route.get("footprint_gain_cells", 0))
        coverage_route_first_turn_frac = float(route.get("first_turn_frac", 0.0))
        coverage_route_corner_count = int(route.get("corner_count", 0))
        coverage_route_geometry_debug = str(route.get("geometry_debug", "geom=none"))
        coverage_route_commit_class = "candidate"
        coverage_route_status = (
            f"wavefront {route['kind']} cost={route['cost']:.2f}m len={route['length']} "
            f"score={coverage_route_score:.1f} comp={coverage_route_component_cells} "
            f"fp={route.get('footprint_gain_cells', 0)}/{route.get('footprint_reclean_ratio', 0.0):.2f} "
            f"cont={coverage_route_continuity_bonus:.1f} wall={route.get('wall_strip_bonus', 0.0):.1f} "
            f"seg={route.get('coverage_segment_debug', 'point')[:20]} "
            f"{coverage_route_geometry_debug[:22]} "
            f"{last_commit_type_debug[:22]} {last_explore_arbitration_debug[:24]} "
            f"{last_exploration_route_debug[:34]} noise={last_objective_noise_filter_debug[:14]} "
            f"defer={last_residual_route_deferred_cells} plan={last_route_planner_ms:.0f}ms/{last_route_planner_nodes}"
        )
        perf_end("planner", t_perf_planner)
        return coverage_goal_world

    # Fallback: if the coarse route grid cannot find a candidate, keep a very
    # conservative local scan. This prevents total planner silence during early
    # mapping when the cleanable region is still tiny.
    coverage_route_map = []
    coverage_route_world = []
    coverage_route_waypoint_map = None
    coverage_route_waypoint_world = None
    coverage_route_cost = float("inf")
    coverage_route_score = float("-inf")
    coverage_route_len = 0
    coverage_route_kind = "none"
    coverage_route_component_id = -1
    coverage_route_component_cells = 0
    coverage_route_status = "fallback/no reachable route"
    coverage_route_top_debug = "none"
    coverage_route_commit_class = "none"
    coverage_route_straight_dist = float("inf")
    coverage_route_lateral_abs = float("inf")
    coverage_route_turn_need = float("inf")
    coverage_route_continuity_bonus = 0.0
    coverage_route_wall_strip_bonus = 0.0
    coverage_route_missed_strip_bonus = 0.0
    coverage_route_segment_gain = 0
    coverage_route_segment_bonus = 0.0
    coverage_route_footprint_gain_cells = 0
    coverage_route_first_turn_frac = 0.0
    coverage_route_corner_count = 0
    coverage_route_geometry_debug = "geom=none"

    robot_mx, robot_my = world_to_map(pose_x, pose_y)
    search_r = int(min(1.4, PLANNER_TARGET_SEARCH_RADIUS_M) * MAP_SCALE)
    x0 = max(0, robot_mx - search_r)
    x1 = min(MAP_SIZE, robot_mx + search_r)
    y0 = max(0, robot_my - search_r)
    y1 = min(MAP_SIZE, robot_my + search_r)
    best = None
    best_score = -1e9
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)
    for my in range(y0, y1, PLANNER_GRID_STEP_PX):
        for mx in range(x0, x1, PLANNER_GRID_STEP_PX):
            if obstacles[my, mx]:
                continue
            # In EXPAND_MAP, fallback must use the same intent as the wavefront
            # planner.  Otherwise a failed frontier route silently falls back to
            # a nearby residual cleanup target and the robot appears to "return
            # for no reason" while the map is still partial.
            front_local_fb = 0
            if planner_intent == PLANNER_INTENT_EXPAND_MAP:
                front_local_fb = frontier_cells_near_map(mx, my, unknown, cleanable)
            if planner_intent == PLANNER_INTENT_EXPAND_MAP and (frontier[my, mx] or front_local_fb >= EXPLORE_FRONTIER_VANTAGE_MIN_LOCAL_CELLS):
                kind = "frontier"
                base = (
                    PLANNER_UNKNOWN_FRONTIER_BONUS
                    + EXPLORE_FRONTIER_REWARD * 0.35
                    + min(EXPLORE_FRONTIER_LOCAL_BONUS_MAX, EXPLORE_FRONTIER_LOCAL_BONUS_SCALE * float(front_local_fb))
                )
            elif (not cleanup_locked) and under_surface_uncleaned[my, mx]:
                # Keep yellow cells local even in fallback mode.
                kind = "under-surface"
                base = UNDER_SURFACE_TARGET_BONUS * 0.75
            elif (not cleanup_locked) and uncleaned[my, mx]:
                kind = "uncleaned"
                base = PLANNER_UNCLEANED_BONUS
            elif frontier[my, mx]:
                kind = "frontier"
                base = PLANNER_UNKNOWN_FRONTIER_BONUS
            else:
                continue
            wx = (mx - MAP_ORIGIN_X) / MAP_SCALE
            wy = (MAP_ORIGIN_Y - my) / MAP_SCALE
            dx = wx - pose_x
            dy = wy - pose_y
            dist = math.hypot(dx, dy)
            if dist < PLANNER_TARGET_REACHED_M:
                continue
            if kind == "uncleaned" and route_target_is_blacklisted(mx, my, "uncleaned"):
                continue
            if kind == "uncleaned" and route_target_is_edge_residual_trap(mx, my, local_mask_count(uncleaned & cleanable, mx, my, RESIDUAL_TARGET_LOCAL_RADIUS_M)):
                last_residual_route_deferred_cells += 1
                continue
            if route_target_is_dock_loiter_trap(mx, my, kind, local_mask_count(uncleaned & cleanable, mx, my, RESIDUAL_TARGET_LOCAL_RADIUS_M)):
                last_residual_route_deferred_cells += 1
                continue
            longitudinal = dx * fx + dy * fy
            lateral_signed = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
            lateral = abs(lateral_signed)
            turn_need = abs(normalize_angle(math.atan2(dy, dx) - pose_theta)) / math.pi
            if kind == "uncleaned":
                local_un_cells = local_mask_count(uncleaned & cleanable & (~under_surface_uncleaned), mx, my, RESIDUAL_TARGET_LOCAL_RADIUS_M)
                if not uncleaned_route_target_allowed_for_phase(mx, my, longitudinal, lateral, dist, local_un_cells):
                    last_residual_route_deferred_cells += 1
                    continue
                allowed_by_intent, intent_penalty, intent_reason = cleanup_target_allowed_under_intent(
                    dist, dist, longitudinal, lateral, turn_need,
                    comp_cells=local_un_cells,
                    fp_gain_cells=local_un_cells,
                    coverage_segment_gain=0,
                    missed_strip_bonus=0.0,
                    wall_strip_bonus=0.0,
                )
                if not allowed_by_intent:
                    last_residual_route_deferred_cells += max(1, int(local_un_cells or 1))
                    last_known_backtrack_debug = f"knownBacktrackPenalty=fallback reject {intent_reason}"
                    continue
                base -= float(intent_penalty)
            if kind == "under-surface" and not under_surface_target_is_direct_route(mx, my, longitudinal, lateral, dist, obstacles, cleanable):
                continue
            forward = longitudinal / max(dist, 1e-6)
            score = base - 2.6 * dist - 0.8 * lateral + 1.2 * max(0.0, forward)
            if RECENT_VISIT_ROUTE_MEMORY_ENABLED:
                score -= RECENT_VISIT_TARGET_PENALTY * min(1.0, float(recent_visit_log_odds[my, mx]) / max(1e-6, RECENT_VISIT_MAX))
            if cleaned_mask[my, mx] > 0:
                score -= 5.0
            if score > best_score:
                best_score = score
                best = (mx, my, wx, wy, kind)

    if best is None:
        coverage_goal_map = None
        coverage_goal_world = None
        coverage_goal_kind = "done" if total > 0 and last_coverage_percent > 95.0 else "none"
        perf_end("planner", t_perf_planner)
        return None

    mx, my, wx, wy, kind = best
    coverage_goal_map = (int(mx), int(my))
    coverage_goal_world = (float(wx), float(wy))
    coverage_goal_kind = kind
    coverage_route_waypoint_map = coverage_goal_map
    coverage_route_waypoint_world = coverage_goal_world
    coverage_route_straight_dist = math.hypot(wx - pose_x, wy - pose_y)
    coverage_route_lateral_abs = abs(-math.sin(pose_theta) * (wx - pose_x) + math.cos(pose_theta) * (wy - pose_y))
    coverage_route_turn_need = abs(normalize_angle(math.atan2(wy - pose_y, wx - pose_x) - pose_theta)) / math.pi
    coverage_route_continuity_bonus = 0.0
    coverage_route_wall_strip_bonus = 0.0
    coverage_route_missed_strip_bonus = 0.0
    coverage_route_segment_gain = 0
    coverage_route_segment_bonus = 0.0
    coverage_route_footprint_gain_cells = 0
    coverage_route_first_turn_frac = min(1.0, float(coverage_route_turn_need))
    coverage_route_corner_count = 0
    coverage_route_geometry_debug = "geom=fallback-direct"
    coverage_route_commit_class = "fallback"

    # If the wavefront graph cannot build a multi-cell route but the fallback
    # picked a nearby frontier viewpoint, do not leave the robot in active=none.
    # Build a bounded direct point-pursuit route only when the footprint sweep says
    # the short segment is still on known free floor.  Unsafe fallback targets stay
    # route-less and are handled by the frontier-only blacklist/watchdog.
    direct_ok = False
    direct_blocked = 1.0
    if (
        FRONTIER_FALLBACK_DIRECT_ROUTE_ENABLED
        and kind == "frontier"
        and coverage_route_straight_dist <= float(FRONTIER_FALLBACK_DIRECT_MAX_DIST_M)
    ):
        try:
            direct_blocked = float(map_line_footprint_blocked_ratio(robot_mx, robot_my, int(mx), int(my), obstacles, cleanable))
            direct_ok = direct_blocked <= float(FRONTIER_FALLBACK_DIRECT_MAX_BLOCKED_RATIO)
        except Exception:
            direct_ok = False
    if direct_ok:
        coverage_route_map = [(int(robot_mx), int(robot_my)), (int(mx), int(my))]
        coverage_route_world = [(float(pose_x), float(pose_y)), (float(wx), float(wy))]
        coverage_route_cost = float(max(coverage_route_straight_dist, 0.05))
        coverage_route_score = float(best_score)
        coverage_route_len = int(len(coverage_route_map))
        coverage_route_kind = kind
        coverage_route_component_id = -1
        coverage_route_component_cells = int(frontier_cells_near_map(int(mx), int(my), unknown, cleanable)) if kind == "frontier" else 0
        coverage_route_status = f"fallback-direct {kind} cost={coverage_route_cost:.2f} blocked={direct_blocked:.2f}"
        last_frontier_target_debug = (
            f"frontierChoice selected=({int(mx)},{int(my)}) score={float(best_score):.1f} "
            f"d={coverage_route_straight_dist:.2f} cost={coverage_route_cost:.2f} "
            f"gray={coverage_route_component_cells} route=direct fail=none reason=local_fallback"
        )
    else:
        coverage_route_status = f"fallback {kind} no-direct d={coverage_route_straight_dist:.2f} blocked={direct_blocked:.2f}"
    perf_end("planner", t_perf_planner)
    return coverage_goal_world


def format_target_debug(target_map, target_world, kind):
    if target_map is None and target_world is None:
        return "none"
    if target_map is not None:
        try:
            mx, my = target_map
            return f"{kind}@({int(mx)},{int(my)})"
        except Exception:
            pass
    try:
        wx, wy = target_world
        return f"{kind}@({float(wx):.2f},{float(wy):.2f})"
    except Exception:
        return str(kind or "none")


def update_map_maturity():
    """Update graded map confidence for route ownership.

    used a single conservative map_mature switch.  keeps that
    switch for full planned coverage, but adds planner_confidence/map_mature_soft
    so Dijkstra can own only short, high-gain local commits before the whole room
    is mature.
    """
    global map_mature, map_mature_soft, map_mature_reason, planner_mode, planner_confidence
    try:
        t = robot.getTime()
    except Exception:
        t = 0.0
    if known_map_coverage_eval_active():
        map_mature = True
        map_mature_soft = True
        planner_confidence = "MATURE"
        map_mature_reason = f"known-map eval cov={float(last_coverage_percent or 0.0):.1f}"
        update_planner_mode_label()
        return map_mature
    total = max(1, int(last_coverage_total_cells or 0))
    frontier_ratio = float(last_frontier_cells or 0) / float(total)
    enough_area = total >= MAP_MATURE_MIN_CLEANABLE_CELLS
    coverage_ready = bool(enough_area and last_coverage_percent >= MAP_MATURE_COVERAGE_PERCENT)
    time_ready = bool(enough_area and t >= MAP_MATURE_MIN_TIME_SEC and last_coverage_percent >= MAP_MATURE_TIME_COVERAGE_PERCENT)
    frontier_low = bool(enough_area and frontier_ratio <= MAP_MATURE_MAX_FRONTIER_RATIO and last_coverage_percent >= MAP_MATURE_TIME_COVERAGE_PERCENT)
    no_longer_map_building = not map_building_active()
    mature = bool(PLANNED_COVERAGE_ENABLED and no_longer_map_building and (coverage_ready or (time_ready and frontier_low)))

    partial_coverage = bool(
        HYBRID_COVERAGE_ENABLED
        and enough_area
        and last_coverage_percent >= PLANNER_CONFIDENCE_PARTIAL_COVERAGE_PERCENT
    )
    partial_time = bool(
        HYBRID_COVERAGE_ENABLED
        and enough_area
        and t >= PLANNER_CONFIDENCE_PARTIAL_MIN_TIME_SEC
        and last_coverage_percent >= SHORT_ROUTE_COMMIT_MIN_COVERAGE_PERCENT
        and frontier_ratio <= PLANNER_CONFIDENCE_PARTIAL_MAX_FRONTIER_RATIO
    )
    soft = bool(partial_coverage or partial_time)

    map_mature = mature
    map_mature_soft = bool(soft or mature)
    if mature:
        planner_confidence = "MATURE"
        if coverage_ready:
            map_mature_reason = f"coverage {last_coverage_percent:.1f}% >= {MAP_MATURE_COVERAGE_PERCENT:.1f}%"
        elif time_ready and frontier_low:
            map_mature_reason = f"time/frontier t={t:.0f}s fr={frontier_ratio:.3f}"
        else:
            map_mature_reason = "mature"
    elif soft:
        planner_confidence = "PARTIAL"
        if partial_coverage:
            map_mature_reason = f"partial cov={last_coverage_percent:.1f}% fr={frontier_ratio:.3f}"
        else:
            map_mature_reason = f"partial time t={t:.0f}s cov={last_coverage_percent:.1f}% fr={frontier_ratio:.3f}"
    else:
        planner_confidence = "EARLY"
        reasons = []
        if not PLANNED_COVERAGE_ENABLED:
            reasons.append("disabled")
        if not enough_area:
            reasons.append(f"area {total}/{MAP_MATURE_MIN_CLEANABLE_CELLS}")
        if map_building_active():
            reasons.append("map-building")
        reasons.append(f"cov={last_coverage_percent:.1f}%")
        reasons.append(f"fr={frontier_ratio:.3f}")
        map_mature_reason = ",".join(reasons[:4])
    update_planner_mode_label()
    return map_mature


def route_commit_active_target_debug():
    return format_target_debug(route_commit_target_map, route_commit_target_world, route_commit_kind) if route_commit_active else "none"


def route_commit_candidate_debug():
    return format_target_debug(coverage_goal_map, coverage_goal_world, coverage_goal_kind)


def goal_map_still_useful(goal_map, kind, obstacles=None, cleanable=None, cleaned=None, uncleaned=None):
    """Return True while a candidate/active target still contains useful work."""
    if goal_map is None:
        return False
    gx, gy = goal_map
    if not map_inside(int(gx), int(gy)):
        return False
    try:
        unknown = None
        if obstacles is None or cleanable is None or cleaned is None or uncleaned is None:
            obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
        if obstacles[int(gy), int(gx)]:
            return False
        if kind == "uncleaned":
            cells = local_mask_count(uncleaned & cleanable, int(gx), int(gy), RESIDUAL_TARGET_LOCAL_RADIUS_M)
            return bool(cells >= ROUTE_COMMIT_LOCAL_UNCLEANED_MIN_CELLS)
        if kind == "under-surface":
            under = under_surface_mask_from_obstacles(obstacles) & cleanable
            cells = local_mask_count(under & uncleaned, int(gx), int(gy), RESIDUAL_TARGET_LOCAL_RADIUS_M)
            return bool(cells >= ROUTE_COMMIT_LOCAL_UNCLEANED_MIN_CELLS)
        if kind == "frontier":
            if unknown is None:
                _obs2, cleanable2, _cleaned2, _uncleaned2, unknown2 = compute_coverage_masks()
                cleanable_for_frontier = cleanable2
                unknown_for_frontier = unknown2
            else:
                cleanable_for_frontier = cleanable
                unknown_for_frontier = unknown
            return bool(frontier_cells_near_map(int(gx), int(gy), unknown_for_frontier, cleanable_for_frontier) >= EXPLORE_FRONTIER_MIN_LOCAL_CELLS)
    except Exception:
        return True
    return True


def route_path_has_useful_footprint(route_map, cleanable, uncleaned, min_cells=None):
    """Useful-work test for segment commits.

    A coverage segment may end on a centreline cell that is already cleaned while
    the route suffix still brings the brush over a wall strip.  Checking only the
    final activeTarget would abort the segment prematurely.  Count uncleaned cells
    visible to the cleaning footprint along the remaining committed route.
    """
    if not route_map:
        return False
    if min_cells is None:
        min_cells = max(5, int(ROUTE_COMMIT_LOCAL_UNCLEANED_MIN_CELLS * 0.35))
    try:
        mask = uncleaned & cleanable
        count = 0
        # Skip every other point on very long routes to keep the live loop cheap.
        stride = 2 if len(route_map) > 18 else 1
        for mx, my in route_map[::stride]:
            if not map_inside(int(mx), int(my)):
                continue
            count += local_mask_count(mask, int(mx), int(my), COVERAGE_RADIUS_M)
            if count >= min_cells:
                return True
    except Exception:
        return False
    return False

def route_commit_goal_still_useful(obstacles=None, cleanable=None, cleaned=None, uncleaned=None):
    """Return True while the active target or segment route contains useful work."""
    if obstacles is None or cleanable is None or cleaned is None or uncleaned is None:
        obstacles, cleanable, cleaned, uncleaned, _unknown = compute_coverage_masks()
    if route_commit_kind == "uncleaned" and route_commit_route_map:
        route_tail = route_commit_route_map[max(0, int(route_commit_progress_idx)):]
        min_cells = 1 if known_map_coverage_eval_active() else None
        if route_path_has_useful_footprint(route_tail, cleanable, uncleaned, min_cells=min_cells):
            return True
    return goal_map_still_useful(route_commit_target_map, route_commit_kind, obstacles, cleanable, cleaned, uncleaned)


def coverage_candidate_goal_still_useful(obstacles=None, cleanable=None, cleaned=None, uncleaned=None):
    """Return True while the displayed candidate target or segment contains useful work."""
    if obstacles is None or cleanable is None or cleaned is None or uncleaned is None:
        obstacles, cleanable, cleaned, uncleaned, _unknown = compute_coverage_masks()
    if coverage_goal_kind == "uncleaned" and coverage_route_map:
        min_cells = 1 if known_map_coverage_eval_active() else None
        if route_path_has_useful_footprint(coverage_route_map, cleanable, uncleaned, min_cells=min_cells):
            return True
    return goal_map_still_useful(coverage_goal_map, coverage_goal_kind, obstacles, cleanable, cleaned, uncleaned)


def abort_route_commit(reason):
    """Cancel only the active route owner; keep the candidate planner free to replan."""
    global route_commit_active, route_commit_target_map, route_commit_target_world, route_commit_kind
    global route_commit_route_map, route_commit_route_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_progress_idx, route_commit_waypoint_idx, route_commit_waypoint_is_corner
    global route_commit_cost, route_commit_score, route_commit_component_id, route_commit_component_cells
    global route_commit_wall_strip_bonus, route_commit_missed_strip_bonus, route_commit_segment_gain, route_commit_footprint_gain_cells
    global route_commit_first_turn_frac, route_commit_corner_count, route_commit_geometry_debug
    global route_commit_last_abort_time, route_commit_reason, route_abort_reason, last_route_commit_debug, planner_mode
    global route_commit_best_target_dist, route_commit_last_progress_time
    global dock_return_active, dock_return_status
    global frontier_route_abort_hold_until, frontier_route_abort_hold_debug
    was_dock = bool(route_commit_kind == "dock" or dock_return_active)
    reason_text = str(reason or "")
    finished_known_map = bool(known_map_coverage_eval_active() and reason_text.startswith("finished"))
    # executor lost the current sweep chunk (for example an over-conservative
    # map-line blocked check), not that the room target is bad.  Blacklisting the
    # active uncleaned endpoint can deadlock the evaluator with candidateTarget
    # visible and activeTarget=none.  Keep blacklist behaviour for live RGB-D
    # exploration/cleanup only.
    if not was_dock and not finished_known_map and not known_map_coverage_eval_active():
        # If the route abort reason is a local clearance/unsafe failure but the
        # bumper did not create contact evidence, keep a small orange runtime
        # obstacle hypothesis.  This prevents repeated passes next to the same
        # invisible/partially visible object without poisoning the contact layer.
        if any(tok in reason_text.lower() for tok in ("unsafe", "blocked", "abort", "front", "body", "clearance")):
            try:
                maybe_mark_near_collision_hypothesis("route abort: " + reason_text[:28])
            except Exception:
                pass
        register_route_target_blacklist(reason)
    route_commit_active = False
    route_commit_target_map = None
    route_commit_target_world = None
    route_commit_kind = "none"
    route_commit_route_map = []
    route_commit_route_world = []
    route_commit_waypoint_map = None
    route_commit_waypoint_world = None
    route_commit_progress_idx = 0
    route_commit_waypoint_idx = 0
    route_commit_waypoint_is_corner = False
    route_commit_cost = float("inf")
    route_commit_score = float("-inf")
    route_commit_component_id = -1
    route_commit_component_cells = 0
    route_commit_wall_strip_bonus = 0.0
    route_commit_missed_strip_bonus = 0.0
    route_commit_segment_gain = 0
    route_commit_footprint_gain_cells = 0
    route_commit_first_turn_frac = 0.0
    route_commit_corner_count = 0
    route_commit_geometry_debug = "geom=none"
    route_commit_best_target_dist = float("inf")
    route_commit_last_progress_time = -999.0
    route_commit_last_abort_time = robot.getTime()
    route_abort_reason = str(reason or "abort")
    if (not was_dock) and planner_intent == PLANNER_INTENT_EXPAND_MAP and str(route_commit_kind) == "frontier":
        frontier_route_abort_hold_until = max(frontier_route_abort_hold_until, route_commit_last_abort_time + float(RUNTIME_ARENA_REPLAN_HOLD_SEC))
        frontier_route_abort_hold_debug = f"routeHold=frontier abort {route_abort_reason[:42]}"
    route_commit_reason = "none"
    last_route_commit_debug = f"abort: {route_abort_reason[:48]}"
    if was_dock:
        dock_return_active = False
        dock_return_status = f"dock abort: {route_abort_reason[:48]}"
        planner_mode = "RETURN_HOME"
    else:
        update_planner_mode_label()
    if str(control_lock.owner) == ControlOwner.ROUTE_COMMIT.value:
        release_control(f"route abort: {route_abort_reason[:48]}")


def finish_route_commit(reason="entry reached"):
    """Route entry reached: either resume local fill or finish docking."""
    global desired_grid_heading, route_abort_reason, last_route_commit_debug
    global dock_return_active, dock_return_completed, dock_return_status, dock_return_reason, coverage_status, nav_state
    global known_map_primary_sweep_completed, known_map_primary_sweep_finish_time, known_map_primary_sweep_finish_coverage
    global known_map_residual_policy_status
    finishing_dock = bool(route_commit_kind == "dock" or dock_return_active)
    finished_known_full_sweep = bool(route_commit_is_known_map_sweep())
    if finishing_dock:
        if auto_map_return_to_dock_active and not auto_map_cleaning_started:
            route_abort_reason = "none"
            abort_route_commit(f"map dock reached: {reason}")
            complete_map_return_to_dock(str(reason or "docked"))
            route_abort_reason = "none"
            last_route_commit_debug = dock_return_status
            return
        dock_return_active = False
        dock_return_completed = True
        dock_return_reason = str(reason or "docked")
        dock_return_status = f"docked: {dock_return_reason[:48]}"
        coverage_status = dock_return_status
        desired_grid_heading = DOCK_FINAL_HEADING
        route_abort_reason = "none"
        abort_route_commit(f"docked: {reason}")
        dock_return_active = False
        dock_return_completed = True
        dock_return_status = f"docked: {dock_return_reason[:48]}"
        coverage_status = dock_return_status
        route_abort_reason = "none"
        last_route_commit_debug = dock_return_status
        hard_stop_motors()
        acquire_control(ControlOwner.PLANNER, DOCK_STOP_HOLD_OWNER_SEC, 0.0, "docked stop")
        return

    if finished_known_full_sweep:
        known_map_primary_sweep_completed = True
        try:
            known_map_primary_sweep_finish_time = float(robot.getTime())
        except Exception:
            known_map_primary_sweep_finish_time = 0.0
        known_map_primary_sweep_finish_coverage = float(last_coverage_percent)
        known_map_residual_policy_status = f"residualPolicy=primary done cov={last_coverage_percent:.1f}%"
    desired_grid_heading = strict_world_grid_heading(pose_theta)
    last_route_commit_debug = f"finish: {reason[:48]}"
    route_abort_reason = "none"
    abort_route_commit(f"finished: {reason}")
    route_abort_reason = "none"
    last_route_commit_debug = f"finish: {reason[:48]}"
    if known_map_coverage_eval_active():
        # In the known-map evaluation mode the drawn route is the experiment.
        # Do not hand motion back to ROW_FORWARD between route commits, otherwise
        # the robot drives somewhere different from the displayed plan.
        try:
            update_coverage_objective()
        except Exception:
            pass
        # Do not bake the dock leg into the long coverage sweep: that was exactly
        # the kind of route-budget coupling that previously produced edge-case
        # commit failures.  After a sweep chunk finishes, decide whether the
        # remaining work is meaningful; if not, hand over to the dedicated dock
        # ROUTE_COMMIT planner.
        try:
            go_home, home_reason = return_home_should_start()
            if go_home:
                start_return_to_dock("after known-map sweep: " + str(home_reason)[:50])
                return
            dock_return_status = "idle: " + str(home_reason)[:50]
        except Exception:
            pass
        coverage_status = f"known-map route finished; replanning {reason[:34]}"
        return
    start_forward_row(f"cell fill after route commit: {reason}", post_lane_lock=False)


def learned_map_arena_exploration_gate():
    """Check that the learned map has observed enough of the room extent.

    This prevents the robot from returning to dock while a reachable bottom band
    is still mostly gray.  The gate is deliberately mild: it is not a known-map
    seed, just a sanity check over the debug arena bounds already used for the
    project visualization.
    """
    if not AUTO_MAP_COMPLETE_ARENA_GATE_ENABLED or not DEBUG_ARENA_BOUNDS_ENABLED:
        return True, "arenaGate=off"
    try:
        actual, _center_no_go, cleanable_floor = build_planning_layers(force=False)
        known = (log_odds < -LO_UNKNOWN_EPS) | (log_odds > LO_OCCUPIED_EPS) | actual.astype(np.bool_)
        if OBSTACLE_HYPOTHESIS_ENABLED:
            known |= hypothesis_obstacle_mask(False)
        x0, y_top = world_to_map(DEBUG_ARENA_X_MIN_M, DEBUG_ARENA_Y_MAX_M)
        x1, y_bottom = world_to_map(DEBUG_ARENA_X_MAX_M, DEBUG_ARENA_Y_MIN_M)
        x0, x1 = sorted((int(x0), int(x1)))
        y_top, y_bottom = sorted((int(y_top), int(y_bottom)))
        pad = max(1, int(round(0.18 * MAP_SCALE)))
        x0 = clamp(x0 + pad, 0, MAP_SIZE - 1)
        x1 = clamp(x1 - pad, 0, MAP_SIZE - 1)
        y_top = clamp(y_top + pad, 0, MAP_SIZE - 1)
        y_bottom = clamp(y_bottom - pad, 0, MAP_SIZE - 1)
        if x1 <= x0 + 8 or y_bottom <= y_top + 8:
            return True, "arenaGate=bad-rect"
        roi = known[y_top:y_bottom + 1, x0:x1 + 1]
        area = int(roi.size)
        known_ratio = float(np.count_nonzero(roi)) / float(max(1, area))
        band_px = max(4, int(round(float(AUTO_MAP_COMPLETE_BOTTOM_BAND_M) * MAP_SCALE)))
        by0 = max(y_top, y_bottom - band_px)
        bottom_roi = known[by0:y_bottom + 1, x0:x1 + 1]
        bottom_ratio = float(np.count_nonzero(bottom_roi)) / float(max(1, int(bottom_roi.size)))
        ok = bool(
            known_ratio >= float(AUTO_MAP_COMPLETE_MIN_ARENA_KNOWN_RATIO)
            and bottom_ratio >= float(AUTO_MAP_COMPLETE_MIN_BOTTOM_BAND_KNOWN_RATIO)
        )
        return ok, f"arenaGate=known {known_ratio:.2f} bottom {bottom_ratio:.2f}"
    except Exception as exc:
        return True, f"arenaGate=err {type(exc).__name__}"


def learned_map_strong_static_filter(obs_mask, free_mask):
    """Remove one-frame/ray-like obstacle noise from the final learned K map.

    The online map may keep weak dots so we can inspect them during EXPLORE.  Once
    the robot is back at dock, K-clean needs a stable static map: confirmed walls,
    compact objects, contact marks and strong hypothesis objects survive; isolated
    diagonal streaks and salt-and-pepper dots are downgraded to free/unknown.
    """
    if not LEARNED_MAP_SANITIZE_STRONG_NOISE_CLEANUP:
        return obs_mask.astype(np.bool_), 0, 0
    try:
        obs = obs_mask.astype(np.bool_).copy()
        free = free_mask.astype(np.bool_)
        contact_occ = contact_log_odds > CONTACT_OCCUPIED_EPS
        structural_strong = (structural_log_odds > STRUCTURAL_OCCUPIED_EPS) if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(obs, dtype=np.bool_)
        visual_strong = visual_log_odds > (CV_DISPLAY_DENSE_EPS + 0.35)
        hyp_strong = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(obs, dtype=np.bool_)
        thin_confirmed = thin_obstacle_log_odds > THIN_OBSTACLE_CONFIRM_EPS if THIN_OBSTACLE_CONFIRM_ENABLED else np.zeros_like(obs, dtype=np.bool_)
        keep = np.zeros_like(obs, dtype=np.bool_)
        removed_cells = 0
        comps = 0
        n, labels, stats, _cent = cv2.connectedComponentsWithStats(obs.astype(np.uint8), 8)
        for cid in range(1, int(n)):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            comps += 1
            left = int(stats[cid, cv2.CC_STAT_LEFT])
            top = int(stats[cid, cv2.CC_STAT_TOP])
            width = int(stats[cid, cv2.CC_STAT_WIDTH])
            height = int(stats[cid, cv2.CC_STAT_HEIGHT])
            span = max(width, height)
            narrow = min(width, height)
            density = float(area) / float(max(1, width * height))
            comp = labels == cid
            touches_arena_border = False
            if DEBUG_ARENA_BOUNDS_ENABLED:
                try:
                    ax0, ayt = world_to_map(DEBUG_ARENA_X_MIN_M, DEBUG_ARENA_Y_MAX_M)
                    ax1, ayb = world_to_map(DEBUG_ARENA_X_MAX_M, DEBUG_ARENA_Y_MIN_M)
                    ax0, ax1 = sorted((int(ax0), int(ax1)))
                    ayt, ayb = sorted((int(ayt), int(ayb)))
                    border_pad = max(2, int(round(0.08 * MAP_SCALE)))
                    touches_arena_border = bool(
                        left <= ax0 + border_pad or top <= ayt + border_pad
                        or left + width >= ax1 - border_pad or top + height >= ayb - border_pad
                    )
                except Exception:
                    touches_arena_border = False
            protected = bool(np.any(comp & (contact_occ | structural_strong | visual_strong | hyp_strong | thin_confirmed)))
            wall_like = bool(touches_arena_border or span >= RAW_MAP_RAY_STREAK_WALL_PROTECT_SPAN_PX or area >= max(420, LEARNED_MAP_SANITIZE_RAY_MAX_AREA_PX * 2))
            ray_like = bool(
                area <= int(LEARNED_MAP_SANITIZE_RAY_MAX_AREA_PX)
                and span >= int(LEARNED_MAP_SANITIZE_RAY_MIN_SPAN_PX)
                and (density <= float(LEARNED_MAP_SANITIZE_RAY_MAX_DENSITY) or narrow <= 2)
            )
            tiny = bool(area < int(LEARNED_MAP_SANITIZE_MIN_OBS_AREA_PX) and span < int(LEARNED_MAP_SANITIZE_MIN_OBS_SPAN_PX))
            if (tiny or (ray_like and not wall_like)) and not protected:
                obs[comp] = False
                removed_cells += area
            else:
                keep[comp] = True
        # Close tiny cracks in surviving static objects so K-planner sees simple blobs, not dotted contours.
        close_r = max(0, int(round(float(LEARNED_MAP_SANITIZE_DENSE_CLOSE_M) * MAP_SCALE)))
        if close_r > 0 and np.any(obs):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
            closed = cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0
            # Do not close over strong free floor corridors.
            strong_free = log_odds < float(LEARNED_MAP_SANITIZE_FREE_LO)
            obs = closed & (~strong_free | obs)
        return obs.astype(np.bool_), removed_cells, comps
    except Exception:
        return obs_mask.astype(np.bool_), 0, 0


def clear_explore_visual_routes_for_learned_k():
    """Clear exploration path overlays before the learned-map K pass.

    The map itself is preserved; only old trajectory/route debug polylines and
    frontier targets are reset so the K-clean view starts from a clean dock state.
    """
    global trajectory, coverage_goal_map, coverage_goal_world, coverage_goal_kind
    global coverage_route_map, coverage_route_world, coverage_route_waypoint_map, coverage_route_waypoint_world
    global coverage_route_cost, coverage_route_score, coverage_route_len, coverage_route_kind, coverage_route_status
    global route_commit_active, route_commit_target_map, route_commit_target_world, route_commit_kind
    global route_commit_route_map, route_commit_route_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_reason, route_abort_reason, last_route_commit_debug
    trajectory = []
    coverage_goal_map = None
    coverage_goal_world = None
    coverage_goal_kind = "none"
    coverage_route_map = []
    coverage_route_world = []
    coverage_route_waypoint_map = None
    coverage_route_waypoint_world = None
    coverage_route_cost = float("inf")
    coverage_route_score = float("-inf")
    coverage_route_len = 0
    coverage_route_kind = "none"
    coverage_route_status = "cleared for learned K"
    route_commit_active = False
    route_commit_target_map = None
    route_commit_target_world = None
    route_commit_kind = "none"
    route_commit_route_map = []
    route_commit_route_world = []
    route_commit_waypoint_map = None
    route_commit_waypoint_world = None
    route_commit_reason = "cleared for learned K"
    route_abort_reason = "cleared"
    last_route_commit_debug = "cleared for learned K"

def learned_map_ready_to_clean():
    """Return True when EXPLORE has built enough map to start the dock->K phase.

    This is a map-sufficiency gate, not a coverage-complete gate. Coverage may
    still be only about half of the final objective because the first run is
    mapping; the second learned-map K pass will clean systematically from dock.
    """
    global auto_map_ready_best_coverage, auto_map_ready_best_time, auto_map_ready_debug
    if not AUTO_LEARNED_MAP_CLEANING_ENABLED:
        auto_map_ready_debug = "autoMap=off"
        return False, auto_map_ready_debug
    if auto_map_cleaning_started or auto_map_return_to_dock_active or dock_return_active or dock_return_completed:
        auto_map_ready_debug = f"autoMap=busy phase={auto_map_mission_phase} dock={dock_return_status[:26]}"
        return False, auto_map_ready_debug
    if known_map_coverage_eval_active():
        auto_map_ready_debug = "autoMap=skip K active"
        return False, auto_map_ready_debug
    try:
        now = float(robot.getTime())
    except Exception:
        now = 0.0
    if now < float(AUTO_MAP_COMPLETE_MIN_TIME_SEC):
        auto_map_ready_debug = f"autoMap=wait time {now:.0f}/{AUTO_MAP_COMPLETE_MIN_TIME_SEC:.0f}s"
        return False, auto_map_ready_debug
    try:
        cov = float(last_coverage_percent or 0.0)
        total = int(last_coverage_total_cells or 0)
        frontiers = int(last_frontier_cells or 0)
        gray_cells = int(last_gray_gap_cells or 0)
        gray_comps = int(last_gray_gap_components or 0)
        frontier_ratio = float(frontiers) / float(max(1, total))
        frontier_or_gray_ratio = float(frontiers + gray_cells) / float(max(1, total))
    except Exception:
        auto_map_ready_debug = "autoMap=bad counters"
        return False, auto_map_ready_debug

    if auto_map_ready_best_time < -900.0 or cov > float(auto_map_ready_best_coverage) + float(AUTO_MAP_COMPLETE_PLATEAU_GAIN_PERCENT):
        auto_map_ready_best_coverage = cov
        auto_map_ready_best_time = now
    plateau_sec = max(0.0, now - float(auto_map_ready_best_time))

    if AUTO_MAP_COMPLETE_REQUIRE_FORWARD_STATE:
        try:
            stable_state = bool(nav_state == NAV_FORWARD and simple_sweep_state in ("MOVE_STRAIGHT", "INIT", "ESCAPE_FORWARD"))
        except Exception:
            stable_state = bool(nav_state == NAV_FORWARD)
        if not stable_state:
            auto_map_ready_debug = f"autoMap=wait stable nav={nav_state} sweep={simple_sweep_state}"
            return False, auto_map_ready_debug

    try:
        obstacles = int(np.count_nonzero(actual_physical_obstacle_mask()))
    except Exception:
        obstacles = int(np.count_nonzero(log_odds > LO_OCCUPIED_EPS))

    if total < int(AUTO_MAP_COMPLETE_MIN_CLEANABLE_CELLS):
        auto_map_ready_debug = f"autoMap=wait area {total}/{AUTO_MAP_COMPLETE_MIN_CLEANABLE_CELLS}"
        return False, auto_map_ready_debug
    if cov < float(AUTO_MAP_COMPLETE_MIN_COVERAGE_PERCENT):
        auto_map_ready_debug = f"autoMap=wait cov {cov:.1f}/{AUTO_MAP_COMPLETE_MIN_COVERAGE_PERCENT:.1f}"
        return False, auto_map_ready_debug
    if obstacles < int(AUTO_MAP_COMPLETE_MIN_OBSTACLE_CELLS):
        auto_map_ready_debug = f"autoMap=wait obs {obstacles}/{AUTO_MAP_COMPLETE_MIN_OBSTACLE_CELLS}"
        return False, auto_map_ready_debug

    gray_ok = bool(
        gray_cells <= int(AUTO_MAP_COMPLETE_MAX_GRAY_GAP_CELLS)
        and gray_comps <= int(AUTO_MAP_COMPLETE_MAX_GRAY_GAP_COMPONENTS)
    )
    if not gray_ok:
        auto_map_ready_debug = f"autoMap=wait gray {gray_cells}/{gray_comps} {last_gray_gap_debug[:28]}"
        return False, auto_map_ready_debug

    arena_ok, arena_dbg = learned_map_arena_exploration_gate()
    if not arena_ok:
        auto_map_ready_debug = f"autoMap=wait {arena_dbg}"
        return False, auto_map_ready_debug

    useful_route_pending = bool(
        coverage_goal_kind == "frontier"
        and coverage_route_kind == "frontier"
        and coverage_goal_map is not None
        and coverage_route_map
        and math.isfinite(float(coverage_route_cost))
        and not route_target_is_blacklisted(int(coverage_goal_map[0]), int(coverage_goal_map[1]), "frontier")
    )
    frontier_ok = bool(
        (not useful_route_pending)
        and (
            frontier_ratio <= float(AUTO_MAP_COMPLETE_MAX_FRONTIER_RATIO)
            or frontiers <= int(AUTO_MAP_COMPLETE_MAX_FRONTIER_CELLS)
            or (frontier_or_gray_ratio <= float(AUTO_MAP_COMPLETE_FRONTIER_OR_GRAY_RATIO) and plateau_sec >= float(AUTO_MAP_COMPLETE_NO_ROUTE_OK_SEC))
        )
    )
    if not frontier_ok:
        route_tag = "route=pending" if useful_route_pending else "route=none/blacklisted"
        auto_map_ready_debug = f"autoMap=wait frontier {frontiers} ratio={frontier_ratio:.3f} gray={gray_cells} {route_tag}"
        return False, auto_map_ready_debug

    plateau_ok = bool(
        plateau_sec >= float(AUTO_MAP_COMPLETE_PLATEAU_SEC)
        or cov >= float(AUTO_MAP_COMPLETE_STRONG_COVERAGE_PERCENT)
    )
    if not plateau_ok:
        auto_map_ready_debug = f"autoMap=wait plateau {plateau_sec:.0f}/{AUTO_MAP_COMPLETE_PLATEAU_SEC:.0f}s cov={cov:.1f}"
        return False, auto_map_ready_debug

    auto_map_ready_debug = (
        f"autoMap=ready cov={cov:.1f} t={now:.0f}s fr={frontiers}/{frontier_ratio:.3f} "
        f"gray={gray_cells}/{gray_comps} area={total} obs={obstacles} plateau={plateau_sec:.0f}s"
    )
    return True, auto_map_ready_debug


def sanitize_learned_map_for_k_cleaning(reason="map complete"):
    """Freeze/polish the learned RGB-D map before K-mode coverage.

    The raw debug map remains visible in saved screenshots, but K-mode should not
    plan on one-frame speckles or inflated purple margins as if they were walls.
    This writes a conservative polished occupancy layer: actual obstacle core is
    stable occupied, learned cleanable floor is stable free, and weak unconfirmed
    obstacle noise outside those masks is decayed.
    """
    global learned_map_sanitized_once, learned_map_sanitize_debug
    global log_odds, visual_log_odds, thin_obstacle_log_odds, contact_log_odds, structural_log_odds, hypothesis_obstacle_log_odds
    try:
        invalidate_heavy_map_caches("learned-map sanitize start")
        try:
            # Freeze K-mode on a polished learned map, not on one-frame streaks.
            cleanup_raw_map_speckles()
            cleanup_raw_map_speckles()
        except Exception:
            pass
        actual, _center_no_go, cleanable_floor = build_planning_layers(force=True)
        cleaned0 = np.zeros_like(cleaned_mask, dtype=np.bool_)
        uncleaned0 = cleanable_floor.astype(np.bool_) & (~actual.astype(np.bool_))
        unknown0 = (~cleanable_floor.astype(np.bool_)) & (~actual.astype(np.bool_))
        try:
            actual2, cleanable2, _cleaned2, _uncleaned2, _unknown2 = filter_objective_noise_masks(
                actual.astype(np.bool_),
                cleanable_floor.astype(np.bool_),
                cleaned0,
                uncleaned0,
                unknown0,
            )
        except Exception:
            actual2 = actual.astype(np.bool_)
            cleanable2 = cleanable_floor.astype(np.bool_) & (~actual2)

        dock_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
        dmx, dmy = world_to_map(DOCK_TARGET_X, DOCK_TARGET_Y)
        if map_inside(dmx, dmy):
            cv2.circle(
                dock_mask,
                (int(dmx), int(dmy)),
                max(2, int(round(float(LEARNED_MAP_DOCK_CLEAR_RADIUS_M) * MAP_SCALE))),
                255,
                -1,
            )
        dock_free = dock_mask > 0
        actual2 = actual2 & (~dock_free)
        cleanable2 = (cleanable2 | dock_free) & (~actual2)

        free_mask = cleanable2 & (~actual2)
        obs_mask = actual2.astype(np.bool_)
        obs_mask, sanitized_removed_noise, sanitized_components = learned_map_strong_static_filter(obs_mask, free_mask)
        cleanable2 = (cleanable2 | (actual2 & (~obs_mask))) & (~obs_mask)
        free_mask = cleanable2 & (~obs_mask)
        if np.any(free_mask):
            log_odds[free_mask] = np.minimum(log_odds[free_mask], float(LEARNED_MAP_SANITIZE_FREE_LO))
            visual_log_odds[free_mask] = np.minimum(visual_log_odds[free_mask], CV_DISPLAY_LIGHT_EPS * 0.15)
            thin_obstacle_log_odds[free_mask] = np.minimum(thin_obstacle_log_odds[free_mask], 0.0)
            contact_log_odds[free_mask] = np.minimum(contact_log_odds[free_mask], 0.0)
            if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
                structural_log_odds[free_mask] = np.minimum(structural_log_odds[free_mask], STRUCTURAL_MIN)
            if OBSTACLE_HYPOTHESIS_ENABLED:
                hypothesis_obstacle_log_odds[free_mask] = 0.0
        if np.any(obs_mask):
            log_odds[obs_mask] = np.maximum(log_odds[obs_mask], float(LEARNED_MAP_SANITIZE_OCC_LO))
            visual_log_odds[obs_mask] = np.maximum(visual_log_odds[obs_mask], float(LEARNED_MAP_SANITIZE_VISUAL_OCC))
            if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
                structural_log_odds[obs_mask] = np.maximum(structural_log_odds[obs_mask], float(LEARNED_MAP_SANITIZE_STRUCTURAL_OCC))
            # If orange hypotheses survived into the sanitized actual obstacle
            # mask, keep them as planning no-go but do not turn them into contact.
            if OBSTACLE_HYPOTHESIS_ENABLED:
                hypothesis_obstacle_log_odds[obs_mask & (hypothesis_obstacle_log_odds > OBSTACLE_HYPOTHESIS_OCC_EPS)] = np.maximum(
                    hypothesis_obstacle_log_odds[obs_mask & (hypothesis_obstacle_log_odds > OBSTACLE_HYPOTHESIS_OCC_EPS)],
                    float(OBSTACLE_HYPOTHESIS_OCC_EPS) + 0.35,
                )

        if LEARNED_MAP_SANITIZE_REMOVE_WEAK_OUTSIDE:
            weak_outside = (log_odds > LO_OCCUPIED_EPS) & (~obs_mask) & (~dock_free)
            if np.any(weak_outside):
                log_odds[weak_outside] = np.minimum(log_odds[weak_outside], 0.0)
                visual_log_odds[weak_outside] = np.minimum(visual_log_odds[weak_outside], CV_DISPLAY_LIGHT_EPS * 0.20)
                thin_obstacle_log_odds[weak_outside] = 0.0
                if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
                    structural_log_odds[weak_outside] = np.minimum(structural_log_odds[weak_outside], STRUCTURAL_MIN)
                if OBSTACLE_HYPOTHESIS_ENABLED:
                    hypothesis_obstacle_log_odds[weak_outside] = np.minimum(hypothesis_obstacle_log_odds[weak_outside], OBSTACLE_HYPOTHESIS_OCC_EPS * 0.5)

        # K cleaning is a second run from the dock on the learned map.
        cleaned_mask[:, :] = 0
        recent_visit_log_odds[:, :] = 0.0
        clear_explore_visual_routes_for_learned_k()
        invalidate_heavy_map_caches("learned-map sanitized")
        learned_map_sanitized_once = True
        learned_map_sanitize_debug = (
            f"learnedMap=sanitized obs={int(np.count_nonzero(obs_mask))} "
            f"free={int(np.count_nonzero(free_mask))} rmNoise={int(sanitized_removed_noise)} "
            f"dockClear={int(np.count_nonzero(dock_free))} routeClear=1 reason={str(reason)[:24]}"
        )
        return True
    except Exception as exc:
        learned_map_sanitized_once = False
        learned_map_sanitize_debug = f"learnedMap=sanitize err {type(exc).__name__}"
        return False


def enable_learned_map_coverage_cleaning(reason="returned to dock"):
    """Start K-mode on the sanitized learned map without seeding WBT geometry."""
    global known_map_eval_runtime_enabled, known_map_eval_status, nav_action_queue
    global auto_map_mission_phase, auto_map_return_to_dock_active, auto_map_cleaning_started
    global dock_return_active, dock_return_completed, dock_return_status, dock_return_reason
    global planner_mode, planner_intent, planner_intent_reason, coverage_status
    global known_map_primary_sweep_completed, known_map_primary_sweep_finish_time, known_map_primary_sweep_finish_coverage
    global known_map_residual_cleanup_commits_started, known_map_residual_cleanup_started_at, known_map_residual_policy_status
    global desired_grid_heading, nav_state, map_freeze_until
    ok = sanitize_learned_map_for_k_cleaning(reason)
    known_map_eval_runtime_enabled = True
    known_map_primary_sweep_completed = False
    known_map_primary_sweep_finish_time = -999.0
    known_map_primary_sweep_finish_coverage = 0.0
    known_map_residual_cleanup_commits_started = 0
    known_map_residual_cleanup_started_at = -999.0
    known_map_residual_policy_status = "residualPolicy=learned primary"
    auto_map_mission_phase = "LEARNED_K_CLEAN"
    auto_map_return_to_dock_active = False
    auto_map_cleaning_started = True
    dock_return_active = False
    dock_return_completed = False
    dock_return_reason = "none"
    dock_return_status = "idle: learned-map K cleaning"
    nav_action_queue = []
    desired_grid_heading = DOCK_FINAL_HEADING
    nav_state = NAV_FORWARD
    map_freeze_until = max(map_freeze_until, robot.getTime() + float(LEARNED_MAP_K_START_HOLD_SEC))
    abort_route_commit("learned-map K start")
    try:
        release_control("learned-map K start")
    except Exception:
        pass
    planner_mode = "K_LEARNED_MAP_PLANNER"
    planner_intent = PLANNER_INTENT_CLEAN_KNOWN
    planner_intent_reason = "learned map sanitized after dock return"
    known_map_eval_status = "knownMap=learned sanitized" if ok else "knownMap=learned sanitize-warning"
    coverage_status = f"start learned-map K cleaning: {learned_map_sanitize_debug[:72]}"
    try:
        update_coverage_objective()
    except Exception:
        pass
    refresh_navigation_phase()
    print("Learned-map K cleaning enabled:", learned_map_sanitize_debug)
    return ok


def complete_map_return_to_dock(reason="map locked at dock"):
    """Complete the intermediate dock stop and immediately switch to K cleaning."""
    global dock_return_active, dock_return_completed, dock_return_status, dock_return_reason, coverage_status
    dock_return_active = False
    dock_return_completed = False
    dock_return_reason = str(reason or "map dock reached")
    dock_return_status = f"dock reached for learned K: {dock_return_reason[:38]}"
    coverage_status = dock_return_status
    hard_stop_motors()
    return enable_learned_map_coverage_cleaning(reason=dock_return_reason)


def start_map_complete_return_to_dock(reason="learned map ready"):
    """Finish EXPLORE, return to dock, then start learned-map K cleaning."""
    global auto_map_mission_phase, auto_map_return_to_dock_active, auto_map_ready_debug
    global simple_sweep_completed, simple_sweep_completion_reason, simple_sweep_last_debug
    global coverage_status, planner_mode, planner_intent, planner_intent_reason
    if not AUTO_LEARNED_MAP_CLEANING_ENABLED:
        return False
    if auto_map_cleaning_started or auto_map_return_to_dock_active or dock_return_active or dock_return_completed:
        return False
    auto_map_mission_phase = "RETURN_TO_DOCK_FOR_K"
    auto_map_return_to_dock_active = True
    auto_map_ready_debug = "autoMap=returnDock " + str(reason)[:70]
    simple_sweep_completed = True
    simple_sweep_completion_reason = "map ready -> dock -> learned K: " + str(reason)[:80]
    simple_sweep_last_debug = "exploreSweep=handoff " + str(reason)[:80]
    planner_mode = "RETURN_DOCK_FOR_LEARNED_K"
    planner_intent = PLANNER_INTENT_FINISH_CLEANUP
    planner_intent_reason = "map ready, return to dock before K cleaning"
    coverage_status = "return dock before learned-map cleaning: " + str(reason)[:72]
    try:
        if str(control_lock.owner) == SIMPLE_SWEEP_OWNER:
            release_control("map ready return to dock")
    except Exception:
        pass
    if math.hypot(pose_x - DOCK_TARGET_X, pose_y - DOCK_TARGET_Y) <= DOCK_TARGET_REACHED_M:
        complete_map_return_to_dock("already at dock: " + str(reason)[:48])
        return True
    return start_return_to_dock("map ready -> learned K: " + str(reason)[:64])


def return_home_should_start():
    """Decide when the mission should stop chasing leftovers and go home."""
    if not RETURN_HOME_ENABLED:
        return False, "disabled"
    if auto_map_return_to_dock_active:
        return True, f"intermediate dock for learned K: {auto_map_ready_debug[:48]}"
    if dock_return_completed or dock_return_active:
        return False, "completed/active"
    try:
        now = robot.getTime()
    except Exception:
        now = 0.0
    if now < DOCK_RETURN_MIN_TIME_SEC:
        return False, f"time {now:.0f}/{DOCK_RETURN_MIN_TIME_SEC:.0f}s"
    if now - dock_return_last_start_time < DOCK_ROUTE_START_COOLDOWN_SEC:
        return False, "dock cooldown"

    if known_map_coverage_eval_active():
        try:
            stop_residual, residual_reason = known_map_residual_cleanup_should_stop(now=now)
            if stop_residual:
                return True, residual_reason
        except Exception:
            pass

    # Normal mission end: high coverage and no coherent uncleaned component worth
    # committing to.  Frontiers alone are not enough reason to keep cleaning once
    # the reachable floor is essentially done.
    high_done = bool(last_coverage_percent >= DOCK_RETURN_COVERAGE_PERCENT)
    no_useful_uncleaned_route = bool(
        coverage_route_kind != "uncleaned"
        or coverage_route_component_cells <= 2
        or not math.isfinite(coverage_route_cost)
    )
    if high_done and no_useful_uncleaned_route:
        return True, f"coverage done {last_coverage_percent:.1f}% route={coverage_route_kind} comp={coverage_route_component_cells}"

    # Late-corner escape: after almost everything is cleaned, repeated recovery in
    # a corner is less useful than returning to dock.  Do not pre-empt a fresh
    # safety action; this starts only after the recovery has been going on for a
    # while or after it hands control back to NAV_FORWARD.
    late = bool(last_coverage_percent >= DOCK_STUCK_RETURN_COVERAGE_PERCENT)
    recovery_long = bool(nav_state in CONTACT_RECOVERY_STATES and (now - contact_recovery_start_time) >= DOCK_STUCK_RECOVERY_SEC)
    if late and recovery_long:
        return True, f"late corner recovery {now - contact_recovery_start_time:.1f}s cov={last_coverage_percent:.1f}%"

    return False, f"cov={last_coverage_percent:.1f}% route={coverage_route_kind} comp={coverage_route_component_cells}"


def start_return_to_dock(reason="mission complete"):
    """Create a ROUTE_COMMIT to the odometry-origin dock pose."""
    global dock_return_active, dock_return_completed, dock_return_reason, dock_return_status
    global dock_return_last_plan_time, dock_return_last_start_time, dock_route_cost
    global route_commit_active, route_commit_target_map, route_commit_target_world, route_commit_kind
    global route_commit_route_map, route_commit_route_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_progress_idx, route_commit_waypoint_idx, route_commit_waypoint_is_corner
    global route_commit_cost, route_commit_score, route_commit_component_id, route_commit_component_cells
    global route_commit_wall_strip_bonus, route_commit_missed_strip_bonus, route_commit_segment_gain, route_commit_footprint_gain_cells
    global route_commit_first_turn_frac, route_commit_corner_count, route_commit_geometry_debug
    global route_commit_started_at, route_commit_reason, route_abort_reason, last_route_commit_debug, planner_mode, coverage_status

    now = robot.getTime()
    if dock_return_completed:
        return True
    if math.hypot(pose_x - DOCK_TARGET_X, pose_y - DOCK_TARGET_Y) <= DOCK_TARGET_REACHED_M:
        if auto_map_return_to_dock_active and not auto_map_cleaning_started:
            return complete_map_return_to_dock(str(reason or "already at dock"))
        dock_return_active = False
        dock_return_completed = True
        dock_return_reason = str(reason or "already at dock")
        dock_return_status = f"docked: {dock_return_reason[:48]}"
        coverage_status = dock_return_status
        hard_stop_motors()
        acquire_control(ControlOwner.PLANNER, DOCK_STOP_HOLD_OWNER_SEC, 0.0, "docked stop")
        return True
    if now - dock_return_last_plan_time < DOCK_ROUTE_REPLAN_INTERVAL_SEC and not dock_return_active:
        dock_return_status = f"dock replan wait: {reason[:36]}"
        return False
    dock_return_last_plan_time = now

    obstacles, cleanable, _cleaned, _uncleaned, _unknown = compute_coverage_masks()
    route = plan_route_to_world_goal((DOCK_TARGET_X, DOCK_TARGET_Y), obstacles, cleanable)
    if route is None:
        dock_return_status = f"dock route failed: {reason[:44]}"
        route_abort_reason = dock_return_status
        return False
    if float(route.get("cost", float("inf"))) > DOCK_RETURN_MAX_ROUTE_COST_M:
        dock_return_status = f"dock route too long {route.get('cost', float('inf')):.2f}m"
        route_abort_reason = dock_return_status
        return False

    dock_return_active = True
    dock_return_reason = str(reason or "return home")
    dock_return_last_start_time = now
    dock_route_cost = float(route.get("cost", float("inf")))

    route_commit_active = True
    route_commit_target_map = route["goal_map"]
    route_commit_target_world = route["goal_world"]
    route_commit_kind = "dock"
    route_commit_route_map = list(route["route_map"])
    route_commit_route_world = list(route["route_world"])
    route_commit_waypoint_map = route.get("waypoint_map")
    route_commit_waypoint_world = route.get("waypoint_world")
    route_commit_progress_idx = 0
    route_commit_waypoint_idx = 0
    route_commit_waypoint_is_corner = False
    route_commit_cost = dock_route_cost
    route_commit_score = 0.0
    route_commit_component_id = -1
    route_commit_component_cells = 0
    route_commit_wall_strip_bonus = 0.0
    route_commit_missed_strip_bonus = 0.0
    route_commit_segment_gain = 0
    route_commit_footprint_gain_cells = 0
    route_commit_started_at = now
    route_commit_reason = "return_to_dock: " + dock_return_reason[:48]
    route_abort_reason = "none"
    planner_mode = "RETURN_HOME"
    dock_return_status = f"returning dock cost={dock_route_cost:.2f} len={route.get('length', 0)} {dock_return_reason[:34]}"
    coverage_status = dock_return_status
    acquire_control(ControlOwner.ROUTE_COMMIT, DOCK_ROUTE_MAX_AGE_SEC, 0.0, f"return dock: {dock_return_reason[:50]}")
    last_route_commit_debug = dock_return_status
    return True


def frontier_single_point_route_is_committable():
    """Allow a near one-cell frontier route to become a bounded direct commit.

    The wavefront planner can legitimately return len=1 when the selected
    viewpoint is already in the robot's local free component.  /116 treated
    that as "route too short", then the frontier-only gate blocked ROW_FORWARD,
    producing owner=NONE/active=none/STOP even though the target was about half a
    meter away.  This helper accepts only short, mostly-forward frontier targets;
    unsafe segments are still rejected by the normal depth/body guards and by the
    map-line blocked-ratio check inside route_commit_speeds().
    """
    if not EXPLORATION_FRONTIER_SINGLE_POINT_COMMIT_ENABLED:
        return False
    if coverage_goal_kind != "frontier" or coverage_route_kind != "frontier":
        return False
    try:
        route_len = len(coverage_route_map or [])
        cost = float(coverage_route_cost)
        straight = float(coverage_route_straight_dist)
        turn = float(coverage_route_turn_need)
    except Exception:
        return False
    if route_len != 1 or not math.isfinite(cost):
        return False
    return bool(
        cost <= float(EXPLORATION_FRONTIER_SINGLE_POINT_MAX_COST_M)
        and straight <= float(EXPLORATION_FRONTIER_SINGLE_POINT_MAX_STRAIGHT_M)
        and turn <= float(EXPLORATION_FRONTIER_SINGLE_POINT_MAX_TURN_FRAC)
    )


def route_candidate_basic_block():
    return policy_first_block([
        (lambda: coverage_goal_map is None or coverage_goal_world is None, "no candidate"),
        (lambda: not math.isfinite(coverage_route_cost), "route cost inf"),
        (
            lambda: not coverage_route_map or (
                len(coverage_route_map) < ROUTE_COMMIT_MIN_ROUTE_LEN
                and not frontier_single_point_route_is_committable()
            ),
            lambda: f"route too short len={len(coverage_route_map or [])}",
        ),
    ])


def frontier_route_commit_policy():
    if coverage_goal_kind != "frontier" or coverage_route_kind != "frontier":
        return policy_deny(f"mixed frontier kind={coverage_goal_kind}/{coverage_route_kind}")
    if not EXPLORE_FRONTIER_COMMIT_ENABLED:
        return policy_deny("frontier commit disabled")
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return policy_deny(f"frontier not priority intent={planner_intent} fr={last_frontier_cells} cov={last_coverage_percent:.1f}")

    if frontier_single_point_route_is_committable():
        return policy_allow(
            f"frontier single-point direct commit cost={coverage_route_cost:.2f} "
            f"d={coverage_route_straight_dist:.2f} turn={coverage_route_turn_need:.2f} fr={last_frontier_cells}"
        )

    blocked, reason = policy_first_block([
        (
            lambda: coverage_route_cost > EXPLORE_FRONTIER_MAX_ROUTE_COST_M,
            lambda: f"frontier route cost {coverage_route_cost:.2f}",
        ),
        (
            lambda: coverage_route_straight_dist > EXPLORE_FRONTIER_MAX_STRAIGHT_DIST_M,
            lambda: f"frontier dist {coverage_route_straight_dist:.2f}",
        ),
        (
            lambda: coverage_route_turn_need > EXPLORE_FRONTIER_MAX_TURN_FRAC,
            lambda: f"frontier turn {coverage_route_turn_need:.2f}",
        ),
        (
            lambda: (
                EXPLORATION_ROUTE_GEOMETRY_STABILITY_ENABLED
                and coverage_route_corner_count > EXPLORATION_FRONTIER_ROW_PRIMARY_MAX_CORNERS + 2
            ),
            lambda: f"frontier route too jagged {coverage_route_geometry_debug}",
        ),
        (
            lambda: (
                not map_mature
                and coverage_route_straight_dist < FRONTIER_COMMIT_MIN_TRANSLATION_M
                and coverage_route_turn_need > FRONTIER_COMMIT_CLOSE_SIDE_TURN_FRAC
            ),
            lambda: (
                f"frontier is close side-view, not route d={coverage_route_straight_dist:.2f} "
                f"turn={coverage_route_turn_need:.2f}"
            ),
        ),
        (
            lambda: coverage_route_score < EXPLORE_FRONTIER_MIN_SCORE,
            lambda: f"frontier score {coverage_route_score:.1f}",
        ),
    ])
    if blocked:
        return policy_deny(reason)
    return policy_allow(
        f"frontier route commit expand-map cost={coverage_route_cost:.2f} "
        f"score={coverage_route_score:.1f} fr={last_frontier_cells}"
    )


def uncleaned_route_components_ok():
    component_ok_short = bool(
        coverage_route_component_cells >= SHORT_ROUTE_COMMIT_MIN_COMPONENT_CELLS
        or coverage_route_continuity_bonus >= 1.2
    )
    component_ok_partial = bool(
        coverage_route_component_cells >= PARTIAL_ROUTE_COMMIT_MIN_COMPONENT_CELLS
        or coverage_route_continuity_bonus >= 1.6
    )
    return component_ok_short, component_ok_partial


def known_map_route_commit_policy():
    blocked, reason = policy_first_block([
        (
            lambda: coverage_route_cost > KNOWN_MAP_EVAL_ROUTE_COMMIT_MAX_COST_M,
            lambda: f"known-map route cost {coverage_route_cost:.2f}>{KNOWN_MAP_EVAL_ROUTE_COMMIT_MAX_COST_M:.1f}",
        ),
        (
            lambda: coverage_route_footprint_gain_cells <= 0 and coverage_route_component_cells <= 0,
            "known-map no useful footprint gain",
        ),
    ])
    if blocked:
        return policy_deny(reason)
    return policy_allow(
        f"known-map sweep route commit cost={coverage_route_cost:.2f} "
        f"score={coverage_route_score:.1f} comp={coverage_route_component_cells} "
        f"fp={coverage_route_footprint_gain_cells}"
    )


def expand_map_cleanup_block_reason():
    if planner_intent != PLANNER_INTENT_EXPAND_MAP:
        return ""
    if exploration_cleanup_locked():
        return "cleanup locked until exploration frontier closes"
    straight, longitudinal, lateral, turn_actual = route_goal_relative_to_pose(coverage_goal_world)
    local_inline = bool(
        coverage_route_cost <= EXPLORE_UNCLEANED_LOCAL_MAX_ROUTE_COST_M
        and straight <= EXPLORE_UNCLEANED_LOCAL_MAX_DIST_M
        and longitudinal >= EXPLORE_UNCLEANED_BACKTRACK_FORWARD_M
        and lateral <= EXPLORE_UNCLEANED_LOCAL_LATERAL_M
        and turn_actual <= EXPLORE_UNCLEANED_BACKTRACK_TURN_FRAC
    )
    important = bool(
        coverage_route_component_cells >= EXPLORE_UNCLEANED_IMPORTANT_COMP_CELLS
        or coverage_route_footprint_gain_cells >= EXPLORE_UNCLEANED_IMPORTANT_FOOTPRINT_GAIN
        or coverage_route_segment_gain >= EXPLORE_UNCLEANED_IMPORTANT_SEGMENT_GAIN
        or coverage_route_missed_strip_bonus >= EXPLORE_UNCLEANED_IMPORTANT_MISSED_BONUS
        or coverage_route_wall_strip_bonus >= ROUTE_COMMIT_WALL_STRIP_BONUS * 0.75
    )
    if local_inline or important:
        return ""
    return (
        f"expand-map blocks cleanup cost={coverage_route_cost:.2f} d={straight:.2f} "
        f"long={longitudinal:.2f} lat={lateral:.2f} comp={coverage_route_component_cells} "
        f"fp={coverage_route_footprint_gain_cells} seg={coverage_route_segment_gain}"
    )


def mature_route_commit_policy():
    blocked, reason = policy_first_block([
        (
            lambda: coverage_route_cost > ROUTE_COMMIT_MAX_ROUTE_COST_M,
            lambda: f"mature route cost {coverage_route_cost:.2f}",
        ),
        (
            lambda: last_coverage_percent < ROUTE_COMMIT_MIN_COVERAGE_PERCENT,
            lambda: f"mature coverage {last_coverage_percent:.1f}%",
        ),
    ])
    return policy_deny(reason) if blocked else policy_allow("mature uncleaned wavefront")


def partial_route_commit_policy(component_ok_partial):
    if not map_mature_soft:
        return False, ""
    if last_coverage_percent < PARTIAL_ROUTE_COMMIT_MIN_COVERAGE_PERCENT:
        return policy_deny(f"partial coverage {last_coverage_percent:.1f}%")
    if (
        coverage_route_cost <= PARTIAL_ROUTE_COMMIT_MAX_ROUTE_COST_M
        and coverage_route_straight_dist <= PARTIAL_ROUTE_COMMIT_MAX_STRAIGHT_DIST_M
        and coverage_route_turn_need <= PARTIAL_ROUTE_COMMIT_MAX_TURN_FRAC
        and coverage_route_score >= PARTIAL_ROUTE_COMMIT_MIN_SCORE
        and component_ok_partial
    ):
        return policy_allow(
            f"partial route commit cost={coverage_route_cost:.2f} "
            f"score={coverage_route_score:.1f} comp={coverage_route_component_cells} "
            f"cont={coverage_route_continuity_bonus:.1f}"
        )
    return False, ""


def short_route_commit_policy(component_ok_short):
    blocked, reason = policy_first_block([
        (
            lambda: last_coverage_percent < SHORT_ROUTE_COMMIT_MIN_COVERAGE_PERCENT,
            lambda: f"early coverage {last_coverage_percent:.1f}%",
        ),
        (
            lambda: coverage_route_cost > SHORT_ROUTE_COMMIT_MAX_ROUTE_COST_M,
            lambda: f"short route cost {coverage_route_cost:.2f}",
        ),
        (
            lambda: coverage_route_straight_dist > SHORT_ROUTE_COMMIT_MAX_STRAIGHT_DIST_M,
            lambda: f"short target dist {coverage_route_straight_dist:.2f}",
        ),
        (
            lambda: coverage_route_turn_need > SHORT_ROUTE_COMMIT_MAX_TURN_FRAC,
            lambda: f"short turn {coverage_route_turn_need:.2f}",
        ),
        (
            lambda: coverage_route_score < SHORT_ROUTE_COMMIT_MIN_SCORE,
            lambda: f"short score {coverage_route_score:.1f}",
        ),
        (
            lambda: not component_ok_short,
            lambda: f"short comp={coverage_route_component_cells} cont={coverage_route_continuity_bonus:.1f}",
        ),
    ])
    if blocked:
        return policy_deny(reason)
    return policy_allow(
        f"short route commit cost={coverage_route_cost:.2f} score={coverage_route_score:.1f} "
        f"comp={coverage_route_component_cells} cont={coverage_route_continuity_bonus:.1f}"
    )


def route_commit_candidate_policy():
    """Return (allowed, reason) for promoting candidateTarget to activeTarget.

    This is the ownership policy.  It deliberately separates goal
    selection from wheel ownership: Dijkstra may always produce candidates, but a
    candidate may pre-empt ROW_FORWARD only if it is mature-planned or a short,
    high-gain local component.
    """
    blocked, reason = route_candidate_basic_block()
    if blocked:
        return policy_deny(reason)

    if coverage_goal_kind == "frontier" or coverage_route_kind == "frontier":
        return frontier_route_commit_policy()

    if coverage_goal_kind != "uncleaned" or coverage_route_kind != "uncleaned":
        return policy_deny(f"candidate kind={coverage_goal_kind}/{coverage_route_kind}")

    if known_map_coverage_eval_active() and KNOWN_MAP_EVAL_FORCE_ROUTE_COMMIT:
        return known_map_route_commit_policy()

    cleanup_block_reason = expand_map_cleanup_block_reason()
    if cleanup_block_reason:
        return policy_deny(cleanup_block_reason)

    if map_mature:
        return mature_route_commit_policy()

    if not HYBRID_COVERAGE_ENABLED or not SHORT_ROUTE_COMMIT_ENABLED:
        return policy_deny(f"map not mature: {map_mature_reason}")

    component_ok_short, component_ok_partial = uncleaned_route_components_ok()
    partial_allowed, partial_reason = partial_route_commit_policy(component_ok_partial)
    if partial_allowed:
        return policy_allow(partial_reason)
    if partial_reason:
        return policy_deny(partial_reason)

    return short_route_commit_policy(component_ok_short)


def candidate_motion_guard_policy():
    blocked, reason = policy_first_block([
        (lambda: not PLANNED_COVERAGE_ENABLED, "disabled"),
        (lambda: nav_state != NAV_FORWARD, lambda: f"nav={nav_state}"),
        (lambda: under_furniture_active or nav_action_queue, "special manoeuvre active"),
        (lambda: last_bumper_left or last_bumper_center or last_bumper_right, "bumper active"),
        (
            lambda: (
                not known_map_coverage_eval_active()
                and robot.getTime() - route_commit_last_abort_time < ROUTE_COMMIT_REPLAN_COOLDOWN_SEC
            ),
            "abort cooldown",
        ),
    ])
    return policy_deny(reason) if blocked else policy_allow("motion guard ok")


def row_primary_frontier_should_hold_forward(front):
    if front is None:
        return False
    return bool(
        EXPLORATION_FRONTIER_ROW_PRIMARY_ENABLED
        and not simple_sweep_completed
        and coverage_goal_kind == "frontier"
        and planner_intent == PLANNER_INTENT_EXPAND_MAP
        and exploration_cleanup_locked()
        and float(front) >= EXPLORATION_FRONTIER_ROW_PRIMARY_FRONT_CLEAR_M
        and (
            coverage_route_first_turn_frac > EXPLORATION_FRONTIER_ROW_PRIMARY_MAX_FIRST_TURN_FRAC
            or coverage_route_corner_count > EXPLORATION_FRONTIER_ROW_PRIMARY_MAX_CORNERS
        )
    )


def candidate_safety_guard_policy(front, center, body_clearance):
    blocked, reason = policy_first_block([
        (
            lambda: row_primary_frontier_should_hold_forward(front),
            lambda: f"row-primary explore: keep forward, route {coverage_route_geometry_debug} front={float(front):.2f}",
        ),
        (
            lambda: front is not None and (
                front < ROUTE_COMMIT_FRONT_ABORT_M
                or (center is not None and center < ROUTE_COMMIT_CENTER_ABORT_M)
            ),
            "front contact risk",
        ),
        (
            lambda: body_clearance is not None and body_clearance < ROUTE_COMMIT_BODY_ABORT_M,
            "body contact risk",
        ),
        (lambda: not coverage_candidate_goal_still_useful(), "candidate disappeared"),
    ])
    return policy_deny(reason) if blocked else policy_allow("candidate safety ok")


def candidate_owner_policy(policy_reason):
    if not control_lock_active() or str(control_lock.owner) == ControlOwner.ROUTE_COMMIT.value:
        return policy_allow(policy_reason)
    if ROUTE_COMMIT_PREEMPT_ROW_FORWARD and str(control_lock.owner) == ControlOwner.ROW_FORWARD.value:
        return policy_allow(f"{policy_reason}; preempt ROW_FORWARD")
    return policy_deny(f"owner={last_control_owner_debug}")


def coverage_candidate_is_committable(front=None, center=None, body_clearance=None):
    """True when the displayed wavefront candidate is allowed to become activeTarget."""
    allowed, guard_reason = candidate_motion_guard_policy()
    if not allowed:
        return False, guard_reason
    allowed, policy_reason = route_commit_candidate_policy()
    if not allowed:
        return False, policy_reason
    allowed, safety_reason = candidate_safety_guard_policy(front, center, body_clearance)
    if not allowed:
        return False, safety_reason
    return candidate_owner_policy(policy_reason)


def start_route_commit_from_candidate(reason):
    """Promote candidateTarget to activeTarget and make ROUTE_COMMIT own wheels."""
    global route_commit_active, route_commit_target_map, route_commit_target_world, route_commit_kind
    global route_commit_route_map, route_commit_route_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_progress_idx, route_commit_waypoint_idx, route_commit_waypoint_is_corner
    global route_commit_cost, route_commit_score, route_commit_component_id, route_commit_component_cells
    global route_commit_wall_strip_bonus, route_commit_missed_strip_bonus, route_commit_segment_gain, route_commit_footprint_gain_cells
    global route_commit_first_turn_frac, route_commit_corner_count, route_commit_geometry_debug
    global route_commit_started_at, route_commit_reason, route_abort_reason, last_route_commit_debug, planner_mode
    global route_commit_best_target_dist, route_commit_last_progress_time
    route_commit_active = True
    route_commit_target_map = coverage_goal_map
    route_commit_target_world = coverage_goal_world
    route_commit_kind = coverage_goal_kind
    route_commit_route_map = list(coverage_route_map or [])
    route_commit_route_world = list(coverage_route_world or [])
    route_commit_waypoint_map = coverage_route_waypoint_map
    route_commit_waypoint_world = coverage_route_waypoint_world
    route_commit_progress_idx = 0
    route_commit_waypoint_idx = 0
    route_commit_waypoint_is_corner = False
    route_commit_cost = float(coverage_route_cost)
    route_commit_score = float(coverage_route_score)
    route_commit_component_id = int(coverage_route_component_id)
    route_commit_component_cells = int(coverage_route_component_cells)
    route_commit_wall_strip_bonus = float(coverage_route_wall_strip_bonus)
    route_commit_missed_strip_bonus = float(coverage_route_missed_strip_bonus)
    route_commit_segment_gain = int(coverage_route_segment_gain)
    route_commit_footprint_gain_cells = int(coverage_route_footprint_gain_cells)
    route_commit_first_turn_frac = float(coverage_route_first_turn_frac)
    route_commit_corner_count = int(coverage_route_corner_count)
    route_commit_geometry_debug = str(coverage_route_geometry_debug)
    route_commit_started_at = robot.getTime()
    try:
        route_commit_best_target_dist = math.hypot(float(route_commit_target_world[0]) - pose_x, float(route_commit_target_world[1]) - pose_y) if route_commit_target_world is not None else float("inf")
    except Exception:
        route_commit_best_target_dist = float("inf")
    route_commit_last_progress_time = route_commit_started_at
    route_commit_reason = str(reason or "route commit")
    route_abort_reason = "none"
    try:
        global known_map_residual_cleanup_commits_started, known_map_residual_cleanup_started_at, known_map_residual_policy_status
        if (
            known_map_coverage_eval_active()
            and bool(known_map_primary_sweep_completed)
            and route_commit_kind == "uncleaned"
            and "known-full-sweep" not in str(route_commit_geometry_debug)
        ):
            known_map_residual_cleanup_commits_started += 1
            if known_map_residual_cleanup_started_at < -100.0:
                known_map_residual_cleanup_started_at = float(robot.getTime())
            known_map_residual_policy_status = (
                f"residualPolicy=mop commit {known_map_residual_cleanup_commits_started}/"
                f"{int(KNOWN_MAP_EVAL_RESIDUAL_CLEANUP_MAX_COMMITS)}"
            )
    except Exception:
        pass
    update_planner_mode_label()
    lock_age = ROUTE_COMMIT_MAX_AGE_SEC
    if known_map_coverage_eval_active() and route_commit_kind == "uncleaned":
        lock_age = min(
            float(KNOWN_MAP_EVAL_ROUTE_MAX_AGE_SEC),
            float(KNOWN_MAP_EVAL_ROUTE_BASE_MAX_AGE_SEC) + float(route_commit_cost) * float(KNOWN_MAP_EVAL_ROUTE_SEC_PER_M),
        )
    acquire_control(
        ControlOwner.ROUTE_COMMIT,
        lock_age,
        0.0,
        f"route commit: {route_commit_reason[:54]}",
    )
    last_route_commit_debug = (
        f"active {route_commit_active_target_debug()} cost={route_commit_cost:.2f} "
        f"score={route_commit_score:.1f} comp={route_commit_component_cells}"
    )
    return True


def route_commit_is_known_map_sweep():
    try:
        return bool(
            known_map_coverage_eval_active()
            and route_commit_kind == "uncleaned"
            and "known-full-sweep" in str(route_commit_geometry_debug)
        )
    except Exception:
        return False

def route_commit_current_segment_cross_track_m():
    """Distance from the robot centre to the currently executed route segment.

    Used only as a tracking gate.  The old known-map executor drove parallel to
    the orange line when the robot was not exactly on the snapped grid centre;
    this tells ROUTE_COMMIT when it must acquire the displayed centreline before
    returning to strict cardinal segment following.
    """
    try:
        if not route_commit_route_map or len(route_commit_route_map) < 2:
            return 0.0
        route = [(int(x), int(y)) for x, y in route_commit_route_map if map_inside(int(x), int(y))]
        if len(route) < 2:
            return 0.0
        n = len(route)
        progress = int(clamp(route_commit_progress_idx, 0, n - 2))
        j = progress + 1
        while j < n and route[j] == route[progress]:
            j += 1
        if j >= n:
            return 0.0
        ax, ay = route[progress]
        bx, by = route[j]
        rx, ry = world_to_map(pose_x, pose_y)
        vx = float(bx - ax)
        vy = float(by - ay)
        den = math.hypot(vx, vy)
        if den <= 1e-6:
            return 0.0
        cte_px = abs((float(rx - ax) * vy - float(ry - ay) * vx) / den)
        return float(cte_px) / float(MAP_SCALE)
    except Exception:
        return 0.0


def route_commit_grid_segment_heading():
    """Return a cardinal heading for the current committed route segment.

    fixed the planned route shape, but the executor still behaved like a
    pure-pursuit controller: it aimed at the centre of every coarse waypoint.  On
    a grid route this creates many small +/-10..20 degree pivots, especially when
    the robot is beside a wall strip and its body cannot sit exactly on the route
    centreline.  executes the same route as Manhattan segments instead:
    align once to the segment direction, drive straight, then pivot only at real
    corners.
    """
    if not ROUTE_COMMIT_SEGMENT_FOLLOW_ENABLED:
        return None, "point"
    if (
        EXPLORATION_FRONTIER_POINT_PURSUIT_ENABLED
        and route_commit_kind == "frontier"
        and planner_intent == PLANNER_INTENT_EXPAND_MAP
    ):
        return None, "frontier-point"
    if not route_commit_route_map or len(route_commit_route_map) < 2:
        return None, "short"
    try:
        route = [(int(x), int(y)) for x, y in route_commit_route_map if map_inside(int(x), int(y))]
        n = len(route)
        if n < 2:
            return None, "short"
        progress = int(clamp(route_commit_progress_idx, 0, n - 2))
        # If the current cell is duplicated or degenerate, skip forward to the
        # next real move.  The planner is coarse, so duplicates can appear after
        # clipping/search-radius changes.
        j = progress + 1
        while j < n and route[j] == route[progress]:
            j += 1
        if j >= n:
            return None, "end"
        dx = int(route[j][0] - route[progress][0])
        dy = int(route[j][1] - route[progress][1])
        # Only cardinal segments are supposed to reach this executor.  If a
        # legacy diagonal slips through, do not follow it as an arbitrary arc;
        # choose the dominant axis and let the next replan clean up the route.
        if abs(dx) >= abs(dy):
            sx = 1 if dx > 0 else -1 if dx < 0 else 0
            sy = 0
        else:
            sx = 0
            sy = 1 if dy > 0 else -1 if dy < 0 else 0
        if sx == 0 and sy == 0:
            return None, "zero"
        # Map y grows downward, world y grows upward.
        heading = math.atan2(float(-sy), float(sx))
        if sx > 0:
            label = "E"
        elif sx < 0:
            label = "W"
        elif sy > 0:
            label = "S"
        else:
            label = "N"
        return normalize_angle(heading), label
    except Exception as exc:
        return None, f"err:{type(exc).__name__}"

def route_commit_select_waypoint():
    """Pick the next active-route waypoint without jumping across corners.

    used a distance-only lookahead from route[1].  That can select an old
    waypoint behind the robot after small drift, or select a point beyond a 90°
    corner.  Both cases look like rounded/strange turns: the robot alternates
    short forward arcs and pivots while the drawn route itself is valid.

    makes waypoint progress monotonic and corner-aware:
    - first snap to the nearest not-yet-passed route cell;
    - advance while the current waypoint is reached;
    - look ahead only along the current straight segment;
    - stop at a corner instead of chord-cutting across it.
    """
    global route_commit_progress_idx, route_commit_waypoint_idx, route_commit_waypoint_is_corner
    if not route_commit_route_map:
        route_commit_progress_idx = 0
        route_commit_waypoint_idx = 0
        route_commit_waypoint_is_corner = False
        return route_commit_target_map, route_commit_target_world

    route = [(int(x), int(y)) for x, y in route_commit_route_map if map_inside(int(x), int(y))]
    n = len(route)
    if n <= 0:
        route_commit_progress_idx = 0
        route_commit_waypoint_idx = 0
        route_commit_waypoint_is_corner = False
        return route_commit_target_map, route_commit_target_world

    rx, ry = world_to_map(pose_x, pose_y)
    reached_px = max(2.0, ROUTE_COMMIT_WAYPOINT_REACHED_M * MAP_SCALE)
    lookahead_px = max(reached_px + 1.0, ROUTE_COMMIT_WAYPOINT_LOOKAHEAD_M * MAP_SCALE)

    progress = int(clamp(route_commit_progress_idx, 0, n - 1))

    # Monotonic nearest-point update. For ordinary one-target routes it is safe
    # to search the whole remaining suffix.  For a full known-map sweep, however,
    # parallel lanes lie close to each other; a global nearest search can jump from
    # the current strip to a later strip and make the robot drive "straight" to a
    # future lane.  Limit snap-ahead to a short local window and let reached
    # waypoints advance the route order.
    nearest_i = progress
    nearest_d = float("inf")
    search_end = n
    if route_commit_is_known_map_sweep():
        try:
            snap_cells = max(2, int(math.ceil(float(KNOWN_MAP_EVAL_ROUTE_PROGRESS_SNAP_WINDOW_M) * MAP_SCALE / max(1.0, float(GLOBAL_ROUTE_GRID_STEP_PX)))))
            search_end = min(n, progress + snap_cells + 1)
        except Exception:
            search_end = min(n, progress + 5)
    for i in range(progress, search_end):
        px, py = route[i]
        d = math.hypot(px - rx, py - ry)
        if d < nearest_d:
            nearest_d = d
            nearest_i = i
    progress = max(progress, nearest_i)

    # parallel lane, do not immediately look ahead to route[progress+1].  First
    # acquire the current route point; otherwise the executor may drive parallel
    # to the orange line, miss the first wall strip, and later abort on a false
    # path-blocked check.
    force_current_point = False
    if route_commit_is_known_map_sweep() and bool(KNOWN_MAP_EVAL_ROUTE_STRICT_POINT_ACQUIRE):
        try:
            px0, py0 = route[int(progress)]
            cur_d_m = math.hypot(float(px0 - rx), float(py0 - ry)) / float(MAP_SCALE)
            cte_m = route_commit_current_segment_cross_track_m()
            force_current_point = bool(
                cur_d_m > float(KNOWN_MAP_EVAL_ROUTE_POINT_REACQUIRE_M)
                and (int(progress) <= 1 or cte_m > float(KNOWN_MAP_EVAL_ROUTE_TRACKING_LATERAL_M))
            )
        except Exception:
            force_current_point = False

    while progress < n - 1 and not force_current_point:
        px, py = route[progress]
        if math.hypot(px - rx, py - ry) > reached_px:
            break
        progress += 1
    route_commit_progress_idx = int(progress)

    def step_dir(a, b):
        dx = int(math.copysign(1, b[0] - a[0])) if b[0] != a[0] else 0
        dy = int(math.copysign(1, b[1] - a[1])) if b[1] != a[1] else 0
        return dx, dy

    selected_i = progress
    is_corner = False
    if force_current_point:
        selected_i = progress
        is_corner = True
    elif progress < n - 1:
        base_dir = step_dir(route[progress], route[progress + 1])
        selected_i = progress + 1
        travelled = math.hypot(route[selected_i][0] - route[progress][0], route[selected_i][1] - route[progress][1])
        j = selected_i
        while travelled < lookahead_px and j < n - 1:
            next_dir = step_dir(route[j], route[j + 1])
            if next_dir != base_dir:
                # The selected point is the corner cell. Do not look beyond it,
                # otherwise the local controller cuts the corner and draws a curve.
                is_corner = True
                break
            travelled += math.hypot(route[j + 1][0] - route[j][0], route[j + 1][1] - route[j][1])
            j += 1
            selected_i = j
        if selected_i < n - 1 and step_dir(route[selected_i], route[selected_i + 1]) != base_dir:
            is_corner = True

    selected = route[selected_i]
    route_commit_waypoint_idx = int(selected_i)
    route_commit_waypoint_is_corner = bool(is_corner)
    wx = (int(selected[0]) - MAP_ORIGIN_X) / MAP_SCALE
    wy = (MAP_ORIGIN_Y - int(selected[1])) / MAP_SCALE
    return (int(selected[0]), int(selected[1])), (float(wx), float(wy))

def route_commit_speeds(front, center, upper_front, left, right, body_clearance):
    """Execute or start ROUTE_COMMIT before ROW_FORWARD can take over."""
    global route_commit_waypoint_map, route_commit_waypoint_world, coverage_status, row_end_candidate_count
    global route_commit_progress_idx, route_commit_waypoint_idx, route_commit_waypoint_is_corner
    global route_commit_best_target_dist, route_commit_last_progress_time
    global last_route_commit_debug, last_optional_block_reason

    if not route_commit_active:
        ok, reason = coverage_candidate_is_committable(front, center, body_clearance)
        if ok:
            start_route_commit_from_candidate(reason)
        else:
            last_route_commit_debug = f"idle: {reason[:52]}"
            return None

    now = robot.getTime()
    if route_commit_kind == "dock":
        max_age = DOCK_ROUTE_MAX_AGE_SEC
    elif known_map_coverage_eval_active():
        max_age = min(
            float(KNOWN_MAP_EVAL_ROUTE_MAX_AGE_SEC),
            float(KNOWN_MAP_EVAL_ROUTE_BASE_MAX_AGE_SEC) + float(route_commit_cost) * float(KNOWN_MAP_EVAL_ROUTE_SEC_PER_M),
        )
    else:
        max_age = ROUTE_COMMIT_MAX_AGE_SEC
    if now - route_commit_started_at > max_age:
        abort_route_commit("timeout")
        return None
    if nav_state != NAV_FORWARD:
        abort_route_commit(f"nav changed to {nav_state}")
        return None
    if last_bumper_left or last_bumper_center or last_bumper_right:
        abort_route_commit("bumper preempt")
        return None
    if route_commit_kind == "uncleaned" and planner_intent == PLANNER_INTENT_EXPAND_MAP:
        if exploration_cleanup_locked():
            abort_route_commit("cleanup locked until exploration frontier closes")
            return None
        active_straight, active_longitudinal, active_lateral, active_turn = route_goal_relative_to_pose(route_commit_target_world)
        active_local = bool(
            route_commit_cost <= EXPLORE_UNCLEANED_LOCAL_MAX_ROUTE_COST_M
            and active_straight <= EXPLORE_UNCLEANED_LOCAL_MAX_DIST_M
            and active_longitudinal >= EXPLORE_UNCLEANED_BACKTRACK_FORWARD_M
            and active_lateral <= EXPLORE_UNCLEANED_LOCAL_LATERAL_M
            and active_turn <= EXPLORE_UNCLEANED_BACKTRACK_TURN_FRAC
        )
        active_important = bool(
            route_commit_component_cells >= EXPLORE_UNCLEANED_IMPORTANT_COMP_CELLS
            or route_commit_footprint_gain_cells >= EXPLORE_UNCLEANED_IMPORTANT_FOOTPRINT_GAIN
            or route_commit_segment_gain >= EXPLORE_UNCLEANED_IMPORTANT_SEGMENT_GAIN
            or route_commit_missed_strip_bonus >= EXPLORE_UNCLEANED_IMPORTANT_MISSED_BONUS
            or route_commit_wall_strip_bonus >= ROUTE_COMMIT_WALL_STRIP_BONUS * 0.75
        )
        if not (active_local or active_important):
            abort_route_commit(
                f"intent expand-map preempts cleanup comp={route_commit_component_cells} cost={route_commit_cost:.2f} "
                f"long={active_longitudinal:.2f} lat={active_lateral:.2f}"
            )
            return None
    if front < ROUTE_COMMIT_FRONT_ABORT_M or center < ROUTE_COMMIT_CENTER_ABORT_M or body_clearance < ROUTE_COMMIT_BODY_ABORT_M:
        abort_route_commit(f"blocked F={front:.2f} C={center:.2f} body={body_clearance:.2f}")
        return None

    obstacles = cleanable = cleaned = uncleaned = None
    try:
        obstacles, cleanable, cleaned, uncleaned, _unknown = compute_coverage_masks()
        if route_commit_is_known_map_sweep():
            try:
                stop_residual, stop_reason = known_map_residual_cleanup_should_stop(
                    now=now, allow_active_sweep=True, cleanable=cleanable, uncleaned=uncleaned
                )
                if stop_residual:
                    finish_route_commit("bounded residual policy: " + str(stop_reason)[:72])
                    return 0.0, 0.0
            except Exception:
                pass
        if route_commit_kind != "dock" and not route_commit_goal_still_useful(obstacles, cleanable, cleaned, uncleaned):
            finish_route_commit("target already cleaned")
            return 0.0, 0.0
    except Exception:
        pass

    if route_commit_target_world is not None:
        tx, ty = route_commit_target_world
        target_dist = math.hypot(tx - pose_x, ty - pose_y)
        reach_m = DOCK_TARGET_REACHED_M if route_commit_kind == "dock" else ROUTE_COMMIT_TARGET_REACHED_M
        if route_commit_kind != "dock" and not route_commit_is_known_map_sweep():
            try:
                if target_dist < route_commit_best_target_dist - ROUTE_COMMIT_NO_PROGRESS_MIN_GAIN_M:
                    route_commit_best_target_dist = float(target_dist)
                    route_commit_last_progress_time = now
                elif now - route_commit_last_progress_time > ROUTE_COMMIT_NO_PROGRESS_TIMEOUT_SEC and target_dist > reach_m + 0.05:
                    abort_route_commit(f"no progress d={target_dist:.2f} best={route_commit_best_target_dist:.2f}")
                    return None
            except Exception:
                pass
        can_finish_by_final_target = True
        if route_commit_route_map:
            try:
                can_finish_by_final_target = int(route_commit_progress_idx) >= len(route_commit_route_map) - 2
            except Exception:
                can_finish_by_final_target = True
        if target_dist <= reach_m and (can_finish_by_final_target or not route_commit_is_known_map_sweep()):
            finish_route_commit(f"entry reached d={target_dist:.2f}m")
            return 0.0, 0.0

    old_progress_idx = int(route_commit_progress_idx)
    wp_map, wp_world = route_commit_select_waypoint()
    if int(route_commit_progress_idx) > old_progress_idx:
        route_commit_last_progress_time = now
    route_commit_waypoint_map = wp_map
    route_commit_waypoint_world = wp_world
    if wp_world is None:
        abort_route_commit("no waypoint")
        return None

    wx, wy = wp_world
    dx = wx - pose_x
    dy = wy - pose_y
    dist = math.hypot(dx, dy)
    if dist <= ROUTE_COMMIT_WAYPOINT_REACHED_M:
        # Do not jump directly to the final target after a waypoint is reached;
        # and ask the corner-aware selector for the next segment.
        try:
            if route_commit_waypoint_idx < len(route_commit_route_map) - 1:
                route_commit_progress_idx = max(route_commit_progress_idx, route_commit_waypoint_idx + 1)
                wp_map, wp_world = route_commit_select_waypoint()
                route_commit_waypoint_map = wp_map
                route_commit_waypoint_world = wp_world
                wx, wy = wp_world
                dx = wx - pose_x
                dy = wy - pose_y
                dist = math.hypot(dx, dy)
        except Exception:
            pass
    reach_m = DOCK_TARGET_REACHED_M if route_commit_kind == "dock" else ROUTE_COMMIT_TARGET_REACHED_M
    # commit, select the first lookahead point within 0.27m, and immediately call
    # finish_route_commit(); HUD then showed candidateTarget but activeTarget=none
    # and owner=NONE/ROW_FORWARD.  Only finish the whole route when the selected
    # waypoint is the final route point.
    at_final_waypoint = True
    try:
        at_final_waypoint = (
            not route_commit_route_map
            or int(route_commit_waypoint_idx) >= len(route_commit_route_map) - 1
        )
    except Exception:
        at_final_waypoint = True
    if dist <= reach_m and at_final_waypoint:
        finish_route_commit(f"entry reached d={dist:.2f}m")
        return 0.0, 0.0

    if obstacles is not None and cleanable is not None:
        try:
            rx, ry = world_to_map(pose_x, pose_y)
            mx, my = world_to_map(wx, wy)
            blocked = map_line_footprint_blocked_ratio(rx, ry, mx, my, obstacles, cleanable)
            if blocked > ROUTE_COMMIT_BLOCKED_RATIO_ABORT:
                if route_commit_is_known_map_sweep():
                    # inflated passability map.  A live line-of-sight check from
                    # the slightly drifted robot pose to the next waypoint can
                    # falsely cut across an inflated obstacle/corner and abort a
                    # valid sweep.  Keep real safety from RangeFinder/bumpers
                    # above, but do not kill the committed known-map route here.
                    last_optional_block_reason = f"known-sweep ignored map-line block {blocked:.2f}"
                else:
                    abort_route_commit(f"path blocked ratio={blocked:.2f}")
                    return None
        except Exception:
            pass

    point_heading = math.atan2(dy, dx)
    segment_heading, segment_label = route_commit_grid_segment_heading()
    target_heading = segment_heading if segment_heading is not None else point_heading
    known_sweep_track = False
    known_sweep_cte_m = 0.0
    if (
        bool(KNOWN_MAP_EVAL_ROUTE_TRACKING_ENABLED)
        and route_commit_is_known_map_sweep()
        and segment_heading is not None
    ):
        known_sweep_cte_m = route_commit_current_segment_cross_track_m()
        point_vs_segment = abs(normalize_angle(point_heading - segment_heading))
        # Acquire the actual displayed route when the robot is laterally off the
        # centreline.  This is bounded point tracking to the current same-segment
        # waypoint, not a free shortcut to the final uncleaned target.
        known_sweep_track = bool(
            known_sweep_cte_m > float(KNOWN_MAP_EVAL_ROUTE_TRACKING_LATERAL_M)
            or point_vs_segment > float(KNOWN_MAP_EVAL_ROUTE_TRACKING_POINT_HEADING_ERR)
            or route_commit_waypoint_is_corner
            or int(route_commit_progress_idx) <= 1
        )
        if known_sweep_track:
            target_heading = point_heading
            segment_label = f"{segment_label}/track{known_sweep_cte_m:.2f}"
    heading_err = normalize_angle(target_heading - pose_theta)
    row_end_candidate_count = 0
    last_optional_block_reason = "ROUTE_COMMIT owns motion"
    coverage_status = (
        f"route commit {route_commit_kind} d={dist:.2f} cost={route_commit_cost:.2f} "
        f"prog={int(route_commit_progress_idx)}/{len(route_commit_route_map) if route_commit_route_map else 0} "
        f"herr={math.degrees(heading_err):.1f} seg={segment_label} wp={route_commit_waypoint_idx}"
        f"{'C' if route_commit_waypoint_is_corner else '-'} comp={route_commit_component_cells} "
        f"{route_commit_geometry_debug[:20]}"
    )
    last_route_commit_debug = coverage_status

    if (
        EXPLORATION_FRONTIER_POINT_PURSUIT_ENABLED
        and route_commit_kind == "frontier"
        and planner_intent == PLANNER_INTENT_EXPAND_MAP
        and segment_heading is None
    ):
        # Frontier exploration is not a coverage strip.  Track the next safe
        # waypoint directly; pivot only for large heading errors.  Small steering
        # is explicitly labelled so the strict motion contract allows this bounded
        # exploration pursuit instead of converting every correction into a pivot.
        if abs(heading_err) > EXPLORATION_FRONTIER_PIVOT_ERR:
            sign = 1.0 if heading_err > 0.0 else -1.0
            speed = clamp(abs(heading_err) * ROUTE_COMMIT_PIVOT_KP, ROUTE_COMMIT_PIVOT_MIN_SPEED, ROUTE_COMMIT_PIVOT_MAX_SPEED)
            coverage_status = f"frontier pursuit pivot err={math.degrees(heading_err):.1f} {route_commit_geometry_debug[:22]}"
            last_route_commit_debug = coverage_status
            note_route_commit_inplace_turn(heading_err, "frontier pursuit pivot")
            return -sign * speed, sign * speed
        if maybe_start_route_commit_post_turn_snapshot("frontier pursuit aligned"):
            return 0.0, 0.0
        base = EXPLORATION_FRONTIER_PURSUIT_SPEED
        if route_commit_waypoint_is_corner and dist < ROUTE_COMMIT_CORNER_SLOWDOWN_M:
            base = EXPLORATION_FRONTIER_PURSUIT_SLOW_SPEED
        if front < ROW_SLOW_DISTANCE or upper_front < ROW_SLOW_DISTANCE or body_clearance < BODY_CORRIDOR_PASS_CLEARANCE + 0.035:
            base = EXPLORATION_FRONTIER_PURSUIT_SLOW_SPEED
        if abs(heading_err) <= EXPLORATION_FRONTIER_STEER_ERR:
            coverage_status = f"frontier pursuit straight d={dist:.2f} wp={route_commit_waypoint_idx} {route_commit_geometry_debug[:18]}"
            last_route_commit_debug = coverage_status
            return base, base
        corr = clamp(EXPLORATION_FRONTIER_STEER_KP * heading_err, -EXPLORATION_FRONTIER_STEER_MAX, EXPLORATION_FRONTIER_STEER_MAX)
        coverage_status = f"frontier pursuit steer err={math.degrees(heading_err):.1f} d={dist:.2f} wp={route_commit_waypoint_idx}"
        last_route_commit_debug = coverage_status
        return base - corr, base + corr

    speed = ROUTE_COMMIT_SPEED
    if route_commit_waypoint_is_corner and dist < ROUTE_COMMIT_CORNER_SLOWDOWN_M:
        speed = ROUTE_COMMIT_SLOW_SPEED
    if front < ROW_SLOW_DISTANCE or upper_front < ROW_SLOW_DISTANCE or body_clearance < BODY_CORRIDOR_PASS_CLEARANCE + 0.035:
        speed = ROUTE_COMMIT_SLOW_SPEED

    if (
        bool(KNOWN_MAP_EVAL_ROUTE_TRACKING_ENABLED)
        and route_commit_is_known_map_sweep()
        and segment_heading is not None
    ):
        # over smooth differential arcs.  If target_heading is point_heading
        # because we are acquiring/reacquiring the drawn route, pivot until the
        # body is aligned to that point and then drive both wheels equally.  Once
        # the centreline is acquired, segment_heading takes over and the robot
        # follows cardinal sweep strips.
        if abs(heading_err) > float(KNOWN_MAP_EVAL_ROUTE_TRACKING_STEER_ERR):
            sign = 1.0 if heading_err > 0.0 else -1.0
            turn_speed = clamp(abs(heading_err) * ROUTE_COMMIT_PIVOT_KP, ROUTE_COMMIT_PIVOT_MIN_SPEED, ROUTE_COMMIT_PIVOT_MAX_SPEED)
            return -sign * turn_speed, sign * turn_speed
        if known_sweep_track:
            return heading_locked_wheel_speeds_to(
                float(KNOWN_MAP_EVAL_ROUTE_ACQUIRE_SPEED),
                target_heading,
                HEADING_LOCK_KP,
                float(KNOWN_MAP_EVAL_ROUTE_ACQUIRE_HEADING_CORR_MAX),
            )
        return heading_locked_wheel_speeds_to(
            speed,
            target_heading,
            HEADING_LOCK_KP,
            float(KNOWN_MAP_EVAL_ROUTE_HEADING_CORR_MAX),
        )

    align_tol = ROUTE_COMMIT_SEGMENT_ALIGN_TOL if segment_heading is not None else ROUTE_COMMIT_ALIGN_TOL
    if abs(heading_err) > align_tol:
        sign = 1.0 if heading_err > 0.0 else -1.0
        speed_turn = clamp(abs(heading_err) * ROUTE_COMMIT_PIVOT_KP, ROUTE_COMMIT_PIVOT_MIN_SPEED, ROUTE_COMMIT_PIVOT_MAX_SPEED)
        note_route_commit_inplace_turn(heading_err, f"route align {route_commit_kind}")
        return -sign * speed_turn, sign * speed_turn

    if maybe_start_route_commit_post_turn_snapshot("route commit aligned"):
        return 0.0, 0.0

    # With ordinary segment following active, do not do continuous point-bearing
    # yaw correction.  Known-map full sweep is handled above because its contract
    # is stronger: the orange route is the experiment route.
    return speed, speed

def coverage_goal_side_bias(side):
    """Return a small bias if the path-aware route waypoint lies on this side."""
    guide = coverage_guidance_world()
    if guide is None:
        return 0.0
    gx, gy = guide
    dx = gx - pose_x
    dy = gy - pose_y
    # local left normal to heading. Positive lateral means target is left.
    longitudinal = math.cos(pose_theta) * dx + math.sin(pose_theta) * dy
    lateral = -math.sin(pose_theta) * dx + math.cos(pose_theta) * dy
    # Early map building must be row-stable.  A waypoint behind or almost
    # sideways is only a hint for later; using it as side bias caused
    # forward-left-right-look-back oscillations while most of the room was still
    # unknown.
    if last_coverage_percent < EARLY_EXPLORATION_COVERAGE_PERCENT:
        angle = abs(math.degrees(math.atan2(abs(lateral), max(1e-6, longitudinal)))) if longitudinal > 0.0 else 180.0
        if longitudinal < ROUTE_GUIDANCE_BEHIND_DISABLE_M or angle > ROUTE_GUIDANCE_MAX_SIDE_ANGLE_DEG:
            return 0.0
    if not coverage_target_replan_allowed(for_side_bias=True):
        return 0.0
    if coverage_goal_kind == "under-surface":
        # Under-furniture zones are cleaned when they are on the current route.
        # They must not bend the whole lawnmower row into a diagonal collision or
        # choose a lane change by themselves. The ordinary row/lawnmower logic
        # remains primary; yellow cells only bias a lane if they are nearly on the
        # same straight approach.
        if longitudinal < 0.10 or longitudinal > UNDER_SURFACE_TARGET_FORWARD_LIMIT_M or abs(lateral) > UNDER_SURFACE_DIRECT_LATERAL_TOL_M:
            return 0.0
    if abs(lateral) < 0.10:
        return 0.0
    target_side = 1.0 if lateral > 0 else -1.0
    return PLANNER_SIDE_BIAS if target_side == (1.0 if side >= 0 else -1.0) else -PLANNER_SIDE_BIAS * 0.45


def coverage_score_for_side(side):
    """Estimate which side is a better next lane using map + cleaned memory.

    Higher score means more unknown/free and not-yet-cleaned cells. Occupied
    cells are penalized. This is deliberately local; it is enough for a diploma
    prototype and much more stable than choosing left/right from one depth frame.
    """
    score = 0.0
    # side vector relative to current heading
    nx = -math.sin(pose_theta) * side
    ny = math.cos(pose_theta) * side
    fx = math.cos(pose_theta)
    fy = math.sin(pose_theta)

    for i in range(1, COVERAGE_SIDE_SAMPLES + 1):
        ahead = 0.15 + 0.12 * i
        lateral = LANE_SPACING
        wx = pose_x + fx * ahead + nx * lateral
        wy = pose_y + fy * ahead + ny * lateral
        mx, my = world_to_map(wx, wy)
        if not map_inside(mx, my):
            score -= 4.0
            continue
        lo = float(log_odds[my, mx])
        vio = float(visual_log_odds[my, mx])
        cleaned = cleaned_mask[my, mx] > 0
        if lo > LO_OCCUPIED_EPS or vio > CV_DISPLAY_DENSE_EPS:
            score -= 8.0
        elif cleaned:
            score -= 2.5
        elif abs(lo) <= LO_UNKNOWN_EPS and vio <= CV_DISPLAY_LIGHT_EPS:
            score += 3.0
        else:
            score += 2.5
    score += coverage_goal_side_bias(side)
    return score


def choose_coverage_side(left_dist, right_dist):
    """Choose lane-change side with hysteresis and map/coverage memory."""
    global lane_side
    left_score = coverage_score_for_side(1.0)
    right_score = coverage_score_for_side(-1.0)

    # RangeFinder side readings are still useful as a hard local safety hint.
    if left_dist < SIDE_DISTANCE + 0.12:
        left_score -= 25.0
    if right_dist < SIDE_DISTANCE + 0.12:
        right_score -= 25.0

    # Hysteresis: if both are close, keep the planned lawnmower side rather than
    # flipping every frame. This was the source of "right-left-right-left" in corners.
    if abs(left_score - right_score) < 5.0:
        return 1.0 if lane_side >= 0 else -1.0
    return 1.0 if left_score > right_score else -1.0


def start_lane_shift(distance=LANE_SPACING):
    global nav_state, lane_shift_start_x, lane_shift_start_y, lane_shift_start_time, lane_shift_target_dist, lane_shift_heading_target
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    # Lane shift must be a straight translation after the first 90-degree pivot,
    # not a curved correction back to a theoretical grid heading.  Freeze the
    # actual yaw at the start of the shift and use only a small heading lock.
    hard_stop_motors()
    nav_state = NAV_LANE_SHIFT
    lane_shift_start_x = pose_x
    lane_shift_start_y = pose_y
    lane_shift_start_time = robot.getTime()
    lane_shift_target_dist = distance
    lane_shift_heading_target = pose_theta
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"straight shift {distance:.2f}m"
    acquire_control(ControlOwner.LANE_SHIFT, LANE_SHIFT_OWNER_TIME_SEC, max(0.0, distance * 0.85), coverage_status)
    # Do not immediately write depth just after a pivot into the next lane.
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.20)


def start_recovery_backup(side, reason="blocked"):
    global nav_state, backup_start_x, backup_start_y, backup_until, backup_after_side
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    nav_state = NAV_RECOVERY_BACKUP
    backup_start_x = pose_x
    backup_start_y = pose_y
    backup_until = robot.getTime() + BACKUP_TIMEOUT_SEC
    backup_after_side = 1.0 if side >= 0 else -1.0
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "backup: " + reason
    acquire_control(ControlOwner.RECOVERY, BACKUP_TIMEOUT_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.20)


def start_leg_escape(side, reason="thin furniture contact", strong=False):
    """Escape from chair/table-leg contact without changing the furniture model.

    side is the desired escape side: +1 = steer/turn left, -1 = steer/turn right.
    This is different from a normal lane change: it first gets the body away
    from the leg, then resumes coverage.

    strong=True is used for real contact traps.  There are now two strong
    variants: a side-trap escape for chair/table legs and a wider front-contact
    escape for low boxes/floor objects that fire the centre or both bumpers.
    """
    global nav_state, leg_escape_side, leg_escape_start_x, leg_escape_start_y, leg_escape_until
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until, last_leg_escape_time
    global leg_escape_backup_distance_current, leg_escape_turn_angle_current, leg_escape_forward_distance_current, leg_escape_forward_speed_current
    global leg_escape_replan_after, under_furniture_until, under_furniture_active, under_furniture_suppressed_until
    global nav_action_queue, row_end_candidate_count, last_contact_trap_time, last_contact_trap_reason, last_turn_variant
    global last_contact_route_kill_until, last_contact_escape_kind
    global last_contact_cluster_time, last_contact_cluster_x, last_contact_cluster_y, last_contact_cluster_count
    now = robot.getTime()
    leg_escape_side = 1.0 if side >= 0 else -1.0

    # finite-state machine create arbitrary 42/78/110-degree paths.  All contact
    # recovery is compact and cardinal: small backup -> one 90-degree pivot ->
    # resume bumper-first exploration.
    if MAP_BUILDING_DISABLE_LEG_ESCAPE_FSM and matrix_first_explore_active() and nav_state not in CONTACT_RECOVERY_STATES:
        info_side, info_reason = choose_explore_contact_turn_side(
            bool(last_bumper_left), bool(last_bumper_right), last_min_left, last_min_right
        )
        # Preserve explicit single-side safety unless this was not a bumper call.
        recovery_side = leg_escape_side if (last_bumper_left or last_bumper_center or last_bumper_right) else info_side
        start_contact_recovery(
            recovery_side,
            "front" if (last_bumper_center or (last_bumper_left and last_bumper_right)) else "side",
            f"map-build compact recovery instead of leg_escape: {reason[:36]}; {info_reason[:54]}",
            EXPLORE_CONTACT_BACKUP_GOAL_M,
            EXPLORE_CONTACT_BACKUP_TIMEOUT_SEC,
            PIVOT_TURN_ANGLE,
            EXPLORE_CONTACT_FORWARD_VERIFY_M,
            CONTACT_ESCAPE_VERIFY_SPEED,
            FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
            False,
        )
        return

    leg_escape_start_x = pose_x
    leg_escape_start_y = pose_y
    repeated_local_contact = False
    if strong:
        # A physical hit invalidates any queued coverage manoeuvre.  Continuing
        # a stale pivot/shift after the escape is exactly how the robot wedges
        # itself under a shelf or against the red low obstacle.
        nav_action_queue = []
        row_end_candidate_count = 0

        cluster_dist = math.hypot(pose_x - last_contact_cluster_x, pose_y - last_contact_cluster_y)
        if now - last_contact_cluster_time <= CONTACT_CLUSTER_TIME_SEC and cluster_dist <= CONTACT_CLUSTER_RADIUS_M:
            last_contact_cluster_count += 1
        else:
            last_contact_cluster_count = 1
            last_contact_cluster_x = pose_x
            last_contact_cluster_y = pose_y
        last_contact_cluster_time = now
        repeated_local_contact = last_contact_cluster_count >= 2

        front_contact = bool(last_bumper_center or (last_bumper_raw_left > BUMPER_ACTIVE_THRESHOLD and last_bumper_raw_right > BUMPER_ACTIVE_THRESHOLD))
        if contact_looks_like_wall_boundary():
            # Boundary-wall contact: release the bumper only slightly, then pivot
            # parallel to the wall and keep cleaning the edge.  Do this before
            # repeated-contact escalation; otherwise a normal wall graze becomes
            # corner-repeat-2/3 and the robot backs far away from the exact strip
            # it should clean.
            leg_escape_backup_distance_current = WALL_CONTACT_BACKUP_DISTANCE
            leg_escape_turn_angle_current = WALL_CONTACT_TURN_ANGLE
            leg_escape_forward_distance_current = WALL_CONTACT_FORWARD_DISTANCE
            leg_escape_forward_speed_current = WALL_CONTACT_FORWARD_SPEED
            leg_escape_until = now + WALL_CONTACT_BACKUP_TIMEOUT_SEC
            last_turn_variant = "wall-edge-88"
            last_contact_escape_kind = "wall"
        elif repeated_local_contact and contact_looks_like_gap_mouth() and last_contact_cluster_count <= 2:
            # First repeated hit in a passable mouth: do NOT retreat 60 cm and
            # rotate 158 degrees.  That is the behaviour that made the robot back
            # out ass-first.  Use a short nudge away from the corner, then force a
            # slow forward commit before any planner/row-end logic can interrupt.
            leg_escape_backup_distance_current = CORNER_NUDGE_BACKUP_DISTANCE
            leg_escape_turn_angle_current = CORNER_NUDGE_TURN_ANGLE
            leg_escape_forward_distance_current = CORNER_NUDGE_FORWARD_DISTANCE
            leg_escape_forward_speed_current = CORNER_NUDGE_FORWARD_SPEED
            leg_escape_until = now + CORNER_NUDGE_BACKUP_TIMEOUT_SEC
            last_turn_variant = f"gap-corner-nudge-{last_contact_cluster_count}"
            last_contact_escape_kind = "gap"
        elif repeated_local_contact:
            # Same corner hit after the nudge failed or when the mouth is not
            # passable: abandon this exact approach, but keep the retreat bounded
            # so the robot does not drive half the room backwards.
            leg_escape_backup_distance_current = CORNER_REPEAT_BACKUP_DISTANCE
            leg_escape_turn_angle_current = CORNER_REPEAT_TURN_ANGLE
            leg_escape_forward_distance_current = CORNER_REPEAT_FORWARD_DISTANCE
            leg_escape_forward_speed_current = CONTACT_ESCAPE_VERIFY_SPEED
            leg_escape_until = now + CORNER_REPEAT_BACKUP_TIMEOUT_SEC
            last_turn_variant = f"corner-repeat-92-{last_contact_cluster_count}"
            last_contact_escape_kind = "corner"
        elif front_contact:
            leg_escape_backup_distance_current = FRONT_CONTACT_TRAP_BACKUP_DISTANCE
            leg_escape_turn_angle_current = FRONT_CONTACT_TRAP_TURN_ANGLE
            leg_escape_forward_distance_current = FRONT_CONTACT_TRAP_FORWARD_DISTANCE
            leg_escape_forward_speed_current = CONTACT_ESCAPE_VERIFY_SPEED
            leg_escape_until = now + FRONT_CONTACT_TRAP_BACKUP_TIMEOUT_SEC
            last_turn_variant = "front-contact-88"
            last_contact_escape_kind = "front"
        else:
            leg_escape_backup_distance_current = LEG_TRAP_BACKUP_DISTANCE
            leg_escape_turn_angle_current = LEG_TRAP_TURN_ANGLE
            leg_escape_forward_distance_current = LEG_TRAP_FORWARD_DISTANCE
            leg_escape_forward_speed_current = TRAP_ESCAPE_VERIFY_SPEED
            leg_escape_until = now + LEG_TRAP_BACKUP_TIMEOUT_SEC
            last_turn_variant = "side-trap-78"
            last_contact_escape_kind = "side"
        leg_escape_replan_after = True
        under_furniture_until = 0.0
        under_furniture_active = False
        under_furniture_suppressed_until = now + UNDER_FURNITURE_TRAP_COOLDOWN_SEC
        if last_contact_escape_kind == "wall":
            kill_sec = WALL_CONTACT_ROUTE_KILL_SEC
        else:
            kill_sec = CONTACT_CLUSTER_ROUTE_KILL_SEC if repeated_local_contact else CONTACT_ROUTE_KILL_SEC
        last_contact_route_kill_until = now + kill_sec
        last_contact_trap_time = now
        last_contact_trap_reason = reason
        coverage_status = "trap escape backup: " + reason
        start_contact_recovery(
            leg_escape_side,
            last_contact_escape_kind,
            reason,
            leg_escape_backup_distance_current,
            max(0.20, leg_escape_until - now),
            leg_escape_turn_angle_current,
            leg_escape_forward_distance_current,
            leg_escape_forward_speed_current,
            contact_recovery_forward_timeout_for_kind(last_contact_escape_kind),
            True,
        )
        return
    else:
        leg_escape_backup_distance_current = LEG_ESCAPE_BACKUP_DISTANCE
        leg_escape_turn_angle_current = LEG_ESCAPE_TURN_ANGLE
        leg_escape_forward_distance_current = LEG_ESCAPE_FORWARD_DISTANCE
        leg_escape_forward_speed_current = LEG_ESCAPE_FORWARD_SPEED
        leg_escape_until = now + LEG_ESCAPE_BACKUP_TIMEOUT_SEC
        leg_escape_replan_after = False
        last_turn_variant = "leg-contact-42"
        last_contact_escape_kind = "leg"
        coverage_status = "leg escape backup: " + reason
    last_leg_escape_time = now
    nav_state = NAV_LEG_ESCAPE_BACKUP
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    # Freeze mapping during escape; contact dynamics near legs create unreliable
    # depth/odometry associations.
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)


def finish_leg_escape_backup():
    global nav_state, leg_escape_turn_target, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    hard_stop_motors()
    leg_escape_turn_target = normalize_angle(pose_theta + leg_escape_side * leg_escape_turn_angle_current)
    nav_state = NAV_LEG_ESCAPE_TURN
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"leg escape turn target={math.degrees(leg_escape_turn_target):.0f}"
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)


def finish_leg_escape_turn():
    global nav_state, leg_escape_forward_start_x, leg_escape_forward_start_y, leg_escape_forward_until
    global desired_grid_heading, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until, last_contact_escape_kind
    hard_stop_motors()
    # Temporarily drive out along the escape heading. After the short escape
    # segment, the normal grid planner will snap back to the nearest 90-degree
    # direction.
    desired_grid_heading = pose_theta
    leg_escape_forward_start_x = pose_x
    leg_escape_forward_start_y = pose_y
    # Strong contact escapes use a short verification segment, not a long blind
    # forward drive. The previous long forward run could immediately hit the red
    # low obstacle after leaving the green shelf/table area.
    if last_contact_escape_kind == "corner":
        leg_escape_forward_until = robot.getTime() + CORNER_REPEAT_FORWARD_TIMEOUT_SEC
    elif last_contact_escape_kind == "wall":
        leg_escape_forward_until = robot.getTime() + WALL_CONTACT_FORWARD_TIMEOUT_SEC
    else:
        leg_escape_forward_until = robot.getTime() + (FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC if leg_escape_replan_after else LEG_ESCAPE_FORWARD_TIMEOUT_SEC)
    nav_state = NAV_LEG_ESCAPE_FORWARD
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "leg escape forward"
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.35)


def finish_leg_escape_forward():
    global desired_grid_heading, row_end_candidate_count, leg_escape_replan_after, last_revisit_lane_change_time
    global post_gap_commit_until, post_gap_commit_start_x, post_gap_commit_start_y
    global post_lane_forward_lock_until, post_lane_forward_lock_start_x, post_lane_forward_lock_start_y
    row_end_candidate_count = 0
    if last_contact_escape_kind == "gap":
        # A gap-mouth nudge is not a normal escape.  Keep the current micro-heading
        # briefly and crawl forward so the robot actually enters the opening
        # instead of snapping to the grid and hitting the same corner again.
        desired_grid_heading = pose_theta
        post_gap_commit_until = robot.getTime() + POST_GAP_COMMIT_SEC
        post_gap_commit_start_x = pose_x
        post_gap_commit_start_y = pose_y
        start_post_recovery_stabilizer("gap", desired_grid_heading, POST_GAP_COMMIT_SEC + 0.65, POST_GAP_COMMIT_DISTANCE_M + 0.08)
        leg_escape_replan_after = False
        start_forward_row("gap mouth commit after corner nudge")
    elif last_contact_escape_kind == "wall":
        # Wall contact recovery must not continue at the arbitrary escape yaw.
        # Snap back to the room grid and rotate in place before translating.
        desired_grid_heading = strict_world_grid_heading(pose_theta)
        start_post_recovery_stabilizer("wall", desired_grid_heading, 0.85, 0.08)
        post_lane_forward_lock_until = robot.getTime() + max(POST_LANE_FORWARD_LOCK_SEC, 0.75)
        post_lane_forward_lock_start_x = pose_x
        post_lane_forward_lock_start_y = pose_y
        leg_escape_replan_after = False
        start_grid_realign(desired_grid_heading, "after wall contact escape", "row forward after wall contact grid realign")
    elif last_contact_escape_kind in ("front", "corner"):
        # Front/corner recovery may rotate away from the obstacle, but it must not
        # become a long diagonal row.  Resume through GRID_REALIGN.
        desired_grid_heading = strict_world_grid_heading(pose_theta)
        start_post_recovery_stabilizer("contact", desired_grid_heading, 0.85, 0.08)
        post_lane_forward_lock_until = robot.getTime() + max(POST_LANE_FORWARD_LOCK_SEC, 0.75)
        post_lane_forward_lock_start_x = pose_x
        post_lane_forward_lock_start_y = pose_y
        last_revisit_lane_change_time = -999.0
        leg_escape_replan_after = False
        start_grid_realign(desired_grid_heading, "after front/corner contact escape", "row forward after front/corner grid realign")
    else:
        # Side/leg contacts are local point-obstacle escapes.  For these it is
        # still valid to snap back to the nearest grid heading, but the snap must
        # happen as an in-place GRID_REALIGN, never as a curved forward re-entry.
        target_heading = snap_to_right_angle(pose_theta)
        start_post_recovery_stabilizer("contact", target_heading)
        if leg_escape_replan_after:
            last_revisit_lane_change_time = -999.0
            leg_escape_replan_after = False
            start_grid_realign(target_heading, "after side/leg trap escape", "row forward after side/leg trap escape; replan to coverage target")
        else:
            start_grid_realign(target_heading, "after leg escape", "row forward after leg escape")


def detect_thin_leg_ahead(depth):
    """Detect a narrow chair/table leg in the direct path and choose a gap side.

    Returns (+1/-1, reason) or (0, reason). +1 means bypass to the robot's left,
    -1 means bypass to the right. This is not semantic recognition of a chair;
    it is geometric RGB-D/depth behavior for thin obstacles under furniture.
    """
    global last_leg_pass_detected, last_leg_pass_side
    last_leg_pass_detected = False
    last_leg_pass_side = 0.0
    if not LEG_PASS_ENABLED or depth is None:
        return 0.0, "disabled"

    r0 = int(RF_H * 0.40)
    r1 = int(RF_H * 0.74)
    band = depth[r0:r1, :]
    col_dist = np.full(RF_W, MAX_VALID_RANGE, dtype=np.float32)
    for c in range(RF_W):
        vals = band[:, c]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > 0.14) & (vals < MAX_VALID_RANGE)]
        if vals.size >= 2:
            col_dist[c] = float(np.percentile(vals, 18))

    center_half = max(4, int(RF_W * FRONT_NARROW_FRACTION * 0.95))
    cc0 = max(0, RF_W // 2 - center_half)
    cc1 = min(RF_W, RF_W // 2 + center_half)
    close = col_dist[cc0:cc1] < LEG_PASS_DETECT_DISTANCE
    close_count = int(np.count_nonzero(close))
    center_width = max(1, cc1 - cc0)
    close_ratio = close_count / center_width
    if close_count == 0 or close_ratio > LEG_PASS_MAX_WIDTH_FRAC:
        return 0.0, f"no thin leg width={close_count}/{center_width}"

    # Make sure it is not a full wall/box: side gaps near the center must be open.
    # Use wider side windows than the center window; chair/table legs often sit just
    # outside the narrow forward corridor, and the old detector was too strict.
    gap_w = max(center_width, int(RF_W * 0.16))
    left_slice = col_dist[max(0, cc0 - gap_w):cc0]
    right_slice = col_dist[cc1:min(RF_W, cc1 + gap_w)]
    valid_left = left_slice[np.isfinite(left_slice)]
    valid_right = right_slice[np.isfinite(right_slice)]
    left_gap = float(np.percentile(valid_left, 35)) if valid_left.size else 0.0
    right_gap = float(np.percentile(valid_right, 35)) if valid_right.size else 0.0
    if max(left_gap, right_gap) < LEG_PASS_SIDE_OPEN_DISTANCE:
        return 0.0, f"gaps closed L={left_gap:.2f} R={right_gap:.2f}"

    side = 1.0 if left_gap >= right_gap else -1.0
    last_leg_pass_detected = True
    last_leg_pass_side = side
    return side, f"thin leg width={close_count}/{center_width} gaps L={left_gap:.2f} R={right_gap:.2f}"


def start_leg_pass(side, reason="thin leg ahead"):
    """Start a small local bypass instead of a full row/lane change."""
    global nav_state, leg_pass_side, leg_pass_target_heading, leg_pass_original_heading
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until, last_leg_pass_time
    now = robot.getTime()
    leg_pass_side = 1.0 if side >= 0 else -1.0
    leg_pass_original_heading = desired_grid_heading
    leg_pass_target_heading = normalize_angle(pose_theta + leg_pass_side * LEG_PASS_TURN_ANGLE)
    last_leg_pass_time = now
    nav_state = NAV_LEG_PASS_TURN
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "leg pass turn: " + reason
    map_freeze_until = max(map_freeze_until, now + 0.12)


def finish_leg_pass_turn():
    global nav_state, leg_pass_start_x, leg_pass_start_y, leg_pass_until
    global desired_grid_heading, prev_cmd_left, prev_cmd_right, coverage_status
    hard_stop_motors()
    desired_grid_heading = leg_pass_target_heading
    leg_pass_start_x = pose_x
    leg_pass_start_y = pose_y
    leg_pass_until = robot.getTime() + LEG_PASS_FORWARD_TIMEOUT_SEC
    nav_state = NAV_LEG_PASS_FORWARD
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "leg pass forward"


def finish_leg_pass_forward():
    global nav_state, desired_grid_heading, prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    hard_stop_motors()
    desired_grid_heading = leg_pass_original_heading
    nav_state = NAV_LEG_PASS_ALIGN
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "leg pass align"
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.12)


def finish_leg_pass_align():
    global row_end_candidate_count, leg_pass_grace_until, coverage_status
    hard_stop_motors()
    row_end_candidate_count = 0
    leg_pass_grace_until = robot.getTime() + LEG_PASS_AFTER_ALIGN_GRACE_SEC
    start_forward_row("row forward after leg pass")
    coverage_status = "row forward after leg pass grace"


def cancel_coverage_lane_queue_for_expand_map(reason="expand-map"):
    """Drop pending 90/shift/90 coverage manoeuvres when EXPAND_MAP regains priority.

    This does not interrupt contact recovery or an already-running safety action.
    It only prevents the second half of a lawnmower lane change from continuing
    after the planner has decided that the current mission is map expansion.
    """
    global nav_action_queue, coverage_status, last_optional_block_reason
    if not nav_action_queue:
        return False
    if not planner_expand_map_motion_active():
        return False
    # Exploration/contact turns are allowed.  The problematic queue is the
    # coverage sequence containing SHIFT/second TURN actions.
    if any(action and action[0] == "SHIFT" for action in nav_action_queue):
        nav_action_queue = []
        coverage_status = f"expand-map cancels coverage lane queue: {reason}"
        last_optional_block_reason = "expand-map cancelled lane queue"
        return True
    return False


def run_next_nav_action(front_for_pivot=None, body_clearance_for_pivot=None):
    """Run the next queued coverage action after a settle pause."""
    global nav_state, nav_action_queue, coverage_status
    if cancel_coverage_lane_queue_for_expand_map("run_next_nav_action"):
        start_forward_row("row forward after expand-map cancelled lane queue", post_lane_lock=True)
        return
    if not nav_action_queue:
        start_forward_row("row forward after completed manoeuvre", post_lane_lock=True)
        return
    action = nav_action_queue.pop(0)
    kind = action[0]
    if kind == "SHIFT":
        start_lane_shift(action[1])
    elif kind == "TURN":
        turn_side = action[1]
        turn_reason = action[2] if len(action) > 2 else "queued turn"
        if pivot_clearance_problem(front_for_pivot, body_clearance_for_pivot):
            start_pre_pivot_backup(turn_side, f"{turn_reason}; queued clearance F={front_for_pivot:.2f} body={body_clearance_for_pivot:.2f}")
        else:
            begin_pivot_turn(turn_side, turn_reason)
    else:
        nav_state = NAV_FORWARD
        coverage_status = "unknown action"


def begin_lawnmower_lane_change(side, reason):
    """Start a full grid-cleaning row transition.

    Sequence: pivot 90 toward next lane -> move sideways one lane -> pivot 90
    the same way. The robot then travels in the opposite direction, producing a
    simple lawnmower/boustrophedon coverage pattern.
    """
    global nav_action_queue, lane_side, last_row_change_time, coverage_status
    side = 1.0 if side >= 0 else -1.0
    if planner_expand_map_motion_active():
        begin_explore_single_turn(side, f"expand-map single turn instead of lane change: {reason}")
        return
    nav_action_queue = [("SHIFT", LANE_SPACING), ("TURN", side, "second 90 for next row")]
    last_row_change_time = robot.getTime()
    coverage_status = f"lane change {'L' if side > 0 else 'R'}"
    begin_pivot_turn(side, reason)
    # Next row should normally offset to the opposite side.
    lane_side = -side


def pivot_clearance_problem(front, body_clearance):
    """True when a 90-degree pivot starts too close to the front shell/contact pads."""
    if front is None:
        return False
    body = body_clearance if body_clearance is not None else MAX_VALID_RANGE
    return bool(front < PIVOT_FRONT_CLEARANCE or body < PIVOT_BODY_CLEARANCE)


def start_pre_pivot_backup(side, reason="pre-pivot clearance"):
    """Back up a little, then resume the intended 90-degree pivot.

    This is not a normal obstacle recovery. It preserves the lane-change action
    queue and only creates a small physical gap for the front shell/contact pads
    before rotating the circular robot body.
    """
    global nav_state, pre_pivot_start_x, pre_pivot_start_y, pre_pivot_until
    global pre_pivot_side, pre_pivot_reason, prev_cmd_left, prev_cmd_right
    global coverage_status, map_freeze_until
    now = robot.getTime()
    pre_pivot_side = 1.0 if side >= 0 else -1.0
    pre_pivot_reason = reason
    pre_pivot_start_x = pose_x
    pre_pivot_start_y = pose_y
    pre_pivot_until = now + PRE_PIVOT_BACKUP_TIMEOUT_SEC
    nav_state = NAV_PRE_PIVOT_BACKUP
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "pre-pivot backup: " + reason
    acquire_control(ControlOwner.RECOVERY, PRE_PIVOT_BACKUP_TIMEOUT_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)


def begin_lawnmower_lane_change_checked(side, reason, front, body_clearance):
    """Start a lane change, but create pivot clearance when needed.

    Calling begin_lawnmower_lane_change() directly near walls was the source of
    the visible loop: the robot tried to pivot while its front shell/contact pads
    were already close to furniture. The IMU yaw then failed to converge and the
    path curled around the same cells.
    """
    global nav_action_queue, lane_side, last_row_change_time, coverage_status
    side = 1.0 if side >= 0 else -1.0
    if planner_expand_map_motion_active():
        begin_explore_single_turn(side, f"expand-map single turn instead of checked lane change: {reason}")
        return
    nav_action_queue = [("SHIFT", LANE_SPACING), ("TURN", side, "second 90 for next row")]
    last_row_change_time = robot.getTime()
    coverage_status = f"lane change {'L' if side > 0 else 'R'}"
    if pivot_clearance_problem(front, body_clearance):
        start_pre_pivot_backup(side, f"{reason}; clearance F={front:.2f} body={body_clearance:.2f}")
    else:
        begin_pivot_turn(side, reason)
    lane_side = -side


def sensor_origin_world(x, y, theta):
    """World coordinates of the RangeFinder/camera, not of the robot center."""
    sx = x + SENSOR_OFFSET_X * math.cos(theta) - SENSOR_OFFSET_Y * math.sin(theta)
    sy = y + SENSOR_OFFSET_X * math.sin(theta) + SENSOR_OFFSET_Y * math.cos(theta)
    return sx, sy


def project_depth_point(x, y, theta, bearing, forward_depth):
    """Project Webots RangeFinder depth into the floor plane.

    Important: RangeFinder values are depth from the camera image. Treating them
    as a polar ray length makes the map skewed and compressed. For the map we
    use: forward = depth, lateral = signed depth * tan(bearing), then rotate by theta.
    """
    sx, sy = sensor_origin_world(x, y, theta)
    forward = forward_depth
    # bearing > 0 corresponds to the right side of the camera image.
    # Robot local +Y is the left side, therefore camera-right must be -Y.
    lateral = IMAGE_RIGHT_TO_ROBOT_Y_SIGN * forward_depth * math.tan(bearing)
    wx = sx + forward * math.cos(theta) - lateral * math.sin(theta)
    wy = sy + forward * math.sin(theta) + lateral * math.cos(theta)
    return wx, wy


def robust_column_depths(depth, row0, row1, edge_skip=None):
    """Return one robust distance estimate per RangeFinder column.

    A raw depth image contains single-pixel spikes, especially when a wall is
    seen at a shallow angle during rotation. We first compress each column to a
    robust percentile value, then validate it against neighboring columns.

    edge_skip is normally RAY_EDGE_SKIP.  passes a smaller value only for
    weak free-space visibility rays, not for obstacle creation.
    """
    edge_skip = RAY_EDGE_SKIP if edge_skip is None else int(edge_skip)
    edge_skip = clamp(edge_skip, 0, max(0, RF_W // 2 - 2))
    col_dist = np.full(RF_W, np.nan, dtype=np.float32)

    for col in range(edge_skip, RF_W - edge_skip):
        vals = depth[row0:row1, col]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > MIN_VALID_RANGE) & (vals < MAX_VALID_RANGE)]
        if vals.size < MIN_VALID_COLUMN_SAMPLES:
            continue
        # 25th percentile follows real nearby obstacles but is less nervous
        # than min/18th percentile.
        col_dist[col] = float(np.percentile(vals, 25))

    filtered = col_dist.copy()
    half = DEPTH_LOCAL_WINDOW // 2
    for col in range(edge_skip + half, RF_W - edge_skip - half):
        d = col_dist[col]
        if not np.isfinite(d):
            continue
        neigh = col_dist[col-half:col+half+1]
        neigh = neigh[np.isfinite(neigh)]
        if neigh.size < 3:
            filtered[col] = np.nan
            continue
        med = float(np.median(neigh))
        # Reject isolated columns that jump far away from their local context.
        # This removes the long black "whiskers" appearing during turns.
        allowed_jump = max(DEPTH_JUMP_REJECT, 0.22 * med)
        if abs(float(d) - med) > allowed_jump:
            filtered[col] = np.nan

    return filtered


def mapping_angular_rate():
    """Use both encoder omega and the last commanded differential speed.

    Webots updates sensor frames and wheel encoders at discrete steps. During a
    spin, measured omega may lag behind the command by one frame. Taking the max
    of measured and commanded angular rate prevents depth frames from being
    stamped with the wrong heading and creating long fan-shaped artifacts.
    """
    cmd_omega = ((prev_cmd_right * RIGHT_SIGN) - (prev_cmd_left * LEFT_SIGN)) * WHEEL_RADIUS / max(WHEEL_BASE, 1e-6)
    if abs(cmd_omega) > abs(current_angular_velocity):
        return cmd_omega
    return current_angular_velocity


def active_scan_mapping_dwell_ready(now=None):
    """True only during the stationary dwell part of SCAN_AROUND.

    allowed NAV_SCAN_AROUND in MAPPING_ALLOWED_STATES, but the first
    frame of a scan stage still arrives before the new turn command is applied.
    That leaked fan-shaped free-space rays while the robot was about to rotate.
    Mapping during active perception is allowed only after the yaw target is
    reached and the controller is deliberately holding zero wheel speed.
    """
    if now is None:
        now = robot.getTime()
    if nav_state != NAV_SCAN_AROUND:
        return False
    if active_scan_dwell_until <= now:
        return False
    if active_scan_dwell_until - now < ACTIVE_SCAN_DWELL_MAP_MIN_REMAIN_SEC:
        return False
    return True


def route_or_scan_rotation_mapping_reason(now=None):
    """Return a reason string if persistent RGB-D mapping must be frozen.

    This guards the real motion owner, not just nav_state. ROUTE_COMMIT can
    request an in-place turn while nav_state still says FORWARD, and SCAN_AROUND
    is a controlled sequence of turn+dwell states. Persistent occupancy updates
    are therefore blocked for any turn-like primitive, command differential,
    measured omega, recovery state, or non-dwell scan state.
    """
    if now is None:
        now = robot.getTime()
    if nav_state == NAV_SCAN_AROUND and not active_scan_mapping_dwell_ready(now):
        return "scan-turn"
    if nav_state in MAPPING_FREEZE_STATES:
        return f"nav={nav_state}"
    cmd_omega = ((prev_cmd_right * RIGHT_SIGN) - (prev_cmd_left * LEFT_SIGN)) * WHEEL_RADIUS / max(WHEEL_BASE, 1e-6)
    req_omega = ((last_requested_right * RIGHT_SIGN) - (last_requested_left * LEFT_SIGN)) * WHEEL_RADIUS / max(WHEEL_BASE, 1e-6)
    omega = current_angular_velocity
    max_omega = max(abs(cmd_omega), abs(req_omega), abs(omega))
    if max_omega > TURN_MAPPING_OMEGA_LIMIT:
        return f"omega={max_omega:.2f}"
    if nav_state != NAV_FORWARD and last_motion_primitive in (
        MotionPrimitive.TURN_IN_PLACE.value,
        MotionPrimitive.FINE_ALIGN.value,
        MotionPrimitive.CONTACT_RELEASE_TURN.value,
    ):
        return f"prim={last_motion_primitive}"
    return ""


def update_mapping_freeze_state(now=None):
    """Synchronize the mapping-freeze hold and return (frozen, reason)."""
    global map_freeze_until
    if now is None:
        now = robot.getTime()
    reason = route_or_scan_rotation_mapping_reason(now)
    if reason:
        map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)
        return True, reason
    if now < map_freeze_until:
        return True, "hold"
    return False, "clear"


def rgbd_hard_occluder_cell(cx, cy):
    """True only for persistent, wall-like evidence that may cut RGB-D rays.

    Hard evidence is allowed to prevent through-wall mapping.  Weak RGB/CV or
    low log-odds speckles are deliberately excluded here; those are handled by
    the soft-density path below so a single noisy point does not leave a large
    visible floor sector gray.
    """
    if not RGBD_OCCLUSION_GUARD_ENABLED or not map_inside(cx, cy):
        return False
    try:
        if contact_log_odds[cy, cx] > RGBD_OCCLUSION_CONTACT_EPS:
            return True
        if STRUCTURAL_OBSTACLE_MEMORY_ENABLED and structural_log_odds[cy, cx] > RGBD_OCCLUSION_STRUCT_EPS:
            return True
        if log_odds[cy, cx] > RGBD_OCCLUSION_HARD_LOG_EPS:
            return True
    except Exception:
        return False
    return False


def rgbd_soft_occluder_cell(cx, cy):
    """Weak local evidence that may be an occluder only when dense/contiguous."""
    if not RGBD_OCCLUSION_GUARD_ENABLED or not map_inside(cx, cy):
        return False
    try:
        return bool(
            rgbd_hard_occluder_cell(cx, cy)
            or log_odds[cy, cx] > RGBD_OCCLUSION_LOG_EPS
            or visual_log_odds[cy, cx] > RGBD_OCCLUSION_VISUAL_EPS
        )
    except Exception:
        return False


def rgbd_soft_occluder_dense(cx, cy):
    """True when weak occlusion evidence looks like a wall, not an isolated dot."""
    if rgbd_hard_occluder_cell(cx, cy):
        return True
    r = int(RGBD_OCCLUSION_SOFT_DENSE_RADIUS)
    x0 = max(0, int(cx) - r)
    x1 = min(MAP_SIZE, int(cx) + r + 1)
    y0 = max(0, int(cy) - r)
    y1 = min(MAP_SIZE, int(cy) + r + 1)
    try:
        weak = (log_odds[y0:y1, x0:x1] > RGBD_OCCLUSION_LOG_EPS) | (visual_log_odds[y0:y1, x0:x1] > RGBD_OCCLUSION_VISUAL_EPS)
        if STRUCTURAL_OBSTACLE_MEMORY_ENABLED:
            weak = weak | (structural_log_odds[y0:y1, x0:x1] > RGBD_OCCLUSION_STRUCT_EPS)
        weak = weak | (contact_log_odds[y0:y1, x0:x1] > RGBD_OCCLUSION_CONTACT_EPS)
        return int(np.count_nonzero(weak)) >= int(RGBD_OCCLUSION_SOFT_DENSE_COUNT)
    except Exception:
        return False


def rgbd_ray_occluder_cell(cx, cy):
    """Compatibility predicate: hard or dense-soft occlusion at a single cell."""
    return bool(rgbd_hard_occluder_cell(cx, cy) or (rgbd_soft_occluder_cell(cx, cy) and rgbd_soft_occluder_dense(cx, cy)))


def rgbd_free_ray_cells(rx, ry, ex, ey):
    """Cells that may be updated as free before the first known obstacle.

    deliberately does not terminate a ray on one weak black/CV speckle.
    A ray is clipped immediately by hard evidence, or by several dense-soft
    cells in a row.  This keeps the through-wall protection while revealing open
    sectors that the front RGB-D camera actually sees.
    """
    global last_rgbd_occlusion_debug
    cells = list(bresenham(int(rx), int(ry), int(ex), int(ey)))
    if not cells:
        return []
    out = []
    soft_run = 0
    clipped_hard = False
    clipped_soft = False
    for i, (cx, cy) in enumerate(cells):
        if not map_inside(cx, cy):
            continue
        check = i >= RGBD_OCCLUSION_START_SKIP_CELLS
        if check and rgbd_hard_occluder_cell(cx, cy):
            clipped_hard = True
            break
        soft_dense = bool(check and rgbd_soft_occluder_cell(cx, cy) and rgbd_soft_occluder_dense(cx, cy))
        if soft_dense:
            soft_run += 1
            if soft_run >= RGBD_OCCLUSION_SOFT_RUN_CELLS:
                clipped_soft = True
                break
            # Do not clear dense-soft obstacle evidence while deciding whether it
            # is a real wall.  Sparse soft evidence is allowed to be cleared by
            # visible free-space support below.
            continue
        soft_run = 0
        out.append((cx, cy))
    if clipped_hard:
        last_rgbd_occlusion_debug = f"occGuard=hard {len(out)}/{len(cells)}"
    elif clipped_soft:
        last_rgbd_occlusion_debug = f"occGuard=soft {len(out)}/{len(cells)}"
    else:
        last_rgbd_occlusion_debug = f"occGuard=open {len(out)}/{len(cells)}"
    return out


def rgbd_ray_to_feature_occluded(rx, ry, mx, my):
    """Return True if a feature/hit lies behind an already known obstacle."""
    if not RGBD_OCCLUSION_GUARD_ENABLED:
        return False
    cells = list(bresenham(int(rx), int(ry), int(mx), int(my)))
    if len(cells) <= RGBD_OCCLUSION_START_SKIP_CELLS + RGBD_OCCLUSION_ENDPOINT_IGNORE_CELLS:
        return False
    stop = max(RGBD_OCCLUSION_START_SKIP_CELLS, len(cells) - RGBD_OCCLUSION_ENDPOINT_IGNORE_CELLS)
    soft_run = 0
    for cx, cy in cells[RGBD_OCCLUSION_START_SKIP_CELLS:stop]:
        if not map_inside(cx, cy):
            continue
        if rgbd_hard_occluder_cell(cx, cy):
            return True
        if rgbd_soft_occluder_cell(cx, cy) and rgbd_soft_occluder_dense(cx, cy):
            soft_run += 1
            if soft_run >= RGBD_OCCLUSION_SOFT_RUN_CELLS:
                return True
        else:
            soft_run = 0
    return False


def rgbd_sensor_origin_mapping_blocked(x, y, theta):
    """Refuse persistent mapping only if the recessed RGB-D origin is in hard wall evidence."""
    sx, sy = sensor_origin_world(x, y, theta)
    rx, ry = world_to_map(sx, sy)
    if not map_inside(rx, ry):
        return True
    return rgbd_hard_occluder_cell(rx, ry)


def thin_column_depths(depth, row0, row1):
    """Find close/narrow obstacles such as table/chair legs.

    The normal wall filter rejects narrow objects because they look like local
    depth jumps. For navigation this is wrong: table legs are exactly narrow
    close obstacles. This function intentionally looks for close vertical-ish
    clusters in a lower band and later writes them as small occupied marks.
    """
    col_dist = np.full(RF_W, np.nan, dtype=np.float32)
    for col in range(THIN_EDGE_SKIP, RF_W - THIN_EDGE_SKIP):
        vals = depth[row0:row1, col]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > MIN_VALID_RANGE) & (vals < THIN_OBSTACLE_MAX_DIST)]
        if vals.size < int(THIN_OBSTACLE_MIN_COLUMN_SAMPLES):
            continue
        # Low percentile catches the nearest part of a thin leg.
        col_dist[col] = float(np.percentile(vals, 12))
    return col_dist


def extract_thin_obstacle_clusters(col_dist, max_dist):
    """Return (center_col, distance, width) for close narrow occupied clusters."""
    clusters = []
    col = THIN_EDGE_SKIP
    while col < RF_W - THIN_EDGE_SKIP:
        d = col_dist[col]
        if not np.isfinite(d) or d > max_dist:
            col += 1
            continue
        start = col
        vals = []
        while col < RF_W - THIN_EDGE_SKIP:
            d = col_dist[col]
            if not np.isfinite(d) or d > max_dist:
                break
            vals.append(float(d))
            col += 1
        width = col - start
        if THIN_OBSTACLE_MIN_WIDTH <= width <= THIN_OBSTACLE_MAX_WIDTH and vals:
            center_col = int((start + col - 1) / 2)
            # Use lower percentile so the obstacle is not placed behind the real leg.
            dist = float(np.percentile(vals, 30))
            clusters.append((center_col, dist, width))
        col += 1
    return clusters


def mark_visibility_free_wedge(x, y, theta, depth, row0, row1):
    """Weakly mark the currently visible front RGB-D sector as free space.

    This fixes the misleading map view where the camera image clearly sees open
    floor/space, but the map shows gray unknown because CV-first mode only wrote
    a narrow central safety ray and rays to RGB feature points.

    Important: this function never writes raw-depth obstacles.  It only lowers
    log-odds along front rays and refuses to erase cells that already look like
    real obstacles/contact evidence.
    """
    if not CV_VISIBILITY_FREE_WEDGE_ENABLED or depth is None:
        return 0

    sx, sy = sensor_origin_world(x, y, theta)
    rx, ry = world_to_map(sx, sy)
    if not map_inside(rx, ry):
        return 0

    col_dist = robust_column_depths(depth, row0, row1, edge_skip=CV_VISIBILITY_EDGE_SKIP)
    marked = 0
    for col in range(CV_VISIBILITY_EDGE_SKIP, RF_W - CV_VISIBILITY_EDGE_SKIP, CV_VISIBILITY_RAY_STRIDE):
        dist = col_dist[col]
        if np.isfinite(dist):
            free_dist = min(max(0.05, float(dist) - CV_VISIBILITY_HIT_MARGIN), CV_VISIBILITY_FREE_RANGE)
        else:
            # Open/no-hit columns are common when the RangeFinder sees far wall/empty
            # space.  Paint only a capped free ray, otherwise a missed hit would
            # incorrectly clear far behind obstacles.
            free_dist = CV_VISIBILITY_NO_HIT_RANGE

        if free_dist <= 0.05:
            continue
        bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV
        ex_w, ey_w = project_depth_point(x, y, theta, bearing, free_dist)
        ex, ey = world_to_map(ex_w, ey_w)
        for cx, cy in rgbd_free_ray_cells(rx, ry, ex, ey):
            # Do not let the visibility layer delete real obstacles.  It is only
            # supposed to turn unknown/weakly-free cells into observed-free cells.
            if log_odds[cy, cx] > CV_VISIBILITY_MAX_OCC_TO_CLEAR:
                continue
            log_odds[cy, cx] = clamp(log_odds[cy, cx] + CV_VISIBILITY_FREE_UPDATE, LO_MIN, LO_MAX)
            clear_contact_evidence_cell(cx, cy, CONTACT_FREE_UPDATE * 0.20)
            marked += 1
    return marked


def update_map_from_depth(pose, depth):
    """Persistent occupancy mapping using filtered RangeFinder rays.

    Important behavior:
    - normal wall/free-space mapping is disabled while the robot is turning;
    - close thin obstacles are handled separately so chair/table legs are not
      removed as "noise";
    - free-space is never painted aggressively during rotation, because that was
      the source of false empty sectors behind real walls.
    """
    global map_freeze_until, last_map_frozen, last_thin_hits
    x, y, theta = pose
    last_thin_hits = 0


    sx, sy = sensor_origin_world(x, y, theta)
    rx, ry = world_to_map(sx, sy)
    robot_mx, robot_my = world_to_map(x, y)
    if not map_inside(rx, ry):
        return 0
    if rgbd_sensor_origin_mapping_blocked(x, y, theta):
        globals()["last_rgbd_occlusion_debug"] = "occGuard=sensor-origin-blocked"
        return 0

    if map_inside(robot_mx, robot_my):
        cv2.circle(log_odds, (robot_mx, robot_my), 6, LO_FREE_UPDATE, -1)

    # ROUTE_COMMIT may be outputting TURN_IN_PLACE while nav_state is still
    # FORWARD, and SCAN_AROUND has both turning and stationary dwell sub-phases.
    omega = mapping_angular_rate()
    now = robot.getTime()
    mapping_frozen, freeze_reason = update_mapping_freeze_state(now)
    last_map_frozen = mapping_frozen
    if mapping_frozen:
        globals()["last_rgbd_occlusion_debug"] = f"mapFreeze={freeze_reason[:18]}"
    turning_fast = mapping_frozen
    turning_moderate = mapping_frozen or abs(omega) > TURN_THIN_OMEGA_LIMIT

    wall_row0 = clamp(int(RF_H * DEPTH_ROWS_FRACTION[0]), 0, RF_H - 1)
    wall_row1 = clamp(int(RF_H * DEPTH_ROWS_FRACTION[1]), wall_row0 + 1, RF_H)
    low_row0 = clamp(int(RF_H * LOW_THIN_ROWS_FRACTION[0]), 0, RF_H - 1)
    low_row1 = clamp(int(RF_H * LOW_THIN_ROWS_FRACTION[1]), low_row0 + 1, RF_H)

    hit_count = 0

    # Hard rule: do not write depth rays while turning or immediately after a
    # turn. This is the only reliable way to remove persistent fan spikes with
    # a front depth camera + wheel odometry. Press R/reset old maps after update.
    if mapping_frozen or nav_state not in MAPPING_ALLOWED_STATES:
        return 0

    # CV-FIRST mode: do not draw the entire depth image as a lidar-like fan.
    # Raw depth only maintains a weak narrow safety/free corridor. Real obstacle
    # geometry is added by update_visual_map_from_rgb_depth(): RGB/OpenCV first
    # finds visual structures, then depth confirms their metric distance.
    if CV_FIRST_MAPPING:
        # front RGB-D camera's visible sector.  Then keep the old narrow central
        # safety corridor for close obstacle support.
        hit_count += mark_visibility_free_wedge(x, y, theta, depth, wall_row0, wall_row1)

        col_dist = robust_column_depths(depth, wall_row0, wall_row1)
        half = max(2, int(RF_W * DEPTH_SAFETY_CENTER_FRACTION * 0.5))
        c0 = max(RAY_EDGE_SKIP, RF_W // 2 - half)
        c1 = min(RF_W - RAY_EDGE_SKIP, RF_W // 2 + half)
        for col in range(c0, c1, DEPTH_SAFETY_RAY_STRIDE):
            dist = col_dist[col]
            if not np.isfinite(dist):
                continue
            dist = float(dist)
            bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV

            # Free-space support is short and weak: enough to show the cleaned
            # corridor, not enough to fake a complete lidar scan.
            free_dist = min(max(0.04, dist - 0.08), DEPTH_SAFETY_FREE_RANGE)
            if free_dist > 0.03:
                end_free_x, end_free_y = project_depth_point(x, y, theta, bearing, free_dist)
                ex, ey = world_to_map(end_free_x, end_free_y)
                for cx, cy in rgbd_free_ray_cells(rx, ry, ex, ey):
                    log_odds[cy, cx] = clamp(log_odds[cy, cx] + DEPTH_SAFETY_FREE_UPDATE, LO_MIN, LO_MAX)
                    clear_contact_evidence_cell(cx, cy, CONTACT_FREE_UPDATE * 0.35)

            # A very close central obstacle is marked weakly for safety only.
            # Furniture/walls farther away must come from RGB-D CV fusion.
            if dist < DEPTH_SAFETY_OCC_RANGE:
                hx, hy = project_depth_point(x, y, theta, bearing, dist)
                mx, my = world_to_map(hx, hy)
                if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
                    cv2.circle(log_odds, (mx, my), 1, DEPTH_SAFETY_OCC_UPDATE, -1)
                    hit_count += 1

        # RangeFinder depth channel. Returning here previously skipped the thin
        # obstacle pass, so small objects seen after a turn often stayed gray
        # unless OpenCV happened to detect an RGB edge. This remains RGB-D mapping:
        # depth contributes bounded close/narrow support, not a lidar-like fan.
        thin_cols = thin_column_depths(depth, low_row0, low_row1)
        for col, dist, width in extract_thin_obstacle_clusters(thin_cols, THIN_OBSTACLE_MAX_DIST):
            bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV
            hx, hy = project_depth_point(x, y, theta, bearing, dist)
            mx, my = world_to_map(hx, hy)
            if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
                if mark_thin_obstacle_candidate(mx, my):
                    hit_count += 1
                    last_thin_hits += 1
        return hit_count

    # 1) Normal map update. Disabled during fast turns, because this is exactly
    # where the long wall "whiskers" are generated.
    if not turning_fast:
        col_dist = robust_column_depths(depth, wall_row0, wall_row1)
        for col in range(RAY_EDGE_SKIP, RF_W - RAY_EDGE_SKIP, RAY_STRIDE):
            dist = col_dist[col]
            if not np.isfinite(dist):
                continue
            dist = float(dist)

            bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV
            is_hit = dist < MAX_VALID_RANGE - OBSTACLE_RANGE_MARGIN

            if is_hit:
                free_dist = max(0.04, dist - 0.07)
            else:
                free_dist = min(dist, NO_HIT_FREE_RANGE)

            # Do not paint a long free wedge when robot is rotating slowly.
            if turning_moderate:
                free_dist = min(free_dist, 0.12)

            if free_dist > 0.03:
                end_free_x, end_free_y = project_depth_point(x, y, theta, bearing, free_dist)
                ex, ey = world_to_map(end_free_x, end_free_y)
                free_update = LO_FREE_UPDATE * (0.35 if turning_moderate else 1.0)
                for cx, cy in rgbd_free_ray_cells(rx, ry, ex, ey):
                    log_odds[cy, cx] = clamp(log_odds[cy, cx] + free_update, LO_MIN, LO_MAX)
                    clear_contact_evidence_cell(cx, cy, CONTACT_FREE_UPDATE * 0.60)

            if is_hit:
                hx, hy = project_depth_point(x, y, theta, bearing, dist)
                mx, my = world_to_map(hx, hy)
                if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
                    cv2.circle(log_odds, (mx, my), 1, LO_OCC_UPDATE, -1)
                    hit_count += 1

    # 2) Thin close obstacle update. This is intentionally separate from the
    # wall filter: narrow table/chair legs must not be rejected as depth spikes.
    # During moderate/fast turns only very close thin obstacles are accepted.
    thin_max_dist = THIN_OBSTACLE_TURN_MAX_DIST if turning_moderate else THIN_OBSTACLE_MAX_DIST
    thin_cols = thin_column_depths(depth, low_row0, low_row1)
    for col, dist, width in extract_thin_obstacle_clusters(thin_cols, thin_max_dist):
        # Side columns are allowed here: table/chair legs often appear at the
        # edge of the frontal camera. Rotation is already handled by freeze.
        bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV
        hx, hy = project_depth_point(x, y, theta, bearing, dist)
        mx, my = world_to_map(hx, hy)
        if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
            # Temporally confirm lower-band thin candidates before they become
            # persistent raw obstacles.
            if mark_thin_obstacle_candidate(mx, my):
                hit_count += 1
                last_thin_hits += 1

    # Do not run a second sparse lower-band pass here. It made the map fill with
    # persistent horizontal black stripes after long runs. Thin obstacles are
    # handled only through validated clusters above.

    return hit_count

def depth_sectors(depth):
    """Return robust left/center/right distances for navigation.

    Earlier version used a wide vertical band and often reacted to the floor,
    robot body, or side wall edges. For navigation we use only the middle rows
    of the RangeFinder image and ignore extremely close peripheral readings.
    """
    global last_min_left, last_min_center, last_min_right

    r0 = int(RF_H * 0.43)
    r1 = int(RF_H * 0.57)
    band = depth[r0:r1, :]

    def robust_distance(arr):
        vals = arr[np.isfinite(arr)]
        vals = vals[(vals > 0.18) & (vals < MAX_VALID_RANGE)]
        if vals.size == 0:
            return MAX_VALID_RANGE

        # Use percentile, not minimum, to avoid single-pixel spikes.
        return float(np.percentile(vals, 18))

    left = robust_distance(band[:, :RF_W // 3])
    center = robust_distance(band[:, RF_W // 3:2 * RF_W // 3])
    right = robust_distance(band[:, 2 * RF_W // 3:])

    last_min_left, last_min_center, last_min_right = left, center, right
    return left, center, right


def depth_front_narrow(depth):
    """Distance in a narrow corridor directly in front of the robot.

    This is intentionally different from the visual/diagnostic center third.
    A wide center sector reacts to side walls and furniture edges and makes the
    robot turn too early. For deciding that a cleaning row has ended, only the
    narrow physical corridor in front of the robot should matter.
    """
    global last_front_narrow
    r0 = int(RF_H * 0.42)
    r1 = int(RF_H * 0.62)
    half = max(2, int(RF_W * FRONT_NARROW_FRACTION * 0.5))
    c0 = max(0, RF_W // 2 - half)
    c1 = min(RF_W, RF_W // 2 + half)
    band = depth[r0:r1, c0:c1]
    vals = band[np.isfinite(band)]
    vals = vals[(vals > 0.16) & (vals < MAX_VALID_RANGE)]
    if vals.size == 0:
        last_front_narrow = MAX_VALID_RANGE
    else:
        # A relatively low percentile detects a real central obstacle while
        # ignoring single bad pixels less than the raw minimum would.
        last_front_narrow = float(np.percentile(vals, 20))
    return last_front_narrow


def depth_front_upper_corridor(depth):
    """Distance in the upper-middle narrow corridor.

    Carpet edges and shadows often appear as close readings only near the floor.
    A real wall/chair/table leg normally also affects the upper-middle rows.
    This helper is used only to suppress false row-end turns, not to build the map.
    """
    global last_front_upper
    r0 = int(RF_H * 0.24)
    r1 = int(RF_H * 0.43)
    half = max(2, int(RF_W * FRONT_NARROW_FRACTION * 0.5))
    c0 = max(0, RF_W // 2 - half)
    c1 = min(RF_W, RF_W // 2 + half)
    band = depth[r0:r1, c0:c1]
    vals = band[np.isfinite(band)]
    vals = vals[(vals > 0.16) & (vals < MAX_VALID_RANGE)]
    if vals.size == 0:
        last_front_upper = MAX_VALID_RANGE
    else:
        last_front_upper = float(np.percentile(vals, 20))
    return last_front_upper


def depth_overhead_structure_distance(depth):
    """Return distance to an upper/front structure such as a table edge.

    This is used only for the under-surface coverage layer. A passable zone
    under furniture should be marked when the lower body corridor is open, but
    the upper band contains a nearby table/chair surface. Empty open space must
    not be labelled as "under furniture" just because it is uncleaned.
    """
    if depth is None:
        return MAX_VALID_RANGE
    r0 = int(RF_H * 0.10)
    r1 = int(RF_H * 0.34)
    half = max(3, int(RF_W * 0.34))
    c0 = max(0, RF_W // 2 - half)
    c1 = min(RF_W, RF_W // 2 + half)
    band = depth[r0:r1, c0:c1]
    vals = band[np.isfinite(band)]
    vals = vals[(vals > 0.16) & (vals < MAX_VALID_RANGE)]
    if vals.size == 0:
        return MAX_VALID_RANGE
    return float(np.percentile(vals, 18))


def depth_body_corridor_clearance(depth):
    """Return the nearest obstacle clearance in the robot body's swept corridor.

    A low front RGB-D camera can see a gap that is visually open along the
    central ray, while the circular body is still too wide to pass. This function
    projects depth columns into the robot frame and checks the whole body radius,
    not only the central camera ray. The result is distance from the physical
    front/side shell to the obstacle, not raw camera depth.
    """
    global last_body_corridor_clearance, last_body_corridor_lateral
    if depth is None:
        last_body_corridor_clearance = MAX_VALID_RANGE
        last_body_corridor_lateral = 0.0
        return last_body_corridor_clearance

    r0 = int(RF_H * BODY_CORRIDOR_ROW_FRACTION[0])
    r1 = int(RF_H * BODY_CORRIDOR_ROW_FRACTION[1])
    r0 = max(0, min(RF_H - 1, r0))
    r1 = max(r0 + 1, min(RF_H, r1))
    band = depth[r0:r1, :]

    best_clearance = MAX_VALID_RANGE
    best_lateral = 0.0
    # Scan every second column; enough for a 160 px RangeFinder and cheaper than
    # processing every pixel. Use a low percentile per column to keep thin legs.
    for col in range(0, RF_W, 2):
        vals = band[:, col]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > BODY_CORRIDOR_VALID_MIN) & (vals < MAX_VALID_RANGE)]
        if vals.size == 0:
            continue
        dist = float(np.percentile(vals, 22))
        bearing = ((col + 0.5) / RF_W - 0.5) * RF_FOV
        lateral = IMAGE_RIGHT_TO_ROBOT_Y_SIGN * dist * math.tan(bearing) + SENSOR_OFFSET_Y
        abs_lat = abs(lateral)
        # Check an inflated swept shell, not only the exact visual cylinder.
        # Without this, a chair/table leg near the front corner can be accepted
        # as "outside the body" and the circular shell then physically clips it.
        body_check_radius = ROBOT_BODY_RADIUS + BODY_ENVELOPE_MARGIN
        if abs_lat > body_check_radius:
            continue

        # Circular front shell. At a given lateral offset the body front is not
        # always x=radius; near the side it is sqrt(r^2-y^2). Use the inflated
        # radius here too, because the robot may yaw/drift a few cm while driving
        # under furniture. This is a virtual safety footprint, not a fake wall.
        lat_for_shell = min(abs_lat, body_check_radius)
        shell_front_x = math.sqrt(max(0.0, body_check_radius * body_check_radius - lat_for_shell * lat_for_shell))
        obstacle_x = SENSOR_OFFSET_X + dist
        clearance = obstacle_x - shell_front_x
        if clearance < best_clearance:
            best_clearance = clearance
            best_lateral = lateral

    last_body_corridor_clearance = best_clearance
    last_body_corridor_lateral = best_lateral
    return best_clearance


def is_floor_or_shadow_false_front_block(front, center, upper_front):
    """True when the row-end detector is probably seeing carpet/shadow, not a wall.

    The previous version required an explicit OpenCV floor/shadow rejection in
    the same frame. That was too fragile: the robot could see a carpet lip in
    depth, but OpenCV did not always count it as a rejected contour, so the
    controller still started a 90-degree turn.

    New rule: if the lower narrow corridor is close but the upper corridor and
    middle depth sector are open, and RGB-D did not confirm a real frontal
    obstacle, keep driving. This rejects carpet/shadow/lower-floor artifacts
    without treating side walls as row endings.
    """
    visual_floor_or_shadow = (last_cv_floor_rejected > 0) or (last_cv_shadow_rejected > 0)
    low_depth_block = front < ROW_END_CONFIRM_DISTANCE
    upper_open = upper_front > FLOOR_FALSE_BLOCK_UPPER_OPEN
    center_open = center > FLOOR_FALSE_BLOCK_CENTER_OPEN
    cv_not_confirming_real_front = last_cv_front_obstacle > ROW_END_CONFIRM_DISTANCE

    # Strong CV case: OpenCV already says the visual feature is floor/shadow.
    # Do not require the middle depth sector to be open here: the sector overlaps
    # the lower rows, so a carpet lip/shadow can make both `front` and `center`
    # look close even though the upper corridor is open. Requiring center_open
    # made the robot treat carpet/shadow artifacts as chair legs.
    cv_floor_case = visual_floor_or_shadow and low_depth_block and upper_open and cv_not_confirming_real_front

    # Geometry-only fallback for carpet lips: close only in the low/narrow
    # corridor while the upper/middle view is clearly open. This is intentionally
    # conservative: it still requires RGB-D not to confirm a frontal object.
    geometry_floor_case = (
        low_depth_block
        and upper_front > (FLOOR_FALSE_BLOCK_UPPER_OPEN + 0.18)
        and center > (FLOOR_FALSE_BLOCK_CENTER_OPEN + 0.18)
        and cv_not_confirming_real_front
    )
    return bool(cv_floor_case or geometry_floor_case)


def forward_under_surface_cells(max_forward_m=UNDER_FURNITURE_AHEAD_SCAN_M, half_width_m=UNDER_FURNITURE_AHEAD_HALF_WIDTH_M):
    """Count remembered under-surface objective cells in the real forward corridor.

    This prevents the controller from entering "under furniture" mode just
    because the front depth looks open.  The mode should be a response to actual
    RGB-D under-surface evidence in the footprint-sized corridor ahead, not to a
    generic gap near a wall or box.  The sampling is local; do not build a full
    1000x1000 mask on every control tick.
    """
    total = 0
    hits = 0
    step_f = max(0.055, 1.0 / MAP_SCALE * 5.0)
    step_l = max(0.045, 1.0 / MAP_SCALE * 4.0)
    f = 0.18
    while f <= max_forward_m:
        l = -half_width_m
        while l <= half_width_m:
            wx = pose_x + math.cos(pose_theta) * f - math.sin(pose_theta) * l
            wy = pose_y + math.sin(pose_theta) * f + math.cos(pose_theta) * l
            mx, my = world_to_map(wx, wy)
            if map_inside(mx, my):
                total += 1
                obstacle = (
                    log_odds[my, mx] > LO_OCCUPIED_EPS
                    or visual_log_odds[my, mx] > CV_DISPLAY_DENSE_EPS
                    or contact_log_odds[my, mx] > CONTACT_OCCUPIED_EPS
                )
                if (not obstacle) and under_surface_log_odds[my, mx] > UNDER_SURFACE_EPS:
                    hits += 1
            l += step_l
        f += step_f
    ratio = hits / max(1, total)
    return hits, ratio

def detect_under_furniture_corridor(front, center, upper_front, left, right):
    """Return True only for a plausible furniture-underpass, not open space.

    The old rule was too permissive: any open forward corridor with visual
    features or a nearby side wall could activate "under furniture pass". That
    is why the robot sometimes drove a strange straight strip while the map said
    target=uncleaned/under-surface. A real under-table/chair opportunity needs
    an open body corridor *and* a furniture cue: an upper/overhead surface, or a
    very close side structure with enough depth-confirmed CV hits.
    """
    if not UNDER_FURNITURE_ENABLED:
        return False

    forward_open = front > UNDER_FURNITURE_FRONT_OPEN
    middle_open = center > UNDER_FURNITURE_CENTER_OPEN
    upper_open = upper_front > UNDER_FURNITURE_UPPER_OPEN
    body_open = last_body_corridor_clearance > (BODY_CORRIDOR_PASS_CLEARANCE + 0.035)
    if not (forward_open and middle_open and upper_open and body_open):
        return False

    overhead_like = UNDER_SURFACE_OVERHEAD_MIN < upper_front < UNDER_FURNITURE_OVERHEAD_NEAR_MAX
    very_near_side = min(left, right, last_cv_left_obstacle, last_cv_right_obstacle) < UNDER_FURNITURE_SIDE_NEAR
    rich_cv_depth = last_cv_depth_confirmed >= UNDER_FURNITURE_STRICT_CV_DEPTH_HITS
    under_hits, under_ratio = forward_under_surface_cells()

    # Empty corridor/wall case: open ahead, one side at ~0.8-1.1 m, many generic
    # CV edges. Do not call that furniture.  Require real remembered under-surface
    # cells in the forward corridor; without them this is normal coverage.
    if overhead_like:
        return under_hits >= UNDER_FURNITURE_MIN_AHEAD_CELLS
    return bool(very_near_side and rich_cv_depth and under_hits >= UNDER_FURNITURE_STRICT_MIN_AHEAD_CELLS)


def start_under_furniture_hold(reason="under furniture corridor"):
    """Temporarily keep driving straight through a likely furniture opening."""
    global under_furniture_until, under_furniture_active, last_under_furniture_time, coverage_status, row_end_candidate_count
    now = robot.getTime()
    under_furniture_until = max(under_furniture_until, now + UNDER_FURNITURE_HOLD_SEC)
    under_furniture_active = True
    last_under_furniture_time = now
    row_end_candidate_count = 0
    coverage_status = reason


def contact_obstacle_side_from_sensors(left_dist, right_dist):
    """Return +1 for a likely obstacle on the robot left, -1 for right, 0 unknown.

    This is a local contact-risk estimate, not semantic furniture recognition.
    It combines the side depth sectors with the body-envelope lateral value.
    """
    if last_body_corridor_clearance < LEG_TRAP_BODY_CLEARANCE and abs(last_body_corridor_lateral) > 0.035:
        return 1.0 if last_body_corridor_lateral > 0.0 else -1.0
    if left_dist < LEG_TRAP_EDGE_GRAZE_DISTANCE and right_dist > left_dist + 0.035:
        return 1.0
    if right_dist < LEG_TRAP_EDGE_GRAZE_DISTANCE and left_dist > right_dist + 0.035:
        return -1.0
    return 0.0


def choose_contact_escape_side(bumper_left, bumper_right, left_dist, right_dist, obstacle_side=0.0):
    """Choose a physical escape side, not a coverage-planning side.

    +1 means turn/escape left, -1 means turn/escape right.  For a centre or
    both-bumper hit the old code often used coverage_target_side_or_default(),
    which could deliberately choose the side where a coverage target exists even
    if that side is physically worse.  Contact recovery must be governed by
    free clearance first; the coverage target can wait.
    """
    if bumper_left and not bumper_right:
        return -1.0
    if bumper_right and not bumper_left:
        return 1.0
    if obstacle_side > 0.0:
        return -1.0
    if obstacle_side < 0.0:
        return 1.0

    # Centre/both-bumper hit: choose the more open side.  Add a tiny hysteresis
    # toward the current lane only when side clearances are almost identical.
    if left_dist > right_dist + 0.04:
        return 1.0
    if right_dist > left_dist + 0.04:
        return -1.0
    return 1.0 if lane_side >= 0 else -1.0


def contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left_dist, right_dist):
    """Estimate where a bumper-confirmed contact belongs on the local map.

    +1 = robot-left/front-left arc, -1 = robot-right/front-right arc, 0 =
    straight front contact. Center contact is deliberately mapped in front, not
    arbitrarily to one side.
    """
    if bumper_left and not bumper_right:
        return 1.0
    if bumper_right and not bumper_left:
        return -1.0
    if last_bumper_center:
        return 0.0
    return contact_obstacle_side_from_sensors(left_dist, right_dist)


def mark_contact_obstacle(obstacle_side, distance=0.20, force_current_pose=False):
    """Write a small contact-confirmed obstacle at the physical impact point.

    This layer is now reserved for *physical* contact only.  Proximity guards,
    hidden-stall heuristics and low-object near misses may start a recovery
    manoeuvre, but they must not create cyan/yellow contact obstacles; otherwise
    the robot invents walls 15-20 cm before touching them.

    A bumper hit is not located at the robot centre and it is not located where
    the robot happens to be after backing up.  Use the latched pose from the
    first bumper frame when available, then project a fixed shell contact point:
    - centre/both bumper: front of the circular shell;
    - left/right bumper: front-side shoulder on the touched side, never behind the wheel axis.

    The distance argument is kept for compatibility with older call sites, but
    it no longer pulls the contact mark back under the robot when depth reports a
    tiny number.  Depth is unreliable during a physical hit; the shell geometry is
    the correct source of the contact point.
    """
    global last_contact_map_cells, last_contact_latch_used

    now = robot.getTime()
    physical_bumper_now = bool(
        last_bumper_raw_left > BUMPER_ACTIVE_THRESHOLD
        or last_bumper_raw_center > BUMPER_ACTIVE_THRESHOLD
        or last_bumper_raw_right > BUMPER_ACTIVE_THRESHOLD
    )
    recent_latch = bool((not last_contact_latch_used) and (now - last_contact_latch_time <= CONTACT_LATCH_VALID_SEC))
    if (not force_current_pose) and (not physical_bumper_now) and (not recent_latch):
        # Do not write a contact-confirmed obstacle from mere RGB-D/depth fear.
        # Normal occupancy/visual layers already describe the wall/object at its
        # sensed position; the contact layer is only for bumper-confirmed hits.
        last_contact_map_cells = int(np.count_nonzero(contact_log_odds > CONTACT_OCCUPIED_EPS))
        return False

    use_latch = (
        (not force_current_pose)
        and (not last_contact_latch_used)
        and (now - last_contact_latch_time <= CONTACT_LATCH_VALID_SEC)
    )
    if use_latch:
        base_x = last_contact_latch_x
        base_y = last_contact_latch_y
        base_theta = last_contact_latch_theta
        # If the current obstacle_side is ambiguous, reconstruct it from the
        # raw bumper that created the latch.
        if abs(obstacle_side) < 0.5:
            if last_contact_latch_center or (last_contact_latch_left and last_contact_latch_right):
                obstacle_side = 0.0
            elif last_contact_latch_left:
                obstacle_side = 1.0
            elif last_contact_latch_right:
                obstacle_side = -1.0
        last_contact_latch_used = True
    else:
        base_x = pose_x
        base_y = pose_y
        base_theta = pose_theta

    if abs(obstacle_side) < 0.5:
        local_x = CONTACT_FRONT_OFFSET_M
        local_y = 0.0
    else:
        side = 1.0 if obstacle_side > 0.0 else -1.0
        local_x = CONTACT_SIDE_FORWARD_OFFSET_M
        local_y = side * CONTACT_SIDE_LATERAL_OFFSET_M

    wx = base_x + math.cos(base_theta) * local_x - math.sin(base_theta) * local_y
    wy = base_y + math.sin(base_theta) * local_x + math.cos(base_theta) * local_y
    mx, my = world_to_map(wx, wy)
    if map_inside(mx, my):
        r = max(1, int(round(CONTACT_MARK_RADIUS_M * MAP_SCALE)))
        cv2.circle(contact_log_odds, (mx, my), r, CONTACT_OCC_UPDATE, -1)
        # Also support the normal obstacle/CV layers, but keep this support very
        # compact. One bumper hit is a point obstacle, not a furniture blob.
        support_r = max(1, r)
        cv2.circle(log_odds, (mx, my), support_r, LO_OCC_UPDATE * 1.15, -1)
        cv2.circle(visual_log_odds, (mx, my), support_r, CV_VISUAL_UPDATE * 0.75, -1)
        last_contact_map_cells = int(np.count_nonzero(contact_log_odds > CONTACT_OCCUPIED_EPS))
        return True
    return False


def passable_side_contact_gap(front, center, upper_front, left, right, body_clearance):
    """True when a side scrape happened at a passable opening.

    In this situation a full trap escape is wrong: the robot has not discovered a
    dead end, it has merely touched one corner of a narrow mouth.  Release the
    bumper with a tiny reverse and then crawl through with centering.
    """
    if not GAP_CONTACT_NUDGE_ENABLED:
        return False
    if last_floor_front_ignore:
        return False
    if upper_front < GAP_MOUTH_UPPER_OPEN_M:
        return False
    if body_clearance < GAP_MOUTH_BODY_MIN_M:
        return False
    if max(left, right) < GAP_MOUTH_MIN_OPEN_SIDE_M:
        return False
    if front < GAP_MOUTH_HARD_STOP_FRONT_M and center < GAP_MOUTH_HARD_STOP_CENTER_M:
        return False
    return bool(
        front > GAP_MOUTH_HARD_STOP_FRONT_M
        or center > GAP_MOUTH_HARD_STOP_CENTER_M
        or abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M
    )


def start_gap_contact_nudge(escape_side, reason):
    """Small release + nudge for a passable gap-mouth side contact."""
    global nav_state, leg_escape_side, leg_escape_start_x, leg_escape_start_y, leg_escape_until
    global leg_escape_backup_distance_current, leg_escape_turn_angle_current, leg_escape_forward_distance_current, leg_escape_forward_speed_current
    global leg_escape_replan_after, last_leg_escape_time, last_turn_variant, last_contact_escape_kind
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until, nav_action_queue, row_end_candidate_count
    now = robot.getTime()
    nav_action_queue = []
    row_end_candidate_count = 0
    leg_escape_side = 1.0 if escape_side >= 0 else -1.0
    leg_escape_start_x = pose_x
    leg_escape_start_y = pose_y
    leg_escape_backup_distance_current = GAP_CONTACT_NUDGE_BACKUP_DISTANCE
    leg_escape_turn_angle_current = GAP_CONTACT_NUDGE_TURN_ANGLE
    leg_escape_forward_distance_current = GAP_CONTACT_NUDGE_FORWARD_DISTANCE
    leg_escape_forward_speed_current = GAP_CONTACT_NUDGE_FORWARD_SPEED
    leg_escape_until = now + GAP_CONTACT_NUDGE_BACKUP_TIMEOUT_SEC
    leg_escape_replan_after = False
    last_leg_escape_time = now
    last_turn_variant = "gap-contact-nudge"
    last_contact_escape_kind = "gap"
    nav_state = NAV_LEG_ESCAPE_BACKUP
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = "gap contact nudge backup: " + reason
    map_freeze_until = max(map_freeze_until, now + TURN_FREEZE_HOLD_SEC)

def start_edge_trap_escape(obstacle_side, left_dist, right_dist, reason):
    """Escape a front-side furniture-leg trap and preserve coverage priority."""
    global last_contact_trap_reason, last_contact_trap_time, row_end_candidate_count, nav_action_queue
    now = robot.getTime()
    if now - last_contact_trap_time < LEG_TRAP_COOLDOWN_SEC:
        return False
    if obstacle_side == 0.0:
        # Unknown contact/risk side: choose physically more open clearance.
        # Coverage bias is deliberately ignored during recovery.
        escape_side = choose_contact_escape_side(False, False, left_dist, right_dist, 0.0)
    else:
        # obstacle_side +1 means contact/risk on robot left => escape right (-1).
        escape_side = -1.0 if obstacle_side > 0.0 else 1.0
    mark_contact_obstacle(obstacle_side if obstacle_side != 0.0 else escape_side, min(left_dist, right_dist, BODY_CORRIDOR_PASS_CLEARANCE))
    nav_action_queue = []
    row_end_candidate_count = 0
    last_contact_trap_time = now
    last_contact_trap_reason = reason
    start_leg_escape(escape_side, reason, strong=True)
    return True


def hard_stop_motors():
    """Immediately stop both motors and clear command smoothing.

    This is needed for accurate 90-degree pivots. With the normal smoothing
    filter, the wheels keep rotating during SETTLE and the map shows turns as
    80/100 degrees instead of a clean 90-degree corner.
    """
    global prev_cmd_left, prev_cmd_right, last_requested_left, last_requested_right
    global last_motion_primitive, last_motion_contract_reason
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    last_requested_left = 0.0
    last_requested_right = 0.0
    last_motion_primitive = MotionPrimitive.STOP.value
    last_motion_contract_reason = "hard stop"
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def begin_pivot_turn(direction, reason):
    """Start a true in-place 90-degree turn.

    The previous implementation used only pose_theta as the stopping condition.
    In Webots the motor command, wheel encoders and depth frame are discrete,
    so the robot often overshot and the map showed a diagonal/arc turn.

    This version records wheel-sensor values and stops after the exact wheel
    rotation required for a 90-degree differential-drive pivot. After the pivot
    it snaps the odometry heading to the target right angle.
    """
    global nav_state, turn_target_theta, turn_direction, turn_settle_until
    global turn_start_left, turn_start_right, turn_start_time, turn_best_abs_error, turn_last_progress_time
    global prev_cmd_left, prev_cmd_right, last_turn_reason, last_turn_variant, map_freeze_until, coverage_status, desired_grid_heading

    turn_direction = 1.0 if direction >= 0 else -1.0

    # Turn from the planned cardinal grid heading.  handles visible
    # non-90 corners by realigning before the turn; using physical_yaw + 90 here
    # would make the next row diagonal and poison the coverage matrix.
    base_heading = desired_grid_heading
    turn_target_theta = strict_world_grid_heading(normalize_angle(base_heading + turn_direction * PIVOT_TURN_ANGLE))

    turn_start_left = left_sensor.getValue()
    turn_start_right = right_sensor.getValue()
    turn_start_time = robot.getTime()
    turn_best_abs_error = abs(normalize_angle(turn_target_theta - pose_theta))
    turn_last_progress_time = turn_start_time
    turn_settle_until = 0.0
    last_turn_reason = reason
    last_turn_variant = "grid90"
    nav_state = NAV_TURN_90
    short_reason = str(reason)[:54]
    coverage_status = f"pivot 90 target={math.degrees(turn_target_theta):.0f} {short_reason}"

    # Kill forward inertia in the command filter and motors before pivoting.
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    acquire_control(ControlOwner.PIVOT_90, PIVOT_OWNER_TIME_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)


def finish_pivot_turn():
    global nav_state, turn_settle_until, prev_cmd_left, prev_cmd_right, map_freeze_until
    global pose_theta, prev_left, prev_right, current_angular_velocity, current_linear_velocity, desired_grid_heading
    global pivot_watchdog_retry_count, post_turn_rgbd_snapshot_pending, last_post_turn_snapshot_debug

    hard_stop_motors()
    pivot_watchdog_retry_count = 0

    # The new desired heading is an exact grid direction. With IMU present we do
    # not fake the current theta; we let the heading-lock controller correct the
    # remaining 1-2 degrees physically. Without IMU, snap odometry as fallback.
    desired_grid_heading = turn_target_theta
    if inertial_unit is None:
        pose_theta = turn_target_theta

    prev_left = left_sensor.getValue()
    prev_right = right_sensor.getValue()
    current_angular_velocity = 0.0
    current_linear_velocity = 0.0

    nav_state = NAV_SETTLE
    turn_settle_until = robot.getTime() + TURN_SETTLE_SEC
    post_turn_rgbd_snapshot_pending = bool(POST_TURN_RGBD_SNAPSHOT_ENABLED and matrix_first_explore_active() and not known_map_coverage_eval_active())
    if post_turn_rgbd_snapshot_pending:
        last_post_turn_snapshot_debug = f"snap=pending after pivot {math.degrees(turn_target_theta):.0f}"
    map_freeze_until = max(map_freeze_until, turn_settle_until + TURN_FREEZE_HOLD_SEC)

def rgb_motion_signature():
    """Very cheap scalar image signature for hidden-contact/stall detection."""
    try:
        # The count/position of CV points is enough. We intentionally do not run
        # optical flow here; this is a low-cost sanity guard, not a RGB-D navigation module.
        pts = last_cv_features.get("points", []) if isinstance(last_cv_features, dict) else []
        if not pts:
            return float(last_cv_map_hits + last_cv_depth_confirmed)
        sx = 0.0
        sy = 0.0
        n = 0
        for p in pts[:80]:
            if len(p) >= 3:
                sx += float(p[0])
                sy += float(p[1])
                n += 1
        if n <= 0:
            return float(last_cv_map_hits + last_cv_depth_confirmed)
        return float(last_cv_map_hits * 3 + last_cv_depth_confirmed * 5 + sx / n * 0.05 + sy / n * 0.05)
    except Exception:
        return None


def maybe_handle_sensor_stall(front, center, left, right, body_clearance):
    """Escape when the robot appears to keep pushing while sensors barely change.

    Wheel odometry alone cannot detect this in Webots: wheels may rotate while the
    circular body is pressed against a table side. Instead we compare a simple
    depth/RGB signature over time while the controller is in a forward-cleaning
    state. This is intentionally conservative and active mainly near furniture
    or CV clutter.
    """
    global last_sensor_stall_time, last_stall_depth_signature, last_stall_rgb_signature, last_sensor_stall_reason
    if not SENSOR_STALL_ENABLED:
        return False

    now = robot.getTime()
    forward_like = nav_state in (NAV_FORWARD, NAV_LEG_PASS_FORWARD)
    if not forward_like or now - row_start_time < SENSOR_STALL_MIN_ROW_AGE_SEC:
        last_sensor_stall_time = now
        last_stall_depth_signature = None
        last_stall_rgb_signature = None
        return False

    # Only watch for hidden contacts in contexts where a collision is plausible.
    near_furniture_context = (
        under_furniture_active
        or coverage_goal_kind == "under-surface"
        or (last_cv_map_hits >= SENSOR_STALL_CONTEXT_CV_HITS and body_clearance < SENSOR_STALL_BODY_CONTEXT_M + 0.03)
        or min(left, right, last_cv_left_obstacle, last_cv_right_obstacle) < SENSOR_STALL_SIDE_CONTEXT_M
        or body_clearance < SENSOR_STALL_BODY_CONTEXT_M
    )
    if wall_hug_candidate(front, center, MAX_VALID_RANGE, left, right, body_clearance):
        # Stable readings while deliberately following a wall are not a hidden
        # collision.  Do not start a reverse escape just because the side view is
        # constant near a long wall.
        last_sensor_stall_time = now
        last_stall_depth_signature = None
        last_stall_rgb_signature = None
        return False

    if not near_furniture_context:
        last_sensor_stall_time = now
        last_stall_depth_signature = None
        last_stall_rgb_signature = None
        return False

    depth_sig = (round(front, 2), round(center, 2), round(left, 2), round(right, 2), round(body_clearance, 2))
    rgb_sig = rgb_motion_signature()
    if last_stall_depth_signature is None:
        last_stall_depth_signature = depth_sig
        last_stall_rgb_signature = rgb_sig
        last_sensor_stall_time = now
        return False

    depth_delta = max(abs(depth_sig[i] - last_stall_depth_signature[i]) for i in range(len(depth_sig)))
    if rgb_sig is None or last_stall_rgb_signature is None:
        rgb_delta = SENSOR_STALL_RGB_DELTA + 1.0
    else:
        rgb_delta = abs(float(rgb_sig) - float(last_stall_rgb_signature))

    if depth_delta > SENSOR_STALL_DEPTH_DELTA or rgb_delta > SENSOR_STALL_RGB_DELTA:
        last_stall_depth_signature = depth_sig
        last_stall_rgb_signature = rgb_sig
        last_sensor_stall_time = now
        return False

    if now - last_sensor_stall_time < SENSOR_STALL_MIN_SEC:
        return False

    # Stalled forward command near furniture: treat this as a recovery condition,
    # but do NOT create a contact-confirmed obstacle unless a bumper/contact latch
    # is actually active.  Stable depth/RGB at 15-20 cm from a wall is just
    # proximity, not proof of collision.
    obstacle_side = contact_obstacle_side_from_sensors(left, right)
    if obstacle_side == 0.0:
        obstacle_side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0
    escape_side = -obstacle_side if obstacle_side != 0.0 else choose_coverage_side(left, right)
    mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
    start_leg_escape(escape_side, f"sensor-stall proximity recovery d={depth_delta:.2f} rgb={rgb_delta:.1f}", strong=True)
    last_sensor_stall_reason = "hidden contact near furniture"
    last_sensor_stall_time = now
    last_stall_depth_signature = None
    last_stall_rgb_signature = None
    return True


def narrow_passage_candidate(front, center, upper_front, left, right, body_clearance):
    """True when the robot is at a doorway/furniture mouth rather than at a wall.

    The key distinction is: front/centre still show a possible corridor, but the
    1:1 body envelope is brushing one side.  A row-end pivot here is wrong; the
    controller must first try to centre the circular body in the opening.
    """
    if not NARROW_PASSAGE_CENTERING_ENABLED or nav_state != NAV_FORWARD:
        return False
    if last_floor_front_ignore:
        return False
    if front < NARROW_PASSAGE_MIN_FRONT_M or center < NARROW_PASSAGE_MIN_CENTER_M:
        return False
    if upper_front < NARROW_PASSAGE_MIN_UPPER_M:
        return False
    if body_clearance < NARROW_PASSAGE_BODY_MIN_M:
        return False

    side_near = min(left, right) < NARROW_PASSAGE_SIDE_NEAR_M
    body_near = body_clearance < NARROW_PASSAGE_BODY_SOFT_M and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M
    cv_side_near = min(last_cv_left_obstacle, last_cv_right_obstacle) < CV_NAV_SIDE_DISTANCE
    return bool(side_near or body_near or cv_side_near or under_furniture_active or coverage_goal_kind == "under-surface")


def narrow_passage_centering_speeds(front, center, upper_front, left, right, body_clearance):
    """Return wheel speeds for a slow aperture-centering pass, or None.

    This is not global path planning.  It is the missing local controller between
    'drive straight' and 'escape turn': when a gap is passable but tight, steer
    away from the near corner while preserving the current row heading.
    """
    global row_end_candidate_count, coverage_status, last_narrow_passage_time
    global last_narrow_passage_start_time, last_narrow_passage_side, last_narrow_passage_reason, last_turn_variant

    if not narrow_passage_candidate(front, center, upper_front, left, right, body_clearance):
        return None

    now = robot.getTime()

    # This is a continuous local controller, not an event trigger.  The previous
    # code returned None on the first frame of a passable opening and during a
    # cooldown window; then the lower-priority edge-trap logic immediately took
    # over and backed the robot away.  Keep producing a slow centering command
    # while the mouth is geometrically passable.
    if last_narrow_passage_start_time < -100.0:
        last_narrow_passage_start_time = now
    elapsed = now - last_narrow_passage_start_time
    if elapsed > NARROW_PASSAGE_MAX_SEC:
        # Reset the attempt timer but do not hand the same frame to escape logic.
        last_narrow_passage_start_time = now
        elapsed = 0.0
    if elapsed > NARROW_PASSAGE_MAX_SEC * 0.82 and body_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
        return None

    if abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M:
        # +lateral = obstacle on robot-left => negative yaw correction => steer right.
        obstacle_side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0
        yaw = -obstacle_side * clamp(
            abs(last_body_corridor_lateral) * NARROW_PASSAGE_LATERAL_KP,
            NARROW_PASSAGE_MIN_YAW,
            NARROW_PASSAGE_MAX_YAW,
        )
        source = f"bodyLat={last_body_corridor_lateral:.2f}"
    elif left < right - 0.035:
        obstacle_side = 1.0
        yaw = -clamp((right - left) * NARROW_PASSAGE_SIDE_KP, NARROW_PASSAGE_MIN_YAW, NARROW_PASSAGE_MAX_YAW)
        source = f"L={left:.2f}<R={right:.2f}"
    elif right < left - 0.035:
        obstacle_side = -1.0
        yaw = clamp((left - right) * NARROW_PASSAGE_SIDE_KP, NARROW_PASSAGE_MIN_YAW, NARROW_PASSAGE_MAX_YAW)
        source = f"R={right:.2f}<L={left:.2f}"
    else:
        obstacle_side = last_narrow_passage_side if abs(last_narrow_passage_side) > 0.1 else 0.0
        yaw = 0.0
        source = "balanced"

    base = NARROW_PASSAGE_UNDER_SPEED if under_furniture_active else NARROW_PASSAGE_SPEED
    # Hard clamp: if the body envelope is already close to the shell, slow down
    # instead of adding a larger yaw that would scrape the corner.
    if body_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
        base *= 0.72
        yaw = clamp(yaw, -NARROW_PASSAGE_MAX_YAW * 0.72, NARROW_PASSAGE_MAX_YAW * 0.72)

    last_narrow_passage_time = now
    last_narrow_passage_side = obstacle_side
    last_narrow_passage_reason = source
    last_turn_variant = "narrow-squeeze"
    row_end_candidate_count = 0
    coverage_status = f"narrow passage squeeze {source} F={front:.2f} C={center:.2f} body={body_clearance:.2f}"
    return base - yaw, base + yaw


def gap_mouth_candidate(front, center, upper_front, left, right, body_clearance):
    """True when the robot is at the corner of a passable mouth/opening.

    This handles the case that narrow_passage_candidate() deliberately rejects:
    the lower direct corridor is already close to a corner, but the upper view
    and one side still indicate a feasible opening.  Treating this as a wall
    creates the repeated pattern: small yaw -> corner hit -> backup -> same hit.
    """
    if not GAP_MOUTH_ALIGN_ENABLED or nav_state != NAV_FORWARD:
        return False
    if last_floor_front_ignore or last_bumper_left or last_bumper_center or last_bumper_right:
        return False
    if upper_front < GAP_MOUTH_UPPER_OPEN_M:
        return False
    if max(left, right) < GAP_MOUTH_MIN_OPEN_SIDE_M:
        return False
    if body_clearance < GAP_MOUTH_BODY_MIN_M:
        return False
    if front < GAP_MOUTH_HARD_STOP_FRONT_M and center < GAP_MOUTH_HARD_STOP_CENTER_M:
        return False

    close_enough = (
        front < GAP_MOUTH_FRONT_SOFT_M
        or center < GAP_MOUTH_CENTER_SOFT_M
        or body_clearance < GAP_MOUTH_BODY_SOFT_M
    )
    asymmetric_opening = abs(left - right) > GAP_MOUTH_ASYMMETRY_M
    one_side_near = min(left, right) < GAP_MOUTH_SIDE_NEAR_M
    body_corner = body_clearance < GAP_MOUTH_BODY_SOFT_M and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M
    return bool(close_enough and (asymmetric_opening or one_side_near or body_corner or coverage_goal_kind == "under-surface"))


def gap_mouth_obstacle_side(left, right, body_clearance):
    """Return +1 if the near corner is on robot-left, -1 if on robot-right."""
    if body_clearance < GAP_MOUTH_BODY_SOFT_M and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M:
        return 1.0 if last_body_corridor_lateral > 0.0 else -1.0
    if left < right - GAP_MOUTH_ASYMMETRY_M:
        return 1.0
    if right < left - GAP_MOUTH_ASYMMETRY_M:
        return -1.0
    if last_gap_mouth_align_side != 0.0:
        return last_gap_mouth_align_side
    return 1.0 if left <= right else -1.0


def gap_mouth_alignment_speeds(front, center, upper_front, left, right, body_clearance):
    """Return slow wheel speeds to centre through a partially blocked mouth."""
    global row_end_candidate_count, coverage_status, last_gap_mouth_align_time
    global last_gap_mouth_align_side, last_gap_mouth_align_reason, last_turn_variant

    if not gap_mouth_candidate(front, center, upper_front, left, right, body_clearance):
        return None
    now = robot.getTime()
    if now - last_gap_mouth_align_time < GAP_MOUTH_COOLDOWN_SEC:
        return None

    obstacle_side = gap_mouth_obstacle_side(left, right, body_clearance)
    # obstacle_side +1 means obstacle on robot-left; yaw must be negative to
    # steer right.  The correction is small: this is a squeeze/crawl, not a pivot.
    if body_clearance < GAP_MOUTH_BODY_SOFT_M and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M:
        yaw_mag = abs(last_body_corridor_lateral) * GAP_MOUTH_BODY_LAT_KP
        source = f"gap bodyLat={last_body_corridor_lateral:.2f}"
    else:
        yaw_mag = abs(left - right) * GAP_MOUTH_YAW_KP
        source = f"gap L={left:.2f} R={right:.2f}"
    yaw_mag = clamp(yaw_mag, GAP_MOUTH_MIN_YAW, GAP_MOUTH_MAX_YAW)
    yaw = -obstacle_side * yaw_mag

    base = GAP_MOUTH_SPEED
    if front < ROW_END_CONFIRM_DISTANCE + 0.06 or center < SAFE_FRONT_DISTANCE + 0.08 or body_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
        base = GAP_MOUTH_SLOW_SPEED
        yaw = clamp(yaw, -GAP_MOUTH_MAX_YAW * 0.72, GAP_MOUTH_MAX_YAW * 0.72)

    last_gap_mouth_align_time = now
    last_gap_mouth_align_side = obstacle_side
    last_gap_mouth_align_reason = source
    last_turn_variant = "gap-mouth-crawl"
    row_end_candidate_count = 0
    coverage_status = f"gap mouth crawl {source} F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}"
    return base - yaw, base + yaw


def contact_looks_like_wall_boundary():
    """True for a real bumper hit against a flat wall/cabinet boundary.

    This is intentionally different from a chair-leg/corner trap.  A wall is a
    useful cleaning boundary: after a tiny release the correct behavior is to
    rotate parallel and follow it closely, not to retreat far and abandon the
    edge strip.  Low obstacles are excluded by requiring the upper depth band to
    be close as well; a low red block usually has an open upper band.
    """
    if not WALL_CONTACT_RECOVERY_ENABLED:
        return False
    if under_furniture_active or coverage_goal_kind == "under-surface":
        return False
    if last_floor_front_ignore:
        return False

    frontal_boundary = (
        last_front_narrow < WALL_CONTACT_FRONT_MAX_M
        and last_min_center < WALL_CONTACT_CENTER_MAX_M
        and last_front_upper < WALL_CONTACT_UPPER_MAX_M
    )
    if not frontal_boundary:
        return False

    # A doorway/gap mouth needs squeeze/nudge logic, not wall-follow.
    if contact_looks_like_gap_mouth():
        return False

    side_context = min(last_min_left, last_min_right, last_cv_left_obstacle, last_cv_right_obstacle) < WALL_CONTACT_SIDE_NEAR_M
    flat_front = abs(last_front_narrow - last_min_center) < 0.18 and abs(last_front_upper - last_min_center) < 0.24
    body_not_pinched = last_body_corridor_clearance > BODY_CORRIDOR_HARD_CLEARANCE + 0.010
    return bool((side_context or flat_front) and body_not_pinched)


def contact_looks_like_gap_mouth():
    """Conservative test used after a bumper hit near a doorway/furniture mouth."""
    if not GAP_MOUTH_ALIGN_ENABLED:
        return False
    if last_front_upper < GAP_MOUTH_UPPER_OPEN_M:
        return False
    if max(last_min_left, last_min_right) < GAP_MOUTH_MIN_OPEN_SIDE_M:
        return False
    if last_body_corridor_clearance < GAP_MOUTH_BODY_MIN_M:
        return False
    return bool(
        abs(last_min_left - last_min_right) > GAP_MOUTH_ASYMMETRY_M
        or min(last_min_left, last_min_right) < GAP_MOUTH_SIDE_NEAR_M
        or abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M
    )


def post_gap_commit_active(front=None, center=None, body_clearance=None):
    """After a corner nudge, force a short slow commit through the mouth."""
    if nav_state != NAV_FORWARD:
        return False
    now = robot.getTime()
    if now >= post_gap_commit_until:
        return False
    moved = math.hypot(pose_x - post_gap_commit_start_x, pose_y - post_gap_commit_start_y)
    if moved >= POST_GAP_COMMIT_DISTANCE_M:
        return False
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return False
    if front is not None and center is not None and front < GAP_MOUTH_HARD_STOP_FRONT_M and center < GAP_MOUTH_HARD_STOP_CENTER_M:
        return False
    if body_clearance is not None and body_clearance < BODY_CORRIDOR_HARD_CLEARANCE:
        return False
    return True


def wall_hug_candidate(front, center, upper_front, left, right, body_clearance):
    """True for an ordinary wall/cabinet edge that should be followed closely.

    This deliberately excludes under-furniture objectives and real front blocks.
    A side wall at 8-15 cm is not a trap; it is exactly where a vacuum should
    keep moving so the cleaning footprint reaches the border.
    """
    if not WALL_HUG_ENABLED or nav_state != NAV_FORWARD:
        return False
    if under_furniture_active or coverage_goal_kind == "under-surface":
        return False
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return False
    if last_floor_front_ignore:
        return False
    if front < WALL_HUG_MIN_FRONT_M or center < WALL_HUG_MIN_CENTER_M or upper_front < WALL_HUG_MIN_UPPER_M:
        return False
    if body_clearance < WALL_HUG_BODY_MIN_M:
        return False
    near_side = min(left, right, last_cv_left_obstacle, last_cv_right_obstacle)
    if near_side > WALL_HUG_SIDE_NEAR_M:
        return False
    body_side = body_clearance < BODY_CORRIDOR_PASS_CLEARANCE and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M
    # Do not use wall-hug to acquire a far wall.  If the wall is 25-40 cm away,
    # driving forward with a steering bias produces the visible diagonal arc and
    # pulls the robot away from the row/coverage contract.  Wall-hug is a
    # boundary-trace controller, not a lateral approach controller.
    if not body_side and near_side > WALL_HUG_REACQUIRE_SIDE_MAX_M:
        return False
    # Require a dominant side.  Symmetric close readings are a frontal/corner
    # situation and should stay with row-end/recovery logic.
    depth_asym = abs(left - right) > WALL_HUG_SIDE_ASYMMETRY_M
    cv_asym = abs(last_cv_left_obstacle - last_cv_right_obstacle) > WALL_HUG_SIDE_ASYMMETRY_M
    return bool(depth_asym or cv_asym or body_side)


def wall_hug_speeds(front, center, upper_front, left, right, body_clearance):
    """Drive parallel to a nearby wall instead of escaping from it.

    Positive yaw turns left.  If the wall is on the left and clearance is larger
    than the target, yaw left a little; if too close, yaw right.  The command is
    intentionally small so the path remains mostly straight, not a visible arc.
    """
    global row_end_candidate_count, coverage_status, last_turn_variant
    if not wall_hug_candidate(front, center, upper_front, left, right, body_clearance):
        return None

    if body_clearance < BODY_CORRIDOR_PASS_CLEARANCE and abs(last_body_corridor_lateral) > NARROW_PASSAGE_LATERAL_DEADBAND_M:
        wall_side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0
        side_clear = body_clearance
        source = f"body={body_clearance:.2f}@{last_body_corridor_lateral:.2f}"
    else:
        # Prefer the physically nearer side; include CV side evidence only as a
        # tie-breaker because raw depth is the metric source.
        left_metric = min(left, last_cv_left_obstacle)
        right_metric = min(right, last_cv_right_obstacle)
        if left_metric <= right_metric:
            wall_side = 1.0
            side_clear = left_metric
            source = f"L={left_metric:.2f}"
        else:
            wall_side = -1.0
            side_clear = right_metric
            source = f"R={right_metric:.2f}"

    # Keep close, but do not scrape. side_clear > target => steer toward wall;
    # side_clear < target => steer away.  At extremely close clearance slow down.
    # If the wall is not acquired yet, do not draw a diagonal acquire arc.
    if side_clear > WALL_HUG_REACQUIRE_SIDE_MAX_M:
        return None
    if side_clear > WALL_HUG_ACQUIRED_SIDE_MAX_M and body_clearance >= BODY_CORRIDOR_PASS_CLEARANCE:
        return None
    err = side_clear - WALL_HUG_TARGET_CLEARANCE_M
    if abs(err) < WALL_HUG_TARGET_DEADBAND_M:
        err = 0.0
    yaw = wall_side * clamp(err * WALL_HUG_KP, -WALL_HUG_MAX_YAW, WALL_HUG_MAX_YAW)
    base = WALL_HUG_SPEED
    if side_clear < WALL_HUG_TOO_CLOSE_M or body_clearance < BODY_CORRIDOR_HARD_CLEARANCE + 0.012:
        base = WALL_HUG_SLOW_SPEED
        yaw = -wall_side * min(WALL_HUG_MAX_YAW, max(0.10, abs(yaw)))
    elif side_clear < WALL_HUG_TARGET_CLEARANCE_M:
        base = WALL_HUG_SLOW_SPEED

    row_end_candidate_count = 0
    last_turn_variant = "wall-hug"
    coverage_status = f"wall trace {source} target={WALL_HUG_TARGET_CLEARANCE_M:.2f} F={front:.2f} C={center:.2f}"
    return base - yaw, base + yaw


def local_forward_alignment_speeds(front, center, upper_front, left, right, body_clearance):
    """Single arbitration point for soft local FORWARD controllers.

    Keep this as one call-site in choose_motion_from_depth().  Duplicating these
    controllers before/after recovery checks makes the final command depend on
    textual if-order rather than behaviour priority.
    """
    gap_cmd = gap_mouth_alignment_speeds(front, center, upper_front, left, right, body_clearance)
    if gap_cmd is not None:
        return gap_cmd

    squeeze_cmd = narrow_passage_centering_speeds(front, center, upper_front, left, right, body_clearance)
    if squeeze_cmd is not None:
        return squeeze_cmd

    wall_cmd = wall_hug_speeds(front, center, upper_front, left, right, body_clearance)
    if wall_cmd is not None:
        return wall_cmd

    return None


def maybe_handle_under_surface_side_risk(front, center, left, right, body_clearance):
    """Prevent sliding into a table/chair leg during an under-surface approach.

    RGB-D may show a free forward corridor while one side of the circular shell
    is already very close to a furniture leg. In that case waiting for a bumper
    contact is unsafe: mark a local contact obstacle and escape.
    """
    global last_side_risk_escape_time
    now = robot.getTime()
    if now - last_side_risk_escape_time < SIDE_RISK_COOLDOWN_SEC:
        return False
    if nav_state != NAV_FORWARD or last_floor_front_ignore:
        return False
    if not (under_furniture_active or coverage_goal_kind == "under-surface"):
        return False

    side = 0.0
    if left < SIDE_RISK_DISTANCE and right > left + 0.05:
        side = 1.0       # obstacle/risk on robot-left
    elif right < SIDE_RISK_DISTANCE and left > right + 0.05:
        side = -1.0      # obstacle/risk on robot-right
    elif body_clearance < SIDE_RISK_BODY_CLEARANCE and abs(last_body_corridor_lateral) > SIDE_RISK_LATERAL:
        side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0

    if side == 0.0:
        return False

    # A tight-but-open doorway/table mouth should be centred through, not treated
    # as a side trap.  Let narrow_passage_centering_speeds() handle it later.
    if narrow_passage_candidate(front, center, last_front_upper, left, right, body_clearance):
        return False

    if max(front, center, last_front_upper) < LEG_ESCAPE_FRONT_OPEN_DISTANCE:
        return False

    side_dist = left if side > 0.0 else right
    mark_contact_obstacle(side, min(side_dist, body_clearance, BODY_CORRIDOR_PASS_CLEARANCE))
    last_side_risk_escape_time = now
    start_leg_escape(-side, f"side-risk under-surface side={'L' if side > 0 else 'R'} L={left:.2f} R={right:.2f} body={body_clearance:.2f}@{last_body_corridor_lateral:.2f}", strong=True)
    return True


def maybe_handle_odometry_stall(front, center, left, right, body_clearance):
    """Fallback stall detector based on commanded forward motion versus pose delta."""
    global last_motion_stall_time, last_motion_stall_x, last_motion_stall_y, last_sensor_stall_reason
    if not ODOM_STALL_ENABLED:
        return False
    now = robot.getTime()
    forward_like = nav_state in (NAV_FORWARD, NAV_LEG_PASS_FORWARD)
    commanded_forward = (prev_cmd_left > ODOM_STALL_CMD_MIN and prev_cmd_right > ODOM_STALL_CMD_MIN)
    near_furniture_context = (
        under_furniture_active
        or coverage_goal_kind == "under-surface"
        or min(left, right, last_cv_left_obstacle, last_cv_right_obstacle) < SIDE_RISK_DISTANCE
        or body_clearance < SIDE_RISK_BODY_CLEARANCE
    )
    if wall_hug_candidate(front, center, MAX_VALID_RANGE, left, right, body_clearance):
        # Near-wall hugging is allowed to have nearly constant side-depth.  Only
        # real bumpers or a hard frontal block should interrupt it.
        last_motion_stall_time = now
        last_motion_stall_x = pose_x
        last_motion_stall_y = pose_y
        return False

    if not (forward_like and commanded_forward and near_furniture_context):
        last_motion_stall_time = now
        last_motion_stall_x = pose_x
        last_motion_stall_y = pose_y
        return False

    if last_motion_stall_time < -100.0:
        last_motion_stall_time = now
        last_motion_stall_x = pose_x
        last_motion_stall_y = pose_y
        return False

    moved = math.hypot(pose_x - last_motion_stall_x, pose_y - last_motion_stall_y)
    if moved > ODOM_STALL_MIN_PROGRESS:
        last_motion_stall_time = now
        last_motion_stall_x = pose_x
        last_motion_stall_y = pose_y
        return False

    if now - last_motion_stall_time < ODOM_STALL_MIN_SEC:
        return False

    obstacle_side = contact_obstacle_side_from_sensors(left, right)
    if obstacle_side == 0.0:
        obstacle_side = 1.0 if last_body_corridor_lateral > 0.0 else -1.0
    mark_contact_obstacle(obstacle_side, min(front, center, body_clearance, BODY_CORRIDOR_PASS_CLEARANCE))
    start_leg_escape(-obstacle_side, f"odometry-stall hidden contact moved={moved:.3f} L={left:.2f} R={right:.2f} body={body_clearance:.2f}", strong=True)
    last_sensor_stall_reason = "odometry stall near furniture"
    last_motion_stall_time = now
    last_motion_stall_x = pose_x
    last_motion_stall_y = pose_y
    return True


def map_building_active():
    """True while the robot is still opening the room map.

    policy: map construction is contact-led and wall-to-wall.  Coverage
    and cleanup planners are not allowed to begin just because the coverage map
    reached ~60%.  That is exactly where the previous build started a small
    snake in the middle of the room.
    """
    if auto_map_return_to_dock_active or auto_map_cleaning_started:
        return False
    if not MATRIX_FIRST_PHASE_SPLIT_ENABLED:
        return False
    try:
        t = robot.getTime()
    except Exception:
        t = 0.0
    if last_coverage_percent < MAP_BUILDING_UNTIL_COVERAGE_PERCENT or t < MAP_BUILDING_MIN_TIME_SEC:
        return True
    # map with many frontiers should still expand toward boundaries instead of
    # starting a coverage lane-change just because the old early-map timer ended.
    if planner_expand_map_motion_active():
        return True
    return False


def matrix_first_explore_active():
    """True while the robot should build the room map, not start a snake pattern.

    The existing project already has the correct data model for this: log_odds,
    cleaned_mask and the derived cleanable/uncleaned/frontier masks are 2-D
    matrices.  The mistake in was using the coverage/lawnmower manoeuvre
    before those matrices were mature enough.  During this phase, row end means
    only a single 90-degree turn, not 90 -> lateral shift -> 90.
    """
    if not MATRIX_FIRST_PHASE_SPLIT_ENABLED:
        return False
    # the map-building gate is stronger than the phase label.  Even if
    # another part of the old controller calls the phase COVERAGE, the robot must
    # not start cleanup/snake behaviour until the room map is mature.
    return map_building_active()


def hard_core_lawnmower_allowed():
    """Allow strict 90/shift/90 strips only after map-first exploration."""
    if not MATRIX_FIRST_PHASE_SPLIT_ENABLED:
        return True
    if planner_expand_map_motion_active():
        return False
    return not matrix_first_explore_active()


def explore_information_gain_for_heading(heading, length_m=EXPLORE_INFO_GAIN_SCAN_AHEAD_M, width_m=EXPLORE_INFO_GAIN_SCAN_WIDTH_M):
    """Return a scalar value for map-building exploration in a cardinal direction.

    This is deliberately not a route planner.  It only ranks the next 90-degree
    turn after a bumper-confirmed wall/contact.  Unknown/frontier and known
    uncleaned cells are useful; cleaned/recent cells and obstacles are not.
    """
    stats = explore_corridor_matrix_stats(heading, length_m=length_m, width_m=width_m)
    if stats.get("total", 0) < EXPLORE_INFO_GAIN_MIN_SAMPLES:
        return -999.0, stats
    raw_unknown = stats.get("unknown", 0.0)
    frontier = stats.get("frontier", 0.0)
    uncleaned = stats.get("uncleaned", 0.0)
    # Raw unknown alone is dangerous: outside the room, behind a wall, is also
    # unknown.  Reward frontier (unknown next to known free/cleanable cells) and
    # known uncleaned floor; penalize raw-unknown regions that have no frontier
    # support.
    outside_wall_unknown = max(0.0, raw_unknown - frontier - 0.18)
    score = (
        EXPLORE_INFO_GAIN_FRONTIER_WEIGHT * frontier
        + 1.15 * uncleaned
        + EXPLORE_INFO_GAIN_RAW_UNKNOWN_WEIGHT * raw_unknown
        - EXPLORE_INFO_GAIN_OUTSIDE_WALL_UNKNOWN_PENALTY * outside_wall_unknown
        - 0.95 * stats.get("cleaned", 0.0)
        - 0.85 * stats.get("recent", 0.0)
        - 2.20 * stats.get("obstacle", 0.0)
    )
    return score, stats


def choose_explore_info_gain_turn_side(left_dist, right_dist, reason_prefix="explore-info"):
    """Choose L/R for EXPLORE by expected matrix information gain.

    +1 means rotate left by 90 degrees; -1 means rotate right by 90 degrees.
    The function is only a side selector for the next atomic turn.  It must not
    generate waypoints or pull the robot toward a local target, because that was
    the failure mode of the previous matrix-route attempt.
    """
    fallback = choose_explore_turn_side_by_depth(left_dist, right_dist)
    if not (EXPLORE_INFO_GAIN_CONTACT_SIDE_ENABLED and matrix_first_explore_active()):
        return fallback, f"{reason_prefix}: depth fallback"

    candidates = []
    for side, side_depth in ((1.0, left_dist), (-1.0, right_dist)):
        if side_depth < EXPLORE_INFO_GAIN_SIDE_DEPTH_MIN_M:
            continue
        heading = strict_world_grid_heading(desired_grid_heading + side * math.pi / 2.0)
        score, stats = explore_information_gain_for_heading(heading)
        depth_bonus = min(max(float(side_depth), 0.0), EXPLORE_INFO_GAIN_DEPTH_BONUS_M) / EXPLORE_INFO_GAIN_DEPTH_BONUS_M * 0.10
        total_score = score + depth_bonus
        candidates.append((total_score, side, heading, stats, side_depth))

    if not candidates:
        return fallback, f"{reason_prefix}: no side samples -> depth fallback"
    candidates.sort(reverse=True, key=lambda item: item[0])
    best = candidates[0]
    if len(candidates) > 1 and (best[0] - candidates[1][0]) < EXPLORE_INFO_GAIN_SIDE_MARGIN:
        # If matrix gain is almost tied, choose the more physically open side.
        return fallback, f"{reason_prefix}: gain tie {best[0]:.2f}/{candidates[1][0]:.2f} -> depth fallback"
    score, side, heading, stats, side_depth = best
    reason = (
        f"{reason_prefix}: {'L' if side > 0 else 'R'} gain={score:.2f} "
        f"fr={stats.get('frontier',0.0):.2f} unk={stats.get('unknown',0.0):.2f} "
        f"un={stats.get('uncleaned',0.0):.2f} cl={stats.get('cleaned',0.0):.2f} "
        f"obs={stats.get('obstacle',0.0):.2f} d={side_depth:.2f}"
    )
    return side, reason


def choose_explore_contact_turn_side(bumper_left, bumper_right, left_dist, right_dist):
    """Physical contact side selection for bumper-first map building.

    this is now boundary tracing, not exploration scoring.  Side bumper
    contacts still turn away from the physical contact.  A centre/both front wall
    contact uses a deterministic right-hand perimeter rule: turn left 90 degrees
    so the contacted wall becomes the robot's right wall.
    """
    center_or_both = bool(last_bumper_center or (bumper_left and bumper_right))
    if bumper_left and not bumper_right and not center_or_both:
        return -1.0, "single left bumper -> turn right"
    if bumper_right and not bumper_left and not center_or_both:
        return 1.0, "single right bumper -> turn left"
    if BOUNDARY_TRACE_PERIMETER_FIRST_ENABLED and matrix_first_explore_active():
        preferred = BOUNDARY_TRACE_FRONT_CONTACT_TURN_SIDE
        # Do not pivot into a visibly closed side; use the opposite side only as
        # a physical fallback, not because of matrix/info-gain.
        if preferred > 0 and left_dist < SIDE_DISTANCE + 0.08 and right_dist > left_dist + BOUNDARY_TRACE_SIDE_OPEN_MARGIN_M:
            preferred = -1.0
        elif preferred < 0 and right_dist < SIDE_DISTANCE + 0.08 and left_dist > right_dist + BOUNDARY_TRACE_SIDE_OPEN_MARGIN_M:
            preferred = 1.0
        return preferred, "boundary trace contact -> deterministic perimeter turn"
    return choose_explore_info_gain_turn_side(left_dist, right_dist, "contact-info")


def choose_explore_turn_side_by_depth(left_dist, right_dist):
    """Stable depth-only fallback for a single exploration 90-degree turn."""
    if left_dist > right_dist + EXPLORE_TURN_SIDE_HYSTERESIS_M:
        return 1.0
    if right_dist > left_dist + EXPLORE_TURN_SIDE_HYSTERESIS_M:
        return -1.0
    return 1.0 if lane_side >= 0 else -1.0

def choose_explore_turn_side(left_dist, right_dist):
    """Choose a single 90-degree turn during early boundary tracing.

    removes the old exploration/info-gain turn selector from the early
    phase.  The robot first traces a perimeter with a stable hand rule; later
    coverage/fill handles uncleaned cells instead of steering the perimeter loop
    toward raw map scores.
    """
    if BOUNDARY_TRACE_PERIMETER_FIRST_ENABLED and matrix_first_explore_active():
        preferred = BOUNDARY_TRACE_FRONT_CONTACT_TURN_SIDE
        if preferred > 0 and left_dist < SIDE_DISTANCE + 0.08 and right_dist > left_dist + BOUNDARY_TRACE_SIDE_OPEN_MARGIN_M:
            return -1.0
        if preferred < 0 and right_dist < SIDE_DISTANCE + 0.08 and left_dist > right_dist + BOUNDARY_TRACE_SIDE_OPEN_MARGIN_M:
            return 1.0
        return preferred
    side, _reason = choose_explore_info_gain_turn_side(left_dist, right_dist, "front-end-info")
    return side


def explore_corridor_matrix_stats(heading, length_m=EXPLORE_ANTI_REVISIT_SCAN_AHEAD_M, width_m=EXPLORE_ANTI_REVISIT_SCAN_WIDTH_M):
    """Small local matrix probe used only to avoid revisiting a fresh strip.

    It samples a rectangular corridor in a chosen cardinal direction and returns
    ratios for cleaned, recent, uncleaned, unknown and obstacle cells.  The
    obstacle/contact decision still belongs to bumper/recovery; this probe is
    only a turn-side bias for map-building.
    """
    try:
        obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
        # Useful frontier: unknown cells adjacent to known cleanable floor. This
        # avoids scoring the infinite unknown outside the walls as a good target.
        cleanable_u8 = cleanable.astype(np.uint8)
        frontier = unknown & (cv2.dilate(cleanable_u8, np.ones((7, 7), np.uint8), iterations=1) > 0)
    except Exception:
        return {"total": 0, "cleaned": 0.0, "recent": 0.0, "uncleaned": 0.0, "unknown": 0.0, "frontier": 0.0, "obstacle": 0.0, "score": -1.0}

    fx = math.cos(heading)
    fy = math.sin(heading)
    nx = -math.sin(heading)
    ny = math.cos(heading)
    total = 0
    cleaned_n = recent_n = uncleaned_n = unknown_n = frontier_n = obstacle_n = 0
    ahead_steps = max(3, int(round(length_m * MAP_SCALE / 4.0)))
    side_steps = max(1, int(round(width_m * MAP_SCALE / 4.0)))
    for ai in range(2, ahead_steps + 1):
        ahead = ai * 4.0 / MAP_SCALE
        for si in range(-side_steps, side_steps + 1):
            side = si * 4.0 / MAP_SCALE
            wx = pose_x + fx * ahead + nx * side
            wy = pose_y + fy * ahead + ny * side
            mx, my = world_to_map(wx, wy)
            if not map_inside(mx, my):
                continue
            total += 1
            if obstacles[my, mx]:
                obstacle_n += 1
            elif cleaned[my, mx]:
                cleaned_n += 1
            elif uncleaned[my, mx]:
                uncleaned_n += 1
            elif unknown[my, mx]:
                unknown_n += 1
            if frontier[my, mx]:
                frontier_n += 1
            if recent_visit_log_odds[my, mx] > 0.6:
                recent_n += 1
    if total <= 0:
        return {"total": 0, "cleaned": 0.0, "recent": 0.0, "uncleaned": 0.0, "unknown": 0.0, "frontier": 0.0, "obstacle": 0.0, "score": -1.0}
    cleaned_r = cleaned_n / total
    recent_r = recent_n / total
    uncleaned_r = uncleaned_n / total
    unknown_r = unknown_n / total
    frontier_r = frontier_n / total
    obstacle_r = obstacle_n / total
    # Frontier/uncleaned should attract.  Cleaned/recent and obstacles repel.
    # Raw unknown is weighted very low because unknown behind an already mapped
    # wall caused repeated perimeter loops in .
    outside_wall_unknown = max(0.0, unknown_r - frontier_r - 0.18)
    score = (2.0 * frontier_r) + (1.45 * uncleaned_r) + (0.12 * unknown_r) - (0.95 * cleaned_r) - (0.85 * recent_r) - (1.9 * obstacle_r) - (0.85 * outside_wall_unknown)
    return {
        "total": total,
        "cleaned": cleaned_r,
        "recent": recent_r,
        "uncleaned": uncleaned_r,
        "unknown": unknown_r,
        "frontier": frontier_r,
        "obstacle": obstacle_r,
        "score": score,
    }


def explore_anti_revisit_turn_side(left_dist, right_dist, front, center, body_clearance):
    """Return +1/-1 if EXPLORE should turn away from a just-cleaned strip.

    This deliberately does not replace bumper-first mapping.  It only prevents
    the robot from spending another long pass on the same already-cleaned row
    when a side direction has more unvisited matrix value.
    """
    late_allowed = bool(last_coverage_percent >= EXPLORE_ANTI_REVISIT_LATE_PERCENT)
    loop_break_allowed = bool(
        EXPLORE_LOOP_BREAKER_ENABLED
        and last_coverage_percent >= EXPLORE_LOOP_BREAKER_MIN_COVERAGE_PERCENT
        and row_distance_from_start() >= EXPLORE_LOOP_BREAKER_MIN_ROW_M
        and min(front, center) >= EXPLORE_LOOP_BREAKER_MIN_FRONT_OPEN_M
    )
    if not ((EXPLORE_ANTI_REVISIT_ENABLED or late_allowed or loop_break_allowed) and EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active()):
        return None, "off"
    now = robot.getTime()
    if now - last_explore_anti_revisit_turn_time < EXPLORE_ANTI_REVISIT_COOLDOWN_SEC:
        return None, "cooldown"
    if row_distance_from_start() < EXPLORE_ANTI_REVISIT_MIN_ROW_M:
        return None, "short-row"
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return None, "bumper-active"
    # Near an expected wall contact, let the bumper/recovery system decide.  The
    # anti-revisit rule is for open repeated lanes, not contact avoidance.
    if min(front, center, body_clearance) < 0.18:
        return None, "near-contact"

    current = explore_corridor_matrix_stats(desired_grid_heading)
    old_ahead = bool(
        current["total"] > 0 and (
            current["cleaned"] >= EXPLORE_ANTI_REVISIT_CLEANED_AHEAD
            or current["recent"] >= EXPLORE_ANTI_REVISIT_RECENT_AHEAD
        )
    )
    long_open_row = bool(loop_break_allowed and row_distance_from_start() >= EXPLORE_LOOP_BREAKER_LONG_OPEN_ROW_M)
    if not old_ahead and not long_open_row:
        return None, f"forward-new c={current['cleaned']:.2f} r={current['recent']:.2f}"

    candidates = []
    for side, depth_side in ((1.0, left_dist), (-1.0, right_dist)):
        if depth_side < EXPLORE_ANTI_REVISIT_MIN_SIDE_DEPTH_M:
            continue
        heading = strict_world_grid_heading(desired_grid_heading + side * math.pi / 2.0)
        stats = explore_corridor_matrix_stats(heading)
        has_useful_side = bool(
            stats.get("frontier", 0.0) >= EXPLORE_LOOP_BREAKER_REQUIRE_SIDE_FRONTIER
            or stats.get("uncleaned", 0.0) >= 0.08
        )
        if not has_useful_side:
            continue
        # Prefer side exits that are actually less repeated than the current row.
        loop_bonus = 0.20 if loop_break_allowed and stats.get("recent", 0.0) < current.get("recent", 0.0) - 0.10 else 0.0
        long_row_bonus = 0.18 if long_open_row else 0.0
        candidates.append((stats["score"] + loop_bonus + long_row_bonus, side, stats))
    if not candidates:
        return None, "sides-blocked"
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_side, best_stats = candidates[0]
    if best_score < EXPLORE_ANTI_REVISIT_MIN_SIDE_SCORE:
        return None, f"side-score-low {best_score:.2f}"
    if (not long_open_row) and best_score < current["score"] + EXPLORE_ANTI_REVISIT_SIDE_MARGIN:
        return None, f"side-not-better {best_score:.2f}/{current['score']:.2f}"
    reason = (
        f"anti-revisit {'long-row' if long_open_row and not old_ahead else 'old forward'} c={current['cleaned']:.2f} r={current['recent']:.2f}; "
        f"side {'L' if best_side > 0 else 'R'} score={best_score:.2f} "
        f"u={best_stats['uncleaned']:.2f} fr={best_stats.get('frontier',0.0):.2f} unk={best_stats['unknown']:.2f}"
    )
    return best_side, reason


def begin_explore_single_turn(side, reason):
    """Start one atomic 90-degree turn for exploration, without lane shifting."""
    global nav_action_queue, lane_side, last_row_change_time, coverage_status, row_end_candidate_count
    side = 1.0 if side >= 0 else -1.0
    nav_action_queue = []
    row_end_candidate_count = 0
    last_row_change_time = robot.getTime()
    lane_side = side
    coverage_status = f"explore single 90 {'L' if side > 0 else 'R'}"
    begin_pivot_turn(side, reason)


def hard_core_row_length_limit():
    """Row length used by the deterministic grid controller."""
    if navigation_phase == NavigationPhase.FINISH_CLEANUP.value:
        return HARD_CORE_FINISH_ROW_M
    if explore_simple_mode_active():
        return HARD_CORE_MAP_FIRST_ROW_M
    return HARD_CORE_COVERAGE_ROW_M


def hard_core_side_for_next_row(left_dist, right_dist):
    """Choose the side for the next strict 90-degree lane transition.

    This deliberately keeps a lawnmower bias, but refuses to pivot into a very
    tight side if the opposite side is clearly safer.  It is much simpler than
    the old target/line-acquire/pocket arbitration and is therefore easier to
    defend in the diploma as a finite-state navigation prototype.
    """
    preferred = 1.0 if lane_side >= 0 else -1.0
    if preferred > 0 and left_dist < HARD_CORE_SIDE_CLOSE_M and right_dist > left_dist + HARD_CORE_SIDE_OPEN_BONUS_M:
        return -1.0
    if preferred < 0 and right_dist < HARD_CORE_SIDE_CLOSE_M and left_dist > right_dist + HARD_CORE_SIDE_OPEN_BONUS_M:
        return 1.0
    return preferred


def hard_core_front_row_end(front, center, upper_front, body_clearance):
    """Return True only for a central/front row end, not a side scrape."""
    central_body = bool(
        body_clearance < HARD_CORE_BODY_BLOCK_M
        and abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M
    )
    lower_front_block = bool(front < HARD_CORE_FRONT_BLOCK_M or center < HARD_CORE_CENTER_BLOCK_M)
    upper_supported_block = bool(
        upper_front < HARD_CORE_UPPER_BLOCK_M
        and (front < HARD_CORE_FRONT_BLOCK_M + 0.10 or center < HARD_CORE_CENTER_BLOCK_M + 0.10)
    )
    return central_body or lower_front_block or upper_supported_block


def explore_bumper_first_forward_speed(front, center, upper_front, body_clearance):
    """Forward speed for bumper-led map building.

    This does not decide that the row ended.  It only reduces impact speed before
    an expected physical bumper hit, so the front pads can trigger cleanly instead
    of wedging the body into the wall.
    """
    range_front = min(float(front), float(center), float(upper_front))
    speed = EXPLORE_BUMPER_FIRST_SPEED
    reason = "cruise"
    if range_front < EXPLORE_PRECONTACT_CREEP_FRONT_M or body_clearance < EXPLORE_PRECONTACT_BODY_CREEP_M:
        speed = EXPLORE_PRECONTACT_CREEP_SPEED
        reason = "creep"
    elif range_front < EXPLORE_PRECONTACT_SLOW_FRONT_M or body_clearance < EXPLORE_PRECONTACT_BODY_SOFT_M:
        speed = EXPLORE_PRECONTACT_SOFT_SPEED
        reason = "slow"
    return speed, reason


def hard_core_controller_speeds(front, center, upper_front, left, right, body_clearance):
    """Deterministic owner for ordinary FORWARD movement.

    Why this exists: after the arbiter, the robot still did not make a
    clean 90-degree turn because the legacy FORWARD path was still allowed to
    interpret a long open depth corridor as permission to keep going.  This
    function bypasses that legacy path in the normal case.  It allows only:
    straight row -> bumper-confirmed exploration contact first; later coverage can use confirmed row ends/lane transitions.
    """
    global row_end_candidate_count, coverage_status, last_revisit_lane_change_time, last_explore_anti_revisit_turn_time
    global last_optional_block_reason, desired_grid_heading

    if not HARD_CORE_CONTROLLER_ENABLED:
        return None
    if nav_state != NAV_FORWARD:
        return None
    if under_furniture_active or robot.getTime() < leg_pass_grace_until:
        # These are explicit special manoeuvres.  Let their existing handlers run.
        return None
    if post_lane_forward_lock_active(front, center, body_clearance) or post_gap_commit_active(front, center, body_clearance):
        return None

    now = robot.getTime()
    row_dist = row_distance_from_start()
    row_limit = hard_core_row_length_limit()
    heading_err = normalize_angle(desired_grid_heading - pose_theta)
    bumper_first_mapping = bool(EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active())

    # Priority inside ordinary FORWARD is: local wall-follow > plain row-forward.
    # The old order asked the hard heading guard first, so the tiny yaw that
    # wall-follow intentionally uses to keep side clearance was repeatedly
    # converted into GRID_REALIGN.  That was a control-ownership bug, not a mapping
    # problem.  Let wall-follow own small heading errors; large errors still pivot.
    early_wall_follow_cmd = None
    if bumper_first_mapping and abs(heading_err) <= MAP_BUILD_WALL_FOLLOW_MAX_HEADING_ERR:
        speed_for_wall, _precontact_mode = explore_bumper_first_forward_speed(front, center, upper_front, body_clearance)
        early_wall_follow_cmd = map_building_wall_follow_speeds(
            front, center, upper_front, left, right, body_clearance, speed_for_wall
        )
        if early_wall_follow_cmd is not None:
            row_end_candidate_count = 0
            return early_wall_follow_cmd

    # Hard rule: ordinary forward motion must not translate with a large yaw error.
    # Also avoid the no-op band: if the error is below GRID_REALIGN_REQUEST_ERR,
    # start_grid_realign() would accept it and reset row ownership every frame.
    if (
        abs(heading_err) > max(HARD_CORE_REALIGN_ERR, GRID_REALIGN_REQUEST_ERR)
        and front > HARD_CORE_REALIGN_MIN_FRONT_M
        and center > HARD_CORE_REALIGN_MIN_FRONT_M
        and not (last_bumper_left or last_bumper_center or last_bumper_right)
    ):
        start_grid_realign(desired_grid_heading, f"hard-core heading err={math.degrees(heading_err):.1f}", "hard-core row after heading realign")
        last_optional_block_reason = "hard-core owns heading realign"
        if nav_state != NAV_FORWARD:
            return 0.0, 0.0
        last_optional_block_reason = "heading realign already satisfied; continue row"
    if bumper_first_mapping:
        # Early room mapping is contact-led: do not turn just because the
        # depth/occupancy corridor looks short.  The bumper layer above this block
        # remains active and will start recovery when the shell really touches.
        front_row_end = False
        row_end_candidate_count = 0
    else:
        front_row_end = hard_core_front_row_end(front, center, upper_front, body_clearance)
        if front_row_end:
            row_end_candidate_count += 1
        else:
            row_end_candidate_count = 0

    # A real front row-end triggers a strict 90-degree lane transition.  This is
    # the place where the robot should turn by 90-ish degrees instead of drifting
    # forward beside the wall.  begin_lawnmower_lane_change_checked() owns the
    # sequence: optional planners cannot interrupt it.
    if row_end_candidate_count >= HARD_CORE_ROW_END_CONFIRM_FRAMES:
        if abs(heading_err) > PRE_TURN_REALIGN_ERR and not (last_bumper_left or last_bumper_center or last_bumper_right):
            start_grid_realign(desired_grid_heading, f"pre-turn realign err={math.degrees(heading_err):.1f}", "row forward after pre-turn realign")
            last_optional_block_reason = "pre-turn grid realign"
            return 0.0, 0.0
        if matrix_first_explore_active() and row_dist >= EXPLORE_MIN_ROW_BEFORE_TURN_M:
            side = choose_explore_turn_side(left, right)
            begin_explore_single_turn(
                side,
                f"matrix-first explore front end F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}",
            )
            last_optional_block_reason = "matrix-first explore: single 90 only"
            return 0.0, 0.0

        side = hard_core_side_for_next_row(left, right)
        row_end_candidate_count = 0
        last_revisit_lane_change_time = now
        begin_lawnmower_lane_change_checked(
            side,
            f"hard-core front row end F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}",
            front,
            body_clearance,
        )
        last_optional_block_reason = "hard-core row-end -> 90/shift/90"
        return 0.0, 0.0

    # If the front depth corridor stays open forever, the old code never turned.
    # In map-first mode that is bad: the robot only paints one long strip.  Force a
    # row transition after a bounded strip length.  This is intentionally simple
    # and deterministic; later global cleanup can optimize residual cells.
    if (
        hard_core_lawnmower_allowed()
        and not WALL_TO_WALL_SWEEP_DISABLE_FIXED_ROW_TURNS
        and row_dist >= row_limit
        and row_dist >= HARD_CORE_MIN_ROW_M
        and now - last_row_change_time > HARD_CORE_TURN_COOLDOWN_SEC
        and front > HARD_CORE_FRONT_BLOCK_M + 0.08
        and center > HARD_CORE_CENTER_BLOCK_M + 0.08
        and body_clearance > BODY_CORRIDOR_HARD_CLEARANCE
    ):
        side = hard_core_side_for_next_row(left, right)
        row_end_candidate_count = 0
        last_revisit_lane_change_time = now
        begin_lawnmower_lane_change_checked(
            side,
            f"hard-core fixed row {row_dist:.2f}/{row_limit:.2f}m L={left:.2f} R={right:.2f}",
            front,
            body_clearance,
        )
        last_optional_block_reason = "hard-core fixed row -> 90/shift/90"
        return 0.0, 0.0

    if bumper_first_mapping:
        revisit_side, revisit_reason = explore_anti_revisit_turn_side(left, right, front, center, body_clearance)
        if revisit_side is not None:
            last_explore_anti_revisit_turn_time = now
            begin_explore_single_turn(revisit_side, revisit_reason)
            last_optional_block_reason = "matrix-first anti-revisit single 90"
            return 0.0, 0.0

    # No legacy local alignment here: equal-ish wheel speeds only, tiny heading
    # correction to the stored cardinal heading.  This removes visual diagonal
    # correction arcs during the normal row.
    if matrix_first_explore_active() and EXPLORE_BUMPER_FIRST_MAPPING:
        speed, precontact_mode = explore_bumper_first_forward_speed(front, center, upper_front, body_clearance)
        last_optional_block_reason = f"perimeter-trace FORWARD/{precontact_mode}"
        coverage_status = (
            f"perimeter-trace {precontact_mode} {row_dist:.2f}m "
            f"F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f} body={body_clearance:.2f} "
            f"cov={last_coverage_percent:.1f}%/{MAP_BUILDING_UNTIL_COVERAGE_PERCENT:.0f}%"
        )
    elif matrix_first_explore_active():
        speed = HARD_CORE_SPEED if front > 0.50 and center > 0.42 else HARD_CORE_SLOW_SPEED
        last_optional_block_reason = "matrix-first explore owns FORWARD"
        coverage_status = f"matrix-first explore straight {row_dist:.2f}m end={row_end_candidate_count}/{HARD_CORE_ROW_END_CONFIRM_FRAMES} cov={last_coverage_percent:.1f}%/{MAP_BUILDING_UNTIL_COVERAGE_PERCENT:.0f}%"
    else:
        speed = HARD_CORE_SPEED if front > 0.50 and center > 0.42 else HARD_CORE_SLOW_SPEED
        last_optional_block_reason = "hard-core owns ordinary FORWARD"
        coverage_status = f"hard-core row forward {row_dist:.2f}/{row_limit:.2f}m end={row_end_candidate_count}/{HARD_CORE_ROW_END_CONFIRM_FRAMES}"
    return heading_locked_wheel_speeds_to(speed, desired_grid_heading, HEADING_LOCK_KP, STRICT_GRID_TINY_CORRECTION)


def simple_sweep_finish_primary(reason):
    """Stop primary deterministic sweep ownership and allow late cleanup/return-home.

    This is the only normal exit from SIMPLE_SWEEP_FSM.  Frontier/route/cleanup
    ownership is deliberately delayed until here so the early robot motion cannot
    fall back to the old reactive sequence: drive to obstacle -> route/turn -> repeat.
    """
    global simple_sweep_completed, simple_sweep_completion_reason, simple_sweep_last_debug
    global coverage_status, known_map_primary_sweep_completed, known_map_primary_sweep_finish_time, known_map_primary_sweep_finish_coverage
    simple_sweep_completed = True
    simple_sweep_completion_reason = str(reason)
    simple_sweep_last_debug = f"exploreSweep=done {reason}"
    coverage_status = simple_sweep_last_debug
    known_map_primary_sweep_completed = True
    known_map_primary_sweep_finish_time = robot.getTime()
    known_map_primary_sweep_finish_coverage = float(last_coverage_percent)
    try:
        if str(control_lock.owner) == SIMPLE_SWEEP_OWNER:
            release_control("simple sweep complete")
    except Exception:
        pass


def simple_sweep_handoff_mode(cov, stall_sec, route_cost):
    plateau_ready = (
        cov >= float(SIMPLE_SWEEP_FRONTIER_HANDOFF_MIN_COVERAGE)
        and stall_sec >= float(SIMPLE_SWEEP_FRONTIER_HANDOFF_STALL_SEC)
    )
    early_short_route_ready = (
        cov >= float(SIMPLE_SWEEP_FRONTIER_HANDOFF_EARLY_MIN_COVERAGE)
        and stall_sec >= float(SIMPLE_SWEEP_FRONTIER_HANDOFF_EARLY_STALL_SEC)
        and route_cost <= float(SIMPLE_SWEEP_FRONTIER_HANDOFF_SHORT_ROUTE_M)
    )
    if early_short_route_ready and not plateau_ready:
        return "short frontier"
    if plateau_ready:
        return "sweep plateau"
    return ""


def simple_sweep_route_handoff_ready():
    """Let frontier ROUTE_COMMIT take over when sweep coverage has plateaued."""
    global simple_sweep_best_coverage_percent, simple_sweep_best_coverage_time
    global simple_sweep_best_coverage_x, simple_sweep_best_coverage_y
    if not SIMPLE_SWEEP_FRONTIER_HANDOFF_ENABLED:
        return False, "disabled"
    now = robot.getTime()
    cov = float(last_coverage_percent or 0.0)
    if simple_sweep_best_coverage_time < -900.0:
        simple_sweep_best_coverage_percent = cov
        simple_sweep_best_coverage_time = now
        simple_sweep_best_coverage_x = pose_x
        simple_sweep_best_coverage_y = pose_y
        return False, "coverage baseline"
    if cov > simple_sweep_best_coverage_percent + float(SIMPLE_SWEEP_FRONTIER_HANDOFF_MIN_GAIN):
        simple_sweep_best_coverage_percent = cov
        simple_sweep_best_coverage_time = now
        simple_sweep_best_coverage_x = pose_x
        simple_sweep_best_coverage_y = pose_y
        return False, f"coverage improving {cov:.1f}%"
    stall_sec = now - simple_sweep_best_coverage_time
    frontiers = int(last_frontier_cells or 0)
    if frontiers < int(SIMPLE_SWEEP_FRONTIER_HANDOFF_MIN_FRONTIERS):
        return False, f"frontiers {frontiers}"
    if planner_intent != PLANNER_INTENT_EXPAND_MAP or not exploration_cleanup_locked():
        return False, f"intent={planner_intent}"
    if coverage_goal_kind != "frontier" or coverage_route_kind != "frontier":
        return False, f"route kind {coverage_goal_kind}/{coverage_route_kind}"
    if not coverage_route_map or len(coverage_route_map) < ROUTE_COMMIT_MIN_ROUTE_LEN:
        return False, "no frontier route"
    if (not math.isfinite(coverage_route_cost)) or coverage_route_cost > float(EXPLORE_FRONTIER_MAX_ROUTE_COST_M):
        return False, f"frontier cost {coverage_route_cost:.2f}"
    if coverage_route_straight_dist > float(EXPLORE_FRONTIER_MAX_STRAIGHT_DIST_M):
        return False, f"frontier dist {coverage_route_straight_dist:.2f}"
    if coverage_route_turn_need > float(EXPLORE_FRONTIER_MAX_TURN_FRAC):
        return False, f"frontier turn {coverage_route_turn_need:.2f}"
    handoff_mode = simple_sweep_handoff_mode(cov, stall_sec, coverage_route_cost)
    if not handoff_mode:
        return False, f"handoff wait cov={cov:.1f}% stall={stall_sec:.1f}s route={coverage_route_cost:.2f}m"
    drift = math.hypot(pose_x - simple_sweep_best_coverage_x, pose_y - simple_sweep_best_coverage_y)
    return True, (
        f"{handoff_mode} cov={cov:.1f}% best={simple_sweep_best_coverage_percent:.1f}% "
        f"stall={stall_sec:.0f}s drift={drift:.2f}m frontier={frontiers} route={coverage_route_cost:.2f}m"
    )


def simple_sweep_should_own():
    """Primary sweep gate: true until 85-90% coverage or lane exhaustion."""
    if not SIMPLE_SWEEP_FSM_ENABLED:
        return False
    if dock_return_completed or dock_return_active:
        return False
    if simple_sweep_completed:
        return False
    if known_map_coverage_eval_active():
        # K mode is the known-map route optimizer. It must be able to take
        # planner/route ownership immediately, instead of fighting the explore
        # sweep FSM. Do not mark the sweep as completed; if K is turned off,
        # explore can resume from its lane memory.
        return False
    ready, ready_reason = learned_map_ready_to_clean()
    if ready:
        start_map_complete_return_to_dock(ready_reason)
        return False
    if float(last_coverage_percent) >= float(SIMPLE_SWEEP_FINISH_COVERAGE_PERCENT):
        simple_sweep_finish_primary(f"coverage {last_coverage_percent:.1f}% >= {SIMPLE_SWEEP_FINISH_COVERAGE_PERCENT:.1f}%")
        return False
    if int(simple_sweep_lane_index) >= int(SIMPLE_SWEEP_MAX_LANES):
        simple_sweep_finish_primary(f"lane limit {simple_sweep_lane_index}/{SIMPLE_SWEEP_MAX_LANES}")
        return False
    handoff_ready, handoff_reason = simple_sweep_route_handoff_ready()
    if handoff_ready:
        simple_sweep_finish_primary(f"frontier handoff: {handoff_reason}")
        return False
    return True


def simple_sweep_clear_legacy_motion(reason="simple sweep owns"):
    """Prevent stale frontier/route/scan queues from owning wheels during sweep."""
    global nav_action_queue, route_commit_active, route_commit_route_map, route_commit_route_world
    global route_commit_target_map, route_commit_target_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_kind, route_commit_reason, last_route_commit_debug
    global active_scan_target_yaws, active_scan_index, active_scan_dwell_until, active_scan_started_at
    global post_turn_rgbd_snapshot_pending, post_turn_rgbd_snapshot_until, last_active_scan_debug, last_post_turn_snapshot_debug
    nav_action_queue = []
    if route_commit_active and route_commit_kind != "dock":
        route_commit_active = False
        route_commit_route_map = []
        route_commit_route_world = []
        route_commit_target_map = None
        route_commit_target_world = None
        route_commit_waypoint_map = None
        route_commit_waypoint_world = None
        route_commit_kind = "none"
        route_commit_reason = str(reason)
        last_route_commit_debug = "inactive: gated by EXPLORE_SWEEP_FSM"
    active_scan_target_yaws = []
    active_scan_index = 0
    active_scan_dwell_until = -999.0
    active_scan_started_at = -999.0
    last_active_scan_debug = "scan=blocked by EXPLORE_SWEEP_FSM"
    post_turn_rgbd_snapshot_pending = False
    post_turn_rgbd_snapshot_until = -999.0
    last_post_turn_snapshot_debug = "snap=blocked by EXPLORE_SWEEP_FSM"


def simple_sweep_desired_lane_heading():
    base = simple_sweep_axis_heading
    if simple_sweep_lane_direction < 0.0:
        base = normalize_angle(base + math.pi)
    return strict_world_grid_heading(base)


def simple_sweep_global_normal_heading(global_side=None):
    """Heading of the fixed sweep-frame lateral normal.

    global_side=+1 means the left normal of simple_sweep_axis_heading,
    global_side=-1 means the right normal.  This is intentionally NOT the
    robot's current left/right side on alternating rows.
    """
    side = simple_sweep_lane_shift_direction if global_side is None else global_side
    side = 1.0 if side >= 0.0 else -1.0
    return normalize_angle(simple_sweep_axis_heading + side * (math.pi * 0.5))


def simple_sweep_lateral_coord(wx=None, wy=None):
    """Signed coordinate across lanes in the sweep frame."""
    if wx is None:
        wx = pose_x
    if wy is None:
        wy = pose_y
    h = simple_sweep_global_normal_heading(+1.0)
    return float(wx) * math.cos(h) + float(wy) * math.sin(h)


def simple_sweep_shift_global_from_robot_side(robot_side):
    """Convert current robot-relative shift side into fixed sweep-frame side."""
    rside = 1.0 if robot_side >= 0.0 else -1.0
    lane_dir = 1.0 if simple_sweep_lane_direction >= 0.0 else -1.0
    return 1.0 if rside * lane_dir >= 0.0 else -1.0


def simple_sweep_plan_next_lane_target(global_side):
    """Choose the absolute lateral coordinate of the next sweep lane.

    e changes the target source: instead of blindly shifting one adjacent
    lane, score nearby parallel strips and pick the nearest line that still has
    unvisited/unknown potential and is not in recent lane history.  This keeps
    the primitive sequence deterministic while preventing repeated entry into
    already-covered orange lines.
    """
    global simple_sweep_pending_lane_target_lateral, simple_sweep_pending_shift_global_side, simple_sweep_lane_score_debug
    side = 1.0 if global_side >= 0.0 else -1.0
    if SIMPLE_SWEEP_LANE_SCORE_ENABLED:
        side, target, skips, score_info, reason = simple_sweep_select_next_lane_target(side)
        simple_sweep_lane_score_debug = (
            f"laneScore side={'L' if side > 0 else 'R'} skip={skips} target={target:.2f} "
            f"pot={score_info.get('potential', 0.0):.1f} fr={score_info.get('frontier', 0)} "
            f"clean={score_info.get('cleaned', 0)} recent={score_info.get('recent', 0)} "
            f"hist={score_info.get('history_penalty', 0.0):.1f} rev={score_info.get('revisit_ratio', 0.0):.2f} {reason}"
        )
    else:
        target = float(simple_sweep_lane_target_lateral) + side * float(SIMPLE_SWEEP_LANE_SHIFT_M)
        skips = 0
        if SIMPLE_SWEEP_LANE_TARGET_MEMORY_ENABLED:
            while skips < 4 and any(abs(target - old) <= SIMPLE_SWEEP_LANE_TARGET_REPEAT_EPS_M for old in simple_sweep_lane_target_history[-12:]):
                target += side * float(SIMPLE_SWEEP_LANE_SHIFT_M)
                skips += 1
        simple_sweep_lane_score_debug = f"laneScore=off side={'L' if side > 0 else 'R'} skip={skips} target={target:.2f}"
    simple_sweep_pending_lane_target_lateral = float(target)
    simple_sweep_pending_shift_global_side = side
    return float(target)


def simple_sweep_commit_pending_lane_target():
    """Commit the next lane target after a lane shift sequence succeeded."""
    global simple_sweep_lane_target_lateral, simple_sweep_pending_lane_target_lateral, simple_sweep_lane_target_history
    global simple_sweep_pending_shift_global_side
    if simple_sweep_pending_lane_target_lateral is None:
        return
    simple_sweep_lane_target_lateral = float(simple_sweep_pending_lane_target_lateral)
    simple_sweep_pending_lane_target_lateral = None
    simple_sweep_pending_shift_global_side = 0.0
    if (not simple_sweep_lane_target_history) or abs(simple_sweep_lane_target_lateral - simple_sweep_lane_target_history[-1]) > 0.03:
        simple_sweep_lane_target_history.append(simple_sweep_lane_target_lateral)
        del simple_sweep_lane_target_history[:-24]


def simple_sweep_lateral_target_error():
    return float(simple_sweep_lane_target_lateral) - simple_sweep_lateral_coord()


def simple_sweep_turn_side_for_global_shift(global_side=None):
    """Convert a fixed world/sweep-frame shift side into robot-relative 90 turns.

    On the return row the robot faces the opposite direction, so the same global
    lane shift must use the opposite robot-relative turn.  This is the core fix
    that prevents left-after-east / left-after-west oscillation over the same two
    stripes.
    """
    side = simple_sweep_lane_shift_direction if global_side is None else global_side
    side = 1.0 if side >= 0.0 else -1.0
    lane_dir = 1.0 if simple_sweep_lane_direction >= 0.0 else -1.0
    return 1.0 if side * lane_dir >= 0.0 else -1.0


def simple_sweep_ray_distance_to_arena(theta):
    """Approximate distance from current pose to the known rectangular arena edge."""
    if not DEBUG_ARENA_BOUNDS_ENABLED:
        return None
    dx = math.cos(theta)
    dy = math.sin(theta)
    candidates = []
    if abs(dx) > 1e-6:
        bx = DEBUG_ARENA_X_MAX_M if dx > 0.0 else DEBUG_ARENA_X_MIN_M
        t = (bx - pose_x) / dx
        y = pose_y + t * dy
        if t > 0.0 and DEBUG_ARENA_Y_MIN_M - 0.20 <= y <= DEBUG_ARENA_Y_MAX_M + 0.20:
            candidates.append(t)
    if abs(dy) > 1e-6:
        by = DEBUG_ARENA_Y_MAX_M if dy > 0.0 else DEBUG_ARENA_Y_MIN_M
        t = (by - pose_y) / dy
        x = pose_x + t * dx
        if t > 0.0 and DEBUG_ARENA_X_MIN_M - 0.20 <= x <= DEBUG_ARENA_X_MAX_M + 0.20:
            candidates.append(t)
    if not candidates:
        return None
    return max(0.0, float(min(candidates)))


def simple_sweep_map_to_world(mx, my):
    """Convert a map cell back to Webots/world coordinates."""
    return (float(mx) - float(MAP_ORIGIN_X)) / float(MAP_SCALE), (float(MAP_ORIGIN_Y) - float(my)) / float(MAP_SCALE)


def simple_sweep_axis_coord(wx=None, wy=None):
    """Signed coordinate along the current sweep axis."""
    if wx is None:
        wx = pose_x
    if wy is None:
        wy = pose_y
    h = simple_sweep_axis_heading
    return float(wx) * math.cos(h) + float(wy) * math.sin(h)


def simple_sweep_lane_recent(target_lat):
    """True if target_lat is basically a line we have already driven/targeted."""
    eps = float(SIMPLE_SWEEP_LANE_SCORE_HISTORY_EPS_M)
    for old in list(simple_sweep_lane_target_history)[-28:]:
        if abs(float(target_lat) - float(old)) <= eps:
            return True
    for old in list(simple_sweep_lane_lateral_history)[-18:]:
        if abs(float(target_lat) - float(old)) <= eps:
            return True
    return False


def simple_sweep_lane_history_penalty(target_lat):
    """Continuous penalty for selecting a line close to already driven lanes."""
    eps = max(1e-3, float(EXPLORE_SWEEP_LANE_BAND_EPS_M))
    penalty = 0.0
    for old in list(simple_sweep_lane_target_history)[-int(EXPLORE_SWEEP_LANE_HISTORY_LIMIT):]:
        d = abs(float(target_lat) - float(old))
        if d <= eps:
            penalty += (1.0 - d / eps) * float(EXPLORE_SWEEP_VISITED_LANE_PENALTY)
    for old in list(simple_sweep_lane_lateral_history)[-int(EXPLORE_SWEEP_LANE_HISTORY_LIMIT):]:
        d = abs(float(target_lat) - float(old))
        if d <= eps:
            penalty += (1.0 - d / eps) * float(EXPLORE_SWEEP_RECENT_LANE_PENALTY)
    return float(penalty)


def simple_sweep_observed_roi_slices():
    """Observed map envelope used by the lane scorer.

    Do not score the entire unknown 1000x1000 canvas.  That would make the
    selector chase map padding instead of the room.  The ROI is derived only
    from current RGB-D/bumper/cleaned evidence, so it is not a known-map prior.
    """
    evidence = (
        (np.abs(log_odds) > LO_UNKNOWN_EPS)
        | (visual_log_odds > CV_DISPLAY_LIGHT_EPS)
        | (structural_log_odds > STRUCTURAL_OCCUPIED_EPS)
        | (contact_log_odds > CONTACT_OCCUPIED_EPS)
        | (cleaned_mask > 0)
    )
    ys, xs = np.nonzero(evidence)
    pad = max(8, int(round(float(SIMPLE_SWEEP_LANE_SCORE_ROI_PAD_M) * MAP_SCALE)))
    if len(xs) == 0:
        mx, my = world_to_map(pose_x, pose_y)
        x0 = max(0, mx - pad)
        x1 = min(MAP_SIZE, mx + pad + 1)
        y0 = max(0, my - pad)
        y1 = min(MAP_SIZE, my + pad + 1)
        return y0, y1, x0, x1
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(MAP_SIZE, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(MAP_SIZE, int(ys.max()) + pad + 1)
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        mx, my = world_to_map(pose_x, pose_y)
        x0 = max(0, mx - pad)
        x1 = min(MAP_SIZE, mx + pad + 1)
        y0 = max(0, my - pad)
        y1 = min(MAP_SIZE, my + pad + 1)
    return y0, y1, x0, x1


def simple_sweep_lane_strip_score(target_lat):
    """Score how useful a fixed lateral lane would be for EXPLORE sweep.

    In explore mode the goal is map discovery, not late coverage optimization.
    Therefore the score prefers lanes that reveal frontier/unknown space and
    known-free cells not yet visited, while penalizing cleaned/recent lane bands
    and furniture/obstacle density.  The function still returns the same fields
    used by the deterministic FSM; no frontier route owner is introduced here.
    """
    if not SIMPLE_SWEEP_LANE_SCORE_ENABLED:
        return {
            "score": 0.0,
            "potential": 0.0,
            "uncleaned": 0,
            "unknown": 0,
            "cleaned": 0,
            "recent": 0,
            "obstacle_ratio": 0.0,
            "revisit_ratio": 0.0,
            "debug": "laneScore=off",
        }
    try:
        obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()
        y0, y1, x0, x1 = simple_sweep_observed_roi_slices()
        step = max(1, int(SIMPLE_SWEEP_LANE_SCORE_SAMPLE_STEP))
        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        if yy.size == 0:
            raise ValueError("empty lane-score ROI")
        wx = (xx.astype(np.float32) - float(MAP_ORIGIN_X)) / float(MAP_SCALE)
        wy = (float(MAP_ORIGIN_Y) - yy.astype(np.float32)) / float(MAP_SCALE)
        nh = simple_sweep_global_normal_heading(+1.0)
        lat = wx * math.cos(nh) + wy * math.sin(nh)
        strip = np.abs(lat - float(target_lat)) <= float(SIMPLE_SWEEP_LANE_SCORE_HALF_WIDTH_M)
        if not bool(np.any(strip)):
            return {
                "score": -999.0,
                "potential": 0.0,
                "uncleaned": 0,
                "unknown": 0,
                "cleaned": 0,
                "recent": 0,
                "obstacle_ratio": 1.0,
                "revisit_ratio": 1.0,
                "debug": f"laneScore empty target={target_lat:.2f}",
            }

        obs_roi = obstacles[y0:y1:step, x0:x1:step]
        cleanable_roi = cleanable[y0:y1:step, x0:x1:step]
        clean_roi = cleaned[y0:y1:step, x0:x1:step]
        unclean_roi = uncleaned[y0:y1:step, x0:x1:step]
        unknown_roi = unknown[y0:y1:step, x0:x1:step]
        recent_roi = recent_visit_log_odds[y0:y1:step, x0:x1:step] > 0.50

        # Discovery gain: unknown is useful mostly at the boundary of known free
        # space.  Pure unknown outside the observed room gets only a tiny weight;
        # frontier-like unknown next to cleanable/free cells gets the main bonus.
        if EXPLORE_SWEEP_INFO_GAIN_SELECTOR_ENABLED:
            try:
                frontier_full = unknown & (cv2.dilate(cleanable.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0)
                frontier_roi = frontier_full[y0:y1:step, x0:x1:step]
            except Exception:
                frontier_roi = unknown_roi
        else:
            frontier_roi = unknown_roi

        total = max(1, int(np.count_nonzero(strip)))
        obs_n = int(np.count_nonzero(obs_roi & strip))
        cleanable_n = int(np.count_nonzero(cleanable_roi & strip))
        cleaned_n = int(np.count_nonzero(clean_roi & strip))
        unclean_n = int(np.count_nonzero(unclean_roi & strip))
        unknown_n = int(np.count_nonzero(unknown_roi & strip))
        frontier_n = int(np.count_nonzero(frontier_roi & strip))
        recent_n = int(np.count_nonzero(recent_roi & strip))
        obstacle_ratio = float(obs_n) / float(total)

        history_penalty = simple_sweep_lane_history_penalty(target_lat)
        if EXPLORE_SWEEP_INFO_GAIN_SELECTOR_ENABLED:
            potential = (
                float(unclean_n) * float(EXPLORE_SWEEP_UNCLEANED_FREE_WEIGHT)
                + float(frontier_n) * float(EXPLORE_SWEEP_FRONTIER_UNKNOWN_WEIGHT)
                + float(max(0, unknown_n - frontier_n)) * float(EXPLORE_SWEEP_RAW_UNKNOWN_WEIGHT)
            )
            min_potential = float(EXPLORE_SWEEP_MIN_FRONTIER_POTENTIAL)
        else:
            potential = float(unclean_n) + float(unknown_n) * SIMPLE_SWEEP_LANE_SCORE_UNKNOWN_WEIGHT
            min_potential = float(SIMPLE_SWEEP_LANE_SCORE_MIN_POTENTIAL)

        revisited = (
            float(cleaned_n)
            + float(recent_n) * SIMPLE_SWEEP_LANE_SCORE_RECENT_WEIGHT
            + float(history_penalty)
        )
        denom = max(1.0, potential + revisited)
        revisit_ratio = revisited / denom
        score = (
            potential
            - float(cleaned_n) * SIMPLE_SWEEP_LANE_SCORE_CLEANED_WEIGHT
            - float(recent_n) * SIMPLE_SWEEP_LANE_SCORE_RECENT_WEIGHT
            - float(history_penalty)
            - float(obs_n) * 1.15
        )
        return {
            "score": float(score),
            "potential": float(potential),
            "min_potential": float(min_potential),
            "uncleaned": unclean_n,
            "unknown": unknown_n,
            "frontier": frontier_n,
            "cleanable": cleanable_n,
            "cleaned": cleaned_n,
            "recent": recent_n,
            "history_penalty": float(history_penalty),
            "obstacle_ratio": float(obstacle_ratio),
            "revisit_ratio": float(revisit_ratio),
            "debug": (
                f"laneScore target={target_lat:.2f} score={score:.1f} pot={potential:.1f} "
                f"u={unclean_n} fr={frontier_n} unk={unknown_n} clean={cleaned_n} recent={recent_n} "
                f"hist={history_penalty:.1f} obs={obstacle_ratio:.2f} rev={revisit_ratio:.2f}"
            ),
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "potential": 0.0,
            "min_potential": float(SIMPLE_SWEEP_LANE_SCORE_MIN_POTENTIAL),
            "uncleaned": 0,
            "unknown": 0,
            "frontier": 0,
            "cleaned": 0,
            "recent": 0,
            "history_penalty": 0.0,
            "obstacle_ratio": 0.0,
            "revisit_ratio": 0.0,
            "debug": f"laneScore error {type(exc).__name__}",
        }


def simple_sweep_lane_candidate_usable(score_info, recent_line=False):
    if recent_line:
        return False
    if float(score_info.get("obstacle_ratio", 1.0)) > float(SIMPLE_SWEEP_LANE_SCORE_MAX_OBSTACLE_RATIO):
        return False
    if float(score_info.get("revisit_ratio", 1.0)) > float(SIMPLE_SWEEP_LANE_SCORE_MAX_REVISIT_RATIO):
        return False
    min_potential = float(score_info.get("min_potential", SIMPLE_SWEEP_LANE_SCORE_MIN_POTENTIAL))
    if float(score_info.get("potential", 0.0)) < min_potential:
        return False
    return True


def simple_sweep_select_next_lane_target(preferred_global_side):
    """Pick the next unvisited lane target near the current sweep line.

    The old deterministic queue remains intact.  This function only replaces the
    naive target=current+spacing with target=nearest useful unvisited strip.  It
    prevents the orange perimeter-loop failure where the robot repeatedly drives
    near the same cleaned wall strip instead of shifting into the blue/unknown
    interior lanes.
    """
    preferred_global_side = 1.0 if preferred_global_side >= 0.0 else -1.0
    current = float(simple_sweep_lane_target_lateral)
    spacing = float(SIMPLE_SWEEP_LANE_SHIFT_M)
    lookahead = max(1, int(SIMPLE_SWEEP_LANE_SCORE_LOOKAHEAD))
    side_order = [preferred_global_side, -preferred_global_side]

    best = None
    best_preferred = None
    best_usable = None
    scan_debug = []
    for side_i, side in enumerate(side_order):
        for k in range(1, lookahead + 1):
            target = current + side * spacing * k
            recent_line = simple_sweep_lane_recent(target)
            score_info = simple_sweep_lane_strip_score(target)
            usable = simple_sweep_lane_candidate_usable(score_info, recent_line)
            # Bias toward the preferred monotonic side, but allow switching if the
            # preferred side is already cleaned/wall and the opposite side has real
            # unvisited potential.
            side_penalty = 0.0 if side_i == 0 else 5.0
            skip_penalty = float(k - 1) * 3.0
            recent_penalty = 999.0 if recent_line else 0.0
            value = float(score_info.get("score", 0.0)) - side_penalty - skip_penalty - recent_penalty
            item = (value, side, target, k, score_info, recent_line, usable)
            if best is None or value > best[0]:
                best = item
            if side_i == 0 and (best_preferred is None or value > best_preferred[0]):
                best_preferred = item
            if usable and (best_usable is None or value > best_usable[0]):
                best_usable = item
            if len(scan_debug) < 4:
                scan_debug.append(
                    f"{'P' if side_i == 0 else 'O'}{k}:{'L' if side > 0 else 'R'} "
                    f"t={target:.2f} pot={score_info.get('potential', 0.0):.0f} "
                    f"rev={score_info.get('revisit_ratio', 0.0):.2f}{' recent' if recent_line else ''}"
                )

    if best_usable is not None:
        value, side, target, k, score_info, recent_line, usable = best_usable
        return side, target, k, score_info, f"best-usable {'; '.join(scan_debug)}"

    # If nothing cleanly passed thresholds, still avoid returning to a known line.
    # Prefer the best scored candidate if it has any potential and is not a recent
    # stripe; otherwise make a monotonic skip over recent history.
    if best is not None:
        value, side, target, k, score_info, recent_line, usable = best
        if (not recent_line) and float(score_info.get("potential", 0.0)) > 0.0:
            return side, target, k, score_info, f"best-effort {'; '.join(scan_debug)}"

    side = preferred_global_side
    target = current + side * spacing
    skips = 1
    while skips <= int(SIMPLE_SWEEP_LANE_SCORE_FALLBACK_SKIP_LIMIT) and simple_sweep_lane_recent(target):
        skips += 1
        target = current + side * spacing * skips
    score_info = simple_sweep_lane_strip_score(target)
    return side, target, skips, score_info, f"memory-fallback {'; '.join(scan_debug)}"


def simple_sweep_pick_initial_global_shift_side():
    """Pick the first shift side from unvisited lane evidence, not arena size."""
    fallback = 1.0 if lane_side >= 0 else -1.0
    if SIMPLE_SWEEP_LANE_SCORE_ENABLED:
        left_target = simple_sweep_lateral_coord() + float(SIMPLE_SWEEP_LANE_SHIFT_M)
        right_target = simple_sweep_lateral_coord() - float(SIMPLE_SWEEP_LANE_SHIFT_M)
        left_score = simple_sweep_lane_strip_score(left_target)
        right_score = simple_sweep_lane_strip_score(right_target)
        left_value = float(left_score.get("score", 0.0))
        right_value = float(right_score.get("score", 0.0))
        # Do not let a tiny score difference flip direction randomly; otherwise
        # use the side with less cleaned/recent and more blue/unknown potential.
        if left_value > right_value * SIMPLE_SWEEP_LANE_SCORE_SIDE_SWITCH_GAIN + 1.0:
            return 1.0
        if right_value > left_value * SIMPLE_SWEEP_LANE_SCORE_SIDE_SWITCH_GAIN + 1.0:
            return -1.0
    if not SIMPLE_SWEEP_ARENA_SIDE_PICK_ENABLED:
        return fallback
    left_h = simple_sweep_global_normal_heading(+1.0)
    right_h = simple_sweep_global_normal_heading(-1.0)
    left_d = simple_sweep_ray_distance_to_arena(left_h)
    right_d = simple_sweep_ray_distance_to_arena(right_h)
    if left_d is None or right_d is None:
        return fallback
    if abs(left_d - right_d) < SIMPLE_SWEEP_ARENA_SIDE_PICK_MARGIN_M:
        return fallback
    return 1.0 if left_d > right_d else -1.0

def simple_sweep_new_lane_revisit_risk():
    """Detect that the lateral shift did not actually leave a recent stripe."""
    lat = simple_sweep_lateral_coord()
    lateral_progress = abs(lat - simple_sweep_current_lane_lateral)
    target_ref = simple_sweep_pending_lane_target_lateral if simple_sweep_pending_lane_target_lateral is not None else simple_sweep_lane_target_lateral
    target_err = abs(lat - target_ref)
    if SIMPLE_SWEEP_LANE_TARGET_MEMORY_ENABLED and target_err > max(0.14, SIMPLE_SWEEP_LANE_TARGET_TOL_M * 2.5):
        return True, f"missed lane target err={target_err:.2f}m tgt={target_ref:.2f} lat={lat:.2f}"
    if lateral_progress < SIMPLE_SWEEP_MIN_LATERAL_PROGRESS_M:
        return True, f"low lateral progress {lateral_progress:.2f}m"
    # Compare with older lanes, not the immediately previous lane.  If the current
    # coordinate matches a recent older lane, the robot is bouncing between lanes
    # instead of monotonically advancing through the sweep.
    for old_lat in list(simple_sweep_lane_lateral_history)[-6:-1]:
        if abs(lat - old_lat) <= SIMPLE_SWEEP_REPEAT_LANE_EPS_M:
            return True, f"revisit lateral {lat:.2f}~{old_lat:.2f}"
    return False, f"ok lateral progress {lateral_progress:.2f}m targetErr={target_err:.2f}m"


def simple_sweep_reset_local_trap_memory():
    global simple_sweep_planned_side_blocked, simple_sweep_opposite_side_blocked, simple_sweep_shift_attempt_side
    simple_sweep_planned_side_blocked = False
    simple_sweep_opposite_side_blocked = False
    simple_sweep_shift_attempt_side = 0.0


def simple_sweep_start_forward(reason="move straight"):
    """Start/continue a sweep lane on the remembered global lane target."""
    global nav_state, coverage_status, row_start_x, row_start_y, row_start_time, desired_grid_heading
    global simple_sweep_state, simple_sweep_last_debug, prev_cmd_left, prev_cmd_right
    global simple_sweep_current_lane_lateral, simple_sweep_lane_lateral_history, simple_sweep_shift_retry_count
    global simple_sweep_lane_score_debug
    desired_grid_heading = simple_sweep_desired_lane_heading()
    nav_state = NAV_FORWARD
    simple_sweep_state = "MOVE_STRAIGHT"
    row_start_x = pose_x
    row_start_y = pose_y
    row_start_time = robot.getTime()
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    simple_sweep_current_lane_lateral = simple_sweep_lateral_coord()
    if (not simple_sweep_lane_lateral_history) or abs(simple_sweep_current_lane_lateral - simple_sweep_lane_lateral_history[-1]) > 0.06:
        simple_sweep_lane_lateral_history.append(simple_sweep_current_lane_lateral)
        del simple_sweep_lane_lateral_history[:-18]
    simple_sweep_shift_retry_count = 0
    turn_side = simple_sweep_turn_side_for_global_shift(simple_sweep_lane_shift_direction)
    lane_err = simple_sweep_lane_target_lateral - simple_sweep_current_lane_lateral
    simple_sweep_last_debug = (
        f"exploreSweep=lane {simple_sweep_lane_index} dir={'+' if simple_sweep_lane_direction > 0 else '-'} "
        f"globalShift={'L' if simple_sweep_lane_shift_direction > 0 else 'R'} "
        f"nextTurn={'L' if turn_side > 0 else 'R'} lat={simple_sweep_current_lane_lateral:.2f} "
        f"targetLat={simple_sweep_lane_target_lateral:.2f} latErr={lane_err:.2f} {simple_sweep_lane_score_debug}"
    )
    coverage_status = f"{simple_sweep_last_debug}: {reason}"
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_OWNER_MIN_SEC, SIMPLE_SWEEP_OWNER_MIN_DIST_M, coverage_status)


def simple_sweep_initialize():
    """Initialize sweep axis from the current robot heading and isolate old owners."""
    global simple_sweep_initialized, simple_sweep_axis_heading, simple_sweep_lane_direction, simple_sweep_lane_shift_direction
    global simple_sweep_lane_index, simple_sweep_state, simple_sweep_queue, simple_sweep_completion_reason
    global desired_grid_heading, lane_side, simple_sweep_lane_lateral_history, simple_sweep_shift_retry_count
    global simple_sweep_initial_lateral, simple_sweep_lane_target_lateral, simple_sweep_pending_lane_target_lateral
    global simple_sweep_shift_target_lateral, simple_sweep_active_shift_global_side, simple_sweep_lane_target_history
    global simple_sweep_pending_shift_global_side, simple_sweep_lane_score_debug
    global simple_sweep_best_coverage_percent, simple_sweep_best_coverage_time, simple_sweep_best_coverage_x, simple_sweep_best_coverage_y
    simple_sweep_initialized = True
    simple_sweep_completion_reason = "running"
    simple_sweep_axis_heading = strict_world_grid_heading(pose_theta)
    desired_grid_heading = simple_sweep_axis_heading
    simple_sweep_lane_direction = 1.0
    # Keep one WORLD/sweep-frame side for the primary sweep.  The robot-relative
    # turn side is recomputed on every row, because the robot faces the opposite
    # direction on return passes.
    simple_sweep_lane_shift_direction = simple_sweep_pick_initial_global_shift_side()
    simple_sweep_lane_index = 0
    simple_sweep_state = "INIT"
    simple_sweep_queue = []
    simple_sweep_initial_lateral = simple_sweep_lateral_coord()
    simple_sweep_lane_target_lateral = simple_sweep_initial_lateral
    simple_sweep_pending_lane_target_lateral = None
    simple_sweep_shift_target_lateral = simple_sweep_lane_target_lateral
    simple_sweep_active_shift_global_side = 0.0
    simple_sweep_pending_shift_global_side = 0.0
    simple_sweep_lane_score_debug = "laneScore=init"
    simple_sweep_lane_target_history = [simple_sweep_lane_target_lateral]
    simple_sweep_lane_lateral_history = []
    simple_sweep_shift_retry_count = 0
    simple_sweep_best_coverage_percent = float(last_coverage_percent or 0.0)
    simple_sweep_best_coverage_time = robot.getTime()
    simple_sweep_best_coverage_x = pose_x
    simple_sweep_best_coverage_y = pose_y
    simple_sweep_reset_local_trap_memory()
    simple_sweep_reset_local_churn("initialize deterministic sweep")
    simple_sweep_clear_legacy_motion("initialize deterministic sweep")
    simple_sweep_start_forward("initialized deterministic boustrophedon sweep")


def simple_sweep_start_backup(reason="backup straight"):
    global nav_state, simple_sweep_state, backup_start_x, backup_start_y, backup_until
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    nav_state = NAV_RECOVERY_BACKUP
    simple_sweep_state = "BACKUP_STRAIGHT"
    backup_start_x = pose_x
    backup_start_y = pose_y
    backup_until = robot.getTime() + SIMPLE_SWEEP_BACKUP_TIMEOUT_SEC
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"exploreSweep backup: {reason}"
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_BACKUP_TIMEOUT_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.20)


def simple_sweep_begin_pivot_90(direction, reason):
    """Start one in-place 90-degree pivot owned by SIMPLE_SWEEP_FSM."""
    global nav_state, turn_target_theta, turn_direction, turn_settle_until
    global turn_start_left, turn_start_right, turn_start_time, turn_best_abs_error, turn_last_progress_time
    global prev_cmd_left, prev_cmd_right, last_turn_reason, last_turn_variant, map_freeze_until, coverage_status, desired_grid_heading
    global simple_sweep_state
    turn_direction = 1.0 if direction >= 0 else -1.0
    base_heading = desired_grid_heading
    turn_target_theta = strict_world_grid_heading(normalize_angle(base_heading + turn_direction * PIVOT_TURN_ANGLE))
    turn_start_left = left_sensor.getValue()
    turn_start_right = right_sensor.getValue()
    turn_start_time = robot.getTime()
    turn_best_abs_error = abs(normalize_angle(turn_target_theta - pose_theta))
    turn_last_progress_time = turn_start_time
    turn_settle_until = 0.0
    last_turn_reason = str(reason)
    last_turn_variant = "simple_sweep_grid90"
    nav_state = NAV_TURN_90
    simple_sweep_state = "TURN_IN_PLACE_90"
    coverage_status = f"exploreSweep pivot90 target={math.degrees(turn_target_theta):.0f}: {str(reason)[:54]}"
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    acquire_control(SIMPLE_SWEEP_OWNER, PIVOT_OWNER_TIME_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)


def simple_sweep_finish_pivot_90():
    global nav_state, turn_settle_until, pose_theta, prev_left, prev_right
    global current_angular_velocity, current_linear_velocity, desired_grid_heading, map_freeze_until
    hard_stop_motors()
    desired_grid_heading = turn_target_theta
    if inertial_unit is None:
        pose_theta = turn_target_theta
    prev_left = left_sensor.getValue()
    prev_right = right_sensor.getValue()
    current_angular_velocity = 0.0
    current_linear_velocity = 0.0
    nav_state = NAV_SETTLE
    turn_settle_until = robot.getTime() + SIMPLE_SWEEP_TURN_SETTLE_SEC
    map_freeze_until = max(map_freeze_until, turn_settle_until + TURN_FREEZE_HOLD_SEC)
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_TURN_SETTLE_SEC, 0.0, "exploreSweep settle after pivot")


def simple_sweep_heading_error(target_heading=None):
    """Signed yaw error to a stored sweep/cardinal heading."""
    target = desired_grid_heading if target_heading is None else target_heading
    return normalize_angle(strict_world_grid_heading(target) - pose_theta)


def simple_sweep_start_align(target_heading=None, reason="heading align"):
    """Start a small in-place heading correction owned by SIMPLE_SWEEP_FSM.

    This deliberately avoids wall-following and curved steering.  It only uses
    the IMU/odometry heading and a cardinal target produced by the deterministic
    sweep FSM.
    """
    global nav_state, simple_sweep_state, simple_sweep_align_target_heading
    global simple_sweep_align_started_at, simple_sweep_align_until, desired_grid_heading
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until

    if not SIMPLE_SWEEP_ALIGN_ENABLED:
        simple_sweep_run_next_action()
        return
    target = desired_grid_heading if target_heading is None else target_heading
    target = strict_world_grid_heading(target)
    err = normalize_angle(target - pose_theta)
    desired_grid_heading = target
    if abs(err) <= SIMPLE_SWEEP_ALIGN_TOL_RAD:
        coverage_status = f"exploreSweep align skip err={math.degrees(err):.1f}: {str(reason)[:46]}"
        simple_sweep_run_next_action()
        return

    hard_stop_motors()
    nav_state = NAV_TURN_90
    simple_sweep_state = "ALIGN_TO_HEADING"
    simple_sweep_align_target_heading = target
    simple_sweep_align_started_at = robot.getTime()
    simple_sweep_align_until = simple_sweep_align_started_at + SIMPLE_SWEEP_ALIGN_TIMEOUT_SEC
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = f"exploreSweep align target={math.degrees(target):.0f} err={math.degrees(err):.1f}: {str(reason)[:46]}"
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_ALIGN_HARD_TIMEOUT_SEC, 0.0, coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + TURN_FREEZE_HOLD_SEC)


def simple_sweep_finish_align(reason="aligned"):
    global nav_state, simple_sweep_state, desired_grid_heading, turn_settle_until
    global map_freeze_until, simple_sweep_last_align_time, current_angular_velocity, current_linear_velocity
    hard_stop_motors()
    desired_grid_heading = strict_world_grid_heading(simple_sweep_align_target_heading)
    current_angular_velocity = 0.0
    current_linear_velocity = 0.0
    simple_sweep_last_align_time = robot.getTime()
    nav_state = NAV_SETTLE
    simple_sweep_state = "ALIGN_SETTLE"
    turn_settle_until = robot.getTime() + SIMPLE_SWEEP_ALIGN_SETTLE_SEC
    map_freeze_until = max(map_freeze_until, turn_settle_until + TURN_FREEZE_HOLD_SEC)
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_ALIGN_SETTLE_SEC, 0.0, f"exploreSweep align settle: {reason}")


def simple_sweep_start_lane_shift(side, distance=SIMPLE_SWEEP_LANE_SHIFT_M):
    global nav_state, simple_sweep_state, simple_sweep_shift_attempt_side, simple_sweep_shift_start_lateral
    global lane_shift_start_x, lane_shift_start_y, lane_shift_start_time, lane_shift_target_dist, lane_shift_heading_target
    global prev_cmd_left, prev_cmd_right, coverage_status, map_freeze_until
    global simple_sweep_active_shift_global_side, simple_sweep_shift_target_lateral, simple_sweep_pending_lane_target_lateral
    global simple_sweep_pending_shift_global_side, simple_sweep_lane_score_debug
    hard_stop_motors()
    nav_state = NAV_LANE_SHIFT
    simple_sweep_state = "LANE_SHIFT_STRAIGHT"
    simple_sweep_shift_attempt_side = 1.0 if side >= 0 else -1.0
    simple_sweep_active_shift_global_side = simple_sweep_shift_global_from_robot_side(simple_sweep_shift_attempt_side)
    simple_sweep_shift_start_lateral = simple_sweep_lateral_coord()
    if SIMPLE_SWEEP_LANE_TARGET_MEMORY_ENABLED:
        if simple_sweep_pending_lane_target_lateral is None:
            simple_sweep_plan_next_lane_target(simple_sweep_active_shift_global_side)
        if simple_sweep_pending_shift_global_side != 0.0:
            simple_sweep_active_shift_global_side = 1.0 if simple_sweep_pending_shift_global_side > 0.0 else -1.0
        simple_sweep_shift_target_lateral = float(simple_sweep_pending_lane_target_lateral)
        distance_to_target = abs(simple_sweep_shift_target_lateral - simple_sweep_shift_start_lateral)
        dynamic_shift_max = max(float(SIMPLE_SWEEP_LANE_SHIFT_MAX_DIST_M), distance_to_target + 0.06)
        lane_shift_target_dist = clamp(distance_to_target, 0.08, dynamic_shift_max)
    else:
        simple_sweep_shift_target_lateral = simple_sweep_shift_start_lateral + simple_sweep_active_shift_global_side * float(distance)
        lane_shift_target_dist = float(distance)
    lane_shift_start_x = pose_x
    lane_shift_start_y = pose_y
    lane_shift_start_time = robot.getTime()
    lane_shift_heading_target = pose_theta
    prev_cmd_left = 0.0
    prev_cmd_right = 0.0
    coverage_status = (
        f"exploreSweep lane shift {'L' if simple_sweep_shift_attempt_side > 0 else 'R'} "
        f"global={'L' if simple_sweep_active_shift_global_side > 0 else 'R'} "
        f"lat={simple_sweep_shift_start_lateral:.2f}->{simple_sweep_shift_target_lateral:.2f} {simple_sweep_lane_score_debug}"
    )
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_LANE_SHIFT_TIMEOUT_SEC, max(0.0, lane_shift_target_dist * 0.65), coverage_status)
    map_freeze_until = max(map_freeze_until, robot.getTime() + 0.20)


def simple_sweep_start_escape_forward(reason="local trap escape"):
    global nav_state, simple_sweep_state, simple_sweep_escape_start_x, simple_sweep_escape_start_y, simple_sweep_escape_until
    global coverage_status, row_start_x, row_start_y, row_start_time
    nav_state = NAV_FORWARD
    simple_sweep_state = "ESCAPE_FORWARD"
    simple_sweep_escape_start_x = pose_x
    simple_sweep_escape_start_y = pose_y
    simple_sweep_escape_until = robot.getTime() + SIMPLE_SWEEP_ESCAPE_FORWARD_TIMEOUT_SEC
    row_start_x = pose_x
    row_start_y = pose_y
    row_start_time = robot.getTime()
    coverage_status = f"exploreSweep straight escape: {reason}"
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_ESCAPE_FORWARD_TIMEOUT_SEC, SIMPLE_SWEEP_ESCAPE_FORWARD_M * 0.70, coverage_status)


def simple_sweep_note_local_churn(kind, reason):
    global simple_sweep_churn_anchor_x, simple_sweep_churn_anchor_y, simple_sweep_churn_start_time
    global simple_sweep_churn_action_count, simple_sweep_churn_last_debug
    if kind not in ("TURN", "ALIGN", "SHIFT"):
        return False
    now = robot.getTime()
    dist = math.hypot(pose_x - simple_sweep_churn_anchor_x, pose_y - simple_sweep_churn_anchor_y)
    age = now - simple_sweep_churn_start_time
    if (
        simple_sweep_churn_action_count <= 0
        or dist > SIMPLE_SWEEP_LOCAL_CHURN_RADIUS_M
        or age > SIMPLE_SWEEP_LOCAL_CHURN_WINDOW_SEC
    ):
        simple_sweep_churn_anchor_x = pose_x
        simple_sweep_churn_anchor_y = pose_y
        simple_sweep_churn_start_time = now
        simple_sweep_churn_action_count = 1
    else:
        simple_sweep_churn_action_count += 1
    simple_sweep_churn_last_debug = (
        f"churn={simple_sweep_churn_action_count}/{SIMPLE_SWEEP_LOCAL_CHURN_MAX_ACTIONS} "
        f"{kind} r={dist:.2f} age={age:.1f}s {str(reason)[:36]}"
    )
    return simple_sweep_churn_action_count >= SIMPLE_SWEEP_LOCAL_CHURN_MAX_ACTIONS


def simple_sweep_reset_local_churn(reason="reset"):
    global simple_sweep_churn_action_count, simple_sweep_churn_start_time, simple_sweep_churn_last_debug
    simple_sweep_churn_action_count = 0
    simple_sweep_churn_start_time = -999.0
    simple_sweep_churn_last_debug = f"churn=reset {str(reason)[:32]}"


def simple_sweep_force_straight_escape(reason):
    """Break repeated TURN/LANE_SHIFT/RECOVERY churn without changing sweep axis."""
    global simple_sweep_queue, simple_sweep_lane_score_debug, simple_sweep_pending_lane_target_lateral
    global simple_sweep_pending_shift_global_side, simple_sweep_planned_side_blocked, simple_sweep_opposite_side_blocked
    # Pick the turn side from the last attempted lane shift, but do not reseed
    # simple_sweep_axis_heading.  The 180 + straight escape is local; after it
    # finishes simple_sweep_after_escape_reset() rejoins the preserved sweep axis.
    turn_side = 1.0 if simple_sweep_shift_attempt_side >= 0.0 else -1.0
    simple_sweep_pending_lane_target_lateral = None
    simple_sweep_pending_shift_global_side = 0.0
    simple_sweep_planned_side_blocked = True
    simple_sweep_opposite_side_blocked = True
    simple_sweep_queue = [
        ("TURN", turn_side, f"local churn 180 turn 1/2: {reason}"),
        ("TURN", turn_side, "local churn 180 turn 2/2"),
        ("ALIGN", "local churn heading before straight escape"),
        ("ESCAPE_FORWARD", f"straight escape after local churn: {reason}"),
    ]
    simple_sweep_lane_score_debug = f"laneScore=churn180Escape {str(reason)[:36]}"
    simple_sweep_reset_local_churn(reason)
    simple_sweep_start_backup(f"local churn release: {reason}")


def simple_sweep_run_next_action():
    """Execute the next atomic sweep primitive from the private FSM queue."""
    global simple_sweep_queue, simple_sweep_lane_direction, simple_sweep_lane_index, desired_grid_heading
    if not simple_sweep_queue:
        simple_sweep_start_forward("queue empty; resume lane")
        return
    action = simple_sweep_queue.pop(0)
    kind = action[0]
    if kind in ("TURN", "ALIGN", "SHIFT"):
        reason = action[2] if kind == "TURN" and len(action) > 2 else (action[1] if len(action) > 1 else kind)
        if simple_sweep_note_local_churn(kind, reason):
            simple_sweep_force_straight_escape(simple_sweep_churn_last_debug)
            return
    if kind == "BACKUP":
        simple_sweep_start_backup(action[1] if len(action) > 1 else "lane end")
    elif kind == "TURN":
        side = action[1]
        reason = action[2] if len(action) > 2 else "sweep 90"
        simple_sweep_begin_pivot_90(side, reason)
    elif kind == "SHIFT":
        side = action[1]
        dist = action[2] if len(action) > 2 else SIMPLE_SWEEP_LANE_SHIFT_M
        simple_sweep_start_lane_shift(side, dist)
    elif kind == "ALIGN":
        reason = action[1] if len(action) > 1 else "queued heading align"
        target = action[2] if len(action) > 2 else desired_grid_heading
        simple_sweep_start_align(target, reason)
    elif kind == "FORWARD":
        simple_sweep_commit_pending_lane_target()
        simple_sweep_lane_direction *= -1.0
        simple_sweep_lane_index += 1
        simple_sweep_reset_local_trap_memory()
        simple_sweep_start_forward(action[1] if len(action) > 1 else "next lane")
    elif kind == "RESUME_FORWARD":
        # Local-obstacle bypass: we moved to a neighbouring lane but should keep
        # the same sweep direction instead of performing a row-end U-turn.
        simple_sweep_commit_pending_lane_target()
        simple_sweep_lane_index += 1
        simple_sweep_reset_local_trap_memory()
        simple_sweep_start_forward(action[1] if len(action) > 1 else "resume current lane direction")
    elif kind == "ESCAPE_FORWARD":
        simple_sweep_start_escape_forward(action[1] if len(action) > 1 else "escape")
    else:
        simple_sweep_start_forward(f"unknown queue action {kind}")


def simple_sweep_schedule_lane_change(global_side, reason):
    """Schedule backup -> 90 -> shift -> 90 -> opposite lane.

    global_side is fixed in the sweep frame.  It is converted to the current
    robot-relative turn side here, so consecutive rows keep advancing across the
    room instead of returning to the previous stripe.
    """
    global simple_sweep_queue, simple_sweep_shift_attempt_side, coverage_status, simple_sweep_last_shift_global_side
    global simple_sweep_lane_shift_direction, simple_sweep_pending_shift_global_side
    global_side = 1.0 if global_side >= 0 else -1.0
    target_lat = simple_sweep_plan_next_lane_target(global_side)
    actual_global_side = simple_sweep_pending_shift_global_side if simple_sweep_pending_shift_global_side != 0.0 else global_side
    actual_global_side = 1.0 if actual_global_side >= 0 else -1.0
    side = simple_sweep_turn_side_for_global_shift(actual_global_side)
    simple_sweep_shift_attempt_side = side
    simple_sweep_last_shift_global_side = actual_global_side
    simple_sweep_lane_shift_direction = actual_global_side
    simple_sweep_queue = [
        ("BACKUP", f"row end before lane shift: {reason}"),
        ("TURN", side, f"first 90 to lane shift: {reason}"),
        ("ALIGN", "after first 90 before lane shift"),
        ("SHIFT", side, SIMPLE_SWEEP_LANE_SHIFT_M),
        ("TURN", side, "second 90 to reverse lane"),
        ("ALIGN", "after second 90 before reverse lane"),
        ("FORWARD", "move straight in reverse lane direction"),
    ]
    coverage_status = (
        f"exploreSweep schedule lane change global={'L' if actual_global_side > 0 else 'R'} "
        f"turn={'L' if side > 0 else 'R'} targetLat={target_lat:.2f}: {reason} | {simple_sweep_lane_score_debug}"
    )
    simple_sweep_run_next_action()


def simple_sweep_contact_is_row_end(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right):
    """Classify a bumper hit during map-discovery sweep.

    A wall/boundary contact ends the current stripe.  A compact low object inside
    the room should be mapped as a contact obstacle and bypassed locally; otherwise
    the robot incorrectly starts a new perimeter-like turn sequence around every
    table leg or low block.
    """
    center_or_both = bool(last_bumper_center or (bumper_left and bumper_right))
    single_side = bool((bumper_left ^ bumper_right) and not center_or_both)
    if contact_looks_like_wall_boundary():
        return True, "wall-boundary"
    try:
        arena_ahead = simple_sweep_ray_distance_to_arena(pose_theta)
        if arena_ahead is not None and arena_ahead < SIMPLE_SWEEP_PRECONTACT_FRONT_M + 0.08:
            return True, f"arena-edge {arena_ahead:.2f}m"
    except Exception:
        pass

    # Low object: lower/bumper contact but the upper RGB-D band is open.  This is
    # exactly the red block/chair-foot case; treating it as a row end makes the
    # FSM turn into the nearest wall.
    if upper_front > max(0.68, WALL_CONTACT_UPPER_MAX_M + 0.10):
        return False, f"local-low upper={upper_front:.2f}"

    full_height_front = bool(
        center_or_both
        and front < WALL_CONTACT_FRONT_MAX_M + 0.08
        and center < WALL_CONTACT_CENTER_MAX_M + 0.08
        and upper_front < WALL_CONTACT_UPPER_MAX_M + 0.08
    )
    if full_height_front:
        return True, f"full-height F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f}"

    if single_side and min(left, right) < WALL_CONTACT_SIDE_NEAR_M and upper_front < WALL_CONTACT_UPPER_MAX_M:
        return True, f"side-wall side={min(left, right):.2f} upper={upper_front:.2f}"

    if body_clearance < BODY_CORRIDOR_HARD_CLEARANCE and upper_front < WALL_CONTACT_UPPER_MAX_M:
        return True, f"body-wall body={body_clearance:.2f} upper={upper_front:.2f}"

    return False, f"local-object F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f} body={body_clearance:.2f}"


def simple_sweep_choose_local_bypass_global_side(left_dist, right_dist, bumper_left, bumper_right):
    """Choose a lateral side for bypassing an internal contact obstacle."""
    center_or_both = bool(last_bumper_center or (bumper_left and bumper_right))
    if bumper_left and not bumper_right and not center_or_both:
        robot_side = -1.0
    elif bumper_right and not bumper_left and not center_or_both:
        robot_side = 1.0
    elif left_dist > right_dist + 0.16:
        robot_side = 1.0
    elif right_dist > left_dist + 0.16:
        robot_side = -1.0
    else:
        robot_side = simple_sweep_turn_side_for_global_shift(simple_sweep_lane_shift_direction)
    return simple_sweep_shift_global_from_robot_side(robot_side)


def simple_sweep_schedule_local_obstacle_bypass(global_side, reason):
    """Bypass a compact internal obstacle without flipping the sweep direction."""
    global simple_sweep_queue, simple_sweep_shift_attempt_side, coverage_status, simple_sweep_last_shift_global_side
    global simple_sweep_lane_shift_direction, simple_sweep_pending_shift_global_side
    global simple_sweep_lane_score_debug
    global_side = 1.0 if global_side >= 0 else -1.0
    target_lat = simple_sweep_plan_next_lane_target(global_side)
    actual_global_side = simple_sweep_pending_shift_global_side if simple_sweep_pending_shift_global_side != 0.0 else global_side
    actual_global_side = 1.0 if actual_global_side >= 0.0 else -1.0
    side = simple_sweep_turn_side_for_global_shift(actual_global_side)
    simple_sweep_shift_attempt_side = side
    simple_sweep_last_shift_global_side = actual_global_side
    simple_sweep_lane_shift_direction = actual_global_side
    simple_sweep_queue = [
        ("BACKUP", f"local obstacle before bypass: {reason}"),
        ("TURN", side, f"local obstacle first 90: {reason}"),
        ("ALIGN", "local obstacle before bypass shift"),
        ("SHIFT", side, SIMPLE_SWEEP_LANE_SHIFT_M),
        ("TURN", -side, "local obstacle second 90; keep lane direction"),
        ("ALIGN", "local obstacle heading before resume"),
        ("RESUME_FORWARD", "local obstacle bypassed; continue same sweep direction"),
    ]
    coverage_status = (
        f"exploreSweep local obstacle bypass global={'L' if actual_global_side > 0 else 'R'} "
        f"turn={'L' if side > 0 else 'R'} targetLat={target_lat:.2f}: {reason} | {simple_sweep_lane_score_debug}"
    )
    simple_sweep_run_next_action()


def simple_sweep_try_opposite_side(reason):
    """When planned side blocks immediately, try the opposite physical side once."""
    global simple_sweep_queue, simple_sweep_planned_side_blocked, coverage_status, simple_sweep_last_shift_global_side
    global simple_sweep_lane_shift_direction, simple_sweep_pending_shift_global_side
    attempted_global = simple_sweep_last_shift_global_side if simple_sweep_last_shift_global_side != 0.0 else simple_sweep_lane_shift_direction
    attempted_global = 1.0 if attempted_global >= 0.0 else -1.0
    opposite_global = -attempted_global
    target_lat = simple_sweep_plan_next_lane_target(opposite_global)
    actual_opposite_global = simple_sweep_pending_shift_global_side if simple_sweep_pending_shift_global_side != 0.0 else opposite_global
    actual_opposite_global = 1.0 if actual_opposite_global >= 0.0 else -1.0
    opposite = simple_sweep_turn_side_for_global_shift(actual_opposite_global)
    simple_sweep_last_shift_global_side = actual_opposite_global
    simple_sweep_lane_shift_direction = actual_opposite_global
    simple_sweep_planned_side_blocked = True
    # Current heading is already perpendicular to the lane. Two opposite 90-degree
    # pivots make the robot face the other lateral direction without drawing an arc.
    simple_sweep_queue = [
        ("BACKUP", f"planned side blocked; release before opposite try: {reason}"),
        ("TURN", opposite, "opposite-side recovery turn 1/2"),
        ("TURN", opposite, "opposite-side recovery turn 2/2"),
        ("ALIGN", "opposite-side heading before shift"),
        ("SHIFT", opposite, SIMPLE_SWEEP_LANE_SHIFT_M),
        ("TURN", opposite, "second 90 after opposite lane shift"),
        ("ALIGN", "opposite-side heading before forward"),
        ("FORWARD", "opposite side worked; resume sweep"),
    ]
    coverage_status = (
        f"exploreSweep planned global side blocked, try opposite "
        f"global={'L' if actual_opposite_global > 0 else 'R'} turn={'L' if opposite > 0 else 'R'} targetLat={target_lat:.2f} | {simple_sweep_lane_score_debug}"
    )
    simple_sweep_run_next_action()


def simple_sweep_start_local_trap_escape(reason):
    """F + planned side + opposite side blocked: backup, turn 180, straight escape."""
    global simple_sweep_queue, simple_sweep_opposite_side_blocked, coverage_status
    simple_sweep_opposite_side_blocked = True
    turn_side = 1.0 if simple_sweep_shift_attempt_side >= 0.0 else -1.0
    simple_sweep_queue = [
        ("BACKUP", f"F/L/R blocked; release: {reason}"),
        ("TURN", turn_side, "local trap 180 turn 1/2"),
        ("TURN", turn_side, "local trap 180 turn 2/2"),
        ("ALIGN", "local trap heading before straight escape"),
        ("ESCAPE_FORWARD", "F/L/R blocked straight escape"),
    ]
    coverage_status = f"exploreSweep local trap escape: {reason}"
    simple_sweep_run_next_action()


def simple_sweep_handle_shift_block(reason):
    """Handle contact/stall during lane shift without handing control to frontier."""
    if not simple_sweep_planned_side_blocked:
        simple_sweep_try_opposite_side(reason)
    else:
        simple_sweep_start_local_trap_escape(reason)


def simple_sweep_after_escape_reset():
    """After a local trap escape, reset local trap state but keep lane memory sane.

    a accidentally allowed a recovery escape to re-seed the sweep axis from
    the robot's *current* heading.  In a bottom/right corner that turned a
    horizontal boustrophedon sweep into a vertical wall chase, producing the
    repeated TURN/LANE_SHIFT loop visible around the lower-right wall.  A trap
    escape may change which direction we travel on the same stripe, but it must
    not redefine the global sweep frame.
    """
    global simple_sweep_axis_heading, simple_sweep_lane_direction, desired_grid_heading, simple_sweep_queue
    global simple_sweep_lane_target_lateral, simple_sweep_pending_lane_target_lateral, simple_sweep_lane_target_history
    global simple_sweep_pending_shift_global_side, simple_sweep_lane_score_debug

    if not SIMPLE_SWEEP_PRESERVE_AXIS_AFTER_ESCAPE:
        simple_sweep_axis_heading = strict_world_grid_heading(pose_theta)
        simple_sweep_lane_direction = 1.0
    else:
        # Keep the original sweep axis.  Rejoin the nearer of +axis / -axis so
        # the robot resumes a stripe instead of continuing into the wall after a
        # corner escape.
        err_plus = abs(normalize_angle(simple_sweep_axis_heading - pose_theta))
        err_minus = abs(normalize_angle(normalize_angle(simple_sweep_axis_heading + math.pi) - pose_theta))
        simple_sweep_lane_direction = 1.0 if err_plus <= err_minus else -1.0

    desired_grid_heading = simple_sweep_desired_lane_heading()
    simple_sweep_lane_target_lateral = simple_sweep_lateral_coord()
    simple_sweep_pending_lane_target_lateral = None
    simple_sweep_pending_shift_global_side = 0.0
    simple_sweep_lane_score_debug = "laneScore=afterEscapePreserveAxis"
    if (not simple_sweep_lane_target_history) or abs(simple_sweep_lane_target_lateral - simple_sweep_lane_target_history[-1]) > 0.05:
        simple_sweep_lane_target_history.append(simple_sweep_lane_target_lateral)
        del simple_sweep_lane_target_history[:-24]
    simple_sweep_reset_local_trap_memory()
    if SIMPLE_SWEEP_REJOIN_ALIGN_AFTER_ESCAPE:
        simple_sweep_queue = [("FORWARD", "local trap escaped; resume preserved sweep axis")]
        simple_sweep_start_align(desired_grid_heading, "rejoin preserved sweep axis after trap escape")
    else:
        simple_sweep_start_forward("local trap memory reset after straight escape")


def simple_sweep_virtual_row_end_reason(front, center, upper_front, body_clearance):
    """Return a conservative virtual row-end reason for EXPLORE sweep.

    This is not a frontier/route planner and it is not normal depth steering.
    It only catches the failure mode where the parent collision shell or odometry
    reaches a wall before the TouchSensor bumper reports a clean contact.
    """
    if not EXPLORE_SWEEP_VIRTUAL_ROW_END_ENABLED:
        return None
    try:
        row_dist = row_distance_from_start()
    except Exception:
        row_dist = 0.0
    if row_dist < float(EXPLORE_SWEEP_VIRTUAL_ROW_END_MIN_ROW_M):
        return None

    # True near-contact from the RGB-D depth channel / body envelope.  This is
    # intentionally very close-range so ordinary depth does not terminate rows.
    try:
        body_hard, body_soft = body_row_end_flags(front, center, upper_front, body_clearance)
    except Exception:
        body_hard, body_soft = False, False
    if body_hard:
        return f"virtual row-end body-hard F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f} body={body_clearance:.2f}"
    if (
        float(front) < float(EXPLORE_SWEEP_VIRTUAL_FRONT_M)
        and float(center) < float(EXPLORE_SWEEP_VIRTUAL_CENTER_M)
        and float(upper_front) < float(EXPLORE_SWEEP_VIRTUAL_UPPER_M)
    ):
        return f"virtual row-end near-contact F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f}"

    # Webots/encoder fail-safe: if the commanded front probe is already outside
    # the known test arena, schedule a lane change instead of continuing to paint
    # an orange line through the wall.  This is a simulation safety fence, not the
    # main mapping source.
    if EXPLORE_SWEEP_ARENA_FRONT_GUARD_ENABLED and DEBUG_ARENA_BOUNDS_ENABLED:
        try:
            hdg = simple_sweep_desired_lane_heading()
            probe = float(EXPLORE_SWEEP_ARENA_FRONT_PROBE_M)
            fx = pose_x + math.cos(hdg) * probe
            fy = pose_y + math.sin(hdg) * probe
            m = float(EXPLORE_SWEEP_ARENA_FRONT_GUARD_MARGIN_M)
            if fx > DEBUG_ARENA_X_MAX_M - m:
                return f"virtual row-end arena+x fx={fx:.2f} max={DEBUG_ARENA_X_MAX_M:.2f}"
            if fx < DEBUG_ARENA_X_MIN_M + m:
                return f"virtual row-end arena-x fx={fx:.2f} min={DEBUG_ARENA_X_MIN_M:.2f}"
            if fy > DEBUG_ARENA_Y_MAX_M - m:
                return f"virtual row-end arena+y fy={fy:.2f} max={DEBUG_ARENA_Y_MAX_M:.2f}"
            if fy < DEBUG_ARENA_Y_MIN_M + m:
                return f"virtual row-end arena-y fy={fy:.2f} min={DEBUG_ARENA_Y_MIN_M:.2f}"
        except Exception:
            pass
    return None


def simple_sweep_fsm_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right):
    """Wheel command for the isolated deterministic boustrophedon sweep FSM.

    Returns None only when the primary sweep is finished or explicitly disabled.
    Otherwise it owns the wheels and blocks frontier/route/scan ownership.
    """
    global row_end_candidate_count, coverage_status, last_optional_block_reason
    global turn_best_abs_error, turn_last_progress_time, simple_sweep_shift_retry_count

    if not simple_sweep_should_own():
        return None
    if nav_state in CONTACT_RECOVERY_STATES:
        # A previously-started hard recovery is allowed to finish.  Once it releases,
        # SIMPLE_SWEEP_FSM will reacquire motion before route/frontier code runs.
        return None
    if not simple_sweep_initialized:
        simple_sweep_initialize()
    simple_sweep_clear_legacy_motion("tick")
    last_optional_block_reason = "EXPLORE_SWEEP_FSM owns map-discovery sweep"

    now = robot.getTime()
    bumper_any = bool(bumper_left or bumper_right or last_bumper_center)

    # Bumper/contact is authoritative, but not every contact is a row end.
    # A low/internal obstacle is first written into the map, then bypassed locally
    # while keeping the global sweep direction.
    if bumper_any:
        obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
        marked_contact = False
        if nav_state in (NAV_FORWARD, NAV_LANE_SHIFT) or simple_sweep_state in ("MOVE_STRAIGHT", "LANE_SHIFT_STRAIGHT"):
            marked_contact = mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
        if marked_contact:
            update_structural_obstacle_memory()
            update_furniture_zone_cache(force=True)
        if nav_state == NAV_FORWARD and simple_sweep_state == "MOVE_STRAIGHT":
            row_end_candidate_count = 0
            is_row_end, contact_kind = simple_sweep_contact_is_row_end(
                front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right
            )
            if is_row_end:
                simple_sweep_schedule_lane_change(
                    simple_sweep_lane_shift_direction,
                    f"bumper row end {contact_kind} L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
                )
            else:
                bypass_global = simple_sweep_choose_local_bypass_global_side(left, right, bumper_left, bumper_right)
                simple_sweep_schedule_local_obstacle_bypass(
                    bypass_global,
                    f"bumper local obstacle {contact_kind} L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
                )
            return 0.0, 0.0
        if nav_state == NAV_TURN_90 and simple_sweep_state == "TURN_IN_PLACE_90":
            # Do not rotate while a bumper is still physically pressed.  Reinsert
            # the interrupted 90-degree pivot after a short straight release.
            simple_sweep_queue.insert(0, ("TURN", turn_direction, "retry 90 after bumper release"))
            simple_sweep_start_backup(
                f"bumper still active before/during pivot L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}"
            )
            return 0.0, 0.0
        if nav_state == NAV_TURN_90 and simple_sweep_state == "ALIGN_TO_HEADING":
            simple_sweep_queue.insert(0, ("ALIGN", "retry heading align after bumper release", simple_sweep_align_target_heading))
            simple_sweep_start_backup(
                f"bumper active during heading align L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}"
            )
            return 0.0, 0.0
        if nav_state == NAV_SETTLE and simple_sweep_state == "TURN_IN_PLACE_90":
            simple_sweep_start_backup(
                f"bumper active during settle L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}"
            )
            return 0.0, 0.0
        if nav_state == NAV_LANE_SHIFT or simple_sweep_state == "LANE_SHIFT_STRAIGHT":
            shifted = math.hypot(pose_x - lane_shift_start_x, pose_y - lane_shift_start_y)
            simple_sweep_handle_shift_block(
                f"side shift contact after {shifted:.2f}m L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}"
            )
            return 0.0, 0.0
        if simple_sweep_state == "ESCAPE_FORWARD":
            simple_sweep_start_local_trap_escape(
                f"escape contact L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}"
            )
            return 0.0, 0.0

    if nav_state == NAV_RECOVERY_BACKUP and simple_sweep_state == "BACKUP_STRAIGHT":
        moved = math.hypot(pose_x - backup_start_x, pose_y - backup_start_y)
        remaining = max(0.0, SIMPLE_SWEEP_BACKUP_DISTANCE_M - moved)
        if moved >= SIMPLE_SWEEP_BACKUP_DISTANCE_M or now >= backup_until:
            simple_sweep_run_next_action()
            return 0.0, 0.0
        if reverse_backup_should_stop(remaining):
            coverage_status = f"exploreSweep backup rear guard {last_rear_guard_clearance:.2f}m {last_rear_guard_reason}"
            simple_sweep_run_next_action()
            return 0.0, 0.0
        coverage_status = f"exploreSweep backup straight {moved:.2f}/{SIMPLE_SWEEP_BACKUP_DISTANCE_M:.2f} rear={last_rear_guard_clearance:.2f}"
        return -BACKUP_SPEED, -BACKUP_SPEED

    if nav_state == NAV_TURN_90 and simple_sweep_state == "TURN_IN_PLACE_90":
        heading_remaining = normalize_angle(turn_target_theta - pose_theta)
        abs_err = abs(heading_remaining)
        coverage_status = f"exploreSweep pivot90 err={math.degrees(heading_remaining):.1f} target={math.degrees(turn_target_theta):.0f}"
        progress_step = math.radians(PIVOT_WATCHDOG_PROGRESS_DEG)
        if abs_err < turn_best_abs_error - progress_step:
            turn_best_abs_error = abs_err
            turn_last_progress_time = now
        if abs_err <= PIVOT_TURN_TOLERANCE:
            simple_sweep_finish_pivot_90()
            return 0.0, 0.0
        if now - turn_start_time > SIMPLE_SWEEP_PIVOT_TIMEOUT_SEC:
            # Do not call the legacy pivot watchdog.  It can escape into ROW_FORWARD.
            # Continue the private queue only when the pivot is already close.
            # A large timeout in the lower/right corner used to be accepted as a
            # successful 90, stacking more TURN/SHIFT actions around one pose.
            if abs_err <= SIMPLE_SWEEP_PIVOT_TIMEOUT_ACCEPT_RAD:
                simple_sweep_finish_pivot_90()
                coverage_status = f"exploreSweep pivot timeout accepted err={math.degrees(abs_err):.1f}"
            else:
                simple_sweep_force_straight_escape(
                    f"pivot timeout err={math.degrees(abs_err):.1f} target={math.degrees(turn_target_theta):.0f}"
                )
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        if abs_err <= PIVOT_FINAL_ZONE:
            speed = clamp(abs_err * PIVOT_TURN_KP, PIVOT_FINAL_MIN_SPEED, PIVOT_FINAL_MAX_SPEED)
        elif abs_err <= PIVOT_FINE_ZONE:
            speed = clamp(abs_err * PIVOT_TURN_KP, PIVOT_FINE_MIN_SPEED, PIVOT_FINE_MAX_SPEED)
        else:
            speed = clamp(abs_err * PIVOT_TURN_KP, PIVOT_TURN_MIN_SPEED, PIVOT_TURN_SPEED)
        return -sign * speed, sign * speed

    if nav_state == NAV_TURN_90 and simple_sweep_state == "ALIGN_TO_HEADING":
        heading_remaining = normalize_angle(simple_sweep_align_target_heading - pose_theta)
        abs_err = abs(heading_remaining)
        coverage_status = (
            f"exploreSweep align err={math.degrees(heading_remaining):.1f} "
            f"target={math.degrees(simple_sweep_align_target_heading):.0f}"
        )
        if abs_err <= SIMPLE_SWEEP_ALIGN_TOL_RAD:
            simple_sweep_finish_align("ok")
            return 0.0, 0.0
        if now >= simple_sweep_align_until and abs_err <= SIMPLE_SWEEP_ALIGN_TIMEOUT_ACCEPT_RAD:
            simple_sweep_finish_align(f"timeout accept err={math.degrees(abs_err):.1f}")
            return 0.0, 0.0
        if now - simple_sweep_align_started_at >= SIMPLE_SWEEP_ALIGN_HARD_TIMEOUT_SEC:
            simple_sweep_finish_align(f"hard timeout err={math.degrees(abs_err):.1f}")
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0.0 else -1.0
        speed = clamp(abs_err * SIMPLE_SWEEP_ALIGN_KP, SIMPLE_SWEEP_ALIGN_MIN_SPEED, SIMPLE_SWEEP_ALIGN_MAX_SPEED)
        return -sign * speed, sign * speed

    if nav_state == NAV_SETTLE and simple_sweep_state == "TURN_IN_PLACE_90":
        if now < turn_settle_until:
            hard_stop_motors()
            coverage_status = "exploreSweep settle after 90"
            return 0.0, 0.0
        hard_stop_motors()
        simple_sweep_run_next_action()
        return 0.0, 0.0

    if nav_state == NAV_SETTLE and simple_sweep_state == "ALIGN_SETTLE":
        if now < turn_settle_until:
            hard_stop_motors()
            coverage_status = "exploreSweep settle after align"
            return 0.0, 0.0
        hard_stop_motors()
        simple_sweep_run_next_action()
        return 0.0, 0.0

    if nav_state == NAV_LANE_SHIFT and simple_sweep_state == "LANE_SHIFT_STRAIGHT":
        shifted = math.hypot(pose_x - lane_shift_start_x, pose_y - lane_shift_start_y)
        lat_now = simple_sweep_lateral_coord()
        lateral_delta = abs(lat_now - simple_sweep_shift_start_lateral)
        shift_side = 1.0 if simple_sweep_active_shift_global_side >= 0.0 else -1.0
        signed_lateral_progress = (lat_now - simple_sweep_shift_start_lateral) * shift_side
        target_remaining = (simple_sweep_shift_target_lateral - lat_now) * shift_side
        reached_target = target_remaining <= SIMPLE_SWEEP_LANE_TARGET_TOL_M
        overshot_guard = shifted >= lane_shift_target_dist + 0.14
        if signed_lateral_progress < -SIMPLE_SWEEP_SHIFT_WRONG_WAY_M:
            simple_sweep_handle_shift_block(
                f"lane shift moving wrong way progress={signed_lateral_progress:.2f} targetRem={target_remaining:.2f}"
            )
            return 0.0, 0.0
        if reached_target:
            simple_sweep_commit_pending_lane_target()
            risk, risk_reason = simple_sweep_new_lane_revisit_risk()
            if risk and simple_sweep_shift_retry_count < SIMPLE_SWEEP_MAX_EXTRA_SHIFT_RETRIES:
                simple_sweep_shift_retry_count += 1
                simple_sweep_queue.insert(0, ("SHIFT", simple_sweep_shift_attempt_side, SIMPLE_SWEEP_LANE_SHIFT_M * 0.75))
                coverage_status = f"exploreSweep extra lane shift {simple_sweep_shift_retry_count}/{SIMPLE_SWEEP_MAX_EXTRA_SHIFT_RETRIES}: {risk_reason}"
                simple_sweep_run_next_action()
                return 0.0, 0.0
            simple_sweep_run_next_action()
            return 0.0, 0.0
        if overshot_guard:
            if simple_sweep_shift_retry_count < SIMPLE_SWEEP_MAX_EXTRA_SHIFT_RETRIES:
                simple_sweep_shift_retry_count += 1
                simple_sweep_queue.insert(0, ("SHIFT", simple_sweep_shift_attempt_side, SIMPLE_SWEEP_LANE_SHIFT_M * 0.75))
                coverage_status = (
                    f"exploreSweep continue lane shift toward remembered target "
                    f"{simple_sweep_shift_retry_count}/{SIMPLE_SWEEP_MAX_EXTRA_SHIFT_RETRIES}: "
                    f"lat={lat_now:.2f} target={simple_sweep_shift_target_lateral:.2f} rem={target_remaining:.2f}"
                )
                simple_sweep_run_next_action()
                return 0.0, 0.0
            simple_sweep_handle_shift_block(
                f"lane target not reached after shift path={shifted:.2f} rem={target_remaining:.2f}"
            )
            return 0.0, 0.0
        if now - lane_shift_start_time > SIMPLE_SWEEP_LANE_SHIFT_TIMEOUT_SEC:
            simple_sweep_handle_shift_block(
                f"side shift timeout after {shifted:.2f}m lat={lateral_delta:.2f}m targetRem={target_remaining:.2f}m"
            )
            return 0.0, 0.0
        # IMU/odometry heading rail only.  It does not look at walls/depth and
        # cannot choose a turn; it just prevents caster/wheel asymmetry from
        # converting a 0.5 m lane shift into a long wall-parallel perimeter run.
        heading_err = normalize_angle(lane_shift_heading_target - pose_theta)
        coverage_status = (
            f"exploreSweep lane shift {'L' if simple_sweep_shift_attempt_side > 0 else 'R'} "
            f"global={'L' if shift_side > 0 else 'R'} lat={lat_now:.2f}->{simple_sweep_shift_target_lateral:.2f} "
            f"rem={target_remaining:.2f} path={shifted:.2f}/{lane_shift_target_dist:.2f} herr={math.degrees(heading_err):.1f}"
        )
        if abs(heading_err) <= math.radians(0.9):
            return SIMPLE_SWEEP_LANE_SHIFT_SPEED, SIMPLE_SWEEP_LANE_SHIFT_SPEED
        return heading_locked_wheel_speeds_to(
            SIMPLE_SWEEP_LANE_SHIFT_SPEED,
            lane_shift_heading_target,
            SIMPLE_SWEEP_SHIFT_HEADING_KP,
            SIMPLE_SWEEP_SHIFT_MAX_CORRECTION,
        )

    if nav_state == NAV_FORWARD and simple_sweep_state == "ESCAPE_FORWARD":
        moved = math.hypot(pose_x - simple_sweep_escape_start_x, pose_y - simple_sweep_escape_start_y)
        if moved >= SIMPLE_SWEEP_ESCAPE_FORWARD_M or now >= simple_sweep_escape_until:
            simple_sweep_after_escape_reset()
            return 0.0, 0.0
        coverage_status = f"exploreSweep straight escape {moved:.2f}/{SIMPLE_SWEEP_ESCAPE_FORWARD_M:.2f}m"
        return SIMPLE_SWEEP_SLOW_SPEED, SIMPLE_SWEEP_SLOW_SPEED

    if nav_state != NAV_FORWARD or simple_sweep_state not in ("MOVE_STRAIGHT", "INIT"):
        # Unknown old state leaked in.  Do not let it fall through to frontier/route;
        # reset the private sweep row instead.
        simple_sweep_start_forward(f"reacquire from nav={nav_state} state={simple_sweep_state}")
        return 0.0, 0.0

    row_end_candidate_count = 0
    row_dist = row_distance_from_start()
    desired_grid_heading = simple_sweep_desired_lane_heading()
    heading_err = simple_sweep_heading_error(desired_grid_heading)
    if (
        SIMPLE_SWEEP_ALIGN_ENABLED
        and row_dist >= SIMPLE_SWEEP_STRAIGHT_REALIGN_MIN_DIST_M
        and abs(heading_err) >= SIMPLE_SWEEP_STRAIGHT_REALIGN_RAD
        and now - simple_sweep_last_align_time >= SIMPLE_SWEEP_ALIGN_COOLDOWN_SEC
    ):
        simple_sweep_start_align(desired_grid_heading, f"mid-lane yaw drift {math.degrees(heading_err):.1f} deg")
        return 0.0, 0.0
    virtual_row_end = simple_sweep_virtual_row_end_reason(front, center, upper_front, body_clearance)
    if virtual_row_end is not None:
        simple_sweep_schedule_lane_change(simple_sweep_lane_shift_direction, virtual_row_end)
        return 0.0, 0.0

    # Depth is normally only a speed hint in the primary sweep.  It cannot
    # terminate a row unless the conservative virtual-contact guard above fires.
    range_front = min(float(front), float(center), float(upper_front))
    speed = SIMPLE_SWEEP_SPEED
    depth_mode = "cruise"
    if range_front < SIMPLE_SWEEP_PRECONTACT_FRONT_M or float(center) < SIMPLE_SWEEP_PRECONTACT_CENTER_M:
        speed = SIMPLE_SWEEP_SLOW_SPEED
        depth_mode = "precontact-slow"
    lane_lat = simple_sweep_lateral_coord()
    lane_err = simple_sweep_lane_target_lateral - lane_lat
    coverage_status = (
        f"exploreSweep MOVE_STRAIGHT lane={simple_sweep_lane_index}/{SIMPLE_SWEEP_MAX_LANES} "
        f"row={row_dist:.2f}m lat={lane_lat:.2f} tgt={simple_sweep_lane_target_lateral:.2f} err={lane_err:.2f} "
        f"{depth_mode} F/C/U={front:.2f}/{center:.2f}/{upper_front:.2f} "
        f"cov={last_coverage_percent:.1f}%/{SIMPLE_SWEEP_FINISH_COVERAGE_PERCENT:.0f}%"
    )
    acquire_control(SIMPLE_SWEEP_OWNER, SIMPLE_SWEEP_OWNER_MIN_SEC, SIMPLE_SWEEP_OWNER_MIN_DIST_M, coverage_status)
    return heading_locked_wheel_speeds_to(
        speed,
        desired_grid_heading,
        SIMPLE_SWEEP_STRAIGHT_HEADING_KP,
        SIMPLE_SWEEP_STRAIGHT_MAX_CORRECTION,
    )


def completed_dock_owner_speeds():
    global coverage_status
    if not dock_return_completed:
        return None
    coverage_status = dock_return_status
    acquire_control(ControlOwner.PLANNER, DOCK_STOP_HOLD_OWNER_SEC, 0.0, "docked stop")
    return 0.0, 0.0


def primary_sweep_owner_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right):
    return simple_sweep_fsm_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right)


def runtime_arena_boundary_guard_speeds(front, center, upper_front, left, right, body_clearance):
    """Prevent frontier/legacy row fallback from driving outside the room.

    This fixes the observed failure where a frontier route abort released wheel
    ownership, hard-core ROW_FORWARD continued straight, active scan started
    reporting outside-map, and wheel odometry ran to y=-50 m.  The guard is
    deliberately before the hard-core forward fallback and after contact safety.
    """
    global runtime_arena_guard_until, runtime_arena_guard_debug, coverage_status
    global frontier_route_abort_hold_until, frontier_route_abort_hold_debug, nav_action_queue, row_end_candidate_count
    if not (RUNTIME_ARENA_GUARD_ENABLED and DEBUG_ARENA_BOUNDS_ENABLED):
        runtime_arena_guard_debug = "arenaGuard=off"
        return None
    now = robot.getTime()
    if dock_return_active or known_map_coverage_eval_active():
        runtime_arena_guard_debug = "arenaGuard=skip dock/known"
        return None
    # If odometry was clamped, force an immediate in-room realign before any
    # fallback can request another forward command.
    clamped_recent = now < odom_arena_guard_until
    outside_now, outside_reason, _px, _py = pose_relative_to_debug_arena(margin_m=-0.02)
    probe_out, probe_reason, _fx, _fy = pose_relative_to_debug_arena(
        probe_heading=pose_theta,
        probe_m=float(RUNTIME_ARENA_FRONT_PROBE_M),
        margin_m=float(RUNTIME_ARENA_GUARD_MARGIN_M),
    )
    if not (clamped_recent or outside_now or (nav_state == NAV_FORWARD and probe_out)):
        runtime_arena_guard_debug = "arenaGuard=open"
        return None

    reason = outside_reason if outside_now else (probe_reason if probe_out else odom_arena_clamp_debug)
    runtime_arena_guard_until = max(runtime_arena_guard_until, now + float(RUNTIME_ARENA_GUARD_HOLD_SEC))
    frontier_route_abort_hold_until = max(frontier_route_abort_hold_until, now + float(RUNTIME_ARENA_REPLAN_HOLD_SEC))
    frontier_route_abort_hold_debug = f"routeHold=arena {reason[:44]}"
    runtime_arena_guard_debug = f"arenaGuard=realign {reason[:54]}"
    coverage_status = runtime_arena_guard_debug
    row_end_candidate_count = 0
    nav_action_queue = []
    if route_commit_active:
        abort_route_commit("arena boundary guard: " + reason)
    if nav_state != NAV_GRID_REALIGN:
        start_grid_realign(heading_to_arena_center_from_pose(), "arena boundary guard: " + reason, "frontier resume after arena guard")
    return 0.0, 0.0


def exploration_route_abort_hold_speeds():
    """Short no-owner hold after frontier route abort, so hard-core row-forward
    cannot immediately reuse the old heading while the planner is rebuilding.
    """
    global frontier_route_abort_hold_debug, coverage_status
    if planner_intent != PLANNER_INTENT_EXPAND_MAP or dock_return_active or route_commit_active:
        return None
    now = robot.getTime()
    if now >= frontier_route_abort_hold_until:
        frontier_route_abort_hold_debug = "routeHold=idle"
        return None
    # Allow explicit recovery/grid actions to finish; the hold is only for the
    # dangerous ordinary forward fallback.
    if nav_state != NAV_FORWARD:
        return None
    coverage_status = frontier_route_abort_hold_debug
    return 0.0, 0.0

def route_and_perception_owner_speeds(front, center, upper_front, left, right, body_clearance):
    """Dispatch non-safety owners after recovery guards have had priority."""
    global dock_return_status, coverage_status

    snapshot_cmd = post_turn_rgbd_snapshot_speeds(front, body_clearance)
    if snapshot_cmd is not None:
        return snapshot_cmd

    scan_cmd = active_rgbd_scan_speeds(front, center, upper_front, left, right, body_clearance)
    if scan_cmd is not None:
        return scan_cmd

    if not route_commit_active:
        map_dock_handoff_started = False
        if not auto_map_cleaning_started and not auto_map_return_to_dock_active:
            map_ready, map_ready_reason = learned_map_ready_to_clean()
            if map_ready:
                map_dock_handoff_started = bool(start_map_complete_return_to_dock(map_ready_reason))
        if not map_dock_handoff_started:
            go_home, home_reason = return_home_should_start()
            if go_home:
                start_return_to_dock(home_reason)
            else:
                dock_return_status = "idle: " + home_reason[:50]

    route_commit_cmd = route_commit_speeds(front, center, upper_front, left, right, body_clearance)
    if route_commit_cmd is not None:
        return route_commit_cmd

    scan_ok, scan_reason = active_rgbd_scan_need(front, body_clearance)
    if scan_ok:
        start_active_rgbd_scan(scan_reason)
        return 0.0, 0.0

    if exploration_frontier_only_owner_gate(front, center, body_clearance, left, right):
        return 0.0, 0.0

    if known_map_coverage_eval_active() and KNOWN_MAP_EVAL_STOP_IF_NO_ROUTE and nav_state == NAV_FORWARD:
        coverage_status = f"known-map wait route: {last_route_commit_debug[:54]}"
        try:
            if str(control_lock.owner) in (ControlOwner.ROW_FORWARD.value, ControlOwner.PLANNER.value):
                release_control("known-map wait route")
        except Exception:
            pass
        return 0.0, 0.0

    return None


def read_navigation_depth_context(depth):
    global last_floor_front_ignore
    left, center, right = depth_sectors(depth)
    front = depth_front_narrow(depth)
    upper_front = depth_front_upper_corridor(depth)
    body_clearance = depth_body_corridor_clearance(depth)
    body_corridor_blocked = body_clearance < BODY_CORRIDOR_PASS_CLEARANCE
    last_floor_front_ignore = False

    cv_front_is_probably_floor_shadow = (
        (last_cv_floor_rejected > 0 or last_cv_shadow_rejected > 0)
        and upper_front > FLOOR_FALSE_BLOCK_UPPER_OPEN
        and center > FLOOR_FALSE_BLOCK_CENTER_OPEN
    )
    if last_cv_front_obstacle < CV_NAV_BLOCK_DISTANCE and not cv_front_is_probably_floor_shadow:
        front = min(front, last_cv_front_obstacle)
        center = min(center, last_cv_front_obstacle)
    if last_cv_left_obstacle < CV_NAV_SIDE_DISTANCE:
        left = min(left, last_cv_left_obstacle)
    if last_cv_right_obstacle < CV_NAV_SIDE_DISTANCE:
        right = min(right, last_cv_right_obstacle)

    return left, center, right, front, upper_front, body_clearance, body_corridor_blocked, cv_front_is_probably_floor_shadow


def refresh_under_furniture_runtime(now):
    global under_furniture_active, last_parallel_aperture_count
    under_furniture_active = (now < under_furniture_until) and (now >= under_furniture_suppressed_until)
    if under_furniture_active or now - last_under_furniture_time <= 1.2:
        return
    if math.hypot(pose_x - last_parallel_aperture_x, pose_y - last_parallel_aperture_y) > PASSAGE_MULTILANE_RESET_DISTANCE_M:
        last_parallel_aperture_count = 0


def filter_bumper_floor_contact(now, front, center, upper_front, bumper_left, bumper_right, cv_front_is_probably_floor_shadow):
    global last_bumper_ignored_as_floor, coverage_status
    last_bumper_ignored_as_floor = False
    if not (bumper_left or bumper_right):
        return bumper_left, bumper_right
    single_side_bump = bool((bumper_left ^ bumper_right) and not last_bumper_center)
    corridor_clear = (
        front > BUMPER_VALID_FRONT_DISTANCE
        and center > BUMPER_VALID_CENTER_DISTANCE
        and upper_front > BUMPER_VALID_FRONT_DISTANCE
        and last_cv_front_obstacle > BUMPER_VALID_FRONT_DISTANCE
    )
    rug_ignore_super_clear = (
        front > 1.05
        and center > 1.05
        and upper_front > 1.05
        and last_cv_front_obstacle > 1.05
    )
    explicit_floor_cv = bool((last_cv_floor_rejected > 0 or last_cv_shadow_rejected > 0) and cv_front_is_probably_floor_shadow)
    raw_contact_age = now - last_contact_latch_time if last_contact_latch_time > 0.0 else 999.0
    rug_ignore_still_safe = (
        raw_contact_age <= BUMPER_RUG_IGNORE_MAX_HOLD_SEC
        and last_contact_map_cells <= BUMPER_RUG_IGNORE_MAX_CONTACT_CELLS
    )
    if single_side_bump and (not under_furniture_active) and corridor_clear and rug_ignore_super_clear and explicit_floor_cv and rug_ignore_still_safe:
        last_bumper_ignored_as_floor = True
        coverage_status = f"short rug-bump grace age={raw_contact_age:.2f}s F={front:.2f} C={center:.2f}"
        return False, False
    return bumper_left, bumper_right


def update_bumper_safety_event_debug():
    global last_safety_event_desc
    safety_event = bumper_event_from_raw(last_bumper_left, last_bumper_center, last_bumper_right, last_bumper_ignored_as_floor)
    last_safety_event_desc = safety_event.description
    return safety_event


def physical_bumper_contact_speeds(now, front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right):
    global last_optional_block_reason
    contact_states = (
        NAV_TURN_90,
        NAV_SETTLE,
        NAV_RECOVERY_BACKUP,
        NAV_LEG_ESCAPE_BACKUP,
        NAV_LEG_ESCAPE_TURN,
        NAV_LEG_ESCAPE_FORWARD,
        NAV_LEG_PASS_TURN,
        NAV_LEG_PASS_FORWARD,
        NAV_LEG_PASS_ALIGN,
        NAV_PRE_PIVOT_BACKUP,
    ) + CONTACT_RECOVERY_STATES
    if nav_state in contact_states or not (bumper_left or bumper_right):
        return None
    if nav_state == NAV_GRID_REALIGN:
        hard_stop_motors()
    obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
    side = choose_contact_escape_side(bumper_left, bumper_right, left, right, obstacle_side)
    mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))

    if EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active():
        map_side, map_reason = choose_explore_contact_turn_side(bumper_left, bumper_right, left, right)
        center_or_both_map = bool(last_bumper_center or (bumper_left and bumper_right))
        start_contact_recovery(
            map_side,
            "front" if center_or_both_map else "side",
            f"map-build bumper row-end: {map_reason}; L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
            EXPLORE_CONTACT_BACKUP_GOAL_M,
            EXPLORE_CONTACT_BACKUP_TIMEOUT_SEC,
            PIVOT_TURN_ANGLE,
            EXPLORE_CONTACT_FORWARD_VERIFY_M,
            CONTACT_ESCAPE_VERIFY_SPEED,
            FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
            False,
        )
        last_optional_block_reason = "perimeter contact -> deterministic 90"
        return 0.0, 0.0

    center_or_both = bool(last_bumper_center or (bumper_left and bumper_right))
    single_side_bumper = bool((bumper_left ^ bumper_right) and not center_or_both)
    if single_side_bumper and passable_side_contact_gap(front, center, upper_front, left, right, body_clearance):
        start_gap_contact_nudge(
            side,
            f"side bumper at passable gap L={int(bumper_left)} R={int(bumper_right)} F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}",
        )
        return 0.0, 0.0

    strong_contact = bool(
        center_or_both
        or under_furniture_active
        or last_body_corridor_clearance < BODY_CORRIDOR_PASS_CLEARANCE
        or abs(last_body_corridor_lateral) > 0.055
        or (front > LEG_ESCAPE_FRONT_OPEN_DISTANCE and center > SAFE_FRONT_DISTANCE)
    )
    urgent_contact = bool(center_or_both or under_furniture_active or strong_contact)
    if urgent_contact or now - last_leg_escape_time > LEG_ESCAPE_COOLDOWN_SEC:
        start_leg_escape(
            side,
            f"bumper contact L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)} F={front:.2f} Cdist={center:.2f} side={'L' if side > 0 else 'R'}",
            strong=strong_contact,
        )
    else:
        start_leg_escape(side, f"repeat bumper contact L={int(bumper_left)} R={int(bumper_right)} F={front:.2f} C={center:.2f}", strong=True)
    return 0.0, 0.0


def contact_safety_stage_speeds(now, front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right, cv_front_is_probably_floor_shadow):
    bumper_left, bumper_right = filter_bumper_floor_contact(
        now,
        front,
        center,
        upper_front,
        bumper_left,
        bumper_right,
        cv_front_is_probably_floor_shadow,
    )
    update_bumper_safety_event_debug()

    contact_cmd = physical_bumper_contact_speeds(now, front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right)
    if contact_cmd is not None:
        return contact_cmd, bumper_left, bumper_right

    if (
        RETURN_HOME_ENABLED
        and (not dock_return_completed)
        and last_coverage_percent >= DOCK_STUCK_RETURN_COVERAGE_PERCENT
        and nav_state in CONTACT_RECOVERY_STATES
        and (now - contact_recovery_start_time) >= DOCK_STUCK_RECOVERY_SEC
        and not (bumper_left or bumper_right or last_bumper_center)
    ):
        start_forward_row("return dock after late contact recovery")
        start_return_to_dock(f"late contact recovery bailout {now - contact_recovery_start_time:.1f}s")
        return (0.0, 0.0), bumper_left, bumper_right

    contact_recovery_cmd = contact_recovery_manager_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right)
    if contact_recovery_cmd is not None:
        return contact_recovery_cmd, bumper_left, bumper_right
    return None, bumper_left, bumper_right


def choose_motion_from_depth(depth):
    global nav_state, turn_settle_until, nav_action_queue, lane_side, coverage_status, desired_grid_heading, grid_realign_until
    global last_row_change_time, row_end_candidate_count, last_bumper_ignored_as_floor, last_revisit_lane_change_time
    global pivot_watchdog_retry_count, turn_best_abs_error, turn_last_progress_time, grid_realign_retry_count
    global last_floor_front_ignore
    global last_stall_depth_signature, last_stall_rgb_signature, last_sensor_stall_time
    global under_furniture_until, under_furniture_active, last_under_furniture_time, under_furniture_suppressed_until
    global last_parallel_aperture_count, under_furniture_confirm_count, last_low_obstacle_guard_time
    global leg_escape_turn_target, leg_pass_target_heading, leg_pass_start_x, leg_pass_start_y, leg_pass_until, leg_pass_grace_until
    global last_contact_latch_used, last_contact_latch_time
    global last_safety_event_desc
    global dock_return_status

    refresh_navigation_phase()

    left, center, right, front, upper_front, body_clearance, body_corridor_blocked, cv_front_is_probably_floor_shadow = read_navigation_depth_context(depth)
    now = robot.getTime()
    refresh_under_furniture_runtime(now)
    bumper_left, bumper_right = read_bumpers()

    dock_cmd = completed_dock_owner_speeds()
    if dock_cmd is not None:
        return dock_cmd

    # placed before legacy bumper/frontier/route handling so the old EXPLORE/ROW
    # fallback cannot turn the sweep back into reactive obstacle-to-obstacle motion.
    simple_sweep_cmd = primary_sweep_owner_speeds(front, center, upper_front, left, right, body_clearance, bumper_left, bumper_right)
    if simple_sweep_cmd is not None:
        return simple_sweep_cmd

    contact_cmd, bumper_left, bumper_right = contact_safety_stage_speeds(
        now,
        front,
        center,
        upper_front,
        left,
        right,
        body_clearance,
        bumper_left,
        bumper_right,
        cv_front_is_probably_floor_shadow,
    )
    if contact_cmd is not None:
        return contact_cmd

    # Pre-contact low-object guard. A shelf edge or the red low obstacle can be
    # below the most reliable upper RGB-D band: the bumper may fire only after the
    # robot has already wedged itself. Treat a collapsing lower/center corridor as
    # a contact-like trap before continuing old coverage/under-furniture logic.
    if (
        LOW_OBSTACLE_GUARD_ENABLED
        and not (EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active())
        and nav_state == NAV_FORWARD
        and now - last_low_obstacle_guard_time > LOW_OBSTACLE_GUARD_COOLDOWN_SEC
        and not last_floor_front_ignore
        and not cv_front_is_probably_floor_shadow
        and not is_floor_or_shadow_false_front_block(front, center, upper_front)
        and not gap_mouth_candidate(front, center, upper_front, left, right, body_clearance)
    ):
        low_front_block = center < LOW_OBSTACLE_GUARD_CENTER_M and front < LOW_OBSTACLE_GUARD_FRONT_M
        # A side wall close to the circular shell is not a low-object trap.  Only
        # a central body hit should pre-emptively escape without bumper contact.
        central_shell_block = (
            body_clearance < LOW_OBSTACLE_GUARD_BODY_M
            and abs(last_body_corridor_lateral) <= BODY_ROW_END_LATERAL_TOL_M
            and upper_front > UNDER_FURNITURE_UPPER_OPEN
        )
        if low_front_block or central_shell_block:
            last_low_obstacle_guard_time = now
            obstacle_side = contact_obstacle_side_from_sensors(left, right)
            side = choose_contact_escape_side(False, False, left, right, obstacle_side)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(
                side,
                f"precontact low-object guard F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}",
                strong=True,
            )
            return 0.0, 0.0

    post_recovery_cmd = post_recovery_stabilizer_speeds(front, center, upper_front, left, right, body_clearance)
    if post_recovery_cmd is not None:
        return post_recovery_cmd

    route_owner_cmd = route_and_perception_owner_speeds(front, center, upper_front, left, right, body_clearance)
    if route_owner_cmd is not None:
        return route_owner_cmd

    arena_guard_cmd = runtime_arena_boundary_guard_speeds(front, center, upper_front, left, right, body_clearance)
    if arena_guard_cmd is not None:
        return arena_guard_cmd

    abort_hold_cmd = exploration_route_abort_hold_speeds()
    if abort_hold_cmd is not None:
        return abort_hold_cmd


    # Legacy fallback only. In the hard-core build, ordinary FORWARD should already
    # have been handled above.  Local alignment is left here only for experimental
    # rollback because it was a major source of diagonal-looking behavior.
    local_alignment_cmd = None
    if (not HARD_CORE_DISABLE_LOCAL_ALIGNMENT) and optional_planner_intercepts_allowed("local_alignment"):
        local_alignment_cmd = local_forward_alignment_speeds(front, center, upper_front, left, right, body_clearance)
    if local_alignment_cmd is not None:
        return local_alignment_cmd

    bumper_first_mapping = bool(EXPLORE_BUMPER_FIRST_MAPPING and matrix_first_explore_active())
    if (not bumper_first_mapping) and maybe_handle_under_surface_side_risk(front, center, left, right, body_clearance):
        return 0.0, 0.0

    # Odometry/sensor-stall detection is still allowed: it means the robot is
    # physically not moving or sensors are frozen, not that the map thinks an
    # obstacle is nearby.
    if maybe_handle_odometry_stall(front, center, left, right, body_clearance):
        return 0.0, 0.0

    if maybe_handle_sensor_stall(front, center, left, right, body_clearance):
        return 0.0, 0.0

    # Near-corner/grazing guard. The frontal depth camera can miss a side rub
    # until contact happens. If one front side is very close, recover before
    # the body starts sliding along the obstacle.
    if (not bumper_first_mapping) and nav_state == NAV_FORWARD and front < WALL_GRAZE_FRONT_LIMIT:
        # Only treat a side reading as a recovery case when it is extremely
        # close. Otherwise the robot used to abandon the row just because a side
        # wall/furniture edge appeared in the camera.
        if left < WALL_GRAZE_DISTANCE and right > left + 0.10:
            start_recovery_backup(-1.0, f"left graze L={left:.2f}")
            return 0.0, 0.0
        if right < WALL_GRAZE_DISTANCE and left > right + 0.10:
            start_recovery_backup(1.0, f"right graze R={right:.2f}")
            return 0.0, 0.0

    # Edge-trap guard for table/chair legs. The previous logic could drive along
    # a leg because the central RGB-D corridor was open. That is not a successful
    # under-chair pass; it is a side-shell contact trap. Back up to the previous
    # pose region and turn away more strongly, then resume coverage planning.
    obstacle_side = contact_obstacle_side_from_sensors(left, right)
    if (
        (not bumper_first_mapping)
        and nav_state == NAV_FORWARD
        and obstacle_side != 0.0
        and not last_floor_front_ignore
        and (under_furniture_active or front < 1.25 or min(left, right) < LEG_TRAP_EDGE_GRAZE_DISTANCE)
        and (upper_front > ROW_END_CONFIRM_DISTANCE + 0.06 or center > SAFE_FRONT_DISTANCE)
    ):
        if passable_side_contact_gap(front, center, upper_front, left, right, body_clearance):
            # A passable mouth with a side-shell warning should be handled by the
            # continuous squeeze controller, not by a reverse escape.  Keep moving
            # slowly through the opening and steer away from the near side.
            squeeze_cmd = narrow_passage_centering_speeds(front, center, upper_front, left, right, body_clearance)
            if squeeze_cmd is not None:
                return squeeze_cmd
            coverage_status = f"hold passable gap side-risk body={body_clearance:.2f} lat={last_body_corridor_lateral:.2f}"
            return heading_locked_wheel_speeds_to(NARROW_PASSAGE_SPEED * 0.55, pose_theta, 0.8, 0.12)
        if start_edge_trap_escape(
            obstacle_side,
            left,
            right,
            f"edge trap side={'L' if obstacle_side > 0 else 'R'} F={front:.2f} C={center:.2f} body={body_clearance:.2f} L={left:.2f} R={right:.2f}",
        ):
            return 0.0, 0.0

    hard_core_cmd = hard_core_controller_speeds(front, center, upper_front, left, right, body_clearance)
    if hard_core_cmd is not None:
        return hard_core_cmd

    # For the first seconds force a gentle forward motion. This prevents the
    # robot from getting stuck near the spawn wall because of a close side ray.
    if now < 2.0 and nav_state == NAV_FORWARD:
        coverage_status = "startup forward"
        return heading_locked_wheel_speeds(CRUISE_SPEED)


    if nav_state == NAV_GRID_REALIGN:
        global grid_realign_best_abs_error, grid_realign_last_progress_time
        if bumper_left or bumper_right or last_bumper_center:
            # Do not rotate while a physical front bumper is pressed. Route/grid
            # alignment loses ownership immediately to contact recovery on the next
            # decision cycle.
            hard_stop_motors()
            coverage_status = "grid realign aborted by bumper"
            return 0.0, 0.0
        heading_remaining = normalize_angle(grid_realign_target - pose_theta)
        abs_heading_remaining = abs(heading_remaining)
        if abs_heading_remaining <= GRID_REALIGN_TOLERANCE:
            finish_grid_realign()
            return 0.0, 0.0

        if abs_heading_remaining < grid_realign_best_abs_error - GRID_REALIGN_PROGRESS_EPS:
            grid_realign_best_abs_error = abs_heading_remaining
            grid_realign_last_progress_time = now

        stalled_small_error = (
            now - grid_realign_last_progress_time >= GRID_REALIGN_STALL_SEC
            and abs_heading_remaining <= GRID_REALIGN_SOFT_ACCEPT_ERR
        )
        timed_out_small_error = now >= grid_realign_until and abs_heading_remaining <= GRID_REALIGN_TIMEOUT_ACCEPT_ERR
        if stalled_small_error or timed_out_small_error:
            # Observed failure mode: err around 5-7 degrees, robot near a wall, and
            # GRID_REALIGN keeps owning the wheels. Finish using the current tangent
            # so FORWARD does not draw a correction arc.
            finish_grid_realign(accept_current_heading=True, reason=f"err={math.degrees(heading_remaining):.1f} stalled={int(stalled_small_error)}")
            return 0.0, 0.0

        if now >= grid_realign_until:
            # Hard-arbiter rule: GRID_REALIGN is an atomic micro-action, not a
            # state allowed to own the robot forever.  additionally prevents
            # the observed deadlock where the target had drifted to a non-cardinal
            # heading (for example -110 deg) and the robot kept stopping/retrying in
            # a corner.  Targets are cardinal; if a bounded retry still cannot
            # converge, back up and make a single exploration turn instead of
            # holding ownership forever.
            if abs_heading_remaining <= GRID_REALIGN_SOFT_ACCEPT_ERR:
                finish_grid_realign(accept_current_heading=False, reason=f"timeout finish err={math.degrees(heading_remaining):.1f}")
                return 0.0, 0.0
            if grid_realign_retry_count < GRID_REALIGN_MAX_RETRIES:
                grid_realign_retry_count += 1
                grid_realign_until = now + GRID_REALIGN_RETRY_SEC
                grid_realign_best_abs_error = abs_heading_remaining
                grid_realign_last_progress_time = now
                hard_stop_motors()
                coverage_status = f"grid realign retry {grid_realign_retry_count}/{GRID_REALIGN_MAX_RETRIES} err={math.degrees(heading_remaining):.1f}"
                acquire_control(ControlOwner.GRID_REALIGN, GRID_REALIGN_RETRY_SEC, 0.0, coverage_status)
                return 0.0, 0.0
            side = 1.0 if heading_remaining >= 0.0 else -1.0
            start_recovery_backup(side, f"grid realign stuck err={math.degrees(heading_remaining):.1f} target={math.degrees(grid_realign_target):.0f}")
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        speed = clamp(abs_heading_remaining * GRID_REALIGN_KP, GRID_REALIGN_MIN_SPEED, GRID_REALIGN_MAX_SPEED)
        coverage_status = f"grid realign in place err={math.degrees(heading_remaining):.1f}"
        return -sign * speed, sign * speed

    # Clearance backup before a pivot. This specifically fixes the loop caused
    # by starting a pivot while the front shell/contact pads are already near
    # furniture. After the small backup, resume the original pivot
    # and keep the existing lane-change queue.
    if nav_state == NAV_PRE_PIVOT_BACKUP:
        moved = math.hypot(pose_x - pre_pivot_start_x, pose_y - pre_pivot_start_y)
        remaining = max(0.0, PRE_PIVOT_BACKUP_DISTANCE - moved)
        if moved >= PRE_PIVOT_BACKUP_DISTANCE or now >= pre_pivot_until:
            begin_pivot_turn(pre_pivot_side, pre_pivot_reason)
            return 0.0, 0.0
        if reverse_backup_should_stop(remaining):
            coverage_status = f"pre-pivot rear guard {last_rear_guard_clearance:.2f}m {last_rear_guard_reason}"
            begin_pivot_turn(pre_pivot_side, pre_pivot_reason + "; rear guard")
            return 0.0, 0.0
        coverage_status = f"pre-pivot backup {moved:.2f}/{PRE_PIVOT_BACKUP_DISTANCE:.2f} rear={last_rear_guard_clearance:.2f}"
        return heading_locked_wheel_speeds(-PRE_PIVOT_BACKUP_SPEED)

    # Furniture-leg escape: local maneuver for thin chair/table legs. This is
    # deliberately separate from the lawnmower lane change. A chair leg is a
    # local contact trap; treating it as "end of row" makes the robot pivot
    # under the chair and stay stuck.
    if nav_state == NAV_LEG_ESCAPE_BACKUP:
        moved = math.hypot(pose_x - leg_escape_start_x, pose_y - leg_escape_start_y)
        remaining = max(0.0, leg_escape_backup_distance_current - moved)
        if moved >= leg_escape_backup_distance_current or now >= leg_escape_until:
            finish_leg_escape_backup()
            return 0.0, 0.0
        label = "wall edge release" if last_contact_escape_kind == "wall" else ("trap escape" if leg_escape_replan_after else "leg escape")
        if reverse_backup_should_stop(remaining):
            # Do not keep reversing into a known rear wall.  Finish the backup
            # phase early and rotate in place away from the original contact.
            coverage_status = f"{label} rear guard {last_rear_guard_clearance:.2f}m {last_rear_guard_reason}"
            finish_leg_escape_backup()
            return 0.0, 0.0
        coverage_status = f"{label} straight backup {moved:.2f}/{leg_escape_backup_distance_current:.2f} rear={last_rear_guard_clearance:.2f}"
        if LEG_ESCAPE_STRAIGHT_BACKUP:
            # Equal wheel speeds: undo the last contact path without drawing a
            # diagonal/curved pre-turn segment.  Wall contacts use a slower,
            # shorter release so the robot does not abandon the edge strip.
            backup_speed = WALL_CONTACT_FORWARD_SPEED * 0.72 if last_contact_escape_kind == "wall" else LEG_ESCAPE_BACKUP_SPEED
            return -backup_speed, -backup_speed
        base = -LEG_ESCAPE_BACKUP_SPEED
        yaw = leg_escape_side * LEG_ESCAPE_REVERSE_TURN
        return base - yaw, base + yaw

    if nav_state == NAV_LEG_ESCAPE_TURN:
        # If a bumper is still active, turning is forbidden.  The older code
        # immediately restarted a stronger escape from here, which could log
        # "escape-turn contact L=1 C=1 R=1" and choose another rotation while the
        # bumper was pressed.  Hand it to the contact RecoveryManager instead.
        if bumper_left or bumper_right:
            side = -1.0 if bumper_left and not bumper_right else (1.0 if bumper_right and not bumper_left else leg_escape_side)
            obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_contact_recovery(
                side,
                "front" if last_bumper_center or (bumper_left and bumper_right) else "side",
                f"escape-turn contact L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
                FRONT_CONTACT_TRAP_BACKUP_DISTANCE,
                FRONT_CONTACT_TRAP_BACKUP_TIMEOUT_SEC,
                FRONT_CONTACT_TRAP_TURN_ANGLE,
                FRONT_CONTACT_TRAP_FORWARD_DISTANCE,
                CONTACT_ESCAPE_VERIFY_SPEED,
                FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
                True,
            )
            return 0.0, 0.0
        heading_remaining = normalize_angle(leg_escape_turn_target - pose_theta)
        if abs(heading_remaining) <= LEG_ESCAPE_TURN_TOLERANCE:
            finish_leg_escape_turn()
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        speed = clamp(abs(heading_remaining) * LEG_ESCAPE_TURN_KP, 0.28, LEG_ESCAPE_TURN_SPEED)
        coverage_status = f"leg escape turn err={math.degrees(heading_remaining):.1f}"
        return -sign * speed, sign * speed

    if nav_state == NAV_LEG_ESCAPE_FORWARD:
        # A new bumper hit during the escape-out segment means the selected
        # heading is not passable. Back up again and choose a stronger angle
        # instead of driving forward through the contact.
        if bumper_left or bumper_right:
            side = -1.0 if bumper_left and not bumper_right else (1.0 if bumper_right and not bumper_left else leg_escape_side)
            obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_contact_recovery(
                side,
                "front" if last_bumper_center or (bumper_left and bumper_right) else "side",
                f"escape-forward contact L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}",
                FRONT_CONTACT_TRAP_BACKUP_DISTANCE,
                FRONT_CONTACT_TRAP_BACKUP_TIMEOUT_SEC,
                FRONT_CONTACT_TRAP_TURN_ANGLE,
                FRONT_CONTACT_TRAP_FORWARD_DISTANCE,
                CONTACT_ESCAPE_VERIFY_SPEED,
                FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC,
                True,
            )
            return 0.0, 0.0
        if (center < LOW_OBSTACLE_GUARD_CENTER_M and front < LOW_OBSTACLE_GUARD_FRONT_M) or body_clearance < LOW_OBSTACLE_GUARD_BODY_M:
            side = choose_contact_escape_side(False, False, left, right, contact_obstacle_side_from_sensors(left, right))
            mark_contact_obstacle(0.0, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(side, f"escape-forward low block F={front:.2f} C={center:.2f} body={body_clearance:.2f}", strong=True)
            return 0.0, 0.0
        moved = math.hypot(pose_x - leg_escape_forward_start_x, pose_y - leg_escape_forward_start_y)
        if moved >= leg_escape_forward_distance_current or now >= leg_escape_forward_until:
            finish_leg_escape_forward()
            return 0.0, 0.0
        label = "wall edge follow" if last_contact_escape_kind == "wall" else ("trap escape" if leg_escape_replan_after else "leg escape")
        coverage_status = f"{label} forward {moved:.2f}/{leg_escape_forward_distance_current:.2f}"
        # Use heading lock to the temporary escape heading.
        return heading_locked_wheel_speeds(leg_escape_forward_speed_current)

    if nav_state == NAV_LEG_PASS_TURN:
        heading_remaining = normalize_angle(leg_pass_target_heading - pose_theta)
        if abs(heading_remaining) <= LEG_PASS_TURN_TOLERANCE:
            finish_leg_pass_turn()
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        speed = clamp(abs(heading_remaining) * LEG_PASS_TURN_KP, 0.22, LEG_PASS_TURN_SPEED)
        coverage_status = f"leg pass turn err={math.degrees(heading_remaining):.1f}"
        return -sign * speed, sign * speed

    if nav_state == NAV_LEG_PASS_FORWARD:
        moved = math.hypot(pose_x - leg_pass_start_x, pose_y - leg_pass_start_y)
        if moved >= LEG_PASS_FORWARD_DISTANCE or now >= leg_pass_until:
            finish_leg_pass_forward()
            return 0.0, 0.0
        # Commit to the furniture-leg bypass. A chair leg directly in front will
        # keep the narrow depth value low, so aborting on ordinary row-end depth
        # makes the robot "get scared" and start a full 90-degree lane turn.
        # Abort only on real physical contact or an extremely close frontal hit.
        if bumper_left or bumper_right:
            side = -1.0 if bumper_left and not bumper_right else (1.0 if bumper_right and not bumper_left else leg_pass_side)
            obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(side, f"leg pass contact L={int(bumper_left)} R={int(bumper_right)} F={front:.2f}", strong=True)
            return 0.0, 0.0
        squeeze_cmd = narrow_passage_centering_speeds(front, center, upper_front, left, right, body_clearance)
        if squeeze_cmd is not None:
            return squeeze_cmd

        obstacle_side = contact_obstacle_side_from_sensors(left, right)
        if obstacle_side != 0.0 and start_edge_trap_escape(
            obstacle_side,
            left,
            right,
            f"leg pass edge trap side={'L' if obstacle_side > 0 else 'R'} F={front:.2f} body={body_clearance:.2f}",
        ):
            return 0.0, 0.0
        if front < LEG_PASS_FRONT_ABORT_DISTANCE and center < 0.20:
            start_recovery_backup(leg_pass_side, f"leg pass blocked F={front:.2f} C={center:.2f}")
            return 0.0, 0.0
        coverage_status = f"leg pass COMMIT {moved:.2f}/{LEG_PASS_FORWARD_DISTANCE:.2f} F={front:.2f}"
        return heading_locked_wheel_speeds(LEG_PASS_FORWARD_SPEED)

    if nav_state == NAV_LEG_PASS_ALIGN:
        heading_remaining = normalize_angle(desired_grid_heading - pose_theta)
        if abs(heading_remaining) <= LEG_PASS_TURN_TOLERANCE:
            finish_leg_pass_align()
            return 0.0, 0.0
        sign = 1.0 if heading_remaining > 0 else -1.0
        speed = clamp(abs(heading_remaining) * LEG_PASS_TURN_KP, 0.20, LEG_PASS_ALIGN_SPEED)
        coverage_status = f"leg pass align err={math.degrees(heading_remaining):.1f}"
        return -sign * speed, sign * speed

    # Recovery: back up a little, then perform a deterministic lane turn. This
    # breaks the corner loop where the robot alternated left/right in place.
    if nav_state == NAV_RECOVERY_BACKUP:
        moved = math.hypot(pose_x - backup_start_x, pose_y - backup_start_y)
        remaining = max(0.0, BACKUP_DISTANCE - moved)
        if moved >= BACKUP_DISTANCE or now >= backup_until:
            if matrix_first_explore_active():
                begin_explore_single_turn(backup_after_side, "after backup recovery; explore single turn")
            else:
                begin_lawnmower_lane_change(backup_after_side, "after backup recovery")
            return 0.0, 0.0
        if reverse_backup_should_stop(remaining):
            coverage_status = f"recovery rear guard {last_rear_guard_clearance:.2f}m {last_rear_guard_reason}"
            if matrix_first_explore_active():
                begin_explore_single_turn(backup_after_side, "after rear-guarded recovery; explore single turn")
            else:
                begin_lawnmower_lane_change(backup_after_side, "after rear-guarded recovery")
            return 0.0, 0.0
        coverage_status = f"recovery backup {moved:.2f}/{BACKUP_DISTANCE:.2f} rear={last_rear_guard_clearance:.2f}"
        return heading_locked_wheel_speeds(-BACKUP_SPEED)

    if nav_state == NAV_SETTLE:
        if now < turn_settle_until:
            coverage_status = "settle"
            hard_stop_motors()
            return 0.0, 0.0
        hard_stop_motors()
        if post_turn_rgbd_snapshot_pending and start_post_turn_rgbd_snapshot("after 90deg pivot"):
            return 0.0, 0.0
        run_next_nav_action(front, body_clearance)
        if nav_state != NAV_FORWARD:
            return 0.0, 0.0

    if nav_state == NAV_TURN_90:
        # IMU-based 90-degree pivot. Encoder-only pivot was the wrong assumption:
        # with wheel slip and the front skid, wheel rotation != body yaw.
        heading_remaining = normalize_angle(turn_target_theta - pose_theta)
        coverage_status = (
            f"pivot yaw err={math.degrees(heading_remaining):.1f}deg "
            f"target={math.degrees(turn_target_theta):.0f}"
        )

        abs_err = abs(heading_remaining)
        progress_step = math.radians(PIVOT_WATCHDOG_PROGRESS_DEG)
        if abs_err < turn_best_abs_error - progress_step:
            turn_best_abs_error = abs_err
            turn_last_progress_time = now

        if abs_err <= PIVOT_TURN_TOLERANCE:
            finish_pivot_turn()
            return 0.0, 0.0

        watchdog_err = math.radians(PIVOT_WATCHDOG_MIN_ERR_DEG)
        pivot_timed_out = (now - turn_start_time) > PIVOT_WATCHDOG_TIMEOUT_SEC and abs_err > watchdog_err
        pivot_stalled = (now - turn_last_progress_time) > PIVOT_WATCHDOG_STALL_SEC and abs_err > watchdog_err
        if pivot_timed_out or pivot_stalled or bumper_left or bumper_right:
            hard_stop_motors()
            reason = "contact" if (bumper_left or bumper_right) else ("timeout" if pivot_timed_out else "stall")
            if pivot_watchdog_retry_count < PIVOT_WATCHDOG_MAX_RETRIES:
                pivot_watchdog_retry_count += 1
                start_pre_pivot_backup(turn_direction, f"pivot watchdog {reason} err={math.degrees(abs_err):.1f}; {last_turn_reason}")
            else:
                # Give up this exact lane-change sequence. Continuing to pursue
                # the same 90-degree pivot is what draws the visible loop. Resume
                # from the nearest grid heading and let the coverage objective
                # choose a new target on the next frames.
                pivot_watchdog_retry_count = 0
                nav_action_queue = []
                desired_grid_heading = snap_to_right_angle(pose_theta)
                last_revisit_lane_change_time = now
                start_forward_row(f"pivot watchdog escape {reason} err={math.degrees(abs_err):.1f}")
            return 0.0, 0.0

        # Direction is determined by the actual remaining yaw, not by stale state.
        sign = 1.0 if heading_remaining > 0 else -1.0

        # the old controller often stopped around 80-85 degrees when the
        # robot was close to a wall.  The cause was not the target calculation:
        # logs showed target=90 but the final 3-10 degrees were commanded with
        # only ~0.18 wheel speed, which is too weak for the Webots contact/skid
        # friction.  Use a stronger bounded fine-pivot zone, but keep the same
        # IMU sign feedback so overshoot is corrected instead of accumulating.
        if abs_err <= PIVOT_FINAL_ZONE:
            speed = clamp(abs(heading_remaining) * PIVOT_TURN_KP, PIVOT_FINAL_MIN_SPEED, PIVOT_FINAL_MAX_SPEED)
            coverage_status += f" finalSpeed={speed:.2f}"
        elif abs_err <= PIVOT_FINE_ZONE:
            speed = clamp(abs(heading_remaining) * PIVOT_TURN_KP, PIVOT_FINE_MIN_SPEED, PIVOT_FINE_MAX_SPEED)
            coverage_status += f" fineSpeed={speed:.2f}"
        else:
            speed = clamp(abs(heading_remaining) * PIVOT_TURN_KP, PIVOT_TURN_MIN_SPEED, PIVOT_TURN_SPEED)
            coverage_status += f" speed={speed:.2f}"
        return -sign * speed, sign * speed

    if nav_state == NAV_LANE_SHIFT:
        shifted = math.hypot(pose_x - lane_shift_start_x, pose_y - lane_shift_start_y)
        # If the side shift is blocked, do not start choosing random turns. Turn
        # into the next row anyway; if the row is also blocked, recovery handles it.
        if shifted >= lane_shift_target_dist or center < SAFE_FRONT_DISTANCE * 0.82 or now - lane_shift_start_time > LANE_SHIFT_TIMEOUT_SEC:
            run_next_nav_action(front, body_clearance)
            return 0.0, 0.0
        heading_err = normalize_angle(lane_shift_heading_target - pose_theta)
        heading_err_deg = math.degrees(heading_err)
        coverage_status = f"lane shift rail {shifted:.2f}/{lane_shift_target_dist:.2f} herr={heading_err_deg:.1f}"
        if abs(heading_err) <= LANE_SHIFT_HEADING_DEADBAND:
            return LANE_SHIFT_SPEED, LANE_SHIFT_SPEED
        return heading_locked_wheel_speeds_to(LANE_SHIFT_SPEED, lane_shift_heading_target, LANE_SHIFT_HEADING_KP, LANE_SHIFT_MAX_CORRECTION)

    # Before any anti-loop or row-end logic, suppress false floor-level blocks.
    # Carpet lips and shadows may make the lower/narrow depth value close, but
    # if the upper corridor is open and OpenCV classifies the visual feature as
    # floor/shadow, the row must continue instead of starting a new lane.
    if nav_state == NAV_FORWARD and is_floor_or_shadow_false_front_block(front, center, upper_front):
        last_floor_front_ignore = True
        row_end_candidate_count = 0
        front = max(front, ROW_END_CONFIRM_DISTANCE + 0.12)
        # Do not let the body-envelope check turn a carpet/shadow strip into a
        # fake furniture collision. The envelope uses middle rows, but Webots
        # depth can still produce low floor artifacts on sharp rug edges.
        if body_clearance > BODY_CORRIDOR_HARD_CLEARANCE:
            body_corridor_blocked = False

    # If the body envelope sees a near obstacle in the swept footprint, clamp the
    # front only when that hit is a real front/centre row-end candidate.  A side
    # shell warning is handled by leg/side-risk logic, not by a full 90-degree
    # pivot. This is the main fix for the visible left-right turning on the spot.
    body_row_hard, body_row_soft = body_row_end_flags(front, center, upper_front, body_clearance)
    if nav_state == NAV_FORWARD and (body_row_hard or body_row_soft):
        front = min(front, body_clearance)
        center = min(center, body_clearance)

    # A completed pivot->shift->pivot sequence must be followed by actual forward
    # travel.  Otherwise stale target/under-furniture/soft row-end signals can
    # schedule another pivot immediately, making the robot spin around one cell.
    if post_lane_forward_lock_active(front, center, body_clearance):
        row_end_candidate_count = 0
        moved, remain_t = post_lane_forward_lock_progress()
        coverage_status = f"post-lane forward lock {moved:.2f}/{POST_LANE_FORWARD_LOCK_DISTANCE_M:.2f}m {remain_t:.1f}s"
        return heading_locked_wheel_speeds(POST_LANE_FORWARD_LOCK_SPEED)

    if post_gap_commit_active(front, center, body_clearance):
        row_end_candidate_count = 0
        moved = math.hypot(pose_x - post_gap_commit_start_x, pose_y - post_gap_commit_start_y)
        coverage_status = f"post-gap commit {moved:.2f}/{POST_GAP_COMMIT_DISTANCE_M:.2f}m"
        return heading_locked_wheel_speeds_to(POST_GAP_COMMIT_SPEED, desired_grid_heading, HEADING_LOCK_KP, max(POST_RECOVERY_MAX_CORRECTION, 0.075))

    # Straight-line guard.  Ordinary coverage rows and post-recovery rows must
    # not bend back to the grid while translating.  If the heading error is
    # large, rotate in place first; heading lock is only for tiny IMU drift.
    if (
        nav_state == NAV_FORWARD
        and not under_furniture_active
        and not (last_contact_escape_kind == "gap" and now < post_gap_commit_until)
        and now - last_contact_trap_time > FORWARD_STRAIGHT_RECOVERY_GRACE_SEC
        and abs(normalize_angle(desired_grid_heading - pose_theta)) > max(FORWARD_STRAIGHT_REALIGN_ERR, GRID_REALIGN_REQUEST_ERR)
        and front > FORWARD_STRAIGHT_REALIGN_MIN_FRONT_M
        and center > FORWARD_STRAIGHT_REALIGN_MIN_FRONT_M
        and not (last_bumper_left or last_bumper_center or last_bumper_right)
    ):
        start_grid_realign(desired_grid_heading, "straight guard before forward", "row forward after straight guard")
        row_end_candidate_count = 0
        return 0.0, 0.0

    owner_guard_cmd = forward_owner_guard_speeds(front, center, body_clearance)
    if owner_guard_cmd is not None:
        row_end_candidate_count = 0
        coverage_status = f"owned straight row: {last_control_owner_debug}"
        return owner_guard_cmd

    # Local coverage efficiency: before committing to a long straight pass,
    # check whether a nearby side corner/pocket is cheaper to fill now. This is
    # exactly the case visible on the coverage map: the robot can keep driving
    # under/along furniture while a small reachable uncleaned corner remains
    # right next to the body.
    if maybe_start_local_pocket_fill(left, right, front, body_clearance):
        row_end_candidate_count = 0
        return 0.0, 0.0

    # is mostly spent and a coherent uncleaned wall/side strip is one lane away,
    # cover it now with a normal 90->shift->90 transition instead of driving far
    # past it and returning later.
    if maybe_start_side_strip_cleanup(front, center, left, right, body_clearance):
        row_end_candidate_count = 0
        return 0.0, 0.0

    # a close lateral uncleaned cluster, start a normal grid-safe lane transition
    # now instead of driving past it and coming back much later.
    if maybe_start_nearby_route_cleanup(front, center, left, right, body_clearance):
        row_end_candidate_count = 0
        return 0.0, 0.0

    # If the robot is aligned with an opening under/through a chair or table,
    # do not let the coverage objective or row-end logic steer it away. This was
    # the reason it looked straight into a passable under-chair gap and then left
    # by an arc: the target/lane logic won over the local furniture opportunity.
    under_corridor_candidate = (
        nav_state == NAV_FORWARD
        and optional_planner_intercepts_allowed("under_furniture")
        and now >= under_furniture_suppressed_until
        and now - last_contact_trap_time > CONTACT_RECOVERY_SUPPRESS_UNDER_SEC
        and now >= last_contact_route_kill_until
        and not body_corridor_blocked
        and detect_under_furniture_corridor(front, center, upper_front, left, right)
    )
    if under_corridor_candidate:
        under_furniture_confirm_count += 1
    elif nav_state == NAV_FORWARD and not under_furniture_active:
        under_furniture_confirm_count = 0

    if under_corridor_candidate and under_furniture_confirm_count >= UNDER_FURNITURE_CONFIRM_FRAMES:
        cleaned_ahead, uncleaned_ahead, _ = coverage_corridor_ahead()
        # Under-furniture mode is a local opportunity, not a permission to mow
        # the same already-cleaned strip forever. Keep it only when the corridor
        # still contains useful uncleaned cells or the selected target/frontier
        # is straight ahead through this opening.
        if cleaned_corridor_should_replan(
            cleaned_ahead,
            uncleaned_ahead,
            UNDER_FURNITURE_REVISIT_CLEANED_RATIO,
            UNDER_FURNITURE_REVISIT_MAX_UNCLEANED_RATIO,
        ):
            coverage_status = f"skip cleaned furniture strip clean={cleaned_ahead:.2f} unclean={uncleaned_ahead:.2f} target={coverage_goal_kind}"
        else:
            start_under_furniture_hold(
                f"under furniture pass confirmed={under_furniture_confirm_count}/{UNDER_FURNITURE_CONFIRM_FRAMES} F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f} L={left:.2f} R={right:.2f}"
            )
    elif nav_state == NAV_FORWARD and under_corridor_candidate:
        coverage_status = f"under furniture candidate {under_furniture_confirm_count}/{UNDER_FURNITURE_CONFIRM_FRAMES} F={front:.2f} C={center:.2f}"
    elif nav_state == NAV_FORWARD and body_corridor_blocked and upper_front > UNDER_FURNITURE_UPPER_OPEN:
        coverage_status = f"body envelope blocks gap body={body_clearance:.2f} lat={last_body_corridor_lateral:.2f}"

    # If a narrow table/chair leg is directly in the path but a nearby side gap
    # exists, do a small local bypass instead of a full lane-change. This makes
    # the robot attempt to clean under/through furniture rather than abandoning
    # the whole area around the chair.
    # Furniture-leg pass is checked before row-end. It uses a higher detection
    # distance than the wall stop threshold: when a thin leg is ahead, the robot
    # should begin a small bypass early instead of driving up to it and then doing
    # a full lane change around the chair/table.
    if (
        nav_state == NAV_FORWARD
        and optional_planner_intercepts_allowed("leg_pass")
        and front < LEG_PASS_DETECT_DISTANCE
        and now - last_leg_pass_time > LEG_PASS_COOLDOWN_SEC
        and not last_floor_front_ignore
        and not cv_front_is_probably_floor_shadow
    ):
        pass_side, pass_reason = detect_thin_leg_ahead(depth)
        if pass_side != 0.0 and upper_front > ROW_END_CONFIRM_DISTANCE + 0.10:
            start_leg_pass(pass_side, pass_reason)
            row_end_candidate_count = 0
            return 0.0, 0.0
        # Fallback only when the narrow forward corridor is actually blocked.
        # The previous version used min(front, center) and triggered from side
        # furniture long before the robot reached the chair/table; that caused
        # pointless diagonal arcs away from passable openings.
        if front < LEG_PASS_DETECT_DISTANCE and upper_front > ROW_END_CONFIRM_DISTANCE + 0.16 and max(left, right) > LEG_PASS_SIDE_OPEN_DISTANCE + LEG_PASS_OPEN_SIDE_PROBE_DISTANCE:
            probe_side = 1.0 if left >= right else -1.0
            start_leg_pass(probe_side, f"open-side probe F={front:.2f} L={left:.2f} R={right:.2f}; {pass_reason}")
            row_end_candidate_count = 0
            return 0.0, 0.0

    # Direct adjacent-strip acquisition.  If a close uncleaned line is physically
    # reachable, merge into it carefully instead of doing a full 90-degree
    # lane-change or deferring it as a residual island.
    line_acquire_cmd = edge_line_acquire_speeds(front, center, upper_front, left, right, body_clearance)
    if line_acquire_cmd is not None:
        return line_acquire_cmd

    # Main coverage mode: drive straight until the current row ends. Do not
    # decide left/right every frame. When blocked, execute one full lane-change
    # maneuver based on coverage memory and local side distances.
    # Anti-loop: after many seconds on already cleaned cells, force the next lane.
    # This is a pragmatic local planner, not full global A*. It prevents the
    # 8-hour repeated-strip failure visible in the previous screenshot.
    if nav_state == NAV_FORWARD and optional_planner_intercepts_allowed("revisit_replan") and now - row_start_time > REVISIT_FORCE_AFTER_SEC:
        revisit = cleaned_ratio_ahead()
        cleaned_ahead, uncleaned_ahead, obstacle_ahead = coverage_corridor_ahead()
        # Coverage objective override: if the corridor ahead is mostly already
        # cleaned while the objective map still has uncleaned cells, stop wasting
        # time on this strip and shift toward the current target. This is the
        # missing link between "we remember cleaned area" and "we try to cover
        # the whole map".
        if (
            not under_furniture_active
            and (not EARLY_EXPLORATION_DISABLE_TARGET_REPLAN or last_coverage_percent >= EARLY_EXPLORATION_COVERAGE_PERCENT)
            and not lane_action_cooldown_active(front)
            and row_distance_from_start() > REVISIT_FORCE_MIN_DISTANCE
            and cleaned_corridor_should_replan(
                cleaned_ahead,
                uncleaned_ahead,
                REVISIT_FORCE_CLEANED_RATIO,
                REVISIT_FORCE_MAX_UNCLEANED_RATIO,
            )
            and coverage_target_replan_allowed(for_side_bias=False)
            and now - last_revisit_lane_change_time > REVISIT_FORCE_COOLDOWN_SEC
            and front > ROW_END_CONFIRM_DISTANCE
            and not row_commit_active(front, body_clearance)
        ):
            side = coverage_target_side_or_default(left, right)
            last_revisit_lane_change_time = now
            begin_lawnmower_lane_change_checked(
                side,
                f"seek uncleaned: cleanedAhead={cleaned_ahead:.2f} uncleanedAhead={uncleaned_ahead:.2f} target={coverage_goal_kind}",
                front,
                body_clearance,
            )
            row_end_candidate_count = 0
            return 0.0, 0.0

        # If the corridor is already mostly cleaned but the only selected target is
        # a detached residual island, do not immediately leave the current sector.
        # Continue the strip/lawnmower order and let those fragments wait for the
        # late cleanup phase; this prevents visible ping-pong to separated spaces.
        if (
            not under_furniture_active
            and cleaned_corridor_should_replan(
                cleaned_ahead,
                uncleaned_ahead,
                REVISIT_FORCE_CLEANED_RATIO,
                REVISIT_FORCE_MAX_UNCLEANED_RATIO,
            )
            and not coverage_target_replan_allowed(for_side_bias=False)
        ):
            coverage_status = f"defer residual island until cleanup {last_coverage_percent:.1f}% def={last_residual_route_deferred_cells}"

        # Last-resort anti-loop if the row is extremely long and nearing a block.
        if row_distance_from_start() > ROW_MAX_DISTANCE and front < ROW_SLOW_DISTANCE and not row_commit_active(front, body_clearance):
            side = coverage_target_side_or_default(left, right)
            begin_lawnmower_lane_change_checked(side, f"anti-loop row={row_distance_from_start():.1f} cleaned={revisit:.2f}", front, body_clearance)
            row_end_candidate_count = 0
            return 0.0, 0.0

    # Active under-furniture traversal: keep the row heading and continue slowly
    # through the passable gap. Only a very close hit or bumper contact cancels it.
    if nav_state == NAV_FORWARD and under_furniture_active:
        if bumper_left or bumper_right:
            side = -1.0 if bumper_left and not bumper_right else (1.0 if bumper_right and not bumper_left else choose_coverage_side(left, right))
            obstacle_side = contact_obstacle_side_from_bumpers(bumper_left, bumper_right, left, right)
            mark_contact_obstacle(obstacle_side, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(side, f"under-furniture contact L={int(bumper_left)} C={int(last_bumper_center)} R={int(bumper_right)}", strong=True)
            return 0.0, 0.0
        obstacle_side = contact_obstacle_side_from_sensors(left, right)
        if obstacle_side != 0.0 and start_edge_trap_escape(
            obstacle_side,
            left,
            right,
            f"under-furniture edge trap side={'L' if obstacle_side > 0 else 'R'} F={front:.2f} body={body_clearance:.2f} L={left:.2f} R={right:.2f}",
        ):
            return 0.0, 0.0
        if front < ROW_END_HARD_DISTANCE and center < SAFE_FRONT_DISTANCE:
            side = choose_contact_escape_side(False, False, left, right, contact_obstacle_side_from_sensors(left, right))
            mark_contact_obstacle(0.0, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(side, f"under-furniture hard block F={front:.2f} C={center:.2f}", strong=True)
            return 0.0, 0.0
        # A low obstacle can be below the most stable upper-depth band while the
        # lower/centre corridor starts collapsing. Treat that as a trap, not as
        # permission to keep pushing under the shelf.
        if (center < LOW_OBSTACLE_GUARD_CENTER_M and front < LOW_OBSTACLE_GUARD_FRONT_M) or body_clearance < LOW_OBSTACLE_GUARD_BODY_M:
            side = choose_contact_escape_side(False, False, left, right, contact_obstacle_side_from_sensors(left, right))
            mark_contact_obstacle(0.0, min(front, center, BODY_CORRIDOR_PASS_CLEARANCE))
            start_leg_escape(side, f"under-furniture low/body block F={front:.2f} C={center:.2f} U={upper_front:.2f} body={body_clearance:.2f}", strong=True)
            return 0.0, 0.0

        if maybe_start_parallel_aperture_pass(front, center, upper_front, left, right, body_clearance):
            return 0.0, 0.0

        cleaned_ahead, uncleaned_ahead, _ = coverage_corridor_ahead()
        if (
            (not EARLY_EXPLORATION_DISABLE_TARGET_REPLAN or last_coverage_percent >= EARLY_EXPLORATION_COVERAGE_PERCENT)
            and not lane_action_cooldown_active(front)
            and cleaned_corridor_should_replan(
                cleaned_ahead,
                uncleaned_ahead,
                UNDER_FURNITURE_REVISIT_CLEANED_RATIO,
                UNDER_FURNITURE_REVISIT_MAX_UNCLEANED_RATIO,
            )
            and row_distance_from_start() > REVISIT_FORCE_MIN_DISTANCE
            and now - last_revisit_lane_change_time > UNDER_FURNITURE_REVISIT_COOLDOWN_SEC
        ):
            under_furniture_until = 0.0
            under_furniture_active = False
            side = coverage_target_side_or_default(left, right)
            last_revisit_lane_change_time = now
            begin_lawnmower_lane_change_checked(
                side,
                f"leave cleaned furniture strip clean={cleaned_ahead:.2f} unclean={uncleaned_ahead:.2f} target={coverage_goal_kind}",
                front,
                body_clearance,
            )
            row_end_candidate_count = 0
            return 0.0, 0.0

        row_end_candidate_count = 0
        if cleaned_ahead > UNDER_FURNITURE_REVISIT_CLEANED_RATIO and coverage_goal_is_ahead():
            coverage_status = f"transit cleaned -> {coverage_goal_kind} {under_furniture_until-now:.1f}s clean={cleaned_ahead:.2f}"
        else:
            coverage_status = f"under furniture pass {under_furniture_until-now:.1f}s F={front:.2f} clean={cleaned_ahead:.2f} unclean={uncleaned_ahead:.2f}"
        return heading_locked_wheel_speeds(UNDER_FURNITURE_SPEED)

    # After a furniture-leg bypass, do not immediately convert the still-visible
    # leg into a row-end pivot. Continue a short distance on the row heading;
    # physical bumpers still override this grace period.
    if nav_state == NAV_FORWARD and now < leg_pass_grace_until:
        if front < LEG_PASS_FRONT_ABORT_DISTANCE and center < 0.20:
            start_recovery_backup(choose_coverage_side(left, right), f"post-leg too close F={front:.2f} C={center:.2f}")
            return 0.0, 0.0
        row_end_candidate_count = 0
        coverage_status = f"post-leg grace {leg_pass_grace_until-now:.1f}s F={front:.2f}"
        return heading_locked_wheel_speeds(SLOW_SPEED)

    # Decide that the row ended only from a narrow forward corridor or from a
    # centre-supported body-footprint hit. A side body-warning near furniture is
    # not allowed to start a full 90-degree lane-change by itself.
    body_row_hard, body_row_soft = body_row_end_flags(front, center, upper_front, body_clearance)
    blocked_hard = front < ROW_END_HARD_DISTANCE or body_row_hard
    blocked_soft = front < ROW_END_CONFIRM_DISTANCE or body_row_soft

    if blocked_soft and lane_action_cooldown_active(front) and not blocked_hard:
        # We have just completed a turn/shift and have not moved far enough to
        # trust another soft row-end. Keep the heading and let the next frames
        # prove whether this is a real wall.
        row_end_candidate_count = 0
        coverage_status = f"debounce soft row-end F={front:.2f} body={body_clearance:.2f}@{last_body_corridor_lateral:.2f}"
        return heading_locked_wheel_speeds(SLOW_SPEED)

    if blocked_soft:
        row_end_candidate_count += 1
    else:
        row_end_candidate_count = 0

    if blocked_hard or row_end_candidate_count >= ROW_END_CONFIRM_FRAMES:
        side = choose_coverage_side(left, right)
        row_end_candidate_count = 0
        # During matrix-first exploration, a row end is a simple corner turn. Do
        # not start a lawnmower shift sequence while the 2-D map is still sparse.
        if matrix_first_explore_active() and row_distance_from_start() >= EXPLORE_MIN_ROW_BEFORE_TURN_M:
            side = choose_explore_turn_side(left, right)
            if front < ROW_END_HARD_DISTANCE or max(left, right) < SIDE_DISTANCE + 0.10:
                start_recovery_backup(side, f"explore corner clearance F={front:.2f} C={center:.2f} L={left:.2f} R={right:.2f}")
                return 0.0, 0.0
            begin_explore_single_turn(side, f"matrix-first legacy row end F={front:.2f} C={center:.2f} L={left:.2f} R={right:.2f}")
            return 0.0, 0.0
        # If obstacle is extremely close or both side sectors are poor, back up
        # first. This is common in corners and avoids alternating pivot commands.
        if front < ROW_END_HARD_DISTANCE or max(left, right) < SIDE_DISTANCE + 0.10:
            start_recovery_backup(side, f"corner F={front:.2f} C={center:.2f} L={left:.2f} R={right:.2f}")
            return 0.0, 0.0
        begin_lawnmower_lane_change_checked(side, f"row end F={front:.2f} C={center:.2f} L={left:.2f} R={right:.2f}", front, body_clearance)
        return 0.0, 0.0

    if row_commit_active(front, body_clearance):
        coverage_status = f"row commit mapping {row_distance_from_start():.2f}m/{ROW_COMMIT_MIN_DISTANCE_M:.2f}; route later"
        return heading_locked_wheel_speeds(CRUISE_SPEED)

    # Slow down near a real forward obstacle but keep the row direction. Side
    # obstacles alone no longer end the row.
    if front < ROW_SLOW_DISTANCE:
        coverage_status = f"row slow F={front:.2f} body={body_clearance:.2f} wait={row_end_candidate_count}/{ROW_END_CONFIRM_FRAMES}"
        return heading_locked_wheel_speeds(SLOW_SPEED)

    coverage_status = "row forward"
    return heading_locked_wheel_speeds(CRUISE_SPEED)

def set_wheel_speeds(left, right):
    global prev_cmd_left, prev_cmd_right
    global last_motion_primitive, last_motion_contract_reason, last_requested_left, last_requested_right
    left = clamp(left, -MAX_SPEED, MAX_SPEED)
    right = clamp(right, -MAX_SPEED, MAX_SPEED)

    motion_contract = apply_strict_motion_contract(
        nav_state=nav_state,
        phase=navigation_phase,
        left=left,
        right=right,
        status=coverage_status,
        enabled=STRICT_MOTION_PRIMITIVES_ENABLED,
        curve_diff_threshold=STRICT_CURVE_DIFF_THRESHOLD,
        pivot_min_speed=STRICT_GRID_PIVOT_MIN_SPEED,
        pivot_max_speed=STRICT_GRID_PIVOT_MAX_SPEED,
    )
    left = clamp(motion_contract.left, -MAX_SPEED, MAX_SPEED)
    right = clamp(motion_contract.right, -MAX_SPEED, MAX_SPEED)
    last_requested_left = left
    last_requested_right = right
    last_motion_primitive = motion_contract.primitive
    last_motion_contract_reason = motion_contract.reason

    # Pivot accuracy is more important than smoothness: if smoothing remains
    # active during a 90-degree turn, the robot keeps rotating during SETTLE.
    if (
        nav_state in (NAV_TURN_90, NAV_SETTLE, NAV_LEG_ESCAPE_TURN, NAV_LEG_PASS_TURN, NAV_LEG_PASS_ALIGN, NAV_PRE_PIVOT_BACKUP, NAV_GRID_REALIGN) + CONTACT_RECOVERY_STATES
        or last_motion_primitive in (MotionPrimitive.TURN_IN_PLACE.value, MotionPrimitive.FINE_ALIGN.value, MotionPrimitive.CONTACT_RELEASE_TURN.value)
        or (abs(left) < 1e-6 and abs(right) < 1e-6)
    ):
        prev_cmd_left = left
        prev_cmd_right = right
    else:
        alpha = CMD_SMOOTH_ALPHA
        prev_cmd_left = (1.0 - alpha) * prev_cmd_left + alpha * left
        prev_cmd_right = (1.0 - alpha) * prev_cmd_right + alpha * right

    left_motor.setVelocity(LEFT_SIGN * prev_cmd_left)
    right_motor.setVelocity(RIGHT_SIGN * prev_cmd_right)


def update_orb_debug(frame):
    global prev_kp, prev_desc, last_orb_matches
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)
    if prev_desc is None or desc is None or prev_kp is None:
        prev_kp, prev_desc = kp, desc
        last_orb_matches = 0
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(prev_desc, desc)
    good = [m for m in sorted(matches, key=lambda m: m.distance)[:80] if m.distance < 70]
    last_orb_matches = len(good)
    prev_kp, prev_desc = kp, desc
    return kp


def depth_at_image_pixel(depth, px, py, patch=CV_DEPTH_PATCH):
    """Depth at an RGB image pixel, using a small RangeFinder patch.

    Camera and RangeFinder have similar frontal alignment in this project. We
    map normalized RGB coordinates to normalized depth coordinates. A median
    patch makes the value stable enough for visual-feature fusion.
    """
    if depth is None:
        return float("nan")
    cx = int(clamp(px / max(1, CAM_W - 1) * (RF_W - 1), 0, RF_W - 1))
    cy = int(clamp(py / max(1, CAM_H - 1) * (RF_H - 1), 0, RF_H - 1))
    x0 = max(0, cx - patch)
    x1 = min(RF_W, cx + patch + 1)
    y0 = max(0, cy - patch)
    y1 = min(RF_H, cy + patch + 1)
    vals = depth[y0:y1, x0:x1]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals > CV_FEATURE_MIN_RANGE) & (vals < CV_FEATURE_MAX_RANGE)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def project_image_feature_to_map(x, y, theta, px, dist):
    """Project an RGB feature column plus depth into map coordinates."""
    bearing = ((float(px) + 0.5) / max(1, CAM_W) - 0.5) * RF_FOV
    wx, wy = project_depth_point(x, y, theta, bearing, dist)
    return world_to_map(wx, wy)


def depth_patch_stats(depth, x0, y0, w, h, pad=2):
    """Depth median/std for an RGB bounding box projected to RangeFinder pixels."""
    if depth is None or w <= 0 or h <= 0:
        return float("nan"), float("nan")
    rx0 = int(clamp((x0 - pad) / max(1, CAM_W - 1) * (RF_W - 1), 0, RF_W - 1))
    rx1 = int(clamp((x0 + w + pad) / max(1, CAM_W - 1) * (RF_W - 1), 0, RF_W - 1))
    ry0 = int(clamp((y0 - pad) / max(1, CAM_H - 1) * (RF_H - 1), 0, RF_H - 1))
    ry1 = int(clamp((y0 + h + pad) / max(1, CAM_H - 1) * (RF_H - 1), 0, RF_H - 1))
    if rx1 <= rx0 or ry1 <= ry0:
        return float("nan"), float("nan")
    vals = depth[ry0:ry1 + 1, rx0:rx1 + 1]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals > CV_FEATURE_MIN_RANGE) & (vals < CV_FEATURE_MAX_RANGE)]
    if vals.size < 4:
        return float("nan"), float("nan")
    return float(np.median(vals)), float(np.std(vals))


def is_shadow_like_patch(frame, depth, x0, y0, w, h):
    """Reject dark floor shadows that produce RGB edges but no real obstacle.

    The important test is not just brightness. A black cube can also be dark.
    We reject only patches that are low-saturation, dark, in the floor/lower
    image area, and have open/flat depth behind them.
    """
    if not CV_SHADOW_FILTER_ENABLED or frame is None or w <= 0 or h <= 0:
        return False
    if y0 < CAM_H * CV_SHADOW_MIN_Y_FRACTION:
        return False
    x0c = int(clamp(x0, 0, CAM_W - 1))
    y0c = int(clamp(y0, 0, CAM_H - 1))
    x1c = int(clamp(x0 + w, 0, CAM_W))
    y1c = int(clamp(y0 + h, 0, CAM_H))
    if x1c <= x0c or y1c <= y0c:
        return False
    roi = frame[y0c:y1c, x0c:x1c]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = float(np.median(hsv[:, :, 1]))
    val = float(np.median(hsv[:, :, 2]))
    if not (val < CV_SHADOW_LOW_VALUE and sat < CV_SHADOW_LOW_SATURATION):
        return False
    d_med, d_std = depth_patch_stats(depth, x0, y0, w, h, pad=4)
    if not np.isfinite(d_med):
        return False
    return d_med > CV_SHADOW_OPEN_DEPTH_MIN and (not np.isfinite(d_std) or d_std < CV_SHADOW_MAX_DEPTH_STD)


def is_shadow_like_line(frame, depth, x1, y1, x2, y2):
    pad = 8
    x0 = min(x1, x2) - pad
    y0 = min(y1, y2) - pad
    w = abs(x2 - x1) + 2 * pad
    h = abs(y2 - y1) + 2 * pad
    return is_shadow_like_patch(frame, depth, x0, y0, w, h)

def is_floor_like_visual_line(x1, y1, x2, y2):
    """Reject carpet/floor texture edges, not real vertical obstacles.

    A low robot camera sees the carpet border as a long horizontal line in the
    lower part of the image. If we map it as an obstacle, the robot appears to
    detect a bump in the carpet. Real furniture/walls still produce vertical or
    diagonal edges and compact contours, so they remain available for RGB-D
    fusion.
    """
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < CV_FLAT_FLOOR_LINE_MIN_LENGTH:
        return False
    mean_y = 0.5 * (float(y1) + float(y2))
    angle = abs(math.degrees(math.atan2(dy, dx)))
    angle = min(angle, abs(180.0 - angle))
    return mean_y > CAM_H * CV_FLOOR_Y_FRACTION and angle < CV_FLOOR_HORIZ_ANGLE_DEG


def is_floor_like_contour_box(x0, y0, w, h):
    """Reject wide, low strips caused by carpet/floor markings.

    This intentionally does not reject narrow vertical boxes, so chair/table
    legs and low furniture edges can still become obstacles.
    """
    if w <= 0 or h <= 0:
        return False
    wide = w > CAM_W * CV_WIDE_LOW_CONTOUR_FRAC
    low = y0 > CAM_H * CV_FLOOR_Y_FRACTION
    flat = h < CAM_H * CV_LOW_STRIP_MAX_HEIGHT_FRAC
    bottom_strip = (y0 + h) > CAM_H * 0.86 and h < CAM_H * 0.26 and w > CAM_W * 0.12
    return (wide and low and flat) or bottom_strip


def is_floor_plane_visual_feature(depth, x0, y0, w, h):
    """Reject carpet/floor visual texture using image position + flat depth.

    This protects the occupancy and CV-navigation layers from treating the dark
    carpet rectangle, white/orange carpet strips, and carpet borders as walls.
    It is intentionally not a semantic carpet detector; it only rejects low,
    wide/flat features with an open, low-variance depth patch.  Narrow high-
    contrast vertical features such as chair/table legs remain valid obstacles.
    """
    if not CV_FLOOR_PLANE_REJECT_ENABLED or w <= 0 or h <= 0:
        return False
    cy = y0 + 0.5 * h
    if cy < CAM_H * CV_FLOOR_PLANE_MIN_Y_FRACTION:
        return False
    wide_or_flat = (w >= CAM_W * CV_FLOOR_PLANE_WIDE_FRAC) or (h <= CAM_H * CV_FLOOR_PLANE_MAX_BOX_HEIGHT_FRAC)
    if not wide_or_flat:
        return False
    d_med, d_std = depth_patch_stats(depth, x0, y0, w, h, pad=5)
    if not np.isfinite(d_med):
        return False
    if d_med < CV_FLOOR_PLANE_MIN_DEPTH:
        return False
    # Flat/open depth means floor texture.  Real furniture edges usually have a
    # stronger depth discontinuity or a compact vertical silhouette.
    return (not np.isfinite(d_std)) or d_std <= CV_FLOOR_PLANE_MAX_STD


def mark_cv_free_ray_to_feature(x, y, theta, px, dist):
    """Mark weak free space to an RGB-D confirmed visual feature.

    This is still CV-first: the ray exists only because RGB found a feature and
    depth confirmed its metric position. It helps the map grow without treating
    every RangeFinder column as a lidar scan.
    """
    if not np.isfinite(dist) or dist <= 0.20:
        return
    sx, sy = sensor_origin_world(x, y, theta)
    rx, ry = world_to_map(sx, sy)
    if not map_inside(rx, ry):
        return
    free_dist = min(max(0.04, float(dist) - 0.10), CV_FREE_RAY_MAX_RANGE)
    if free_dist <= 0.04:
        return
    bearing = ((float(px) + 0.5) / max(1, CAM_W) - 0.5) * RF_FOV
    ex_w, ey_w = project_depth_point(x, y, theta, bearing, free_dist)
    ex, ey = world_to_map(ex_w, ey_w)
    for cx, cy in rgbd_free_ray_cells(rx, ry, ex, ey):
        log_odds[cy, cx] = clamp(log_odds[cy, cx] + CV_FREE_RAY_UPDATE, LO_MIN, LO_MAX)
        clear_contact_evidence_cell(cx, cy, CONTACT_FREE_UPDATE * 0.45)


def detect_rgb_cv_features(frame, depth):
    global last_cv_floor_rejected, last_cv_shadow_rejected
    last_cv_floor_rejected = 0
    last_cv_shadow_rejected = 0
    """Extract RGB visual features and attach depth when possible.

    Returns a dict of:
    - lines: Hough line segments with depth-confirmed sample points;
    - boxes: contour boxes with depth-confirmed representative points;
    - points: all depth-confirmed RGB feature points for map fusion.

    This is not semantic recognition. It is RGB-D visual mapping: RGB decides
    *what looks like an object/wall edge*, depth decides *where it is in meters*.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Slightly stronger Canny thresholds reduce floor texture noise while still
    # keeping wall/furniture boundaries in the simple Webots scene.
    edges = cv2.Canny(blur, 60, 145)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    features = {"lines": [], "boxes": [], "points": []}

    # --- Line structures: wall edges, furniture edges, table borders. ---
    lines = cv2.HoughLinesP(closed, 1, np.pi / 180.0, threshold=38,
                            minLineLength=CV_LINE_MIN_LENGTH, maxLineGap=10)
    if lines is not None:
        line_items = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < CV_LINE_MIN_LENGTH:
                continue
            # Ignore top HUD-like lines if any and near-bottom depth overlay area.
            if max(y1, y2) > CAM_H * 0.92:
                continue
            # Carpet/floor borders are visual texture, not obstacles. They are
            # typically long horizontal lines in the lower half of the image.
            if is_floor_like_visual_line(x1, y1, x2, y2):
                last_cv_floor_rejected += 1
                continue
            if is_floor_plane_visual_feature(depth, min(x1, x2) - 6, min(y1, y2) - 6, abs(x2 - x1) + 12, abs(y2 - y1) + 12):
                last_cv_floor_rejected += 1
                continue
            if is_shadow_like_line(frame, depth, x1, y1, x2, y2):
                last_cv_shadow_rejected += 1
                continue
            pts = []
            # Sample line with a fixed pixel step. Five samples were too sparse,
            # so walls appeared as isolated dots instead of map structures.
            sample_count = int(clamp(length / max(1, CV_LINE_SAMPLE_STEP_PX), 3, 12))
            for i in range(sample_count):
                t = (i + 0.5) / sample_count
                px = int(round(x1 + (x2 - x1) * t))
                py = int(round(y1 + (y2 - y1) * t))
                d = depth_at_image_pixel(depth, px, py)
                if np.isfinite(d):
                    pts.append((px, py, float(d)))
            if not pts:
                continue
            if len(pts) >= 3:
                d_arr = np.array([float(p[2]) for p in pts], dtype=np.float32)
                d_span = float(np.nanmax(d_arr) - np.nanmin(d_arr))
                d_std = float(np.nanstd(d_arr))
                if d_span > CV_LINE_MAX_DEPTH_SPAN_M or d_std > CV_LINE_MAX_DEPTH_STD_M:
                    # A single Hough line across table shadows / floor edges at
                    # different depths is not a metric wall/object edge.  Reject
                    # it before it becomes dotted persistent obstacle noise.
                    last_cv_floor_rejected += 1
                    continue
            # Prefer long, depth-confirmed lines.
            line_items.append((length, (x1, y1, x2, y2, pts)))
        for _, item in sorted(line_items, key=lambda it: it[0], reverse=True)[:CV_LINE_MAX_COUNT]:
            features["lines"].append(item)
            for px, py, d in item[4]:
                features["points"].append((px, py, d, "line"))

    # --- Contour regions: non-semantic visual object candidates. ---
    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CV_CONTOUR_MIN_AREA:
            continue
        x0, y0, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue
        if w * h > CV_CONTOUR_MAX_FRAC * CAM_W * CAM_H:
            continue
        if y0 + h > CAM_H * 0.94:
            continue
        if is_floor_like_contour_box(x0, y0, w, h):
            last_cv_floor_rejected += 1
            continue
        if is_floor_plane_visual_feature(depth, x0, y0, w, h):
            last_cv_floor_rejected += 1
            continue
        if is_shadow_like_patch(frame, depth, x0, y0, w, h):
            last_cv_shadow_rejected += 1
            continue

        samples = [
            (x0 + w * 0.50, y0 + h * 0.50),
            (x0 + w * 0.30, y0 + h * 0.62),
            (x0 + w * 0.70, y0 + h * 0.62),
            (x0 + w * 0.35, y0 + h * 0.80),
            (x0 + w * 0.65, y0 + h * 0.80),
        ]
        dvals = []
        for sx, sy in samples:
            d = depth_at_image_pixel(depth, int(sx), int(sy), patch=CV_DEPTH_PATCH + 1)
            if np.isfinite(d):
                dvals.append((int(sx), int(sy), float(d)))
        if not dvals:
            continue
        # Representative distance of the visual region.
        rep = min(dvals, key=lambda p: p[2])
        boxes.append((area, x0, y0, w, h, rep, dvals))

    for _, x0, y0, w, h, rep, dvals in sorted(boxes, key=lambda b: b[0], reverse=True)[:CV_BOX_MAX_COUNT]:
        features["boxes"].append((x0, y0, w, h, rep))
        for px, py, d in dvals:
            features["points"].append((px, py, d, "contour"))

    return features


def update_visual_map_from_rgb_depth(pose, frame, depth):
    """Fuse RGB OpenCV features with depth and add them to a visual map layer.

    This is the strengthened CV component:
    RGB features are not merely visualized. Depth-confirmed RGB edges/contours
    are projected into the persistent map as a separate visual obstacle layer and
    weakly support the main occupancy grid.
    """
    global last_cv_features, last_rgb_contours, last_rgb_lines
    global last_cv_map_hits, last_cv_depth_confirmed, visual_log_odds
    global last_cv_left_obstacle, last_cv_front_obstacle, last_cv_right_obstacle
    global last_cv_floor_rejected

    if not CV_FUSION_ENABLED:
        last_cv_map_hits = 0
        last_cv_depth_confirmed = 0
        last_cv_features = {"lines": [], "boxes": [], "points": []}
        return 0

    features = detect_rgb_cv_features(frame, depth)
    last_cv_features = features
    last_rgb_lines = len(features["lines"])
    last_rgb_contours = len(features["boxes"])
    last_cv_depth_confirmed = len(features["points"])
    last_cv_map_hits = 0

    # CV-derived obstacle sectors for navigation. This is deliberately separate
    # from raw depth sectors: it answers "did RGB see an object/edge here, and
    # can depth attach a distance to it?"
    last_cv_left_obstacle = MAX_VALID_RANGE
    last_cv_front_obstacle = MAX_VALID_RANGE
    last_cv_right_obstacle = MAX_VALID_RANGE
    for px, py, d, kind in features.get("points", []):
        if not np.isfinite(d):
            continue
        # Ignore very high image features for navigation; they are often wall
        # decorations/top edges, not obstacles in the robot's path.
        if py < CAM_H * 0.22:
            continue
        if px < CAM_W / 3:
            last_cv_left_obstacle = min(last_cv_left_obstacle, float(d))
        elif px > 2 * CAM_W / 3:
            last_cv_right_obstacle = min(last_cv_right_obstacle, float(d))
        else:
            last_cv_front_obstacle = min(last_cv_front_obstacle, float(d))

    # Do not integrate visual evidence while the RGB-D pose is rotationally
    # unstable.  Do not rely on last_map_frozen: depth and CV updates are
    # throttled independently, so the depth updater may not have run this tick.
    mapping_frozen, freeze_reason = update_mapping_freeze_state(robot.getTime())
    if mapping_frozen or nav_state not in MAPPING_ALLOWED_STATES:
        globals()["last_rgbd_occlusion_debug"] = f"cvFreeze={freeze_reason[:18]}"
        return 0

    x, y, theta = pose
    sx, sy = sensor_origin_world(x, y, theta)
    rx, ry = world_to_map(sx, sy)
    if not map_inside(rx, ry) or rgbd_sensor_origin_mapping_blocked(x, y, theta):
        return 0

    # Lines: add visual evidence along projected depth-confirmed line samples.
    for x1, y1, x2, y2, pts in features["lines"]:
        projected = []
        for px, py, d in pts:
            mx, my = project_image_feature_to_map(x, y, theta, px, d)
            if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
                projected.append((px, py, d, mx, my))
        if not projected:
            continue
        for px, py, d, mx, my in projected:
            mark_cv_free_ray_to_feature(x, y, theta, px, d)
            cv2.circle(visual_log_odds, (mx, my), 3, CV_VISUAL_LINE_UPDATE, -1)
            cv2.circle(log_odds, (mx, my), 2, CV_OCCUPANCY_SUPPORT_UPDATE, -1)
            last_cv_map_hits += 1
        if len(projected) >= 2:
            p0 = (projected[0][3], projected[0][4])
            p1 = (projected[-1][3], projected[-1][4])
            # Only connect short projected segments; long connections are often
            # perspective artifacts across different depths.
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 80:
                for cx, cy in bresenham(p0[0], p0[1], p1[0], p1[1]):
                    if map_inside(cx, cy):
                        visual_log_odds[cy, cx] = clamp(visual_log_odds[cy, cx] + CV_VISUAL_LINE_UPDATE * 0.35, LO_MIN, LO_MAX)

    # Contours/boxes: add compact obstacle candidates.
    for x0, y0, w, h, rep in features["boxes"]:
        px, py, d = rep
        mx, my = project_image_feature_to_map(x, y, theta, px, d)
        if map_inside(mx, my) and not rgbd_ray_to_feature_occluded(rx, ry, mx, my):
            mark_cv_free_ray_to_feature(x, y, theta, px, d)
            radius = 3 if d > 1.0 else 4
            cv2.circle(visual_log_odds, (mx, my), radius, CV_VISUAL_UPDATE, -1)
            cv2.circle(log_odds, (mx, my), 2, CV_OCCUPANCY_SUPPORT_UPDATE, -1)
            last_cv_map_hits += 1

    return last_cv_map_hits


def draw_rgb_cv_features(debug):
    """Draw cached RGB-CV detections from the current frame."""
    # Draw depth-confirmed line features.
    for x1, y1, x2, y2, pts in last_cv_features.get("lines", [])[:CV_LINE_MAX_COUNT]:
        cv2.line(debug, (x1, y1), (x2, y2), CV_LINE_COLOR, 1)
        for px, py, d in pts:
            cv2.circle(debug, (int(px), int(py)), 2, (255, 120, 255), -1)

    # Draw contour boxes and a distance marker at the representative point.
    for x, y, w, h, rep in last_cv_features.get("boxes", [])[:CV_BOX_MAX_COUNT]:
        cv2.rectangle(debug, (x, y), (x + w, y + h), CV_BOX_COLOR, 1)
        px, py, d = rep
        cv2.circle(debug, (int(px), int(py)), 3, (255, 120, 255), -1)
        cv2.putText(debug, f"{d:.1f}m", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CV_BOX_COLOR, 1)
    return debug


def draw_cv_debug(frame, depth):
    debug = frame.copy()
    debug = draw_rgb_cv_features(debug)
    # Overlay a coarse depth view at bottom-left.
    depth_vis = depth.copy()
    depth_vis[~np.isfinite(depth_vis)] = MAX_VALID_RANGE
    depth_vis = np.clip(depth_vis, 0, MAX_VALID_RANGE)
    depth_u8 = (255 * (1.0 - depth_vis / MAX_VALID_RANGE)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    depth_color = cv2.resize(depth_color, (CAM_W // 3, CAM_H // 4), interpolation=cv2.INTER_NEAREST)
    debug[CAM_H - depth_color.shape[0]:CAM_H, 0:depth_color.shape[1]] = depth_color

    # Navigation zones.
    cv2.line(debug, (CAM_W // 3, 0), (CAM_W // 3, CAM_H), (80, 200, 80), 1)
    cv2.line(debug, (2 * CAM_W // 3, 0), (2 * CAM_W // 3, CAM_H), (80, 200, 80), 1)
    # Compact HUD. After lowering camera resolution to 320x240 the old text
    # covered most of the RGB image and made debugging useless. Keep only the
    # information that explains navigation decisions.
    hud_color = (40, 255, 40)
    fs = 0.38
    th = 1
    cv2.rectangle(debug, (0, 0), (CAM_W, 54), (0, 0, 0), -1)
    cv2.putText(debug, f"D L/F/U/C/R {last_min_left:.2f}/{last_front_narrow:.2f}/{last_front_upper:.2f}/{last_min_center:.2f}/{last_min_right:.2f}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, fs, hud_color, th)
    cv2.putText(debug, f"CV box={last_rgb_contours} line={last_rgb_lines} fused={last_cv_map_hits}/{last_cv_depth_confirmed}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, fs, hud_color, th)
    cv2.putText(debug, f"floor={last_cv_floor_rejected} shadow={last_cv_shadow_rejected} nav {last_cv_left_obstacle:.1f}/{last_cv_front_obstacle:.1f}/{last_cv_right_obstacle:.1f}", (5, 46), cv2.FONT_HERSHEY_SIMPLEX, fs, hud_color, th)
    return debug


def map_view_status_text():
    mode = "full" if not map_view_auto_crop else ("follow" if map_view_follow_robot else "auto")
    pan = ""
    if abs(map_view_pan_x) > 1e-3 or abs(map_view_pan_y) > 1e-3:
        pan = f" pan={map_view_pan_x:.0f},{map_view_pan_y:.0f}"
    return f"mapView={mode} zoom={map_view_zoom:.2f}x{pan} crop={last_map_view_crop_debug} keys:+/- C F IJKL 0"


def current_stop_or_safety_hint():
    """Human-readable explanation for 'why did it stop before touching?'.

    This is intentionally diagnostic-only. It separates physical bumper contact
    from RGB-D/depth precontact and virtual body-envelope row-end logic.
    """
    if last_bumper_left or last_bumper_center or last_bumper_right:
        return f"BUMPER L/C/R={int(last_bumper_left)}/{int(last_bumper_center)}/{int(last_bumper_right)}"
    if nav_state in CONTACT_RECOVERY_STATES:
        return f"CONTACT-RECOVERY {str(contact_recovery_reason)[:32]} rear={last_rear_guard_reason}"
    if nav_state in (NAV_RECOVERY_BACKUP, NAV_LEG_ESCAPE_BACKUP, NAV_LEG_ESCAPE_TURN, NAV_LEG_ESCAPE_FORWARD):
        return f"RECOVERY {last_contact_escape_kind[:26]} rear={last_rear_guard_reason}"
    if last_motion_primitive == MotionPrimitive.STOP.value or (abs(last_requested_left) < 1e-6 and abs(last_requested_right) < 1e-6):
        if last_front_narrow < SAFE_FRONT_DISTANCE or last_min_center < SAFE_FRONT_DISTANCE:
            return f"DEPTH PRECONTACT F/C={last_front_narrow:.2f}/{last_min_center:.2f}"
        if last_body_corridor_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
            return f"BODY-ENVELOPE precontact {last_body_corridor_clearance:.2f}@{last_body_corridor_lateral:.2f}"
        if row_end_candidate_count > 0:
            return f"ROW-END confirm {row_end_candidate_count}/{ROW_END_CONFIRM_FRAMES} F={last_front_narrow:.2f} body={last_body_corridor_clearance:.2f}"
        return f"STOP by state: {coverage_status[:45]}"
    if last_front_narrow < SLOW_FRONT_DISTANCE:
        return f"DEPTH SLOWDOWN F={last_front_narrow:.2f} C={last_min_center:.2f}"
    if last_body_corridor_clearance < BODY_CORRIDOR_PASS_CLEARANCE:
        return f"BODY SLOW/GUARD {last_body_corridor_clearance:.2f}@{last_body_corridor_lateral:.2f}"
    return "clear"


def padded_debug_crop(img, ix0, iy0, ix1, iy1, fill_color=UNKNOWN_COLOR):
    """Crop a map window, allowing the crop to go outside the map canvas.

    OpenCV map images live in a fixed MAP_SIZE x MAP_SIZE array.  When the room
    is close to the array edge, a normal clamped crop cannot keep it centered.
    This helper copies the valid intersection into a padded canvas, so auto-crop
    and manual pan behave like a real camera viewport over a larger world.
    """
    crop_w = max(1, int(ix1 - ix0))
    crop_h = max(1, int(iy1 - iy0))
    out = np.empty((crop_h, crop_w, 3), dtype=img.dtype)
    out[:, :] = fill_color

    sx0 = max(0, int(ix0))
    sy0 = max(0, int(iy0))
    sx1 = min(MAP_SIZE, int(ix1))
    sy1 = min(MAP_SIZE, int(iy1))
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    dx0 = sx0 - int(ix0)
    dy0 = sy0 - int(iy0)
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


def apply_debug_map_crop(img, points, auto_crop=True):
    """Apply user-controlled map crop/zoom/pan for debug windows only.

    This is visualization-only.  It never changes map coordinates, coverage data,
    planner targets, route ownership or motion.
    """
    global last_map_view_crop_debug
    if (not auto_crop) or (not map_view_auto_crop) or (not points):
        last_map_view_crop_debug = "full"
        return img

    x0 = max(0, min(p[0] for p in points) - 90)
    y0 = max(0, min(p[1] for p in points) - 90)
    x1 = min(MAP_SIZE, max(p[2] for p in points) + 90)
    y1 = min(MAP_SIZE, max(p[3] for p in points) + 90)
    if x1 - x0 <= 32 or y1 - y0 <= 32:
        last_map_view_crop_debug = "tiny"
        return img

    # Default center follows the known map content.  Optional follow mode uses
    # the robot pose as the viewport center while retaining the content-derived
    # crop size, so the robot does not disappear when debugging local behavior.
    if map_view_follow_robot:
        rx, ry = world_to_map(pose_x, pose_y)
        cx = float(rx)
        cy = float(ry)
        center_mode = "robot"
    else:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        center_mode = "content"

    cx += float(map_view_pan_x)
    cy += float(map_view_pan_y)

    base_w = max(220.0, float(x1 - x0))
    base_h = max(220.0, float(y1 - y0))
    zoom = max(MAP_VIEW_ZOOM_MIN, min(MAP_VIEW_ZOOM_MAX, float(map_view_zoom)))
    crop_w = base_w / zoom
    crop_h = base_h / zoom

    # Keep the crop aspect ratio close to the display window so zooming does not
    # distort the room.  Padding outside the map is allowed below, so a room near
    # the map boundary can still remain visually centered.
    aspect = float(MAP_VIEW_W) / float(MAP_VIEW_H)
    if crop_w / max(1.0, crop_h) < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect

    crop_w = max(220.0, crop_w)
    crop_h = max(220.0, crop_h)
    # Do not let an accidental extreme zoom-out create a huge padded canvas.
    crop_w = min(float(MAP_SIZE) * 1.35, crop_w)
    crop_h = min(float(MAP_SIZE) * 1.35, crop_h)

    ix0 = int(round(cx - crop_w * 0.5))
    iy0 = int(round(cy - crop_h * 0.5))
    ix1 = ix0 + int(round(crop_w))
    iy1 = iy0 + int(round(crop_h))

    if ix1 - ix0 <= 32 or iy1 - iy0 <= 32:
        last_map_view_crop_debug = "bad"
        return img

    last_map_view_crop_debug = f"{center_mode}:{ix0},{iy0}-{ix1},{iy1}"
    return padded_debug_crop(img, ix0, iy0, ix1, iy1)


def draw_map_hud_and_legend(img):
    """Draw readable debug info and a defense-friendly legend after crop/resize.

    The map itself is in metric coordinates; this overlay is screen-space so it
    remains readable even when auto-cropping follows the robot.
    """
    global last_contact_map_cells
    h, w = img.shape[:2]
    last_contact_map_cells = int(np.count_nonzero(contact_log_odds > CONTACT_OCCUPIED_EPS))

    def panel(x, y, ww, hh, alpha=0.72):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (min(w - 1, x + ww), min(h - 1, y + hh)), (245, 245, 245), -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
        cv2.rectangle(img, (x, y), (min(w - 1, x + ww), min(h - 1, y + hh)), (80, 80, 80), 1)

    # Compact status panel. Keep this separate from the map geometry so it is
    # not confused with an obstacle or CV feature.
    panel(8, 8, 900, 160, 0.70)
    cv2.putText(img, f"pose x={pose_x:.2f} y={pose_y:.2f} theta={math.degrees(pose_theta):.1f} deg | target={math.degrees(desired_grid_heading):.0f}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"mode=RGB-D CV-FIRST | phase={navigation_phase} | nav={nav_state} | prim={last_motion_primitive} | map={'FROZEN' if last_map_frozen else 'active'} | rowEnd={row_end_candidate_count}/{ROW_END_CONFIRM_FRAMES} | bump L/C/R={int(last_bumper_left)}/{int(last_bumper_center)}/{int(last_bumper_right)} contactCells={last_contact_map_cells} trap={int(robot.getTime()-last_contact_trap_time < 1.2)} rugIgnore={int(last_bumper_ignored_as_floor)}, floorIgnore={int(last_floor_front_ignore)} legPass={int(last_leg_pass_detected)}/{'L' if last_leg_pass_side>0 else ('R' if last_leg_pass_side<0 else '-')} under={int(under_furniture_active)} shadowFilt={last_cv_shadow_rejected}",
                (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"depth L/F/U/C/R={last_min_left:.2f}/{last_front_narrow:.2f}/{last_front_upper:.2f}/{last_min_center:.2f}/{last_min_right:.2f} m | bodyClear={last_body_corridor_clearance:.2f} lat={last_body_corridor_lateral:.2f} | CV fused={last_cv_map_hits}/{last_cv_depth_confirmed}, floorFilt={last_cv_floor_rejected}, shadowFilt={last_cv_shadow_rejected}, L/F/R={last_cv_left_obstacle:.2f}/{last_cv_front_obstacle:.2f}/{last_cv_right_obstacle:.2f} m",
                (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"coverage={coverage_status[:40]}, scan={last_active_scan_debug[:24]}, {last_mapping_write_debug[:34]} | {map_view_status_text()} | S save R reset Q quit",
                (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"objective: {last_coverage_percent:.1f}% cleaned, uncleaned={last_uncleaned_cells}, frontier={last_frontier_cells}, mapMature={int(map_mature)} conf={planner_confidence} intent={planner_intent} {planner_mode} {known_map_eval_status[:22]} {auto_map_mission_phase[:14]}",
                (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"candidateTarget={route_commit_candidate_debug()} activeTarget={route_commit_active_target_debug()} abort={route_abort_reason[:28]}",
                (18, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"{last_furniture_zone_debug[:70]} | {last_raw_map_speckle_debug[:28]}",
                (18, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)

    if not SHOW_MAP_LEGEND:
        return

    # Legend: deliberately placed at bottom-left, small and fixed-size.
    lx, ly = 12, h - 226
    panel(lx, ly, 470, 210, 0.76)
    cv2.putText(img, "Legend / map layers", (lx + 10, ly + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 1, cv2.LINE_AA)

    items = [
        (UNKNOWN_COLOR, "gray = unknown / not observed"),
        (FREE_COLOR, "white = free/passable space"),
        (OCC_COLOR, "black = confirmed occupied obstacle"),
        (CONTACT_COLOR, "yellow = bumper-confirmed contact obstacle"),
        (HYPOTHESIS_OBSTACLE_COLOR, "orange = inferred obstacle / occlusion no-go"),
        (UNDER_SURFACE_COLOR, "pale yellow = passable under-furniture floor"),
        (CV_SEEN_COLOR, "light pink = RGB-D visual evidence"),
        (CV_DENSE_COLOR, "dark purple = dense CV boundary"),
        (TRAJ_COLOR, "orange line = robot trajectory"),
        (ROBOT_COLOR, "red = robot pose; green = camera/depth FOV"),
    ]
    if FURNITURE_ZONE_DRAW_OVERLAY:
        items.insert(8, (FURNITURE_ZONE_COLOR, "magenta = internal furniture/no-go island"))
    yy = ly + 45
    for color, label in items:
        cv2.rectangle(img, (lx + 12, yy - 10), (lx + 34, yy + 6), color, -1)
        cv2.rectangle(img, (lx + 12, yy - 10), (lx + 34, yy + 6), (50, 50, 50), 1)
        cv2.putText(img, label, (lx + 44, yy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        yy += 17

def draw_furniture_zone_overlay(img, alpha_inflated=0.36, draw_labels=True):
    """Draw detected internal furniture/complex zones on a map image."""
    if not (FURNITURE_ZONE_DETECTION_ENABLED and FURNITURE_ZONE_DRAW_OVERLAY):
        return img
    draw_labels = bool(draw_labels and FURNITURE_ZONE_DRAW_LABELS)
    try:
        core, inflated = furniture_zone_masks()
        if inflated is not None and np.any(inflated):
            overlay = img.copy()
            overlay[inflated] = FURNITURE_ZONE_INFLATED_COLOR
            img = cv2.addWeighted(overlay, alpha_inflated, img, 1.0 - alpha_inflated, 0)
        if core is not None and np.any(core):
            core_u8 = core.astype(np.uint8)
            contours, _hier = cv2.findContours(core_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours[:24]:
                if cnt is None or len(cnt) < 3:
                    continue
                cv2.drawContours(img, [cnt], -1, FURNITURE_ZONE_COLOR, 2)
                if draw_labels:
                    m = cv2.moments(cnt)
                    if abs(m.get('m00', 0.0)) > 1e-6:
                        cx = int(m['m10'] / m['m00'])
                        cy = int(m['m01'] / m['m00'])
                        cv2.putText(img, "furn", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, FURNITURE_ZONE_COLOR, 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def draw_hypothesis_obstacle_overlay(img, alpha=0.62, draw_labels=False):
    """Draw orange inferred-obstacle/occlusion-shadow cells."""
    if not OBSTACLE_HYPOTHESIS_ENABLED:
        return img
    try:
        hyp = hypothesis_obstacle_mask(force=False)
        if hyp is None or not np.any(hyp):
            return img
        overlay = img.copy()
        overlay[hyp] = HYPOTHESIS_OBSTACLE_COLOR
        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        hyp_u8 = hyp.astype(np.uint8)
        contours, _hier = cv2.findContours(hyp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[:40]:
            if cnt is None or len(cnt) < 3:
                continue
            cv2.drawContours(img, [cnt], -1, HYPOTHESIS_EDGE_COLOR, 1)
            if draw_labels and OBSTACLE_HYPOTHESIS_DRAW_LABELS:
                m = cv2.moments(cnt)
                if abs(m.get('m00', 0.0)) > 1e-6:
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    cv2.putText(img, "hyp", (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, HYPOTHESIS_EDGE_COLOR, 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def render_map(auto_crop=True):
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    unknown = np.abs(log_odds) <= LO_UNKNOWN_EPS
    free = log_odds < -LO_UNKNOWN_EPS
    contact_occ = contact_log_odds > CONTACT_OCCUPIED_EPS
    structural_occ = structural_log_odds > STRUCTURAL_OCCUPIED_EPS if STRUCTURAL_OBSTACLE_MEMORY_ENABLED else np.zeros_like(contact_occ, dtype=np.bool_)
    occ = (log_odds > LO_OCCUPIED_EPS) | contact_occ | structural_occ
    img[unknown] = UNKNOWN_COLOR
    img[free] = FREE_COLOR
    img[occ] = OCC_COLOR

    # RGB-D CV layer has two visual levels:
    # - light pink: RGB/OpenCV feature confirmed by depth, but not necessarily an obstacle;
    # - dark purple: dense/confirmed visual boundary, usually a wall/furniture edge.
    # This prevents the map from looking like a noisy lidar fan and makes the role
    # of computer vision explainable during the defense.
    visual_seen = visual_log_odds > CV_DISPLAY_LIGHT_EPS
    visual_dense = visual_log_odds > CV_DISPLAY_DENSE_EPS
    if np.any(visual_seen):
        overlay = img.copy()
        overlay[visual_seen] = CV_SEEN_COLOR
        img = cv2.addWeighted(overlay, 0.34, img, 0.66, 0)
    if np.any(visual_dense):
        overlay = img.copy()
        overlay[visual_dense] = CV_DENSE_COLOR
        img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)
    if np.any(contact_occ):
        overlay = img.copy()
        overlay[contact_occ] = CONTACT_COLOR
        img = cv2.addWeighted(overlay, 0.80, img, 0.20, 0)
    # Hypothesis overlay is deliberately separate from black confirmed obstacles.
    img = draw_hypothesis_obstacle_overlay(img, alpha=0.66, draw_labels=True)
    under_surface = under_surface_mask_from_obstacles(occ | visual_dense)
    if np.any(under_surface):
        overlay = img.copy()
        overlay[under_surface] = UNDER_SURFACE_COLOR
        img = cv2.addWeighted(overlay, 0.34, img, 0.66, 0)

    img = draw_furniture_zone_overlay(img, alpha_inflated=0.34, draw_labels=True)

    # Do NOT paint the cleaned footprint on the main occupancy/CV map.
    # The main map is for geometry: unknown/free/obstacle/CV evidence + trajectory.
    # Cleaned area is visualized only in the separate Coverage objective map,
    # otherwise the two windows duplicate the same colored footprint and the
    # RGB-D CV layer becomes harder to read.

    # Metric grid, 1 m.
    for gx in np.arange(-2, 14, 1.0):
        x1, y1 = world_to_map(gx, -7)
        x2, y2 = world_to_map(gx, 7)
        cv2.line(img, (x1, y1), (x2, y2), GRID_COLOR, 1)
    for gy in np.arange(-7, 8, 1.0):
        x1, y1 = world_to_map(-2, gy)
        x2, y2 = world_to_map(14, gy)
        cv2.line(img, (x1, y1), (x2, y2), GRID_COLOR, 1)

    draw_debug_arena_bounds(img)

    for i in range(1, len(trajectory)):
        cv2.line(img, trajectory[i-1], trajectory[i], TRAJ_COLOR, 2)

    mx, my = world_to_map(pose_x, pose_y)
    if map_inside(mx, my):
        # FOV rays. The side labels are deliberately based on robot-frame
        # left/right, not raw image columns; this helps catch mirror mistakes.
        fov_rays = [(-RF_FOV/2, "R"), (0, ""), (RF_FOV/2, "L")]
        for a, label in fov_rays:
            # positive robot-frame bearing means local +Y, i.e. left. Since
            # image-right was inverted in project_depth_point, the drawn FOV
            # remains physically left/right correct around the robot heading.
            ex, ey = world_to_map(pose_x + 1.8 * math.cos(pose_theta + a), pose_y + 1.8 * math.sin(pose_theta + a))
            cv2.line(img, (mx, my), (ex, ey), FOV_COLOR, 1)
            if label:
                cv2.putText(img, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.45, FOV_COLOR, 1)
        cv2.circle(img, (mx, my), 8, ROBOT_COLOR, -1)
        hx = int(mx + 25 * math.cos(pose_theta))
        hy = int(my - 25 * math.sin(pose_theta))
        cv2.arrowedLine(img, (mx, my), (hx, hy), (0, 0, 180), 2, tipLength=0.35)

    # Visual-only candidate target.  Do not draw it like an active waypoint: the
    # real wheel owner is shown separately as activeTarget on the coverage map/HUD.
    if coverage_goal_map is not None:
        gx, gy = coverage_goal_map
        if map_inside(gx, gy):
            cv2.circle(img, (gx, gy), 5, CANDIDATE_TARGET_COLOR, 1)
            cv2.drawMarker(img, (gx, gy), CANDIDATE_TARGET_COLOR, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
            cv2.putText(img, "cand:" + str(coverage_goal_kind), (gx + 7, gy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CANDIDATE_TARGET_COLOR, 1)

    if RETURN_HOME_ENABLED:
        dxm, dym = world_to_map(DOCK_TARGET_X, DOCK_TARGET_Y)
        if map_inside(dxm, dym):
            cv2.drawMarker(img, (dxm, dym), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=18, thickness=2)
            cv2.putText(img, "dock", (dxm + 10, dym - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 0, 0), 1)

    # Debug/HUD text is drawn after auto-crop and resize by draw_map_hud_and_legend().

    crop_points = []
    if auto_crop:
        hyp_crop = hypothesis_obstacle_mask(False) if OBSTACLE_HYPOTHESIS_ENABLED else np.zeros_like(unknown, dtype=np.bool_)
        known = np.where((~unknown) | hyp_crop)
        if known[0].size > 0:
            crop_points.append((known[1].min(), known[0].min(), known[1].max(), known[0].max()))
        if trajectory:
            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]
            crop_points.append((min(xs), min(ys), max(xs), max(ys)))
        append_debug_arena_crop_point(crop_points)
    img = apply_debug_map_crop(img, crop_points, auto_crop=auto_crop)

    img = cv2.resize(img, (MAP_VIEW_W, MAP_VIEW_H), interpolation=cv2.INTER_AREA)
    draw_map_hud_and_legend(img)
    return img


def render_coverage_planner_map(auto_crop=True):
    """Render the high-level cleaning objective map.

    This is separate from the raw occupancy/CV map: it explains the robot's
    mission goal — cover all reachable known space, remember cleaned cells, and
    seek remaining uncleaned/frontier regions.
    """
    obstacles, cleanable, cleaned, uncleaned, unknown = compute_coverage_masks()

    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    img[unknown] = UNKNOWN_COLOR
    img[cleanable] = FREE_COLOR
    img[uncleaned] = UNCLEANED_COLOR
    img[cleaned] = CLEANED_COLOR
    under_surface = under_surface_mask_from_obstacles(obstacles) & cleanable
    under_surface_uncleaned = under_surface & uncleaned
    if np.any(under_surface_uncleaned):
        overlay = img.copy()
        overlay[under_surface_uncleaned] = UNDER_SURFACE_COLOR
        img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
    img[obstacles] = OCC_COLOR
    img = draw_hypothesis_obstacle_overlay(img, alpha=0.70, draw_labels=False)
    img = draw_furniture_zone_overlay(img, alpha_inflated=0.42, draw_labels=True)

    # Frontier cells are unknown cells adjacent to known cleanable space.
    cleanable_u8 = cleanable.astype(np.uint8)
    frontier = unknown & (cv2.dilate(cleanable_u8, np.ones((7, 7), np.uint8), iterations=1) > 0)
    if np.any(frontier):
        overlay = img.copy()
        overlay[frontier] = FRONTIER_COLOR
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    # Metric grid.
    for gxw in np.arange(-2, 14, 1.0):
        x1, y1 = world_to_map(gxw, -7)
        x2, y2 = world_to_map(gxw, 7)
        cv2.line(img, (x1, y1), (x2, y2), GRID_COLOR, 1)
    for gyw in np.arange(-7, 8, 1.0):
        x1, y1 = world_to_map(-2, gyw)
        x2, y2 = world_to_map(14, gyw)
        cv2.line(img, (x1, y1), (x2, y2), GRID_COLOR, 1)

    draw_debug_arena_bounds(img)

    for i in range(1, len(trajectory)):
        cv2.line(img, trajectory[i-1], trajectory[i], TRAJ_COLOR, 2)

    # Path-aware coverage route. Magenta polyline is the coarse wavefront route;
    # orange dot is the immediate waypoint used for side/lane decisions. This makes
    # the planner explainable: the target is not just a diagonal straight-line lure.
    if coverage_route_map and len(coverage_route_map) >= 2:
        pts = [(int(x), int(y)) for x, y in coverage_route_map if map_inside(int(x), int(y))]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], ROUTE_COLOR, 1)
    if coverage_route_waypoint_map is not None:
        wxp, wyp = coverage_route_waypoint_map
        if map_inside(wxp, wyp):
            cv2.circle(img, (int(wxp), int(wyp)), 6, ROUTE_WAYPOINT_COLOR, -1)

    if coverage_goal_map is not None:
        gx, gy = coverage_goal_map
        if map_inside(gx, gy):
            cv2.circle(img, (gx, gy), 7, CANDIDATE_TARGET_COLOR, 1)
            cv2.drawMarker(img, (gx, gy), CANDIDATE_TARGET_COLOR, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
            cv2.putText(img, "cand:" + str(coverage_goal_kind), (gx + 10, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CANDIDATE_TARGET_COLOR, 1)

    if route_commit_active and route_commit_route_map and len(route_commit_route_map) >= 2:
        pts = [(int(x), int(y)) for x, y in route_commit_route_map if map_inside(int(x), int(y))]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], ACTIVE_TARGET_COLOR, 2)
    if route_commit_active and route_commit_target_map is not None:
        ax, ay = route_commit_target_map
        if map_inside(int(ax), int(ay)):
            cv2.circle(img, (int(ax), int(ay)), 13, ACTIVE_TARGET_COLOR, 2)
            cv2.drawMarker(img, (int(ax), int(ay)), ACTIVE_TARGET_COLOR, markerType=cv2.MARKER_TILTED_CROSS, markerSize=26, thickness=2)
            cv2.putText(img, "active:" + str(route_commit_kind), (int(ax) + 12, int(ay) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ACTIVE_TARGET_COLOR, 1)

    if RETURN_HOME_ENABLED:
        dxm, dym = world_to_map(DOCK_TARGET_X, DOCK_TARGET_Y)
        if map_inside(dxm, dym):
            cv2.drawMarker(img, (dxm, dym), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=20, thickness=2)
            cv2.putText(img, "dock", (dxm + 12, dym - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)

    # Draw 1:1 predicted robot footprint.  The circle is the real body radius in
    # map scale, not a symbolic dot.  Small footprints along the magenta route show
    # whether the planned corridor can physically accept the round body.
    if FOOTPRINT_PREDICTION_ENABLED and coverage_route_map and len(coverage_route_map) >= 2:
        r_px = max(2, int(round(FOOTPRINT_RADIUS_M * MAP_SCALE)))
        for idx, (rxp, ryp) in enumerate(coverage_route_map[::max(1, FOOTPRINT_ROUTE_DRAW_STRIDE)]):
            if not map_inside(int(rxp), int(ryp)):
                continue
            if idx > 14:
                break
            ok, _ratio = footprint_fits_at(int(rxp), int(ryp), obstacles, cleanable, margin_m=FOOTPRINT_ROUTE_MARGIN_M)
            cv2.circle(img, (int(rxp), int(ryp)), r_px, FOOTPRINT_OK_COLOR if ok else FOOTPRINT_BAD_COLOR, 1)

    mx, my = world_to_map(pose_x, pose_y)
    if map_inside(mx, my):
        r_px = max(2, int(round(FOOTPRINT_RADIUS_M * MAP_SCALE)))
        ok_here, ratio_here = footprint_fits_at(mx, my, obstacles, cleanable, margin_m=FOOTPRINT_ROUTE_MARGIN_M)
        cv2.circle(img, (mx, my), r_px, FOOTPRINT_COLOR if ok_here else FOOTPRINT_BAD_COLOR, 2)
        cv2.circle(img, (mx, my), 8, ROBOT_COLOR, -1)
        sx = int(mx + SENSOR_OFFSET_X * MAP_SCALE * math.cos(pose_theta) - SENSOR_OFFSET_Y * MAP_SCALE * math.sin(pose_theta))
        sy = int(my - (SENSOR_OFFSET_X * MAP_SCALE * math.sin(pose_theta) + SENSOR_OFFSET_Y * MAP_SCALE * math.cos(pose_theta)))
        if map_inside(sx, sy):
            cv2.circle(img, (sx, sy), 4, (0, 0, 0), -1)
        hx = int(mx + 25 * math.cos(pose_theta))
        hy = int(my - 25 * math.sin(pose_theta))
        cv2.arrowedLine(img, (mx, my), (hx, hy), (0, 0, 180), 2, tipLength=0.35)

    crop_points = []
    if auto_crop:
        known = np.where(cleanable | obstacles | cleaned | frontier)
        if known[0].size > 0:
            crop_points.append((known[1].min(), known[0].min(), known[1].max(), known[0].max()))
        if trajectory:
            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]
            crop_points.append((min(xs), min(ys), max(xs), max(ys)))
        if coverage_goal_map is not None:
            gx, gy = coverage_goal_map
            crop_points.append((gx, gy, gx, gy))
        append_debug_arena_crop_point(crop_points)
    img = apply_debug_map_crop(img, crop_points, auto_crop=auto_crop)

    map_img = cv2.resize(img, (MAP_VIEW_W, MAP_VIEW_H), interpolation=cv2.INTER_AREA)

    # Keep a compact HUD outside the map.  The older verbose debug block used
    # seven dense text rows and made the planner view look noisy during demos.
    # Detailed diagnostics are still available in the console; the visual map now
    # keeps only mission-critical status: coverage, mode, owner, active target and dock.
    hud_h = 92
    canvas = np.full((MAP_VIEW_H + hud_h, MAP_VIEW_W, 3), 238, dtype=np.uint8)
    canvas[hud_h:hud_h + MAP_VIEW_H, :, :] = map_img
    cv2.rectangle(canvas, (0, 0), (MAP_VIEW_W - 1, hud_h - 1), (246, 246, 246), -1)
    cv2.rectangle(canvas, (0, 0), (MAP_VIEW_W - 1, hud_h - 1), (95, 95, 95), 1)

    mode_text = f"{planner_mode} | {known_map_eval_status[:24]} | {map_view_status_text()}"
    cv2.putText(canvas,
                f"Coverage: {last_coverage_percent:.1f}% ({last_coverage_cleaned_cells}/{last_coverage_total_cells}) | conf={planner_confidence} intent={planner_intent} | {mode_text}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"phase={navigation_phase} | nav={nav_state} | owner={last_control_owner_debug[:18]} | {owner_source_debug()[:22]} | {last_mapping_write_debug[:30]} | {load_shedding_debug_text()[:18]}",
                (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"candidate={route_commit_candidate_debug()} | active={route_commit_active_target_debug()} | route={coverage_route_status[:34]} | dock={dock_return_status[:18]} | {auto_map_ready_debug[:24]}",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_map():
    path = maps_dir / f"occupancy_map_{int(robot.getTime()*1000):07d}.png"
    cv2.imwrite(str(path), render_map(auto_crop=False))
    planner_path = maps_dir / f"coverage_objective_map_{int(robot.getTime()*1000):07d}.png"
    cv2.imwrite(str(planner_path), render_coverage_planner_map(auto_crop=False))
    print("Map saved:", path)
    print("Coverage objective map saved:", planner_path)
    print("Save is passive: motion owner remains", last_control_owner_debug, auto_map_ready_debug, learned_map_sanitize_debug)


def reset_map():
    global log_odds, visual_log_odds, thin_obstacle_log_odds, contact_log_odds, structural_log_odds, under_surface_log_odds, hypothesis_obstacle_log_odds
    global hypothesis_obstacle_cache, hypothesis_obstacle_cache_step, last_hypothesis_obstacle_debug, last_near_collision_hypothesis_time, last_near_collision_hypothesis_debug
    global trajectory, cleaned_mask, recent_visit_log_odds, row_start_x, row_start_y, row_start_time
    global coverage_goal_map, coverage_goal_world, coverage_goal_kind, last_contact_map_cells, last_under_surface_cells, last_under_surface_target_cells
    global last_bumper_any_raw_active, last_contact_latch_time, last_contact_latch_used
    global under_furniture_confirm_count, last_contact_route_kill_until, last_low_obstacle_guard_time, last_contact_escape_kind, last_rear_guard_clearance, last_rear_guard_reason
    global last_contact_cluster_time, last_contact_cluster_x, last_contact_cluster_y, last_contact_cluster_count
    global last_narrow_passage_time, last_narrow_passage_start_time, last_narrow_passage_side, last_narrow_passage_reason
    global last_gap_mouth_align_time, last_gap_mouth_align_side, last_gap_mouth_align_reason, post_gap_commit_until, post_gap_commit_start_x, post_gap_commit_start_y, lane_shift_heading_target
    global grid_realign_target, grid_realign_until, grid_realign_after_status, last_unreachable_target_trim_cells
    global grid_realign_start_time, grid_realign_best_abs_error, grid_realign_last_progress_time, grid_realign_soft_finish, grid_realign_retry_count
    global post_recovery_stabilize_until, post_recovery_stabilize_start_x, post_recovery_stabilize_start_y
    global post_recovery_stabilize_heading, post_recovery_stabilize_distance, post_recovery_stabilize_kind
    global contact_recovery_kind, contact_recovery_reason, contact_recovery_side, contact_recovery_start_x, contact_recovery_start_y
    global contact_recovery_backup_distance, contact_recovery_backup_until, contact_recovery_clear_since, contact_recovery_wait_until
    global contact_recovery_turn_angle, contact_recovery_turn_target, contact_recovery_forward_start_x, contact_recovery_forward_start_y
    global contact_recovery_forward_distance, contact_recovery_forward_speed, contact_recovery_forward_until, contact_recovery_forward_timeout
    global contact_recovery_replan_after, contact_recovery_bumper_hold_since
    global coverage_route_map, coverage_route_world, coverage_route_waypoint_map, coverage_route_waypoint_world, coverage_route_cost, coverage_route_score, coverage_route_len, coverage_route_kind, coverage_route_component_id, coverage_route_component_cells, coverage_route_status, last_residual_route_deferred_cells
    global coverage_route_top_debug, coverage_route_commit_class, coverage_route_straight_dist, coverage_route_lateral_abs, coverage_route_turn_need, coverage_route_continuity_bonus
    global coverage_route_wall_strip_bonus, coverage_route_missed_strip_bonus, coverage_route_segment_gain, coverage_route_segment_bonus, coverage_route_footprint_gain_cells
    global coverage_route_first_turn_frac, coverage_route_corner_count, coverage_route_geometry_debug
    global last_missed_strip_recovery_debug, last_rejected_components_debug, last_missed_strip_promoted_cells
    global map_mature, map_mature_soft, map_mature_reason, planner_confidence, planner_mode, planner_intent, planner_intent_reason, route_commit_active, route_commit_target_map, route_commit_target_world, route_commit_kind
    global route_commit_route_map, route_commit_route_world, route_commit_waypoint_map, route_commit_waypoint_world
    global route_commit_cost, route_commit_score, route_commit_component_id, route_commit_component_cells
    global route_commit_wall_strip_bonus, route_commit_missed_strip_bonus, route_commit_segment_gain, route_commit_footprint_gain_cells
    global route_commit_first_turn_frac, route_commit_corner_count, route_commit_geometry_debug
    global route_commit_started_at, route_commit_last_abort_time, route_commit_reason, route_abort_reason, last_route_commit_debug
    global route_commit_best_target_dist, route_commit_last_progress_time, route_commit_target_blacklist, last_route_target_blacklist_debug
    global dock_return_active, dock_return_completed, dock_return_reason, dock_return_status, dock_return_last_plan_time, dock_return_last_start_time, dock_route_cost
    global auto_map_mission_phase, auto_map_return_to_dock_active, auto_map_cleaning_started, auto_map_final_dock_requested
    global auto_map_ready_best_coverage, auto_map_ready_best_time, auto_map_ready_debug, learned_map_sanitized_once, learned_map_sanitize_debug
    global last_gray_gap_cells, last_gray_gap_components, last_gray_gap_debug
    global simple_sweep_best_coverage_percent, simple_sweep_best_coverage_time, simple_sweep_best_coverage_x, simple_sweep_best_coverage_y
    global last_side_strip_cleanup_time, last_side_strip_cleanup_reason
    global last_nearby_route_cleanup_time, last_nearby_route_cleanup_reason
    global last_line_acquire_time, last_line_acquire_side, last_line_acquire_reason
    global last_parallel_aperture_time, last_parallel_aperture_count, last_parallel_aperture_side, last_parallel_aperture_score, last_parallel_aperture_blocked_ratio
    global last_parallel_aperture_x, last_parallel_aperture_y
    global nav_state
    global active_scan_target_yaws, active_scan_index, active_scan_dwell_until, active_scan_started_at
    global active_scan_started_x, active_scan_started_y, active_scan_reason, active_scan_last_completed_at
    global active_scan_last_x, active_scan_last_y, active_scan_start_frontier_local, active_scan_start_unknown_local
    global active_scan_area_memory, last_active_scan_debug, last_rgbd_occlusion_debug, last_exploration_cleanup_lock_debug, last_exploration_route_debug
    global post_turn_rgbd_snapshot_until, post_turn_rgbd_snapshot_started_at, post_turn_rgbd_snapshot_frames, post_turn_rgbd_snapshot_write_frames, post_turn_rgbd_snapshot_pending, last_post_turn_snapshot_debug
    global simple_sweep_churn_anchor_x, simple_sweep_churn_anchor_y, simple_sweep_churn_start_time
    global simple_sweep_churn_action_count, simple_sweep_churn_last_debug
    global last_owner_source_debug
    last_owner_source_debug = "ownerSource=reset"
    log_odds[:, :] = 0.0
    visual_log_odds[:, :] = 0.0
    thin_obstacle_log_odds[:, :] = 0.0
    contact_log_odds[:, :] = 0.0
    structural_log_odds[:, :] = 0.0
    under_surface_log_odds[:, :] = 0.0
    hypothesis_obstacle_log_odds[:, :] = 0.0
    hypothesis_obstacle_cache = None
    hypothesis_obstacle_cache_step = -999999
    last_hypothesis_obstacle_debug = "hypObs=reset"
    last_near_collision_hypothesis_time = -999.0
    last_near_collision_hypothesis_debug = "nearHyp=reset"
    last_contact_map_cells = 0
    last_bumper_any_raw_active = False
    last_contact_latch_time = -999.0
    last_contact_latch_used = False
    last_under_surface_cells = 0
    last_under_surface_target_cells = 0
    under_furniture_confirm_count = 0
    last_contact_route_kill_until = -999.0
    last_low_obstacle_guard_time = -999.0
    last_contact_escape_kind = "none"
    last_rear_guard_clearance = 999.0
    last_rear_guard_reason = "clear"
    last_contact_cluster_time = -999.0
    last_contact_cluster_x = 1e9
    last_contact_cluster_y = 1e9
    last_contact_cluster_count = 0
    last_narrow_passage_time = -999.0
    last_narrow_passage_start_time = -999.0
    last_narrow_passage_side = 0.0
    last_narrow_passage_reason = ""
    last_gap_mouth_align_time = -999.0
    last_gap_mouth_align_side = 0.0
    last_gap_mouth_align_reason = ""
    post_gap_commit_until = -999.0
    post_gap_commit_start_x = 0.0
    post_gap_commit_start_y = 0.0
    lane_shift_heading_target = 0.0
    grid_realign_target = 0.0
    grid_realign_until = 0.0
    grid_realign_after_status = "row forward after grid realign"
    grid_realign_start_time = -999.0
    grid_realign_best_abs_error = float("inf")
    grid_realign_last_progress_time = -999.0
    grid_realign_soft_finish = False
    grid_realign_retry_count = 0
    post_recovery_stabilize_until = -999.0
    post_recovery_stabilize_start_x = 0.0
    post_recovery_stabilize_start_y = 0.0
    post_recovery_stabilize_heading = 0.0
    post_recovery_stabilize_distance = POST_RECOVERY_STABILIZE_DISTANCE_M
    post_recovery_stabilize_kind = "none"
    contact_recovery_kind = "none"
    contact_recovery_reason = ""
    contact_recovery_side = 1.0
    contact_recovery_start_x = 0.0
    contact_recovery_start_y = 0.0
    contact_recovery_backup_distance = FRONT_CONTACT_TRAP_BACKUP_DISTANCE
    contact_recovery_backup_until = 0.0
    contact_recovery_clear_since = -999.0
    contact_recovery_wait_until = 0.0
    contact_recovery_turn_angle = FRONT_CONTACT_TRAP_TURN_ANGLE
    contact_recovery_turn_target = 0.0
    contact_recovery_forward_start_x = 0.0
    contact_recovery_forward_start_y = 0.0
    contact_recovery_forward_distance = FRONT_CONTACT_TRAP_FORWARD_DISTANCE
    contact_recovery_forward_speed = CONTACT_ESCAPE_VERIFY_SPEED
    contact_recovery_forward_until = 0.0
    contact_recovery_forward_timeout = FRONT_CONTACT_TRAP_FORWARD_TIMEOUT_SEC
    contact_recovery_replan_after = False
    contact_recovery_bumper_hold_since = -999.0
    last_unreachable_target_trim_cells = 0
    last_parallel_aperture_time = -999.0
    last_parallel_aperture_count = 0
    last_parallel_aperture_side = 0.0
    last_parallel_aperture_score = 0
    last_parallel_aperture_blocked_ratio = 1.0
    last_parallel_aperture_x = 1e9
    last_parallel_aperture_y = 1e9
    active_scan_target_yaws = []
    active_scan_index = 0
    active_scan_dwell_until = -999.0
    active_scan_started_at = -999.0
    active_scan_started_x = 0.0
    active_scan_started_y = 0.0
    active_scan_reason = "none"
    active_scan_last_completed_at = -999.0
    active_scan_last_x = 1e9
    active_scan_last_y = 1e9
    last_active_scan_debug = "scan=reset"
    last_rgbd_occlusion_debug = "occGuard=reset"
    nav_state = NAV_FORWARD
    cleaned_mask[:, :] = 0
    recent_visit_log_odds[:, :] = 0.0
    if known_map_coverage_eval_active() and KNOWN_MAP_EVAL_RESEED_EVERY_RESET:
        seed_known_map_coverage_eval(reason="reset")
    trajectory = []
    coverage_goal_map = None
    coverage_goal_world = None
    coverage_goal_kind = "none"
    coverage_route_map = []
    coverage_route_world = []
    coverage_route_waypoint_map = None
    coverage_route_waypoint_world = None
    coverage_route_cost = float("inf")
    coverage_route_score = float("-inf")
    coverage_route_len = 0
    coverage_route_kind = "none"
    coverage_route_component_id = -1
    coverage_route_component_cells = 0
    coverage_route_status = "reset"
    coverage_route_top_debug = "none"
    coverage_route_commit_class = "none"
    coverage_route_straight_dist = float("inf")
    coverage_route_lateral_abs = float("inf")
    coverage_route_turn_need = float("inf")
    coverage_route_continuity_bonus = 0.0
    coverage_route_wall_strip_bonus = 0.0
    coverage_route_missed_strip_bonus = 0.0
    coverage_route_segment_gain = 0
    coverage_route_segment_bonus = 0.0
    coverage_route_footprint_gain_cells = 0
    coverage_route_first_turn_frac = 0.0
    coverage_route_corner_count = 0
    coverage_route_geometry_debug = "geom=none"
    map_mature = False
    map_mature_soft = False
    map_mature_reason = "reset"
    planner_confidence = "EARLY"
    planner_mode = "KNOWN_MAP_COVERAGE_EVAL" if known_map_coverage_eval_active() else "ROW_COVERAGE"
    planner_intent = PLANNER_INTENT_CLEAN_KNOWN if known_map_coverage_eval_active() else PLANNER_INTENT_EXPAND_MAP
    planner_intent_reason = "known-map reset" if known_map_coverage_eval_active() else "reset"
    last_exploration_cleanup_lock_debug = "cleanupLock=reset"
    last_exploration_route_debug = "exploreRoute=reset"
    active_scan_target_yaws = []
    active_scan_index = 0
    active_scan_dwell_until = -999.0
    active_scan_started_at = -999.0
    active_scan_last_completed_at = -999.0
    active_scan_last_x = 1e9
    active_scan_last_y = 1e9
    active_scan_start_frontier_local = 0
    active_scan_start_unknown_local = 0
    active_scan_area_memory = []
    last_active_scan_debug = "scan=reset"
    post_turn_rgbd_snapshot_until = -999.0
    post_turn_rgbd_snapshot_started_at = -999.0
    post_turn_rgbd_snapshot_frames = 0
    post_turn_rgbd_snapshot_write_frames = 0
    post_turn_rgbd_snapshot_pending = False
    last_post_turn_snapshot_debug = "snap=reset"
    simple_sweep_churn_anchor_x = pose_x
    simple_sweep_churn_anchor_y = pose_y
    simple_sweep_churn_start_time = -999.0
    simple_sweep_churn_action_count = 0
    simple_sweep_churn_last_debug = "churn=reset map"
    simple_sweep_best_coverage_percent = 0.0
    simple_sweep_best_coverage_time = -999.0
    simple_sweep_best_coverage_x = pose_x
    simple_sweep_best_coverage_y = pose_y
    route_commit_active = False
    route_commit_target_map = None
    route_commit_target_world = None
    route_commit_kind = "none"
    route_commit_route_map = []
    route_commit_route_world = []
    route_commit_waypoint_map = None
    route_commit_waypoint_world = None
    route_commit_cost = float("inf")
    route_commit_score = float("-inf")
    route_commit_component_id = -1
    route_commit_component_cells = 0
    route_commit_wall_strip_bonus = 0.0
    route_commit_missed_strip_bonus = 0.0
    route_commit_segment_gain = 0
    route_commit_footprint_gain_cells = 0
    route_commit_first_turn_frac = 0.0
    route_commit_corner_count = 0
    route_commit_geometry_debug = "geom=none"
    route_commit_started_at = -999.0
    route_commit_last_abort_time = -999.0
    route_commit_best_target_dist = float("inf")
    route_commit_last_progress_time = -999.0
    route_commit_target_blacklist = []
    last_route_target_blacklist_debug = "none"
    route_commit_reason = "none"
    route_abort_reason = "none"
    last_route_commit_debug = "reset"
    dock_return_active = False
    dock_return_completed = False
    dock_return_reason = "none"
    dock_return_status = "reset"
    dock_return_last_plan_time = -999.0
    dock_return_last_start_time = -999.0
    dock_route_cost = float("inf")
    auto_map_mission_phase = "EXPLORE_MAP"
    auto_map_return_to_dock_active = False
    auto_map_cleaning_started = False
    auto_map_final_dock_requested = False
    auto_map_ready_best_coverage = -1.0
    auto_map_ready_best_time = -999.0
    auto_map_ready_debug = "autoMap=reset"
    last_gray_gap_cells = 0
    last_gray_gap_components = 0
    last_gray_gap_debug = "grayGap=reset"
    global frontier_only_stall_since, frontier_only_stall_blacklists, frontier_only_last_blacklist_time, frontier_only_last_stall_debug
    frontier_only_stall_since = -999.0
    frontier_only_stall_blacklists = 0
    frontier_only_last_blacklist_time = -999.0
    frontier_only_last_stall_debug = "stall=reset"
    learned_map_sanitized_once = False
    learned_map_sanitize_debug = "learnedMap=raw"
    last_nearby_route_cleanup_time = -999.0
    last_nearby_route_cleanup_reason = ""
    last_side_strip_cleanup_time = -999.0
    last_side_strip_cleanup_reason = ""
    last_residual_route_deferred_cells = 0
    last_line_acquire_time = -999.0
    last_line_acquire_side = 0.0
    last_line_acquire_reason = ""
    row_start_x = pose_x
    row_start_y = pose_y
    row_start_time = robot.getTime()
    refresh_navigation_phase()
    invalidate_heavy_map_caches("reset")
    print("Map reset")

# ---------------- Main loop ----------------
if SHOW_WINDOWS:
    if ASYNC_DEBUG_VIEWER_ENABLED:
        debug_viewer_client = AsyncDebugViewerClient(Path(__file__).with_name("async_debug_viewer.py"), enabled=True)
        last_debug_viewer_status = "viewer=async started" if debug_viewer_client.alive() else "viewer=async failed"
        print(f"Async debug viewer: {last_debug_viewer_status}")
    else:
        if SHOW_RGB_DEBUG_WINDOW:
            cv2.namedWindow("RGB camera + depth debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RGB camera + depth debug", 480, 360)
        if SHOW_OCCUPANCY_MAP_WINDOW:
            cv2.namedWindow("Persistent occupancy map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Persistent occupancy map", MAP_VIEW_W, MAP_VIEW_H)
        if SHOW_COVERAGE_PLANNER_WINDOW:
            cv2.namedWindow("Coverage objective map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Coverage objective map", MAP_VIEW_W, MAP_VIEW_H + 132)

if known_map_coverage_eval_active():
    seed_known_map_coverage_eval(reason="startup")
    # Build the first objective from the known map before the robot moves.
    update_coverage_objective()
    refresh_navigation_phase()

while robot.step(timestep) != -1:
    loop_t0 = perf_start()
    pose = profiled_call("odom", update_odometry)
    t0 = perf_start()
    frame = read_camera_frame()
    depth = read_depth_image()
    perf_end("io", t0)
    if frame is None or depth is None:
        set_wheel_speeds(0, 0)
        continue

    profiled_call("clean", mark_cleaned_footprint, pose_x, pose_y)

    # Expensive diagnostics/mapping are throttled. Safety decisions still use
    # the fresh depth frame below in choose_motion_from_depth().
    if ORB_DEBUG_ENABLED and task_due(ORB_DEBUG_UPDATE_STEPS):
        profiled_call("orb", update_orb_debug, frame)

    hit_count, cv_hit_count, under_marked = run_mapping_stage(pose, frame, depth)
    append_current_trajectory_point()
    run_planner_stage()
    run_motion_stage(depth)
    save_frame_if_due(frame)

    run_window_stage(frame, depth)

    last_perf_loop_ms = perf_end("loop", loop_t0)
    last_perf_debug = perf_text(150)
    print_debug_status_if_due()

    step_id += 1

set_wheel_speeds(0, 0)
cv2.destroyAllWindows()
