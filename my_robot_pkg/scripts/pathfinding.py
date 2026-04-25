import math
import heapq
import numpy as np
import odometry
import lidar_processing
import costmap

WHEEL_RADIUS    = 0.15
WHEEL_BASE      = 0.58          # track width (y-distance between left/right wheels)
MAX_WHEEL_SPEED = 6.0           # rad/s  (motor limit)

# Derived top speeds
MAX_LINEAR  = MAX_WHEEL_SPEED * WHEEL_RADIUS          # 0.90 m/s
MAX_ANGULAR = (2.0 * MAX_WHEEL_SPEED * WHEEL_RADIUS) / WHEEL_BASE  # ~3.1 rad/s


# --- Goal ---
GOAL_TOLERANCE = 0.20           # meters, stop when this close to goal

# --- Turning ---
HEADING_TOLERANCE  = math.radians(15)   # below this, switch from TURN to DRIVE
TURN_COMMIT_ANGLE  = math.radians(30)   # above this heading error while driving, stop and turn
TURN_SPEED_FACTOR  = 0.45               # fraction of MAX_ANGULAR for in-place turns
TURN_P_GAIN        = 3.0                # proportional gain for turn speed (clamped)

# --- Driving ---
DRIVE_SPEED        = 0.20
DRIVE_SPEED_NEAR   = 0.10               # m/s when close to goal (< 0.8 m)
STEER_P_GAIN       = 2.5                # proportional steering correction while driving
SLOW_ZONE          = 0.80               # meters, slow down when this close to goal

# --- Pure pursuit ---
LOOKAHEAD_BASE     = 0.50               # minimum lookahead distance (meters)
LOOKAHEAD_SCALE    = 0.3                # lookahead grows with speed: base + scale * v

# --- Obstacle reaction ---
OBSTACLE_STOP_DIST  = 0.70              # front LiDAR distance, stop (meters)
OBSTACLE_SLOW_DIST  = 1.30              # front LiDAR distance, slow down
BACKUP_SPEED        = -0.30             # m/s during backup 
BACKUP_DURATION     = 25                # steps to back up (~0.8 s at 32 ms/step)

# --- A* global planner ---
REPLAN_INTERVAL     = 90                # steps between scheduled replans
MIN_REPLAN_COOLDOWN = 30                # minimum steps between ANY two replans
WAYPOINT_SPACING    = 3                 # keep every Nth A* cell (path thinning)
WAYPOINT_REACH_DIST = 0.35             # meters advance to next waypoint
ASTAR_BLOCKED_COST  = 45                

# --- Stuck detection ---
STUCK_STEPS         = 150               # steps without progress,  force replan
STUCK_DIST          = 0.20              # metres of movement needed to reset counter
STUCK_BACKUP_STEPS  = 40                # longer backup when truly stuck

# state 

_goal_x = 5.0
_goal_y = 3.0

_state = "IDLE"   # IDLE | TURNING | DRIVING | BACKUP | ARRIVED

# Global path
_global_path     = []     # list of (wx, wy) waypoints from A*
_waypoint_idx    = 0

# Replanning
_last_replan_step     = -REPLAN_INTERVAL
_last_any_replan_step = -MIN_REPLAN_COOLDOWN

# Stuck detection
_steps_since_progress = 0
_last_progress_pos    = (0.0, 0.0)

# Backup maneuver
_backup_counter = 0

# For visualization / status
_current_v     = 0.0
_current_omega = 0.0


# public

def setup(goal_x=5.0, goal_y=3.0):
    global _goal_x, _goal_y
    _goal_x = goal_x
    _goal_y = goal_y


def start():
    global _state, _global_path, _waypoint_idx
    global _last_replan_step, _last_any_replan_step
    global _steps_since_progress, _last_progress_pos, _backup_counter
    rx, ry, _ = odometry.get_pose()
    _state             = "TURNING"
    _global_path       = []
    _waypoint_idx      = 0
    _last_replan_step        = -REPLAN_INTERVAL
    _last_any_replan_step    = -MIN_REPLAN_COOLDOWN
    _steps_since_progress    = 0
    _last_progress_pos       = (rx, ry)
    _backup_counter          = 0
    print(f"[NAV] Navigation started → goal ({_goal_x:.2f}, {_goal_y:.2f})")


def update(step):
    """Called every timestep.  Returns (left_wheel_speed, right_wheel_speed)."""
    global _state, _backup_counter, _current_v, _current_omega

    rx, ry, rh = odometry.get_pose()

    # Goal check 
    dist_to_goal = math.hypot(_goal_x - rx, _goal_y - ry)
    if dist_to_goal < GOAL_TOLERANCE:
        if _state != "ARRIVED":
            _state = "ARRIVED"
            print(f"\n[NAV] ★ Goal reached at ({rx:.3f}, {ry:.3f})")
        return (0.0, 0.0)

    if _state in ("IDLE", "ARRIVED"):
        return (0.0, 0.0)

    # Global planner (A*) 
    _maybe_replan(rx, ry, step)

    # Get target point (pure pursuit lookahead) 
    target_x, target_y = _get_pursuit_target(rx, ry)

    # Heading error to target 
    desired_heading = math.atan2(target_y - ry, target_x - rx)
    heading_error   = _normalize_angle(desired_heading - rh)

    # Front obstacle check 
    front_dist = lidar_processing.get_front_distance()

    # BACKUP state 
    if _state == "BACKUP":
        _backup_counter -= 1
        if _backup_counter <= 0:
            _state = "TURNING"
            _force_replan(rx, ry, step, "post-backup")
        v, omega = BACKUP_SPEED, 0.0
        _current_v, _current_omega = v, omega
        return _to_wheel_speeds(v, omega)

    # Obstacle too close, initiate backup 
    if front_dist < OBSTACLE_STOP_DIST and _state != "BACKUP":
        _state = "BACKUP"
        _backup_counter = BACKUP_DURATION
        _current_v, _current_omega = BACKUP_SPEED, 0.0
        return _to_wheel_speeds(BACKUP_SPEED, 0.0)

    # TURNING state: spin in place until heading is good
    if _state == "TURNING":
        if abs(heading_error) < HEADING_TOLERANCE:
            _state = "DRIVING"
            # fall through to DRIVING below
        else:
            # Proportional turn speed, clamped
            omega = TURN_P_GAIN * heading_error
            max_turn = TURN_SPEED_FACTOR * MAX_ANGULAR
            omega = max(-max_turn, min(max_turn, omega))
            # Minimum turn speed so we don't stall on friction
            min_omega = 0.4
            if abs(omega) < min_omega:
                omega = min_omega if heading_error > 0 else -min_omega
            _current_v, _current_omega = 0.0, omega
            return _to_wheel_speeds(0.0, omega)

    # DRIVING state: move forward with proportional steering
    if _state == "DRIVING":
        # If heading drifts too far, go back to TURNING
        if abs(heading_error) > TURN_COMMIT_ANGLE:
            _state = "TURNING"
            _current_v, _current_omega = 0.0, 0.0
            return _to_wheel_speeds(0.0, 0.0)

        # Speed selection
        if dist_to_goal < SLOW_ZONE:
            v = DRIVE_SPEED_NEAR
        elif front_dist < OBSTACLE_SLOW_DIST:
            # Slow down near obstacles but don't stop
            v = DRIVE_SPEED * (front_dist / OBSTACLE_SLOW_DIST)
            v = max(0.15, v)
        else:
            v = DRIVE_SPEED

        # Proportional steering
        omega = STEER_P_GAIN * heading_error

        # Check costmap along immediate path for safety
        if _check_immediate_path_blocked(rx, ry, rh, v):
            # Something in the way that LiDAR front cone might miss
            _state = "TURNING"
            _force_replan(rx, ry, step, "path-blocked")
            return _to_wheel_speeds(0.0, 0.0)

        _current_v, _current_omega = v, omega
        return _to_wheel_speeds(v, omega)

    return (0.0, 0.0)


# pursuit

def _get_pursuit_target(rx, ry):
    """Return the (x, y) point the robot should steer toward."""
    global _waypoint_idx

    if not _global_path:
        return (_goal_x, _goal_y)

    # Advance past waypoints we've already reached
    while _waypoint_idx < len(_global_path) - 1:
        wx, wy = _global_path[_waypoint_idx]
        if math.hypot(rx - wx, ry - wy) < WAYPOINT_REACH_DIST:
            _waypoint_idx += 1
        else:
            break

    # Compute lookahead distance
    lookahead = LOOKAHEAD_BASE + LOOKAHEAD_SCALE * abs(_current_v)

    # Walk along the path from current waypoint to find the lookahead point
    # Start from closest waypoint and look forward
    best_point = _global_path[-1]  # default: end of path

    for i in range(_waypoint_idx, len(_global_path)):
        wx, wy = _global_path[i]
        d = math.hypot(rx - wx, ry - wy)
        if d >= lookahead:
            best_point = (wx, wy)
            break
    else:
        # All waypoints are within lookahead, target the last one
        best_point = _global_path[-1]

    return best_point


# path safety check

def _check_immediate_path_blocked(rx, ry, rh, speed):
    """Check points ahead and slightly to the sides on the costmap."""
    check_dist = max(0.5, speed * 2.0)   # look ~2 seconds ahead
    for frac in (0.25, 0.5, 0.75, 1.0):
        cx = rx + check_dist * frac * math.cos(rh)
        cy = ry + check_dist * frac * math.sin(rh)
        if costmap.get_cost_at_world(cx, cy) >= costmap.COST_OCCUPIED:
            return True
        # Also check slightly left and right of center (robot width clearance)
        offset = 0.30  # about half robot width
        lx = cx + offset * math.cos(rh + math.pi/2)
        ly = cy + offset * math.sin(rh + math.pi/2)
        rx2 = cx + offset * math.cos(rh - math.pi/2)
        ry2 = cy + offset * math.sin(rh - math.pi/2)
        if (costmap.get_cost_at_world(lx, ly) >= costmap.COST_OCCUPIED or
            costmap.get_cost_at_world(rx2, ry2) >= costmap.COST_OCCUPIED):
            return True
    return False


# replan

def _maybe_replan(rx, ry, step):
    global _last_replan_step, _last_any_replan_step

    cooldown_ok = (step - _last_any_replan_step) >= MIN_REPLAN_COOLDOWN
    if not cooldown_ok:
        return

    is_stuck = _check_stuck(rx, ry)
    needs_replan = (
        len(_global_path) == 0
        or (step - _last_replan_step) >= REPLAN_INTERVAL
        or _is_path_blocked()
        or is_stuck
    )

    if needs_replan:
        reason = ("init"    if len(_global_path) == 0 else
                  "stuck"   if is_stuck               else
                  "blocked" if _is_path_blocked()      else
                  "periodic")
        _do_replan(rx, ry, step, reason)


def _force_replan(rx, ry, step, reason):
    global _last_any_replan_step
    _do_replan(rx, ry, step, reason)


def _do_replan(rx, ry, step, reason):
    global _last_replan_step, _last_any_replan_step
    print(f"[NAV] Replanning ({reason})")
    _plan_global_path(rx, ry)
    _last_replan_step     = step
    _last_any_replan_step = step


def _check_stuck(rx, ry):
    """Return True if the robot hasn't moved enough recently."""
    global _steps_since_progress, _last_progress_pos, _state, _backup_counter
    lx, ly = _last_progress_pos
    if math.hypot(rx - lx, ry - ly) > STUCK_DIST:
        _last_progress_pos    = (rx, ry)
        _steps_since_progress = 0
        return False
    _steps_since_progress += 1
    if _steps_since_progress >= STUCK_STEPS:
        _steps_since_progress = 0
        _last_progress_pos = (rx, ry)
        # Force a longer backup to escape
        if _state != "BACKUP":
            _state = "BACKUP"
            _backup_counter = STUCK_BACKUP_STEPS
        return True
    return False


def _is_path_blocked():
    """Return True if the next few waypoints are on occupied cells."""
    if not _global_path:
        return False
    end = min(_waypoint_idx + 4, len(_global_path))
    for wx, wy in _global_path[_waypoint_idx:end]:
        if costmap.get_cost_at_world(wx, wy) >= costmap.COST_OCCUPIED:
            return True
    return False


# A* global planner

def _plan_global_path(rx, ry):
    """Run A* on the costmap grid and store the resulting waypoints."""
    global _global_path, _waypoint_idx

    params = costmap.get_grid_params()
    res = params['resolution']
    ox  = params['origin_x']
    oy  = params['origin_y']
    W   = params['width']
    H   = params['height']

    def w2g(wx, wy):
        return (int((wx - ox) / res), int((wy - oy) / res))

    sx, sy = w2g(rx, ry)
    gx, gy = w2g(_goal_x, _goal_y)

    # Clamp to grid
    sx = max(0, min(W - 1, sx))
    sy = max(0, min(H - 1, sy))
    gx = max(0, min(W - 1, gx))
    gy = max(0, min(H - 1, gy))

    cell_path = _run_astar(sx, sy, gx, gy, W, H)

    if not cell_path:
        print("[NAV] A* found no path – aiming directly at goal")
        _global_path  = [(_goal_x, _goal_y)]
        _waypoint_idx = 0
        return

    # Thin the path and convert to world coords
    world_path = []
    for i, (cgx, cgy) in enumerate(cell_path):
        if i % WAYPOINT_SPACING == 0 or i == len(cell_path) - 1:
            wx = ox + (cgx + 0.5) * res
            wy = oy + (cgy + 0.5) * res
            world_path.append((wx, wy))

    # Always end exactly at the goal
    world_path.append((_goal_x, _goal_y))

    # Smooth the path to remove jagged A* artifacts
    world_path = _smooth_path(world_path)

    _global_path  = world_path
    _waypoint_idx = 0
    print(f"[NAV] A* path: {len(cell_path)} cells → {len(world_path)} waypoints")


def _run_astar(sx, sy, gx, gy, W, H):
    """A* on the costmap grid.  Returns list of (gx, gy) or []."""
    grid = costmap.get_grid()
    if grid is None:
        return []

    if sx == gx and sy == gy:
        return [(gx, gy)]

    # 8-directional moves
    DIRS = [
        (-1,  0, 1.0),  ( 1,  0, 1.0),  ( 0, -1, 1.0),  ( 0,  1, 1.0),
        (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414),
    ]

    open_heap = []
    g_score   = {}
    came_from = {}

    g_score[(sx, sy)] = 0.0
    h0 = math.hypot(gx - sx, gy - sy)
    heapq.heappush(open_heap, (h0, 0.0, sx, sy))

    while open_heap:
        f, g, cx, cy = heapq.heappop(open_heap)

        if cx == gx and cy == gy:
            path = []
            node = (cx, cy)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path

        if g > g_score.get((cx, cy), float('inf')) + 1e-6:
            continue

        for dx, dy, dc in DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            cell_cost = float(grid[ny, nx])
            if cell_cost >= ASTAR_BLOCKED_COST:
                continue

            # Penalty: prefer open space over cells near obstacles
            # Heavy penalty: strongly prefer wide-open space over skirting obstacles
            penalty = (cell_cost / costmap.COST_OCCUPIED) * 10.0
            ng = g + dc + penalty

            if ng < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                h = math.hypot(gx - nx, gy - ny)
                heapq.heappush(open_heap, (ng + h, ng, nx, ny))

    return []


def _smooth_path(path):
    """Simple path smoother: iteratively pull waypoints toward the line
    between their neighbours while checking the costmap for collisions.
    Removes jitter from the A* grid alignment."""
    if len(path) <= 2:
        return path

    smoothed = [p for p in path]  # copy
    weight_smooth = 0.3
    weight_data   = 0.5

    for _ in range(10):  # iterations (reduced from 30 for speed)
        for i in range(1, len(smoothed) - 1):
            ox, oy = smoothed[i]
            # Pull toward original
            nx = ox + weight_data * (path[i][0] - ox)
            ny = oy + weight_data * (path[i][1] - oy)
            # Pull toward midpoint of neighbours
            mx = (smoothed[i-1][0] + smoothed[i+1][0]) / 2.0
            my = (smoothed[i-1][1] + smoothed[i+1][1]) / 2.0
            nx += weight_smooth * (mx - ox)
            ny += weight_smooth * (my - oy)
            # Only accept if the new position is still in free space
            if costmap.get_cost_at_world(nx, ny) < ASTAR_BLOCKED_COST:
                smoothed[i] = (nx, ny)

    return smoothed


# helpers

def _normalize_angle(a):
    """Wrap angle to [-π, π]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def _to_wheel_speeds(v, omega):
    """Convert (linear velocity, angular velocity) to (left, right) wheel speeds."""
    left  = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    # Clamp
    left  = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left))
    right = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right))
    return (left, right)


# public getters

def get_path():
    """Return remaining waypoints for visualization."""
    if _global_path and _waypoint_idx < len(_global_path):
        rx, ry, _ = odometry.get_pose()
        return [(rx, ry)] + list(_global_path[_waypoint_idx:])
    rx, ry, _ = odometry.get_pose()
    return [(rx, ry), (_goal_x, _goal_y)]


def get_status():
    rx, ry, _ = odometry.get_pose()
    dist  = math.hypot(_goal_x - rx, _goal_y - ry)
    front = lidar_processing.get_front_distance()
    wp    = f"WP {_waypoint_idx}/{len(_global_path)}" if _global_path else "no path"
    return (f"{_state:10s} | Goal: {dist:.2f}m | Front: {front:.2f}m "
            f"| {wp} | v:{_current_v:.2f} ω:{_current_omega:.2f}")
