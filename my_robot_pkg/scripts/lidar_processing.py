import math
import random
import odometry
import numpy as np
from scipy.spatial import cKDTree

SENSOR_HEIGHT = 0.30        # LiDAR height above ground (m)

# Height classification thresholds
GROUND_MAX    = 0.08        # below this = ground
OBSTACLE_MAX  = 0.60        # above this = overhead

BODY_FILTER_RANGE  = 0.70   # secondary body filter: covers 0.60–0.70 m annular zone
BODY_FILTER_HEIGHT = 0.25   # max world height classified as chassis return in that zone

MAX_RANGE = 40.0
MIN_RANGE = 0.60            # self-return filter, robot body is within this radius

# VFH polar histogram
NUM_SECTORS   = 72          # 5 degrees per sector
OBSTACLE_DIST = 2.5

# Clustering
CLUSTER_EPS      = 0.30
CLUSTER_MIN_PTS  = 3
CLUSTER_INTERVAL = 6        # run clustering every N frames
MAX_CLUSTER_PTS  = 400

# Arena bounds in odometry frame (conservative)
ARENA_X_MIN = 0.0
ARENA_X_MAX = 7.24
ARENA_Y_MIN = -0.1
ARENA_Y_MAX =  4.60
WALL_MARGIN  = 0.50         # reject clusters within this distance of any wall

# Rock shape filters 
MAX_ROCK_DIAMETER  = 0.80
MIN_ROCK_DIAMETER  = 0.10
MIN_ROCK_Z_SPAN    = 0.04   # must have vertical extent (not flat ground noise)
MAX_ROCK_PLANARITY = 8.0    # 2D eigenvalue ratio, walls >15, rocks <8
MAX_ROCK_POINTS    = 80

# Landmark locking
# Requires MIN_LANDMARK_OBS total observations AND MIN_SPATIAL_OBS from
# spatially distinct rover positions.
LANDMARK_MATCH_DIST = 0.50
LANDMARK_MERGE_DIST = 0.40
MIN_LANDMARK_OBS    = 3
MIN_SPATIAL_OBS     = 2
MIN_OBS_SPACING     = 0.40  # min rover displacement between recorded obs positions
CORRECTION_GAIN     = 0.25
MATCH_MIN_LOCKED    = 2     # locked landmarks needed to fire a correction

FRONT_CONE_HALF_ANGLE = math.radians(25)
FRONT_CONE_MAX_DIST   = 3.0

_lidar           = None
_last_front_dist = float('inf')
_landmarks       = []
_frame_count     = 0

_sector_angles   = []
_sector_dists    = []
_scan_world      = []
_obstacle_pts    = []
_detected_rocks  = []


def setup(lidar_device=None):
    """Initialise the lidar_processing module.

    lidar_device: optional Webots device handle (used only in Webots mode).
                  Pass nothing (or None) when running under ROS 2.
    """
    global _lidar
    _lidar = lidar_device
    if _lidar is not None:
        try:
            _lidar.enablePointCloud()
        except Exception:
            pass
    print(f"[LIDAR] 3D mode, {NUM_SECTORS} sectors, "
          f"range [{MIN_RANGE:.2f}, {MAX_RANGE:.1f}]m, "
          f"obstacle height [>{GROUND_MAX:.2f}, {OBSTACLE_MAX:.2f}]m")
    print(f"[LIDAR] Arena bounds x=[{ARENA_X_MIN},{ARENA_X_MAX}] "
          f"y=[{ARENA_Y_MIN},{ARENA_Y_MAX}], wall margin={WALL_MARGIN}m")


def update(cloud):
    global _last_front_dist, _sector_angles, _sector_dists
    global _scan_world, _obstacle_pts, _detected_rocks, _frame_count

    _frame_count += 1

    if cloud is None or len(cloud) == 0:
        return

    rx, ry, rh = odometry.get_pose()
    cos_h = math.cos(rh)
    sin_h = math.sin(rh)

    do_cluster = (_frame_count % CLUSTER_INTERVAL == 0)

    if hasattr(cloud[0], 'x'):
        lxs = np.array([pt.x for pt in cloud], dtype=np.float32)
        lys = np.array([pt.y for pt in cloud], dtype=np.float32)
        lzs = np.array([pt.z for pt in cloud], dtype=np.float32)
    else:
        arr = np.array(cloud, dtype=np.float32)
        lxs, lys, lzs = arr[:, 0], arr[:, 1], arr[:, 2]

    valid = np.isfinite(lxs) & np.isfinite(lys) & np.isfinite(lzs)
    horiz_sq = lxs * lxs + lys * lys
    valid &= (horiz_sq >= MIN_RANGE * MIN_RANGE) & (horiz_sq <= MAX_RANGE * MAX_RANGE)

    world_z = SENSOR_HEIGHT + lzs
    valid &= (world_z <= OBSTACLE_MAX)

    lxs = lxs[valid]; lys = lys[valid]; lzs = lzs[valid]
    world_z = world_z[valid]; horiz_sq = horiz_sq[valid]

    # Rotate to world frame
    wxs = rx + lxs * cos_h - lys * sin_h
    wys = ry + lxs * sin_h + lys * cos_h

    ground_mask   = world_z < GROUND_MAX
    obstacle_mask = ~ground_mask

    # Remove close-range chassis returns
    body_mask = (horiz_sq < BODY_FILTER_RANGE * BODY_FILTER_RANGE) & (world_z < BODY_FILTER_HEIGHT)
    obstacle_mask &= ~body_mask

    gx_arr = wxs[ground_mask][::4]
    gy_arr = wys[ground_mask][::4]
    ox_arr = wxs[obstacle_mask]
    oy_arr = wys[obstacle_mask]

    scan_world = ([(float(x), float(y), 'ground') for x, y in zip(gx_arr, gy_arr)] +
                  [(float(x), float(y), 'obstacle') for x, y in zip(ox_arr, oy_arr)])
    obstacle_pts = list(zip(ox_arr.tolist(), oy_arr.tolist()))

    olxs = lxs[obstacle_mask]
    olys = lys[obstacle_mask]
    o_horiz_sq = horiz_sq[obstacle_mask]
    o_horiz = np.sqrt(o_horiz_sq)
    local_angles = np.arctan2(olys, olxs)
    front_mask = (np.abs(local_angles) <= FRONT_CONE_HALF_ANGLE) & (o_horiz <= FRONT_CONE_MAX_DIST)
    front_min = float(np.min(o_horiz[front_mask])) if np.any(front_mask) else float('inf')

    # VFH histogram
    frac = (local_angles + math.pi) / (2.0 * math.pi)
    sectors = np.clip((frac * NUM_SECTORS).astype(np.int32), 0, NUM_SECTORS - 1)
    sector_min = np.full(NUM_SECTORS, float('inf'), dtype=np.float64)
    np.minimum.at(sector_min, sectors, o_horiz)

    _sector_angles = (rh + (-math.pi + (np.arange(NUM_SECTORS) + 0.5) / NUM_SECTORS * 2.0 * math.pi)).tolist()
    _sector_dists    = sector_min.tolist()
    _scan_world      = scan_world
    _obstacle_pts    = obstacle_pts
    _last_front_dist = front_min

    _detected_rocks = []
    if do_cluster and int(np.sum(obstacle_mask)) >= CLUSTER_MIN_PTS:
        oz_arr = world_z[obstacle_mask]
        obstacle_pts_3d = list(zip(ox_arr.tolist(), oy_arr.tolist(), oz_arr.tolist()))
        _run_rock_detection(obstacle_pts_3d)


def get_histogram():
    return (_sector_angles, _sector_dists)

def get_scan_world():
    return _scan_world

def get_obstacle_points():
    return _obstacle_pts

def get_landmarks():
    return _landmarks

def get_locked_landmarks():
    return [lm for lm in _landmarks if _is_locked(lm)]

def get_front_distance():
    return _last_front_dist

def get_detected_rocks():
    return _detected_rocks


def _run_rock_detection(pts_3d):
    if len(pts_3d) > MAX_CLUSTER_PTS:
        pts_3d = random.sample(pts_3d, MAX_CLUSTER_PTS)

    pts_2d   = [(p[0], p[1]) for p in pts_3d]
    clusters = _cluster_points(pts_2d)

    rock_observations = []
    for indices in clusters:
        cluster_3d = [pts_3d[i] for i in indices]
        result     = _analyze_cluster_3d(cluster_3d)
        if result is None:
            continue

        cx, cy, h_diameter, z_span, planarity = result

        is_rock = (
            not _is_near_boundary(cx, cy)
            and h_diameter >= MIN_ROCK_DIAMETER
            and h_diameter <= MAX_ROCK_DIAMETER
            and z_span     >= MIN_ROCK_Z_SPAN
            and planarity  <= MAX_ROCK_PLANARITY
            and len(cluster_3d) <= MAX_ROCK_POINTS
        )

        if is_rock:
            est_radius = max(h_diameter / 2.0, 0.15)
            rock_observations.append((cx, cy))
            _detected_rocks.append({'x': cx, 'y': cy, 'radius': est_radius})

    if rock_observations:
        _update_landmarks(rock_observations)


def _is_near_boundary(cx, cy):
    return (cx < ARENA_X_MIN + WALL_MARGIN
            or cx > ARENA_X_MAX - WALL_MARGIN
            or cy < ARENA_Y_MIN + WALL_MARGIN
            or cy > ARENA_Y_MAX - WALL_MARGIN)


def _analyze_cluster_3d(pts_3d):
    """Returns (cx, cy, h_diameter, z_span, planarity) or None.
    planarity is the 2D eigenvalue ratio — high = wall-like, low = compact."""
    n = len(pts_3d)
    if n < 3:
        return None

    sx = sy = 0.0
    xlo = ylo = zlo =  1e9
    xhi = yhi = zhi = -1e9
    for x, y, z in pts_3d:
        sx += x;  sy += y
        if x < xlo: xlo = x
        if x > xhi: xhi = x
        if y < ylo: ylo = y
        if y > yhi: yhi = y
        if z < zlo: zlo = z
        if z > zhi: zhi = z

    cx = sx / n
    cy = sy / n
    h_diameter = max(xhi - xlo, yhi - ylo)
    z_span     = zhi - zlo

    # 2D covariance → eigenvalue ratio (planarity)
    cxx = cyy = cxy = 0.0
    for x, y, _ in pts_3d:
        dx = x - cx;  dy = y - cy
        cxx += dx * dx
        cyy += dy * dy
        cxy += dx * dy
    cxx /= n;  cyy /= n;  cxy /= n

    trace = cxx + cyy
    det   = cxx * cyy - cxy * cxy
    disc  = max(trace * trace * 0.25 - det, 0.0)
    sq    = math.sqrt(disc)
    e1    = trace * 0.5 + sq
    e2    = max(trace * 0.5 - sq, 1e-8)
    planarity = e1 / e2

    return cx, cy, h_diameter, z_span, planarity


def _cluster_points(pts):
    """DBSCAN clustering via cKDTree."""
    n = len(pts)
    pts_arr = np.array(pts, dtype=np.float32)
    tree = cKDTree(pts_arr)
    neighbor_lists = tree.query_ball_point(pts_arr, r=CLUSTER_EPS)

    used = [False] * n
    clusters = []
    for i in range(n):
        if used[i]:
            continue
        neighbors = neighbor_lists[i]
        if len(neighbors) < CLUSTER_MIN_PTS:
            continue
        cluster = []
        queue = [i]
        used[i] = True
        while queue:
            ci = queue.pop()
            cluster.append(ci)
            for j in neighbor_lists[ci]:
                if not used[j]:
                    used[j] = True
                    queue.append(j)
        if len(cluster) >= CLUSTER_MIN_PTS:
            clusters.append(cluster)
    return clusters


def _is_locked(lm):
    """Locked = enough observations from spatially distinct positions."""
    return (lm['obs_count'] >= MIN_LANDMARK_OBS
            and len(lm['obs_positions']) >= MIN_SPATIAL_OBS)


def _update_landmarks(rock_observations):
    """Match observations to landmarks, update running-average positions,
    and apply a pose correction when enough locked landmarks are re-seen."""
    global _landmarks

    rx, ry, _ = odometry.get_pose()
    matches   = []
    unmatched = []

    for obs_x, obs_y in rock_observations:
        best_dist = float('inf')
        best_idx  = -1
        for idx, lm in enumerate(_landmarks):
            d = math.hypot(obs_x - lm['x'], obs_y - lm['y'])
            if d < best_dist:
                best_dist = d
                best_idx  = idx

        if best_dist < LANDMARK_MATCH_DIST and best_idx >= 0:
            lm = _landmarks[best_idx]
            n  = lm['obs_count']

            lm['x'] = (lm['x'] * n + obs_x) / (n + 1)
            lm['y'] = (lm['y'] * n + obs_y) / (n + 1)
            lm['obs_count'] = n + 1

            # Record rover position if far enough from prior ones
            if all(math.hypot(rx - px, ry - py) >= MIN_OBS_SPACING
                   for px, py in lm['obs_positions']):
                lm['obs_positions'].append((rx, ry))

            if _is_locked(lm):
                matches.append((obs_x, obs_y, lm['x'], lm['y']))
        else:
            unmatched.append((obs_x, obs_y))

    for ox, oy in unmatched:
        too_close = any(
            math.hypot(ox - lm['x'], oy - lm['y']) < LANDMARK_MERGE_DIST
            for lm in _landmarks
        )
        if not too_close:
            _landmarks.append({
                'x': ox, 'y': oy,
                'obs_count': 1,
                'obs_positions': [(rx, ry)]
            })

    if len(matches) >= MATCH_MIN_LOCKED:
        dx_sum = sum(m[2] - m[0] for m in matches)
        dy_sum = sum(m[3] - m[1] for m in matches)
        n      = len(matches)
        corr_x = (dx_sum / n) * CORRECTION_GAIN
        corr_y = (dy_sum / n) * CORRECTION_GAIN
        odometry.apply_correction(corr_x, corr_y)
        print(f"[LIDAR] Rock correction: "
              f"dx={corr_x:+.3f}m  dy={corr_y:+.3f}m  "
              f"({len(matches)} locked rocks, "
              f"{sum(1 for lm in _landmarks if _is_locked(lm))} total locked)")
