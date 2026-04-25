import math

try:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

import lidar_processing

# Grid parameters
RESOLUTION  = 0.10          # meters per cell
GRID_WIDTH  = 100           # cells
GRID_HEIGHT = 100           # cells
ORIGIN_X    = -2.0          # world X at grid (0,0)
ORIGIN_Y    = -2.0          # world Y at grid (0,0)

INFLATE_RADIUS = 0.70       # obstacle inflation radius (m)
INFLATE_CELLS  = int(math.ceil(INFLATE_RADIUS / RESOLUTION))

COST_FREE     = 0
COST_OCCUPIED = 100
COST_INFLATED = 60
COST_UNKNOWN  = 0

DECAY_RATE = 15             # cost units lost per frame (clears stale obstacles)

_grid = None


def setup():
    global _grid

    if not NUMPY_AVAILABLE:
        print("[COSTMAP] numpy not available, costmap disabled")
        return

    _grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    print(f"[COSTMAP] Grid {GRID_WIDTH}x{GRID_HEIGHT} at {RESOLUTION}m/cell, "
          f"inflate={INFLATE_RADIUS}m ({INFLATE_CELLS} cells)")


def update():
    global _grid

    if _grid is None:
        return

    # Decay toward zero each frame
    np.subtract(_grid, DECAY_RATE, out=_grid)
    np.maximum(_grid, 0, out=_grid)

    obs_pts = lidar_processing.get_obstacle_points()
    if not obs_pts:
        return

    obs_arr = np.array(obs_pts, dtype=np.float32)
    gxs = ((obs_arr[:, 0] - ORIGIN_X) / RESOLUTION).astype(np.int32)
    gys = ((obs_arr[:, 1] - ORIGIN_Y) / RESOLUTION).astype(np.int32)
    valid = (gxs >= 0) & (gxs < GRID_WIDTH) & (gys >= 0) & (gys < GRID_HEIGHT)
    gxs = gxs[valid]
    gys = gys[valid]

    if len(gxs) == 0:
        return

    new_occupied = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    new_occupied[gys, gxs] = True

    dist_m = distance_transform_edt(~new_occupied) * RESOLUTION

    # Linear cost falloff within inflation radius
    inflate_cost = np.where(
        dist_m < INFLATE_RADIUS,
        COST_INFLATED * (1.0 - dist_m / INFLATE_RADIUS),
        0.0
    ).astype(np.float32)

    np.maximum(_grid, inflate_cost, out=_grid)
    _grid[gys, gxs] = COST_OCCUPIED


def get_cost_at_world(wx, wy):
    if _grid is None:
        return COST_FREE
    gx, gy = _world_to_grid(wx, wy)
    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
        return float(_grid[gy, gx])
    return COST_FREE


def get_grid():
    return _grid


def get_grid_params():
    return {
        'resolution': RESOLUTION,
        'width':      GRID_WIDTH,
        'height':     GRID_HEIGHT,
        'origin_x':   ORIGIN_X,
        'origin_y':   ORIGIN_Y,
    }


def _world_to_grid(wx, wy):
    gx = int((wx - ORIGIN_X) / RESOLUTION)
    gy = int((wy - ORIGIN_Y) / RESOLUTION)
    return gx, gy

