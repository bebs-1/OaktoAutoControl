import math

START_X = 0.6
START_Y = 0.8

_WHEEL_RADIUS = 0.15
_WHEEL_BASE   = 0.58

x = 0.0
y = 0.0
heading = 0.0

_prev_FL = None
_prev_FR = None
_prev_RL = None
_prev_RR = None
_encoders_ok = False
_imu_ok = False


def setup(encoders_ok=True, imu_ok=True, start_x=None, start_y=None):
    global x, y, heading, _encoders_ok, _imu_ok
    global _prev_FL, _prev_FR, _prev_RL, _prev_RR

    _encoders_ok = encoders_ok
    _imu_ok = imu_ok
    x = start_x if start_x is not None else START_X
    y = start_y if start_y is not None else START_Y
    heading = 0.0
    _prev_FL = _prev_FR = _prev_RL = _prev_RR = None


def update(fl, fr, rl, rr, yaw, dt):
    global x, y, heading
    global _prev_FL, _prev_FR, _prev_RL, _prev_RR

    if not _encoders_ok or not _imu_ok:
        return

    if _prev_FL is None:
        _prev_FL, _prev_FR = fl, fr
        _prev_RL, _prev_RR = rl, rr
        return

    enc_vel_left  = ((fl - _prev_FL) + (rl - _prev_RL)) / (2.0 * dt)
    enc_vel_right = ((fr - _prev_FR) + (rr - _prev_RR)) / (2.0 * dt)
    v_forward = (enc_vel_left + enc_vel_right) / 2.0 * _WHEEL_RADIUS

    heading = yaw
    x += v_forward * math.cos(heading) * dt
    y += v_forward * math.sin(heading) * dt

    _prev_FL, _prev_FR = fl, fr
    _prev_RL, _prev_RR = rl, rr


def get_pose():
    return (x, y, heading)


def apply_correction(dx, dy, dheading=0.0):
    global x, y, heading
    x += dx
    y += dy
    heading += dheading
