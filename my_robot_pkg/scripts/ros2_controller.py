#!/usr/bin/env python3
"""
ROS 2 controller node — replaces Webots_Test_1_Controller.py

Expects the SparkMax motor driver node (MotorDriver.cpp) to be running.

Subscriptions
─────────────
  /scan/points                (sensor_msgs/PointCloud2)   3-D LiDAR
  /camera/depth/color/points  (sensor_msgs/PointCloud2)   RealSense depth
  /camera/imu                 (sensor_msgs/Imu)            RealSense IMU
  motor_data                  (sensor_msgs/JointState)    SparkMax feedback
                                 .name     = [motor1, motor2, motor3, motor4]
                                 .velocity = RPM per motor
                                 .effort   = current (A) per motor
  /cmd_vel                    (geometry_msgs/Twist)        tele-op override
  /joy                        (sensor_msgs/Joy)             Xbox controller
                                 Button 7 (Start) toggles autonomous nav on/off

Publications
────────────
  motor_cmd  (std_msgs/Int32MultiArray)  [FL, FR, RL, RR] in RPM
                 index 0 = motor1 = FL
                 index 1 = motor2 = FR
                 index 2 = motor3 = RL
                 index 3 = motor4 = RR

Services
────────
  /start_navigation  (std_srvs/Trigger)  start / abort the full dig-dump cycle

Mission cycle (autonomous)
──────────────────────────
  NAV_TO_DIG → DIGGING → NAV_TO_DUMP → DUMPING → (repeat)
  Start button / service toggles the cycle on/off.
  D-pad overrides the actuator sequence at any time and aborts the mission.

All topic names and numeric values can be overridden with ROS parameters
(see declare_parameter calls below).
"""

import math
import os

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, Joy, JointState, PointCloud2
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2

import costmap
import lidar_processing
import mapping
import odometry
import pathfinding

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Robot kinematics — must match your hardware
WHEEL_RADIUS = 0.15   # metres
WHEEL_BASE   = 0.58   # metres (lateral distance between left and right wheels)

# Ordered list mapping SparkMax joint names to motor keys.
# Order must match the Int32MultiArray command indices [0..3].
# motor1=FL, motor2=FR, motor3=RL, motor4=RR  (matches MotorDriver.cpp)
MOTOR_JOINT_NAMES: list[tuple[str, str]] = [
    ('motor1', 'FL'),
    ('motor2', 'FR'),
    ('motor3', 'RL'),
    ('motor4', 'RR'),
]

# Radians/s → RPM
_RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


class RobotController(Node):
    def __init__(self) -> None:
        super().__init__('robot_controller')

        # ── ROS parameters ────────────────────────────────────────────
        self.declare_parameter('start_x',    0.6)
        self.declare_parameter('start_y',    0.8)
        self.declare_parameter('goal_x',     6.0)
        self.declare_parameter('goal_y',     0.5)
        self.declare_parameter('control_hz', 31.25)   # ≈ 32 ms timestep

        # Topic names
        self.declare_parameter('lidar_topic',   '/livox/lidar')
        self.declare_parameter('depth_topic',   '/camera/depth/color/points')
        self.declare_parameter('imu_topic',     '/camera/imu')
        self.declare_parameter('cmd_vel_topic',        '/cmd_vel')
        self.declare_parameter('joy_topic',             '/joy')
        self.declare_parameter('motor_cmd_topic',       'motor_cmd')
        self.declare_parameter('motor_data_topic',      'motor_data')
        # Xbox button index used to toggle autonomous navigation.
        # Default 7 = Start button on most Linux Xbox mappings.
        self.declare_parameter('auto_toggle_button', 7)

        # LiDAR mounting orientation
        # lidar_upside_down=True → sensor is mounted inverted (180° around X/forward axis).
        # Applies y → -y, z → -z so obstacle heights and lateral positions are correct.
        self.declare_parameter('lidar_upside_down', False)

        # Depth-camera options
        # depth_optical_frame=True  → cloud arrives in the RealSense optical
        #   frame (Z-forward, X-right, Y-down).  The node rotates it to the
        #   robot base_link convention (X-forward, Y-left, Z-up).
        # depth_optical_frame=False → cloud already in base_link convention.
        self.declare_parameter('depth_optical_frame', True)
        self.declare_parameter('depth_cam_height',    0.25)   # metres above ground
        self.declare_parameter('depth_max_points',    5000)   # subsample cap

        # Serial port for mechanism control (D-pad) and auto-digging
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('serial_baud', 9600)

        # Joy axes / drive parameters
        self.declare_parameter('drive_rpm',   500)    # RPM used for manual driving
        self.declare_parameter('joy_deadband', 0.2)

        # Auto-dig sequence target positions (inches, relative to Arduino startup pos=0)
        # Tune these to match your robot's physical geometry.
        #   Actuator 1 = arm (lift):   positive = up,   negative = down
        #   Actuator 2 = bucket (tilt): 0 = curled/hold (top tilted back),
        #                               positive = tilt top forward (open / dump)
        self.declare_parameter('dig_arm_lower_pos',    0.0)  # lower arm to dig depth
        self.declare_parameter('dig_bucket_open_pos',   3.0)  # tilt bucket forward to scoop
        self.declare_parameter('dig_arm_carry_pos',     2.0)  # raise arm to carry height
        self.declare_parameter('dig_bucket_hold_pos',   0.5)  # tilt bucket back to retain material
        self.declare_parameter('dig_pos_tolerance',     0.3)  # inches — how close is 'close enough'
        # RPM to drive forward while pressing into material (LOWER_ARM and RAISE_ARM).
        # Set to 0 to disable drive-assist during digging.
        self.declare_parameter('dig_drive_rpm', 150)
        # Maximum seconds allowed per dig phase before the sequence is aborted.
        # Protects against a stuck actuator or lost serial feedback.
        self.declare_parameter('dig_phase_timeout', 15.0)

        # Dump site coordinates (default = start position; set to your bin location)
        self.declare_parameter('dump_x', 0.6)
        self.declare_parameter('dump_y', 0.8)
        # Dump sequence actuator positions (inches, same origin as dig positions)
        self.declare_parameter('dump_arm_pos',            1.0)   # arm height over dump bin
        self.declare_parameter('dump_bucket_release_pos',  5.0)  # tilt bucket forward to drop load
        self.declare_parameter('dump_arm_reset_pos',      0.0)   # arm carry height after dump
        self.declare_parameter('dump_bucket_reset_pos',   0.5)   # tilt bucket back for travel

        p = self.get_parameter
        start_x   = p('start_x').value
        start_y   = p('start_y').value
        goal_x    = p('goal_x').value
        goal_y    = p('goal_y').value
        ctrl_hz   = p('control_hz').value
        lidar_t      = p('lidar_topic').value
        depth_t      = p('depth_topic').value
        imu_t        = p('imu_topic').value
        cmdvel_t          = p('cmd_vel_topic').value
        joy_t             = p('joy_topic').value
        motor_cmd_t       = p('motor_cmd_topic').value
        motor_data_t      = p('motor_data_topic').value
        self._auto_btn    = int(p('auto_toggle_button').value)

        self._lidar_upside_down = bool(p('lidar_upside_down').value)
        self._depth_optical    = p('depth_optical_frame').value
        self._depth_cam_height = p('depth_cam_height').value
        self._depth_max_pts    = int(p('depth_max_points').value)
        self.dt                = 1.0 / ctrl_hz
        self._drive_rpm        = int(p('drive_rpm').value)
        self._joy_deadband     = float(p('joy_deadband').value)
        self._dig_arm_lower    = float(p('dig_arm_lower_pos').value)
        self._dig_bucket_open  = float(p('dig_bucket_open_pos').value)
        self._dig_arm_carry    = float(p('dig_arm_carry_pos').value)
        self._dig_bucket_hold  = float(p('dig_bucket_hold_pos').value)
        self._dig_pos_tol      = float(p('dig_pos_tolerance').value)
        self._dig_drive_rpm    = int(p('dig_drive_rpm').value)
        self._dig_phase_timeout   = float(p('dig_phase_timeout').value)
        # Store dig site (goal_x/goal_y) and dump site for mission cycling
        self._dig_x               = goal_x
        self._dig_y               = goal_y
        self._dump_x              = float(p('dump_x').value)
        self._dump_y              = float(p('dump_y').value)
        self._dump_arm_pos        = float(p('dump_arm_pos').value)
        self._dump_bucket_release = float(p('dump_bucket_release_pos').value)
        self._dump_arm_reset      = float(p('dump_arm_reset_pos').value)
        self._dump_bucket_reset   = float(p('dump_bucket_reset_pos').value)

        # ── Serial port (mechanisms + auto-dig) ─────────────────
        self._ser: 'serial.Serial | None' = None
        if _SERIAL_AVAILABLE:
            try:
                self._ser = serial.Serial(
                    p('serial_port').value,
                    p('serial_baud').value,
                    timeout=0.1,
                )
                self.get_logger().info(
                    f'Serial port opened: {p("serial_port").value}')
            except Exception as e:
                self.get_logger().warn(f'Serial port unavailable: {e}')
        else:
            self.get_logger().warn(
                'pyserial not installed — mechanism control disabled')

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(PointCloud2, lidar_t,     self._lidar_cb,  10)
        self.create_subscription(PointCloud2, depth_t,     self._depth_cb,  10)
        self.create_subscription(Imu,         imu_t,       self._imu_cb,    10)
        self.create_subscription(JointState,  motor_data_t, self._joint_cb, 10)
        self.create_subscription(Twist,       cmdvel_t,    self._cmdvel_cb, 10)
        self.create_subscription(Joy,         joy_t,       self._joy_cb,    10)

        # ── Motor command publisher (single Int32MultiArray → SparkMax) ─
        self._motor_cmd_pub = self.create_publisher(
            Int32MultiArray, motor_cmd_t, 10)

        # ── Start-navigation service ──────────────────────────────────
        self._nav_srv = self.create_service(
            Trigger, 'start_navigation', self._start_nav_cb)

        # ── Sensor state ──────────────────────────────────────────────
        self._yaw: float             = 0.0
        self._imu_ready: bool        = False
        # Simulated cumulative encoder positions (rad) — integrated from RPM
        self._wheel_pos: dict        = {key: 0.0 for key in ('FL', 'FR', 'RL', 'RR')}
        self._wheel_vel_rpm: dict    = {key: 0.0 for key in ('FL', 'FR', 'RL', 'RR')}
        self._joint_index: dict      = {}           # joint name → motor key
        self._latest_lidar: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None

        # Tele-op state (set by /joy or /cmd_vel)
        self._teleop_override: bool       = False
        self._teleop_left:  float         = 0.0
        self._teleop_right: float         = 0.0
        # Direct 4-wheel RPM from joy — overrides _teleop_left/_right when set
        self._teleop_rpm: list[int] | None = None

        # Joy button edge-detection (toggle fires on press, not hold)
        self._prev_joy_buttons: list[int] = []

        # Mechanism serial state (deduplicated — only write on change)
        self._mech1_state: str = ''
        self._mech2_state: str = ''

        # Actuator positions fed back from Arduino serial (inches, relative to startup=0)
        self._act1_pos: float = 0.0
        self._act2_pos: float = 0.0

        # Actuator sequence state machine.
        # Dig states:  LOWER_ARM | OPEN_BUCKET | RAISE_ARM | CURL_BUCKET
        # Dump states: DUMP_LOWER_ARM | DUMP_RELEASE | DUMP_RESET_ARM | DUMP_RESET_BUCKET
        # Shared:      IDLE (inactive) | HOLD (sequence done, awaiting mission transition)
        # _auto_digging suppresses neutral D-pad stop while a sequence owns channel 1.
        self._auto_digging: bool     = False
        self._dig_state: str         = 'IDLE'
        self._dig_phase_start: float = 0.0   # ROS clock seconds when current phase began

        # ── Mission state machine ─────────────────────────────────────────
        # IDLE | NAV_TO_DIG | DIGGING | NAV_TO_DUMP | DUMPING
        self._mission_state: str = 'IDLE'
        self._step = 0
        # When the mission is paused by Start, this stores the phase to enter
        # on the next Start press so manual driving can bridge a travel leg.
        self._resume_phase: str | None = None

        # ── Module initialisation ─────────────────────────────────────
        odometry.setup(encoders_ok=True, imu_ok=True,
                       start_x=start_x, start_y=start_y)
        mapping.setup(script_dir=SCRIPT_DIR, goal_x=goal_x, goal_y=goal_y)
        lidar_processing.setup()
        costmap.setup()
        pathfinding.setup(goal_x=goal_x, goal_y=goal_y)

        # ── Control timer ─────────────────────────────────────────────
        self._timer = self.create_timer(self.dt, self._control_loop)
        # ── Serial position read timer (20 Hz) ───────────────────────────
        # Drains the Arduino's "pos1,pos2\n" stream without blocking the
        # control loop.  Uses in_waiting so it only reads when data is ready.
        self._serial_timer = self.create_timer(0.05, self._read_serial_positions)
        self.get_logger().info(
            f'robot_controller ready | '
            f'start=({start_x}, {start_y})  goal=({goal_x}, {goal_y})')
        self.get_logger().info(
            f'Topics: lidar={lidar_t}  depth={depth_t}  '
            f'imu={imu_t}  motor_data={motor_data_t}  '
            f'motor_cmd={motor_cmd_t}')
        self.get_logger().info(
            'Call service /start_navigation (std_srvs/Trigger) to begin autonomous nav')

    # ── Sensor callbacks ──────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu) -> None:
        """Extract yaw from the IMU quaternion (RealSense or any sensor_msgs/Imu)."""
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)
        self._imu_ready = True

    def _lidar_cb(self, msg: PointCloud2) -> None:
        """Parse a 3-D LiDAR PointCloud2 into an (N, 3) float32 array."""
        pts = np.array(
            list(pc2.read_points(msg, field_names=('x', 'y', 'z'),
                                 skip_nans=True)),
            dtype=np.float32,
        )
        if pts.ndim == 2 and pts.shape[0] > 0:
            if self._lidar_upside_down:
                # 180° rotation around X (forward): y → -y, z → -z
                pts[:, 1] = -pts[:, 1]
                pts[:, 2] = -pts[:, 2]
            self._latest_lidar = pts

    def _depth_cb(self, msg: PointCloud2) -> None:
        """Parse the RealSense depth PointCloud2 and re-express it in the
        same coordinate convention as the LiDAR (X-forward, Y-left, Z-up)."""
        pts = np.array(
            list(pc2.read_points(msg, field_names=('x', 'y', 'z'),
                                 skip_nans=True)),
            dtype=np.float32,
        )
        if pts.ndim != 2 or pts.shape[0] == 0:
            return

        # Subsample to bound CPU cost (depth clouds can be 300 k+ points)
        if pts.shape[0] > self._depth_max_pts:
            idx = np.random.choice(pts.shape[0], self._depth_max_pts,
                                   replace=False)
            pts = pts[idx]

        if self._depth_optical:
            # RealSense optical frame: Z = forward, X = right, Y = down
            # Robot base_link frame:   X = forward, Y = left,  Z = up
            pts = np.stack([
                 pts[:, 2],   # cam Z  → robot X (forward)
                -pts[:, 0],   # cam -X → robot Y (left)
                -pts[:, 1],   # cam -Y → robot Z (up)
            ], axis=1)

        # Shift Z so height thresholds inside lidar_processing stay valid.
        # SENSOR_HEIGHT is where the LiDAR sits; depth_cam_height is where
        # the RealSense sits — both measured from the ground plane.
        height_shift = self._depth_cam_height - lidar_processing.SENSOR_HEIGHT
        pts[:, 2] += height_shift

        self._latest_depth = pts

    def _joint_cb(self, msg: JointState) -> None:
        """Receive SparkMax RPM feedback and integrate to simulated encoder
        positions so that odometry.update() (which expects cumulative rad)
        stays unchanged."""
        # Build the name→key index once on the first message
        if not self._joint_index:
            for ros_name, motor_key in MOTOR_JOINT_NAMES:
                if ros_name in msg.name:
                    self._joint_index[ros_name] = (msg.name.index(ros_name), motor_key)

        for ros_name, (idx, motor_key) in self._joint_index.items():
            if idx < len(msg.velocity):
                rpm = msg.velocity[idx]
                self._wheel_vel_rpm[motor_key] = rpm
                # Integrate: Δpos (rad) = ω (rad/s) × dt
                self._wheel_pos[motor_key] += (rpm / _RAD_S_TO_RPM) * self.dt

    def _cmdvel_cb(self, msg: Twist) -> None:
        """Convert a Twist tele-op command to individual wheel speeds (rad/s)."""
        v     = msg.linear.x       # m/s forward
        omega = msg.angular.z      # rad/s CCW positive
        left  = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        self._teleop_left   = left
        self._teleop_right  = right
        self._teleop_override = (abs(v) > 1e-3 or abs(omega) > 1e-3)

    def _read_serial_positions(self) -> None:
        """Non-blocking drain of Arduino position feedback.

        Arduino sends "pos1,pos2\n" every loop iteration (inches, float).
        We read all complete lines currently in the buffer so the control
        loop always has a fresh position without blocking.
        """
        if self._ser is None or not self._ser.is_open:
            return
        try:
            while self._ser.in_waiting:
                raw = self._ser.readline().decode('ascii', errors='ignore').strip()
                if ',' in raw:
                    parts = raw.split(',')
                    if len(parts) == 2:
                        self._act1_pos = float(parts[0])
                        self._act2_pos = float(parts[1])
        except Exception as exc:
            self.get_logger().warn(
                f'Serial read error: {exc}', throttle_duration_sec=5.0)

    def _joy_cb(self, msg: Joy) -> None:
        """Handle all Xbox controller input.

        Axes
        ----
          axes[1]  Left stick Y   → forward / backward
          axes[0]  Left stick X   → turn left / right
          axes[6]  D-pad X        → mechanism 1  (serial: u1 / d1 / stop)
          axes[7]  D-pad Y        → mechanism 2  (serial: u2 / d2 / stop)

        Buttons
        -------
          buttons[auto_toggle_button]  (default Start, index 7)
                                       → toggle autonomous navigation
        """
        buttons = list(msg.buttons)
        axes    = list(msg.axes)

        DEAD = self._joy_deadband
        RPM  = self._drive_rpm

        # ── Autonomous toggle (rising-edge on Start button) ────────────
        prev_val = (self._prev_joy_buttons[self._auto_btn]
                    if self._auto_btn < len(self._prev_joy_buttons) else 0)
        curr_val = buttons[self._auto_btn] if self._auto_btn < len(buttons) else 0

        if curr_val == 1 and prev_val == 0:
            if self._mission_state == 'IDLE':
                if self._resume_phase is not None:
                    # Resume at the phase saved when the mission was paused
                    phase = self._resume_phase
                    self._resume_phase = None
                    self._start_phase(phase)
                else:
                    # Fresh start from the beginning
                    self._mission_state = 'NAV_TO_DIG'
                    self._navigate_to(self._dig_x, self._dig_y)
                    self.get_logger().info(
                        f'Mission STARTED → navigating to dig site '
                        f'({self._dig_x:.2f}, {self._dig_y:.2f})')
            else:
                # Pause: record what phase should start on the next press,
                # then hand control back to the driver.
                _NEXT = {
                    'NAV_TO_DIG':  'DIGGING',
                    'DIGGING':     'NAV_TO_DUMP',
                    'NAV_TO_DUMP': 'DUMPING',
                    'DUMPING':     'NAV_TO_DIG',
                }
                self._resume_phase  = _NEXT.get(self._mission_state, 'NAV_TO_DIG')
                self._mission_state = 'IDLE'
                self._teleop_rpm    = None
                self._auto_digging  = False
                self._dig_state     = 'IDLE'
                self._serial_cmd('stop', channel=1)
                self._serial_cmd('stop', channel=2)
                self._stop_motors()
                self.get_logger().info(
                    f'Mission PAUSED — manual control active. '
                    f'Press Start to begin {self._resume_phase}.')

        self._prev_joy_buttons = buttons

        # ── Driving axes (only active when not in autonomous mode) ──────
        fwd  = axes[1] if len(axes) > 1 else 0.0
        turn = axes[0] if len(axes) > 0 else 0.0

        if self._mission_state == 'IDLE':
            if abs(fwd) > abs(turn):
                if fwd > DEAD:
                    self._teleop_rpm = [ RPM,  RPM,  RPM,  RPM]   # forward
                elif fwd < -DEAD:
                    self._teleop_rpm = [-RPM, -RPM, -RPM, -RPM]   # backward
                else:
                    self._teleop_rpm = [0, 0, 0, 0]
            else:
                if turn > DEAD:
                    self._teleop_rpm = [ RPM,  RPM, -RPM, -RPM]   # left
                elif turn < -DEAD:
                    self._teleop_rpm = [-RPM, -RPM,  RPM,  RPM]   # right
                else:
                    self._teleop_rpm = [0, 0, 0, 0]

            moving = self._teleop_rpm is not None and any(
                v != 0 for v in self._teleop_rpm)
            self._teleop_override = moving

        # ── Mechanism control via D-pad (serial) ────────────────────────
        # D-pad left/right (axes[6]): actuator 1  — u1 = up, d1 = down
        # D-pad up/down   (axes[7]): actuator 2  — u2 = up, d2 = down
        # Neutral D-pad only sends 'stop' when no auto-dig is holding the channel.
        act1 = axes[6] if len(axes) > 6 else 0.0
        act2 = axes[7] if len(axes) > 7 else 0.0

        # Actuator 1 (arm / lift)
        # Any active D-pad press cancels an in-progress sequence AND aborts the mission.
        if act1 > DEAD:
            if self._dig_state not in ('IDLE', 'HOLD'):
                self._dig_state     = 'IDLE'
                self._mission_state = 'IDLE'
                self._resume_phase  = None   # D-pad full abort — clear any saved resume
                self.get_logger().info('[SEQ] Sequence cancelled by D-pad — mission aborted')
            self._auto_digging = False
            self._serial_cmd('u1', channel=1)
        elif act1 < -DEAD:
            if self._dig_state not in ('IDLE', 'HOLD'):
                self._dig_state     = 'IDLE'
                self._mission_state = 'IDLE'
                self._resume_phase  = None   # D-pad full abort — clear any saved resume
                self.get_logger().info('[SEQ] Sequence cancelled by D-pad — mission aborted')
            self._auto_digging = False
            self._serial_cmd('d1', channel=1)
        elif self._dig_state == 'IDLE':       # neutral — stop only when no sequence running
            self._serial_cmd('stop', channel=1)

        # Actuator 2 (bucket / tilt)
        if act2 > DEAD:
            if self._dig_state not in ('IDLE', 'HOLD'):
                self._dig_state     = 'IDLE'
                self._mission_state = 'IDLE'
            self._serial_cmd('u2', channel=2)
        elif act2 < -DEAD:
            if self._dig_state not in ('IDLE', 'HOLD'):
                self._dig_state     = 'IDLE'
                self._mission_state = 'IDLE'
            self._serial_cmd('d2', channel=2)
        elif self._dig_state == 'IDLE':       # neutral — stop only when no sequence running
            self._serial_cmd('stop', channel=2)

    # ── Navigation service ────────────────────────────────────────────────

    def _start_phase(self, phase: str) -> None:
        """Enter a specific mission phase directly (used after a manual pause)."""
        if phase == 'DIGGING':
            self._mission_state   = 'DIGGING'
            self._dig_state       = 'OPEN_BUCKET'
            self._auto_digging    = True
            self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('[RESUME] Starting dig sequence at current position')
        elif phase == 'NAV_TO_DUMP':
            self._mission_state = 'NAV_TO_DUMP'
            self._navigate_to(self._dump_x, self._dump_y)
            self.get_logger().info(
                f'[RESUME] Navigating to dump site '
                f'({self._dump_x:.2f}, {self._dump_y:.2f})')
        elif phase == 'DUMPING':
            self._mission_state   = 'DUMPING'
            self._dig_state       = 'DUMP_LOWER_ARM'
            self._auto_digging    = True
            self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('[RESUME] Starting dump sequence at current position')
        elif phase == 'NAV_TO_DIG':
            self._mission_state = 'NAV_TO_DIG'
            self._navigate_to(self._dig_x, self._dig_y)
            self.get_logger().info(
                f'[RESUME] Navigating to dig site '
                f'({self._dig_x:.2f}, {self._dig_y:.2f})')

    def _start_nav_cb(self, _request, response):
        if self._mission_state == 'IDLE':
            self._mission_state = 'NAV_TO_DIG'
            self._navigate_to(self._dig_x, self._dig_y)
            self.get_logger().info('Mission started via service')
            response.success = True
            response.message = 'Navigating to dig site'
        else:
            response.success = False
            response.message = f'Mission already running: {self._mission_state}'
        return response

    def _navigate_to(self, goal_x: float, goal_y: float) -> None:
        """Re-configure pathfinding and mapping for a new goal and start navigating."""
        pathfinding.setup(goal_x=goal_x, goal_y=goal_y)
        pathfinding.start()
        mapping.setup(script_dir=SCRIPT_DIR, goal_x=goal_x, goal_y=goal_y)

    def _advance_dig_sequence(self) -> None:
        """Called every control tick while _dig_state is an active step.

        Dig sequence
        ────────────
        OPEN_BUCKET      →  u2, stationary (tilt bucket top forward while arm is high)
        LOWER_ARM        →  d1 + drive forward (lower open bucket into material and push)
        CURL_BUCKET      →  d2 + drive forward (curl bucket back to scoop/capture material)
        RAISE_ARM        →  u1, stationary (raise arm to carry height with load secured)

        Dump sequence
        ─────────────
        DUMP_LOWER_ARM   →  arm to dump_arm_pos, stationary
        DUMP_RELEASE     →  d2 to dump_bucket_release_pos, stationary
        DUMP_RESET_ARM   →  arm to dump_arm_reset_pos, stationary
        DUMP_RESET_BUCKET→  bucket to dump_bucket_reset_pos, stationary

        HOLD  →  sequence complete; control loop transitions mission state.

        Returns (left_rpm, right_rpm) when the robot should drive, else None.
        Timeout aborts and resets to IDLE if the position target is never reached.
        """
        TOL  = self._dig_pos_tol
        DRPM = self._dig_drive_rpm   # forward RPM during active digging phases

        # ── Per-phase timeout ─────────────────────────────────────────
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._dig_phase_start
        if elapsed > self._dig_phase_timeout:
            self._serial_cmd('stop', channel=1)
            self._serial_cmd('stop', channel=2)
            self._dig_state     = 'IDLE'
            self._mission_state = 'IDLE'
            self._auto_digging  = False
            self.get_logger().error(
                f'[SEQ] Phase timeout ({elapsed:.1f}s) — actuators stopped, mission aborted. '
                f'Check serial connection and actuator movement.')
            return None

        if self._dig_state == 'OPEN_BUCKET':
            self._serial_cmd('u2', channel=2)
            if self._act2_pos >= self._dig_bucket_open - TOL:
                self._serial_cmd('stop', channel=2)   # hold bucket open
                self._dig_state = 'LOWER_ARM'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DIG] Bucket open at {self._act2_pos:.2f}" → lowering into material')
            return None                               # stationary while opening bucket

        elif self._dig_state == 'LOWER_ARM':
            self._serial_cmd('d1', channel=1)
            if self._act1_pos <= self._dig_arm_lower + TOL:
                self._serial_cmd('stop', channel=1)   # hold arm at dig depth
                self._dig_state = 'CURL_BUCKET'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DIG] Arm at {self._act1_pos:.2f}" → curling bucket to scoop')
                return None                           # stop driving between phases
            return (DRPM, DRPM)                       # drive forward while lowering into material

        elif self._dig_state == 'CURL_BUCKET':
            self._serial_cmd('d2', channel=2)
            if self._act2_pos <= self._dig_bucket_hold + TOL:
                self._serial_cmd('stop', channel=2)
                self._dig_state = 'RAISE_ARM'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DIG] Bucket curled at {self._act2_pos:.2f}" → raising arm')
                return None                           # stop driving between phases
            return (DRPM, DRPM)                       # drive forward while scooping

        elif self._dig_state == 'RAISE_ARM':
            self._serial_cmd('u1', channel=1)
            if self._act1_pos >= self._dig_arm_carry - TOL:
                self._serial_cmd('stop', channel=1)
                self._serial_cmd('stop', channel=2)
                self._dig_state    = 'HOLD'
                self._auto_digging = False
                self.get_logger().info(
                    f'[DIG] Digging complete — arm={self._act1_pos:.2f}" '
                    f'bucket={self._act2_pos:.2f}"')
            return None                               # stationary while raising arm

        elif self._dig_state == 'DUMP_LOWER_ARM':
            # Move arm to position over the dump bin (direction from current pos)
            cmd = 'u1' if self._dump_arm_pos > self._act1_pos else 'd1'
            self._serial_cmd(cmd, channel=1)
            if abs(self._act1_pos - self._dump_arm_pos) <= TOL:
                self._serial_cmd('stop', channel=1)
                self._dig_state = 'DUMP_RELEASE'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DUMP] Arm at {self._act1_pos:.2f}" → releasing bucket')
            return None

        elif self._dig_state == 'DUMP_RELEASE':
            self._serial_cmd('u2', channel=2)
            if self._act2_pos >= self._dump_bucket_release - TOL:
                self._serial_cmd('stop', channel=2)
                self._dig_state = 'DUMP_RESET_ARM'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DUMP] Bucket released at {self._act2_pos:.2f}" → resetting arm')
            return None

        elif self._dig_state == 'DUMP_RESET_ARM':
            # Raise arm back to carry height for travel
            cmd = 'u1' if self._dump_arm_reset > self._act1_pos else 'd1'
            self._serial_cmd(cmd, channel=1)
            if abs(self._act1_pos - self._dump_arm_reset) <= TOL:
                self._serial_cmd('stop', channel=1)
                self._dig_state = 'DUMP_RESET_BUCKET'
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DUMP] Arm reset at {self._act1_pos:.2f}" → resetting bucket')
            return None

        elif self._dig_state == 'DUMP_RESET_BUCKET':
            # Curl bucket back to travel position
            cmd = 'u2' if self._dump_bucket_reset > self._act2_pos else 'd2'
            self._serial_cmd(cmd, channel=2)
            if abs(self._act2_pos - self._dump_bucket_reset) <= TOL:
                self._serial_cmd('stop', channel=2)
                self._dig_state    = 'HOLD'
                self._auto_digging = False
                self.get_logger().info(
                    f'[DUMP] Dump sequence complete — arm={self._act1_pos:.2f}" '
                    f'bucket={self._act2_pos:.2f}"')
            return None

        return None

    # ── Main control loop ─────────────────────────────────────────────────

    def _control_loop(self) -> None:
        self._step += 1

        # Odometry: requires both IMU yaw and all four wheel encoder values
        if (self._imu_ready and
                all(v is not None for v in self._wheel_pos.values())):
            odometry.update(
                self._wheel_pos['FL'], self._wheel_pos['FR'],
                self._wheel_pos['RL'], self._wheel_pos['RR'],
                self._yaw, self.dt,
            )

        # Perception: merge LiDAR and depth clouds before processing
        clouds = [c for c in (self._latest_lidar, self._latest_depth)
                  if c is not None]
        if clouds:
            combined = np.vstack(clouds) if len(clouds) > 1 else clouds[0]
            lidar_processing.update(combined)

        costmap.update()
        mapping.update()

        # Motion: tele-op takes priority, then autonomous, then stop
        if self._teleop_override and self._teleop_rpm is not None:
            self._publish_rpm(*self._teleop_rpm)
        elif self._teleop_override:
            self._set_motors(self._teleop_left, self._teleop_right)
        elif self._mission_state == 'NAV_TO_DIG':
            left, right = pathfinding.update(self._step)
            self._set_motors(left, right)
            if 'ARRIVED' in pathfinding.get_status():
                self._mission_state   = 'DIGGING'
                self._dig_state       = 'OPEN_BUCKET'
                self._auto_digging    = True
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f'[DIG] Dig site reached — opening bucket before lowering arm')

        elif self._mission_state == 'DIGGING':
            if self._dig_state == 'HOLD':
                # Digging complete — navigate to dump site
                self._dig_state     = 'IDLE'
                self._auto_digging  = False
                self._mission_state = 'NAV_TO_DUMP'
                self._navigate_to(self._dump_x, self._dump_y)
                self.get_logger().info(
                    f'[DIG] Complete — navigating to dump '
                    f'({self._dump_x:.2f}, {self._dump_y:.2f})')
            elif self._dig_state != 'IDLE':
                dig_drive = self._advance_dig_sequence()
                if dig_drive is not None:
                    self._publish_rpm(dig_drive[0], dig_drive[1],
                                      dig_drive[0], dig_drive[1])
                else:
                    self._stop_motors()

        elif self._mission_state == 'NAV_TO_DUMP':
            left, right = pathfinding.update(self._step)
            self._set_motors(left, right)
            if 'ARRIVED' in pathfinding.get_status():
                self._mission_state   = 'DUMPING'
                self._dig_state       = 'DUMP_LOWER_ARM'
                self._auto_digging    = True
                self._dig_phase_start = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info('[DUMP] Dump site reached — starting dump sequence')

        elif self._mission_state == 'DUMPING':
            if self._dig_state == 'HOLD':
                # Dump complete — return to dig site for another cycle
                self._dig_state     = 'IDLE'
                self._auto_digging  = False
                self._mission_state = 'NAV_TO_DIG'
                self._navigate_to(self._dig_x, self._dig_y)
                self.get_logger().info(
                    f'[DUMP] Complete — returning to dig site '
                    f'({self._dig_x:.2f}, {self._dig_y:.2f})')
            elif self._dig_state != 'IDLE':
                self._advance_dig_sequence()   # dump phases never return drive RPM
                self._stop_motors()

        else:
            self._auto_digging = False
            self._stop_motors()

        # Periodic status log
        if self._step % 25 == 0:
            rx, ry, rh = odometry.get_pose()
            if self._mission_state != 'IDLE':
                status = f'{self._mission_state}:{self._dig_state} | {pathfinding.get_status()}'
            else:
                status = 'IDLE — press Start or call /start_navigation'
            self.get_logger().info(
                f'Step {self._step:5d} | '
                f'({rx:+.2f}, {ry:+.2f}) {math.degrees(rh):+.1f} deg | {status}')

    # ── Motor helpers ─────────────────────────────────────────────────────

    def _set_motors(self, left: float, right: float) -> None:
        """Convert rad/s to RPM integers and publish to motor_cmd.

        Int32MultiArray layout (matches MotorDriver.cpp ordering):
          [0] motor1 = FL,  [1] motor2 = FR
          [2] motor3 = RL,  [3] motor4 = RR
          add gear ratio and convert from rad/s to RPM: RPM = rad/s × 60 / (2π) × gear_ratio
        """
        fl = int(round(left  * _RAD_S_TO_RPM))
        fr = int(round(right * _RAD_S_TO_RPM))
        rl = int(round(left  * _RAD_S_TO_RPM))
        rr = int(round(right * _RAD_S_TO_RPM))
        self._publish_rpm(fl, fr, rl, rr)

    def _publish_rpm(self, fl: int, fr: int, rl: int, rr: int) -> None:
        """Publish four individual wheel RPM values directly to motor_cmd."""
        msg = Int32MultiArray()
        msg.data = [fl, fr, rl, rr]
        self._motor_cmd_pub.publish(msg)

    def _serial_cmd(self, cmd: str, channel: int) -> None:
        """Write a mechanism command to the serial port, but only if the
        state for that channel has changed (avoids flooding the serial bus)."""
        if self._ser is None:
            return
        if channel == 1:
            if cmd == self._mech1_state:
                return
            self._mech1_state = cmd
        else:
            if cmd == self._mech2_state:
                return
            self._mech2_state = cmd
        try:
            self._ser.write(cmd.encode())
        except Exception as e:
            self.get_logger().warn(f'Serial write error: {e}')

    def _stop_motors(self) -> None:
        self._publish_rpm(0, 0, 0, 0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_motors()
        if node._ser is not None and node._ser.is_open:
            node._ser.write(b'stop')
            node._ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
