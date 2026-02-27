"""
lqr_pid_controller.py
=====================
Discrete-time LQR steering + PID throttle controller for a ground vehicle
following a reference path in ROS2.

SYSTEM MODEL
------------
We use the linearized lateral error dynamics of a kinematic bicycle model,
expressed in the path-tangent frame. The state vector is:

    x = [e_lat, e_yaw, ė_lat]

where:
    e_lat  = lateral deviation from the reference path  [m]
    e_yaw  = heading error (vehicle yaw - path yaw)     [rad]
    ė_lat  = lateral velocity (time derivative of e_lat) [m/s]

The continuous-time linearized dynamics are:

    ẋ = A_c x + B_c u

with:
         [0   v    1]          [   0   ]
    A_c= [0   0    0]   B_c =  [ v / L ]
         [0   0    0]          [   0   ]

where v is the linearization speed [m/s] and L is the vehicle wheelbase [m].

We discretize with Zero-Order Hold (ZOH) at a fixed timestep DT using:
    A_d = expm(A_c * DT)
    B_d = (A_d - I) @ inv(A_c) @ B_c     (or scipy's cont2discrete)

The optimal gain K is found by solving the Discrete Algebraic Riccati
Equation (DARE):
    P = A_dᵀ P A_d - (A_dᵀ P B_d)(R + B_dᵀ P B_d)⁻¹(B_dᵀ P A_d) + Q
    K = (R + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d

Control law:
    u = -K x     (steering angle command)

THROTTLE
--------
A simple PID loop tracks the reference speed. Integral windup is clamped.

USAGE / TUNING GUIDE
--------------------
1. Set VEHICLE_WHEELBASE_M and LINEARIZATION_SPEED_MPS to match your robot.
2. Adjust CONTROL_DT_S to your desired loop rate (must be <= state estimator rate).
3. Tune Q_MATRIX and R_MATRIX:
   - Increase Q[0,0] to penalize lateral offset more aggressively.
   - Increase Q[1,1] to penalize heading error more aggressively.
   - Increase R_MATRIX to produce smoother (smaller) steering commands.
4. Tune PID gains for throttle: PID_KP, PID_KI, PID_KD.
5. Set PID_INTEGRAL_MAX to prevent windup.

DEPENDENCIES
------------
    pip install scipy numpy
    ROS2 (rclpy, nav_msgs, geometry_msgs, std_msgs)
"""

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from std_msgs.msg import Float64


# =============================================================================
# VEHICLE PHYSICAL PARAMETERS  — change these to match your robot
# =============================================================================

# Distance between front and rear axles [meters].
# Typical values: RC car ~0.25 m, go-kart ~1.0 m, sedan ~2.7 m
VEHICLE_WHEELBASE_M: float = 0.30  # <-- SET THIS

# Speed at which the bicycle model is linearized [m/s].
# The LQR gain K is computed at this single operating point.
# If your robot operates over a wide speed range, consider gain scheduling
# (recompute K at multiple speeds and interpolate).
# For gain scheduling, call _compute_lqr_gain(speed) at runtime.
LINEARIZATION_SPEED_MPS: float = 1.0  # <-- SET THIS (nominal cruise speed)


# =============================================================================
# CONTROL LOOP TIMING  — decouple from ROS topic rates
# =============================================================================

# Fixed timestep for the discrete LQR controller [seconds].
# The control loop runs on a ROS timer at this rate, NOT on topic callbacks.
# Rule of thumb: DT ~ 1/(5 to 10 × bandwidth of your mechanical system).
# At 1 m/s with L=0.3 m, bandwidth ≈ v/L = 3.3 rad/s → DT ~ 0.03–0.06 s.
CONTROL_DT_S: float = 0.05  # 20 Hz  <-- TUNE THIS


# =============================================================================
# LQR COST MATRICES  — the primary tuning knobs for steering
# =============================================================================

# State cost matrix Q (3×3 diagonal).
# States: [lateral_error (m), heading_error (rad), lateral_velocity (m/s)]
#
# Interpretation:
#   Q[0,0] — cost per m² of lateral offset.   Increase → tighter path tracking.
#   Q[1,1] — cost per rad² of heading error.  Increase → faster heading correction.
#   Q[2,2] — cost per (m/s)² of lateral vel.  Increase → damps oscillation.
#
# Start with Q = diag(1, 1, 0) and increase Q[0,0] until tracking is tight,
# then raise Q[2,2] to reduce overshoot.
Q_MATRIX: np.ndarray = np.diag([
    10.0,   # lateral error weight       [1/m²]   <-- TUNE: higher = tighter tracking
    5.0,    # heading error weight       [1/rad²] <-- TUNE: higher = faster heading fix
    0.1,    # lateral velocity weight    [1/(m/s)²] <-- TUNE: higher = less oscillation
])

# Input cost scalar R (1×1 here since we have one input: steering angle).
# Higher R → smoother, smaller steering commands (gentler driving).
# Lower R → more aggressive corrections (may cause oscillation or actuator saturation).
R_MATRIX: np.ndarray = np.array([[1.0]])  # [1/rad²]  <-- TUNE THIS


# =============================================================================
# STEERING ACTUATOR LIMITS
# =============================================================================

# Maximum achievable steering angle [radians].
# Exceeding this physically damages the servo or is kinematically impossible.
# ~0.52 rad ≈ 30 deg,  ~0.70 rad ≈ 40 deg
MAX_STEERING_ANGLE_RAD: float = 0.52  # <-- SET THIS


# =============================================================================
# PID THROTTLE GAINS  — for speed tracking
# =============================================================================

# Proportional gain: primary response to speed error [throttle / (m/s)]
PID_KP: float = 1.5  # <-- TUNE THIS

# Integral gain: eliminates steady-state speed error [throttle / (m/s·s)]
# Set to 0 first, add slowly to eliminate steady-state offset.
PID_KI: float = 0.2  # <-- TUNE THIS

# Derivative gain: damps speed oscillations [throttle / (m/s²)]
# Use sparingly; amplifies noise. Start at 0.
PID_KD: float = 0.05  # <-- TUNE THIS

# Anti-windup clamp on the integral accumulator [throttle units].
# Prevents the integrator from building up when the throttle is saturated.
PID_INTEGRAL_MAX: float = 1.0  # <-- TUNE THIS (usually ~ throttle_max / ki)

# Throttle output limits [dimensionless, e.g. 0.0–1.0 or -1.0–1.0]
THROTTLE_MIN: float = 0.0   # <-- SET THIS
THROTTLE_MAX: float = 1.0   # <-- SET THIS


# =============================================================================
# HELPER
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_continuous_system(v: float, L: float):
    """
    Return the continuous-time (A_c, B_c) for the linearized lateral error
    dynamics of a kinematic bicycle model at speed v and wheelbase L.

    State:  x = [e_lat, e_yaw, ė_lat]
    Input:  u = δ  (front steering angle)

    Derivation sketch:
        ė_lat  = v·sin(e_yaw) + ė_lat_disturbance  ≈  v·e_yaw  (small angle)
        ė_yaw  = (v / L)·tan(δ)                     ≈  (v/L)·δ  (small angle)
        ë_lat  ≈  v·ė_yaw  = (v²/L)·δ   — captured through the state coupling

    The simplified 3-state continuous model used here is:

        d/dt [e_lat ]   [0  v   1] [e_lat ]   [  0  ]
             [e_yaw ] = [0  0   0] [e_yaw ] + [v/L  ] · δ
             [ė_lat ]   [0  0   0] [ė_lat ]   [  0  ]

    NOTE: This is a simplified linearization. A more complete model would
    include cross-coupling between ė_yaw and ė_lat. Good enough for low speed.
    """
    A_c = np.array([
        [0.0,  v,   1.0],   # ė_lat  = v·e_yaw + ė_lat  (lateral kinematics)
        [0.0,  0.0, 0.0],   # ë_yaw  = 0   (yaw rate driven by input, not state)
        [0.0,  0.0, 0.0],   # ë_lat  = 0   (lateral accel not modeled)
    ], dtype=float)

    B_c = np.array([
        [0.0   ],   # e_lat not directly driven by steering
        [v / L ],   # ė_yaw = (v/L)·δ  (bicycle model yaw rate)
        [0.0   ],   # ė_lat not directly driven by steering
    ], dtype=float)

    return A_c, B_c


def discretize_system(A_c: np.ndarray, B_c: np.ndarray, dt: float):
    """
    Discretize (A_c, B_c) using Zero-Order Hold (ZOH) at timestep dt.

    Returns (A_d, B_d) suitable for the DARE solve.
    scipy.signal.cont2discrete handles the matrix exponential correctly.
    """
    # cont2discrete returns (A_d, B_d, C_d, D_d, dt) — we only need first two
    A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, np.eye(3), np.zeros((3, 1))),
                                       dt=dt, method="zoh")
    return A_d, B_d


def compute_lqr_gain(A_d: np.ndarray, B_d: np.ndarray,
                     Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the Discrete Algebraic Riccati Equation (DARE) and return the
    optimal state-feedback gain matrix K.

    DARE:  P = A_dᵀ P A_d - (A_dᵀ P B_d)(R + B_dᵀ P B_d)⁻¹(B_dᵀ P A_d) + Q
    Gain:  K = (R + B_dᵀ P B_d)⁻¹ B_dᵀ P A_d

    Control law:  u* = -K x   (minimizes Σ xᵀQx + uᵀRu over infinite horizon)

    Args:
        A_d: Discrete state transition matrix (n×n)
        B_d: Discrete input matrix (n×m)
        Q:   State cost matrix (n×n, positive semi-definite)
        R:   Input cost matrix (m×m, positive definite)

    Returns:
        K: Optimal gain matrix (m×n)
    """
    # Solve DARE for the cost-to-go matrix P
    P = solve_discrete_are(A_d, B_d, Q, R)

    # Compute optimal gain
    # K = (R + B_dᵀ P B_d)⁻¹ (B_dᵀ P A_d)
    K = np.linalg.solve(
        R + B_d.T @ P @ B_d,   # scalar (or small matrix) to invert
        B_d.T @ P @ A_d        # (m×n) numerator
    )
    return K  # shape: (1, 3) for our system


# =============================================================================
# ROS2 NODE
# =============================================================================

class LqrPidController(Node):
    """
    ROS2 node implementing:
      - Discrete LQR for lateral steering control
      - PID with anti-windup for longitudinal speed control

    Subscribes:
      /state_estimate         (nav_msgs/Odometry)   — vehicle state
      /path_reference         (geometry_msgs/PoseStamped) — reference pose
      /path_reference_velocity (std_msgs/Float64)   — reference speed

    Publishes:
      /steering_cmd           (std_msgs/Float64)    — steering angle [rad]
      /throttle_cmd           (std_msgs/Float64)    — throttle [0, 1]
    """

    def __init__(self) -> None:
        super().__init__("lqr_pid_controller")

        # ------------------------------------------------------------------
        # ROS Parameters (can be overridden via YAML or command line)
        # ------------------------------------------------------------------
        self.declare_parameter("state_topic",              "/state_estimate")
        self.declare_parameter("reference_pose_topic",     "/path_reference")
        self.declare_parameter("reference_velocity_topic", "/path_reference_velocity")
        self.declare_parameter("steering_topic",           "/steering_cmd")
        self.declare_parameter("throttle_topic",           "/throttle_cmd")

        # Physical / timing parameters (override module-level constants via ROS params if desired)
        self.declare_parameter("wheelbase_m",              VEHICLE_WHEELBASE_M)
        self.declare_parameter("linearization_speed_mps",  LINEARIZATION_SPEED_MPS)
        self.declare_parameter("control_dt_s",             CONTROL_DT_S)
        self.declare_parameter("max_steering_angle_rad",   MAX_STEERING_ANGLE_RAD)
        self.declare_parameter("throttle_min",             THROTTLE_MIN)
        self.declare_parameter("throttle_max",             THROTTLE_MAX)
        self.declare_parameter("pid_integral_max",         PID_INTEGRAL_MAX)

        # LQR cost weights — flat list [Q00, Q11, Q22, R00]
        # (passing full matrices through ROS params is awkward; use constants above for fine-tuning)
        self.declare_parameter("pid_kp", PID_KP)
        self.declare_parameter("pid_ki", PID_KI)
        self.declare_parameter("pid_kd", PID_KD)

        # Load parameters
        p = lambda name: self.get_parameter(name).value  # shorthand

        self.state_topic    = p("state_topic")
        self.ref_pose_topic = p("reference_pose_topic")
        self.ref_vel_topic  = p("reference_velocity_topic")
        self.steering_topic = p("steering_topic")
        self.throttle_topic = p("throttle_topic")

        self.wheelbase      = float(p("wheelbase_m"))
        self.v_lin          = float(p("linearization_speed_mps"))
        self.dt             = float(p("control_dt_s"))
        self.max_steering   = float(p("max_steering_angle_rad"))
        self.throttle_min   = float(p("throttle_min"))
        self.throttle_max   = float(p("throttle_max"))
        self.pid_integral_max = float(p("pid_integral_max"))
        self.pid_kp         = float(p("pid_kp"))
        self.pid_ki         = float(p("pid_ki"))
        self.pid_kd         = float(p("pid_kd"))

        # ------------------------------------------------------------------
        # Build discrete LQR gain K at startup (offline computation)
        # ------------------------------------------------------------------
        # Step 1: Linearize the bicycle model at the nominal speed
        A_c, B_c = build_continuous_system(v=self.v_lin, L=self.wheelbase)
        self.get_logger().info(
            f"Continuous system A_c:\n{A_c}\nB_c:\n{B_c}"
        )

        # Step 2: Discretize with ZOH at the control timestep
        self.A_d, self.B_d = discretize_system(A_c, B_c, self.dt)
        self.get_logger().info(
            f"Discrete system A_d:\n{self.A_d}\nB_d:\n{self.B_d}"
        )

        # Step 3: Solve DARE and compute K
        self.K = compute_lqr_gain(self.A_d, self.B_d, Q_MATRIX, R_MATRIX)
        self.get_logger().info(f"LQR gain K = {self.K}")

        # ------------------------------------------------------------------
        # State storage (populated by subscriber callbacks)
        # ------------------------------------------------------------------
        self.current_state: Optional[dict] = None   # latest odometry
        self.reference:     Optional[dict] = None   # latest reference pose
        self.reference_velocity: Optional[float] = None  # latest ref speed

        # ------------------------------------------------------------------
        # PID integrator state for throttle control
        # ------------------------------------------------------------------
        self._speed_integral: float = 0.0      # accumulated integral term
        self._last_speed_error: float = 0.0    # for derivative calculation

        # ------------------------------------------------------------------
        # Publishers / Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(Odometry,    self.state_topic,    self._on_state,    10)
        self.create_subscription(PoseStamped, self.ref_pose_topic, self._on_ref_pose, 10)
        self.create_subscription(Float64,     self.ref_vel_topic,  self._on_ref_vel,  10)

        self.steering_pub = self.create_publisher(Float64, self.steering_topic, 10)
        self.throttle_pub = self.create_publisher(Float64, self.throttle_topic, 10)

        # ------------------------------------------------------------------
        # Fixed-rate control timer  — decoupled from topic publish rates
        # ------------------------------------------------------------------
        # The LQR controller runs strictly at 1/DT Hz, using whatever the
        # most recent state and reference happen to be.
        # This avoids recomputing control on every topic arrival and ensures
        # the discrete-time assumption (constant dt) actually holds.
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f"LQR-PID controller initialized at {1.0/self.dt:.1f} Hz. "
            f"Wheelbase={self.wheelbase} m, v_lin={self.v_lin} m/s, dt={self.dt} s."
        )

    # ------------------------------------------------------------------
    # Subscriber callbacks — just store the latest data, no control here
    # ------------------------------------------------------------------

    def _on_state(self, msg: Odometry) -> None:
        """Cache the latest vehicle state from odometry."""
        yaw = _yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.current_state = {
            "x":   msg.pose.pose.position.x,
            "y":   msg.pose.pose.position.y,
            "yaw": yaw,
            "vx":  msg.twist.twist.linear.x,   # longitudinal velocity [m/s]
            "vy":  msg.twist.twist.linear.y,    # lateral velocity [m/s]
        }

    def _on_ref_pose(self, msg: PoseStamped) -> None:
        """Cache the latest reference pose (position + heading)."""
        psi_ref = _yaw_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self.reference = {
            "x_ref":   msg.pose.position.x,
            "y_ref":   msg.pose.position.y,
            "psi_ref": psi_ref,   # reference heading [rad]
        }

    def _on_ref_vel(self, msg: Float64) -> None:
        """Cache the latest reference speed."""
        self.reference_velocity = float(msg.data)

    # ------------------------------------------------------------------
    # Main control loop — runs on a fixed timer at CONTROL_DT_S
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """
        Compute and publish steering (LQR) and throttle (PID) commands.

        Called at a fixed rate by the ROS timer. Uses the most recently
        received state and reference; does nothing if data is not yet available.
        """
        # Guard: wait until all data is available
        if self.current_state is None:
            self.get_logger().warning("Waiting for state estimate...", throttle_duration_sec=5.0)
            return
        if self.reference is None:
            self.get_logger().warning("Waiting for reference pose...", throttle_duration_sec=5.0)
            return
        if self.reference_velocity is None:
            self.get_logger().warning("Waiting for reference velocity...", throttle_duration_sec=5.0)
            return

        # --------------------------------------------------------------
        # 1. Compute LQR state vector in the path-tangent frame
        # --------------------------------------------------------------
        dx = self.reference["x_ref"] - self.current_state["x"]
        dy = self.reference["y_ref"] - self.current_state["y"]
        psi_ref = self.reference["psi_ref"]

        # Lateral error: signed distance from vehicle to the reference path,
        # positive when vehicle is to the LEFT of the path direction.
        # Projection of (dx, dy) onto the path-normal direction.
        e_lat = -dx * math.sin(psi_ref) + dy * math.cos(psi_ref)   # [m]

        # Heading error: how much the vehicle yaw differs from path heading.
        # Normalized to [-π, π] to avoid wrap-around discontinuities.
        e_yaw = normalize_angle(self.current_state["yaw"] - psi_ref)  # [rad]

        # Lateral velocity: component of vehicle velocity perpendicular to path.
        # Used by LQR to damp oscillatory lateral motion.
        e_lat_dot = (
            -self.current_state["vx"] * math.sin(psi_ref)
            + self.current_state["vy"] * math.cos(psi_ref)
        )  # [m/s]

        # State vector for the LQR: x = [e_lat, e_yaw, ė_lat]
        x_err = np.array([e_lat, e_yaw, e_lat_dot], dtype=float)

        # --------------------------------------------------------------
        # 2. LQR steering command:  u = -K x
        # K has shape (1, 3), so result is shape (1,) — extract scalar.
        # Negative sign because we want to drive x → 0.
        # --------------------------------------------------------------
        steering = float(-(self.K @ x_err)[0])

        # Clamp to physical actuator limits
        steering = float(np.clip(steering, -self.max_steering, self.max_steering))

        # --------------------------------------------------------------
        # 3. PID throttle command
        # --------------------------------------------------------------
        # Current speed (magnitude, not signed)
        speed = math.hypot(self.current_state["vx"], self.current_state["vy"])  # [m/s]
        speed_error = self.reference_velocity - speed   # positive → too slow

        throttle = self._pid_step(speed_error)

        # Clamp throttle to [min, max]
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))

        # --------------------------------------------------------------
        # 4. Publish
        # --------------------------------------------------------------
        steering_msg = Float64()
        steering_msg.data = steering
        self.steering_pub.publish(steering_msg)

        throttle_msg = Float64()
        throttle_msg.data = throttle
        self.throttle_pub.publish(throttle_msg)

        # Debug log (throttled to avoid flooding)
        self.get_logger().debug(
            f"e_lat={e_lat:.3f} m  e_yaw={math.degrees(e_yaw):.1f}°  "
            f"e_lat_dot={e_lat_dot:.3f} m/s  "
            f"steer={math.degrees(steering):.1f}°  throttle={throttle:.3f}"
        )

    # ------------------------------------------------------------------
    # PID with anti-windup
    # ------------------------------------------------------------------

    def _pid_step(self, error: float) -> float:
        """
        Compute one PID step for the throttle controller.

        Uses the fixed control timestep DT (self.dt) rather than measuring
        wall-clock time, since we're called from a fixed-rate timer.

        Anti-windup: the integral term is clamped to ±PID_INTEGRAL_MAX
        to prevent accumulation during saturation.

        Args:
            error: speed_error = v_ref - v_current  [m/s]

        Returns:
            throttle output (before clamping to [min, max])
        """
        dt = self.dt   # constant timestep — no need to measure

        # Integral term with anti-windup clamping
        self._speed_integral += error * dt
        self._speed_integral = float(np.clip(
            self._speed_integral, -self.pid_integral_max, self.pid_integral_max
        ))

        # Derivative term (backward difference)
        derivative = (error - self._last_speed_error) / dt

        # PID output
        output = (
            self.pid_kp * error                         # proportional
            + self.pid_ki * self._speed_integral        # integral
            + self.pid_kd * derivative                  # derivative
        )

        self._last_speed_error = error
        return output


# =============================================================================
# STATIC UTILITIES
# =============================================================================

def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """
    Extract yaw (rotation about Z) from a unit quaternion [x, y, z, w].

    Uses the standard formula:
        yaw = atan2(2(wz + xy), 1 - 2(y² + z²))

    Returns yaw in [-π, π] radians.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LqrPidController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
