import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    return wrapped


class LqrPidController(Node):
    def __init__(self) -> None:
        super().__init__("lqr_pid_controller")

        # Topics and limits
        self.declare_parameter("state_topic", "/state_estimate")
        self.declare_parameter("reference_pose_topic", "/path_reference")
        self.declare_parameter("reference_velocity_topic", "/path_reference_velocity")
        self.declare_parameter("steering_topic", "/steering_cmd")
        self.declare_parameter("throttle_topic", "/throttle_cmd")
        self.declare_parameter("max_steering_angle", 0.6)  # radians
        self.declare_parameter("throttle_min", 0.0)
        self.declare_parameter("throttle_max", 1.0)

        # LQR gains applied to [lateral_error, heading_error, lateral_velocity]
        self.declare_parameter("lqr_gains", [0.8, 1.2, 0.05])

        # PID gains for throttle control (tracking reference speed)
        self.declare_parameter("pid_kp", 1.5)
        self.declare_parameter("pid_ki", 0.2)
        self.declare_parameter("pid_kd", 0.05)

        # Load parameters
        self.state_topic = self.get_parameter("state_topic").get_parameter_value().string_value
        self.ref_pose_topic = (
            self.get_parameter("reference_pose_topic").get_parameter_value().string_value
        )
        self.ref_vel_topic = (
            self.get_parameter("reference_velocity_topic").get_parameter_value().string_value
        )
        self.steering_topic = self.get_parameter("steering_topic").get_parameter_value().string_value
        self.throttle_topic = self.get_parameter("throttle_topic").get_parameter_value().string_value
        self.max_steering = float(
            self.get_parameter("max_steering_angle").get_parameter_value().double_value
        )
        self.throttle_min = float(
            self.get_parameter("throttle_min").get_parameter_value().double_value
        )
        self.throttle_max = float(
            self.get_parameter("throttle_max").get_parameter_value().double_value
        )
        self.lqr_gains = np.array(
            self.get_parameter("lqr_gains").get_parameter_value().double_array_value, dtype=float
        )
        if self.lqr_gains.size != 3:
            self.get_logger().warn(
                "lqr_gains should have 3 elements for [lateral_error, heading_error, lateral_velocity]."
            )
        self.pid_kp = self.get_parameter("pid_kp").get_parameter_value().double_value
        self.pid_ki = self.get_parameter("pid_ki").get_parameter_value().double_value
        self.pid_kd = self.get_parameter("pid_kd").get_parameter_value().double_value

        # Publishers/Subscribers
        self.state_sub = self.create_subscription(Odometry, self.state_topic, self._on_state, 10)
        self.ref_pose_sub = self.create_subscription(
            PoseStamped, self.ref_pose_topic, self._on_reference_pose, 10
        )
        self.ref_vel_sub = self.create_subscription(
            Float64, self.ref_vel_topic, self._on_reference_velocity, 10
        )
        self.steering_pub = self.create_publisher(Float64, self.steering_topic, 10)
        self.throttle_pub = self.create_publisher(Float64, self.throttle_topic, 10)

        # State storage
        self.current_state: Optional[dict] = None
        self.reference: Optional[dict] = None
        self.reference_velocity: Optional[float] = None

        # PID integrator state
        self._last_control_time = None
        self._last_speed_error = 0.0
        self._integral_error = 0.0

        self.get_logger().info(
            f"Listening for state on {self.state_topic}, reference pose on {self.ref_pose_topic}, "
            f"reference velocity on {self.ref_vel_topic}. Publishing steering to {self.steering_topic} "
            f"and throttle to {self.throttle_topic}."
        )

    def _on_state(self, msg: Odometry) -> None:
        yaw = self._yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self.current_state = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "yaw": yaw,
            "vx": msg.twist.twist.linear.x,
            "vy": msg.twist.twist.linear.y,
        }
        self._compute_and_publish()

    def _on_reference_pose(self, msg: PoseStamped) -> None:
        psi_ref = self._yaw_from_quaternion(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self.reference = {
            "x_ref": msg.pose.position.x,
            "y_ref": msg.pose.position.y,
            "psi_ref": psi_ref,
        }
        self._compute_and_publish()

    def _on_reference_velocity(self, msg: Float64) -> None:
        self.reference_velocity = float(msg.data)
        self._compute_and_publish()

    def _compute_and_publish(self) -> None:
        # Ensure we have state and reference pose
        if self.current_state is None or self.reference is None:
            return

        if self.reference_velocity is None:
            # If we have no reference velocity yet, do not publish throttle to avoid surprises.
            self.get_logger().throttle(self.get_clock(), 10.0, "Waiting for reference velocity...")
            return

        now = self.get_clock().now()
        # Position errors in reference frame (lateral_error is positive to the left of the ref heading)
        dx = self.reference["x_ref"] - self.current_state["x"]
        dy = self.reference["y_ref"] - self.current_state["y"]
        psi_ref = self.reference["psi_ref"]

        lateral_error = -dx * math.sin(psi_ref) + dy * math.cos(psi_ref)
        heading_error = normalize_angle(self.current_state["yaw"] - psi_ref)
        lateral_velocity = -self.current_state["vx"] * math.sin(psi_ref) + self.current_state["vy"] * math.cos(
            psi_ref
        )

        state_vec = np.array([lateral_error, heading_error, lateral_velocity], dtype=float)
        steering = float(-np.dot(self.lqr_gains, state_vec))
        steering = max(-self.max_steering, min(self.max_steering, steering))

        speed = math.hypot(self.current_state["vx"], self.current_state["vy"])
        speed_error = self.reference_velocity - speed

        throttle = self._pid_step(speed_error, now)
        throttle = max(self.throttle_min, min(self.throttle_max, throttle))

        steering_msg = Float64()
        steering_msg.data = steering
        throttle_msg = Float64()
        throttle_msg.data = throttle

        self.steering_pub.publish(steering_msg)
        self.throttle_pub.publish(throttle_msg)

    def _pid_step(self, error: float, now) -> float:
        if self._last_control_time is None:
            self._last_control_time = now
            self._last_speed_error = error
            return 0.0

        dt = (now - self._last_control_time).nanoseconds / 1e9
        if dt <= 0.0:
            return 0.0

        self._integral_error += error * dt
        derivative = (error - self._last_speed_error) / dt

        output = self.pid_kp * error + self.pid_ki * self._integral_error + self.pid_kd * derivative

        self._last_control_time = now
        self._last_speed_error = error
        return output

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        # Quaternion to yaw (assuming q = [x, y, z, w])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


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
