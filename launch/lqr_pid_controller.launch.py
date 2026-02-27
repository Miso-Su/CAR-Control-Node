from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="lqr_pid_controller",
                executable="controller_node",
                name="lqr_pid_controller",
                output="screen",
                parameters=[
                    {
                        # Override these in a yaml or via CLI as needed.
                        "state_topic": "/state_estimate",
                        "reference_pose_topic": "/path_reference",
                        "reference_velocity_topic": "/path_reference_velocity",
                        "steering_topic": "/steering_cmd",
                        "throttle_topic": "/throttle_cmd",
                        "lqr_gains": [0.8, 1.2, 0.05],
                        "max_steering_angle": 0.6,
                        "pid_kp": 1.5,
                        "pid_ki": 0.2,
                        "pid_kd": 0.05,
                        "throttle_min": 0.0,
                        "throttle_max": 1.0,
                    }
                ],
            )
        ]
    )
