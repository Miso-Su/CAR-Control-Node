# CAR-Control-Node
This is the ROS 2 Node for the CAR team's control system

# Log 26th of February
Initial commit by Ian Pichs, first pass at the decoupled PID Throttle and LQR for Steering with placeholders for topics from other subteams. Code is currently undergoing review. Development is underway on current issues.


# Issues to address:

No integral windup protection in _pid_step. If the robot is stuck or saturated, _integral_error grows unbounded. Add a clamp:

python   self._integral_error = max(-integral_max, min(integral_max, self._integral_error + error * dt))

_compute_and_publish called 3x per cycle — once per subscriber callback. You're recomputing and republishing commands whenever any of the three topics arrives, even if only one updated. Consider using a timer-based control loop instead, publishing at a fixed rate.

self.get_logger().throttle(...) API — the ROS2 Python API for throttled logging is self.get_logger().warning(msg, throttle_duration_sec=10.0), not the C++-style call you have. Your current call will likely throw an error.

No timestamp validation — if state and reference poses have very different timestamps (stale data), you'll compute on mismatched data silently. Worth logging a warning if the stamps differ by more than some threshold.

lateral_error sign convention comment says "positive to the left" but whether that's consistent with your steering sign -np.dot(...) depends on your vehicle model. Worth a unit test or sanity check on a simple straight-line case.

Minor: lqr_gains size check warns but doesn't fail — if someone passes 2 values, np.dot will silently crash later. Better to raise early.
