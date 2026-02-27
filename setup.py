from setuptools import setup

package_name = "lqr_pid_controller"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/lqr_pid_controller.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="pichs",
    maintainer_email="pichs@todo.com",
    description="ROS 2 node that computes steering via LQR and throttle via PID.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_node = lqr_pid_controller.lqr_pid_controller:main",
        ],
    },
)
