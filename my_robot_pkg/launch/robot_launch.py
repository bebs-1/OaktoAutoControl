import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ros2_controller_node = Node(
        package='my_robot_pkg',
        executable='ros2_controller.py',
        name='robot_controller'
    )

    motor_driver_node = Node(
        package='my_robot_pkg',
        executable='motor_driver',
        name='motor_driver'
    )

    return LaunchDescription([
        ros2_controller_node,
        motor_driver_node
    ])
