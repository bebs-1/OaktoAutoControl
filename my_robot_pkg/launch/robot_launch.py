import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'dev': '/dev/input/js0'}]
    )

    command_sender_node = Node(
        package='my_robot_pkg',
        executable='command_sender.py',
        name='motor_commander'
    )

    motor_driver_node = Node(
        package='my_robot_pkg',
        executable='motor_driver',
        name='motor_driver'
    )

    return LaunchDescription([
        joy_node,
        command_sender_node,
        motor_driver_node
    ])