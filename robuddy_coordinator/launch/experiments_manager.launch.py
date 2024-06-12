import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='robuddy_coordinator',
            executable='automatic_test_suite',
            name='automatic_test_suite',
            output='screen'
        )
    ])