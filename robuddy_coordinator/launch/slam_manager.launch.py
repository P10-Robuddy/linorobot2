import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
            Node(
                package='robuddy_coordinator',
                executable='slam_node_manager',
                name='slam_node_manager',
                output='screen'
            )
    ])
