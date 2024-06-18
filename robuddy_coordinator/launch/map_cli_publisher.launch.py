import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
            Node(
                package='robuddy_coordinator',
                executable='map_cli_publisher',
                name='map_cli_publisher',
                output='screen'
            )
    ])
