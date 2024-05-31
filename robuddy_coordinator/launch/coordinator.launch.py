import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

            Node(
                package='robuddy_coordinator',
                executable='exploration_listener',
                name='exploration_listener',
                output='screen',
                ),

            Node(
                package='robuddy_coordinator',
                executable='robuddy_coordinator',
                name='robuddy_coordinator',
                output='screen',
            ),

            Node(
                package='robuddy_coordinator',
                executable='map_polygonization',
                name='map_polygonization',
                output='screen',
            ),

            Node(
                package='robuddy_coordinator',
                executable='patrolling_publisher',
                name='patrolling_publisher',
                output='screen',
            )
    ])