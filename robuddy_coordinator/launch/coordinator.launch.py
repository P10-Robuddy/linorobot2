import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'cycles',
            default_value='2',
            description='Number of cycles for the patrolling'
        ),

        DeclareLaunchArgument(
            'partitions',
            default_value='1',
            description='Number of graph partitions'
        ),

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
            parameters=[{'partitions': LaunchConfiguration('partitions')}]
        ),

        Node(
            package='robuddy_coordinator',
            executable='patrolling_publisher',
            name='patrolling_publisher',
            output='screen',
            parameters=[{'cycles': LaunchConfiguration('cycles')}]
        )
    ])