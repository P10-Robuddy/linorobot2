import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from ament_index_python import get_package_share_directory
from nav2_common.launch import RewrittenYaml

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
    ])