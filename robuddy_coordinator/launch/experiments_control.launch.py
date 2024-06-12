import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():

    gazebo_launch_path =  PathJoinSubstitution(
        [FindPackageShare('linorobot2_gazebo'), 'launch', 'gazebo.launch.py']
    )
    world_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_gazebo"), "worlds","experiment_rooms", "worlds", "room1", "world.model"]
    )

    nav2_launch_path = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'),
         'launch', 'navigation_launch.py']
    )

    nav_path = PathJoinSubstitution(
        [
            FindPackageShare('linorobot2_navigation'),'launch','navigation.launch.py'
        ]
    )

    slam_launch_path = PathJoinSubstitution(
        [
            FindPackageShare('linorobot2_navigation'),'launch','slam.launch.py'
        ]
    )

    # explore_lite_path = PathJoinSubstitution(
    #     [
    #         FindPackageShare('explore_lite'), 'launch', 'explore.launch.py'
    #     ]
    # )

    nav2_params_file = '/home/polybotdesktop/linorobot2_ws/src/linorobot2/linorobot2_navigation/config/navigation.yaml'

    return LaunchDescription([
        DeclareLaunchArgument(
            name='sim',
            default_value='false',
            description='Enable use_sime_time to true'
        ),

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

        DeclareLaunchArgument(
            name='world', 
            default_value=world_path,
            description='Gazebo world'
        ),


        Node(
            package='robuddy_coordinator',
            executable='exploration_listener',
            name='exploration_listener',
            output='screen',
            ),

        # Node(
        #         package='robuddy_coordinator',
        #         executable='slam_node_manager',
        #         name='slam_node_manager'
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            launch_arguments={
                'sim': LaunchConfiguration("sim"),
                'params_file': nav2_params_file
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_path),
            launch_arguments={
                'sim': LaunchConfiguration("sim"),
                'rviz': 'True'
            }.items()
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
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            launch_arguments={
                    'world':LaunchConfiguration('world')
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch_path),
            launch_arguments={
                    'sim': LaunchConfiguration("sim"),
                    'mode': 'localization'
            }.items()
        ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(explore_lite_path)
        # )
    ])