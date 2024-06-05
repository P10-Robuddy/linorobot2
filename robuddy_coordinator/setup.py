from setuptools import find_packages, setup
import os

package_name = 'robuddy_coordinator'

scripts = [
    os.path.join('scripts', 'polygonization.py'),
    os.path.join('scripts', 'goal_pose_provider.py'),
    os.path.join('scripts', 'patrolling_publisher.py') 
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={'': ['launch/*.launch.py']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/coordinator.launch.py']),
        ('share/' + package_name + '/launch',['launch/slam_manager.launch.py']),
        ('share/' + package_name + '/launch',['launch/waypoint_publisher.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='polybotdesktop',
    maintainer_email='jacob57@live.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exploration_listener = robuddy_coordinator.exploration_listener:main',
            'robuddy_coordinator = robuddy_coordinator.robuddy_coordinator:main',
            'map_polygonization = robuddy_coordinator.map_polygonization:main',
            'patrolling_publisher = robuddy_coordinator.patrolling_publisher:main',
            'slam_node_manager = robuddy_coordinator.slam_node_manager:main',
            'waypoint_publisher = robuddy_coordinator.waypoint_publisher:main'
        ],
    },
)