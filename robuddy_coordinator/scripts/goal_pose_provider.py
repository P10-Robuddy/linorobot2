#! /usr/bin/env python3
# Copyright 2021 Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from datetime import datetime
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration  # Handles time for ROS 2
import rclpy  # Python client library for ROS 2

from scripts.robot_navigator import BasicNavigator, NavigationResult # Helper Module

import csv
import numpy as np
class goal_provider:
    def main():
        
        # Start the ROS 2 Python Client Library
        #rclpy.init()
        
        # Launch the ROS 2 Navigation Stack
        navigator = BasicNavigator()
        
        # Set the robot's initial pose if necessary
        # initial_pose = PoseStamped()
        # initial_pose.header.frame_id = 'map'
        # initial_pose.header.stamp = navigator.get_clock().now().to_msg()
        # initial_pose.pose.position.x = 0.0
        # initial_pose.pose.position.y = 0.0
        # initial_pose.pose.position.z = 0.0
        # initial_pose.pose.orientation.x = 0.0
        # initial_pose.pose.orientation.y = 0.0
        # initial_pose.pose.orientation.z = 0.0
        # initial_pose.pose.orientation.w = 1.0
        # navigator.setInitialPose(initial_pose)
        
        # Activate navigation, if not autostarted. This should be called after setInitialPose()
        # or this will initialize at the origin of the map and update the costmap with bogus readings.
        # If autostart, you should `waitUntilNav2Active()` instead.
        # navigator.lifecycleStartup()
        
        # Wait for navigation to fully activate. Use this line if autostart is set to true.
        #navigator.waitUntilNav2Active(localizer="bt_navigator")
        
        # If desired, you can change or load the map as well
        #navigator.changeMap('/path/to/map.yaml')
        
        # You may use the navigator to clear or obtain costmaps
        # navigator.clearAllCostmaps()  # also have clearLocalCostmap() and clearGlobalCostmap()
        # global_costmap = navigator.getGlobalCostmap()
        # local_costmap = navigator.getLocalCostmap()
        waypoint_file = open('/home/polybotdesktop/linorobot2_ws/src/linorobot2/robuddy_coordinator/robuddy_coordinator/generated_files/waypoints.csv', 'r')

        file = csv.reader(waypoint_file)



        next(file, None)
        goal_poses = []
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = navigator.get_clock().now().to_msg()
        goal_pose.pose.orientation.z = 1.0
        goal_pose.pose.orientation.w = 0.0
        for pt in file:
                goal_pose.pose.position.x = float(pt[1])
                goal_pose.pose.position.y = float(pt[2])
                goal_poses.append(deepcopy(goal_pose))

        #nav_start = navigator.get_clock().now()
        start_time = datetime.now()
        navigator.followWaypoints(goal_poses)
        # start a timer here to get patrolling time for statistics
                

        # Test print the rows to see list of waypoins.
        #print(goal_poses)
        
        
        # Do something during our route (e.x. AI to analyze stock information or upload to the cloud)
            # Simply the current waypoint ID for the demonstation
        i = 0
        while not navigator.isNavComplete():
            i = i + 1
            feedback = navigator.getFeedback()
            if feedback and i % 5 == 0:
                print('Executing current waypoint: ' +
                        str(feedback.current_waypoint + 1) + '/' + str(len(goal_poses)))

        result = navigator.getResult()
        if result == NavigationResult.SUCCEEDED:
            print('Inspection of waypoints done! Returning to start...')
            end_time = datetime.now()
            timeDelta = end_time-start_time
            minutes, seconds = divmod((end_time-start_time).total_seconds(), 60)
            print(str(minutes) + "minutes" + str(seconds) + "second(s)")
            # Stop timer to get full patrolling time
        elif result == NavigationResult.CANCELED:
            print('Inspection all waypoints canceled. Returning to start...')
            exit(1)
        elif result == NavigationResult.FAILED:
            print('Inspection of waypoints failed! Returning to start...')

        # go back to start
        initial_pose.header.stamp = navigator.get_clock().now().to_msg()
        navigator.goToPose(initial_pose)
        while not navigator.isNavComplete():
            pass

        exit(0)