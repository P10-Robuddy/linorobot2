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
    def main(self):
        
        # Start the ROS 2 Python Client Library
        #rclpy.init()
        
        # Launch the ROS 2 Navigation Stack
        navigator = BasicNavigator()
        
        # Set the robot's initial pose if necessary
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.position.z = 0.0
        initial_pose.pose.orientation.x = 0.0
        initial_pose.pose.orientation.y = 0.0
        initial_pose.pose.orientation.z = 0.0
        initial_pose.pose.orientation.w = 1.0
        navigator.setInitialPose(initial_pose)
        
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
        goal_poses.append(deepcopy(initial_pose))
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = navigator.get_clock().now().to_msg()
        goal_pose.pose.orientation.z = 1.0
        goal_pose.pose.orientation.w = 0.0
        for pt in file:
                goal_pose.pose.position.x = float(pt[1])
                goal_pose.pose.position.y = float(pt[2])
                goal_poses.append(deepcopy(goal_pose))


        # Initialize idleness dictionary to store lists of time deltas
        idleness_dict = {i: [] for i in range(len(goal_poses))}
        last_visited_dict = {i: None for i in range(len(goal_poses))}

        numCycles = 3 # Number of cycles to get idleness values
        #nav_start = navigator.get_clock().now()
        for cycle in range(numCycles):
            print(f"Starting rotation {cycle + 1}")
            # start a timer here to get patrolling time for statistics
            start_time = datetime.now()
            navigator.followWaypoints(goal_poses)

            previous_waypoint = -1

            while not navigator.isNavComplete():
                feedback = navigator.getFeedback()
                if feedback:
                    current_waypoint = feedback.current_waypoint
                    now = datetime.now()
                    if current_waypoint != previous_waypoint:   
                        # Calculate and store idleness
                        if last_visited_dict[current_waypoint] is not None:
                            idleness = (now - last_visited_dict[current_waypoint]).total_seconds()
                            idleness_dict[current_waypoint].append(idleness)
                            print(f"Just saved idleness: {idleness}, for waypoint {current_waypoint}")
                            last_visited_dict[current_waypoint] = now
                            previous_waypoint = current_waypoint
       
                    print(f'Executing current waypoint: {current_waypoint + 1}/{len(goal_poses)} in cycle {cycle} of {numCycles}')


            result = navigator.getResult()
            if result == NavigationResult.SUCCEEDED:
                print('Inspection of waypoints done! Returning to start...')
            elif result == NavigationResult.CANCELED:
                print('Inspection of waypoints canceled. Returning to start...')
                break
            elif result == NavigationResult.FAILED:
                print('Inspection of waypoints failed! Returning to start...')
                break


            initial_pose.header.stamp = navigator.get_clock().now().to_msg()
            navigator.goToPose(initial_pose)
            while not navigator.isNavComplete():
                pass

            end_time = datetime.now()
            minutes, seconds = divmod((end_time - start_time).total_seconds(), 60)
            print(f"Rotation {cycle + 1} completed in {int(minutes)} minutes and {int(seconds)} seconds")

        print("All rotations completed. Shutting down...")
        rclpy.shutdown()

        # Save the idleness dictionary to a CSV file
        self.save_idleness_to_csv(idleness_dict, 'idleness.csv')


    def save_idleness_to_csv(self, idleness_dict, filename):
        with open('/home/polybotdesktop/linorobot2_ws/src/linorobot2/robuddy_coordinator/robuddy_coordinator/generated_files' + filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Waypoint', 'Idleness_Values'])
            # Write the data
            for waypoint, idleness_list in idleness_dict.items():
                writer.writerow([waypoint] + idleness_list)