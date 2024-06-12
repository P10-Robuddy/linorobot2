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
import rclpy  # Python client library for ROS 2
import os

from scripts.robot_navigator import BasicNavigator, NavigationResult # Helper Module

import csv
import numpy as np
class goal_provider:
    map_dir = ''
    num_cycles = 5
    def main(self, map_dir, cycles):
        
        # Start the ROS 2 Python Client Library
        # rclpy.init()
        
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

        goal_provider.map_dir = map_dir
        goal_provider.num_cycles = cycles

        
        waypoint_path = os.path.join(map_dir, 'waypoints.csv')
        print('the waypoint path is: ' + waypoint_path)


        with open(waypoint_path, 'r') as waypoint_file:
            file = list(csv.reader(waypoint_file))

        file_length = len(file)

        file = file[1:]


   
        start_time = datetime.now()

        field_names = ['Cycle','TimeDelta']
        cycle_dictionary = {'Cycle': 0, 'TimeDelta': 0
        }
        num_partitions = 2
        for row in file:
            if int(row[3]) > num_partitions:
                num_partitions = int(row[3])

        file = file[1:]

        for partition in range(num_partitions+1):
            # Initialize dictionaries and additional variables
            idleness_dict = {i: [] for i in range(file_length)}
            last_visited_dict = {i: None for i in range(file_length)}  
             
            for cycle in range(goal_provider.num_cycles):
                goal_poses = []
                goal_poses.append(deepcopy(initial_pose))
                for row in file:
                    if int(row[3]) == partition:
                        print(f'creating goal pose for waypoint {row[0]}')
                        #print(f'Waypoint: {row[0]}, X: {row[1]}, Y: {row[2]}, Partition: {row[3]}')
                        goal_pose = PoseStamped()
                        goal_pose.header.frame_id = 'map'
                        goal_pose.header.stamp = navigator.get_clock().now().to_msg()
                        goal_pose.pose.orientation.z = 1.0
                        goal_pose.pose.orientation.w = 0.0
                        goal_pose.pose.position.x = float(row[1])
                        goal_pose.pose.position.y = float(row[2])
                        goal_poses.append(deepcopy(goal_pose))


                print(f"Starting rotation {cycle + 1}")
                # start a timer here to get patrolling time for statistics
                cycle_start_time = datetime.now()
                navigator.followWaypoints(goal_poses)
                for pose in goal_poses:
                    print(pose)



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
                                print(f'Waypoint {current_waypoint} idleness: {idleness:.2f} seconds')
                            else:
                                # Record the initial visit time
                                last_visited_dict[current_waypoint] = now
                                print(f'Waypoint {current_waypoint} visited for the first time.')

                            # Update last visited time
                            last_visited_dict[current_waypoint] = now
                            previous_waypoint = current_waypoint
                        print(f'Executing current waypoint: {current_waypoint + 1}/{len(goal_poses)} in cycle {cycle +1} of {goal_provider.num_cycles} in path: {partition}')


                result = navigator.getResult()
                if result == NavigationResult.SUCCEEDED:
                    end_time = datetime.now()
                    minutes, seconds = divmod((end_time - cycle_start_time).total_seconds(), 60)
                    print(f"Rotation {cycle + 1} completed in {int(minutes)} minutes and {int(seconds)} seconds")
                    cycle_dictionary['Cycle'],cycle_dictionary['TimeDelta'] = cycle, int((end_time - cycle_start_time).total_seconds())
                    self.save_cycle_to_csv(field_names, cycle_dictionary)

                elif result == NavigationResult.CANCELED:
                    print('Inspection of waypoints canceled. Returning to start...')
                    break
                elif result == NavigationResult.FAILED:
                    print('Inspection of waypoints failed! Returning to start...')
                    break
            self.save_idleness_to_csv(idleness_dict, 'idleness.csv')
            


        print("All rotations completed. Shutting down...")
        rclpy.shutdown()
        # initial_pose.header.stamp = navigator.get_clock().now().to_msg()
        # navigator.goToPose(initial_pose)





    def save_idleness_to_csv(self, idleness_dict, filename):
        file_path = os.path.join(goal_provider.map_dir, filename)
        
        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writerow(['Waypoint', 'Idleness_Values'])
                
            # Write the data
            for waypoint, idleness_list in idleness_dict.items():
                writer.writerow([waypoint] + idleness_list)
            file.close()


    def save_cycle_to_csv(self, fields, dictionary):
        file_path = os.path.join(goal_provider.map_dir, 'PatrollingCycleTime.csv')
        
        # Check if the file exists
        file_exists = os.path.isfile(file_path)
        
        # Open the file in append mode
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            
            # Write the header if the file does not exist or is empty
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writeheader()
            
            # Write the data row
            writer.writerow(dictionary)