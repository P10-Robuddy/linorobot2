#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os
from std_msgs.msg import String

class WaypointsPublisher(Node):
    gen_files_dir = None
    marker_array = None
    def __init__(self):
        super().__init__('waypoints_publisher')
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints', 10)
        timer_period = 1.0  # seconds
        self.mapSubscriber = self.create_subscription(String, "map_polygonization", self.callback_mapSubscriber ,10)
        
        while WaypointsPublisher.gen_files_dir is not None:
            self.timer = self.create_timer(timer_period, self.publish_waypoints)

            waypoint_file = open(os.path.join(WaypointsPublisher.gen_files_dir, 'waypoints.csv'), 'r')

            waypoints = csv.reader(waypoint_file)
                                
            next(waypoints, None)


            self.marker_array = MarkerArray()
            for waypoint in waypoints:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.ns = "waypoints"
                marker.id = int(waypoint[0])
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(waypoint[1])
                marker.pose.position.y = float(waypoint[2])
                marker.pose.position.z = 0.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                self.marker_array.markers.append(marker)

    def publish_waypoints(self):
        for marker in self.marker_array.markers:
            marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.marker_array)
    def callback_mapSubscriber(self, msg):
        WaypointsPublisher.gen_files_dir = msg.data
def main(args=None):
    rclpy.init(args=args)
    waypoints_publisher = WaypointsPublisher()

    rclpy.spin(waypoints_publisher)
    waypoints_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
