#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys

class SendMessage(Node):

    def __init__(self, linear_velocity, angular_velocity):
        super().__init__("send_message")

        # Initalise node
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.5, self.send_velocity_command)

        self.get_logger().info("Send Message Node Started")

        # Set linear and angular velocities
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    # Give a heartbeat and show elapsed time
    def send_velocity_command(self):
        msg = Twist()
        msg.linear.x = self.linear_velocity
        msg.angular.z = self.angular_velocity
        self.cmd_vel_pub.publish(msg)

        self.get_logger().info("Sent command: linear velocity: " + str(self.linear_velocity) + ", angular velocity: " + str(self.angular_velocity))

def main(args=None):
    rclpy.init(args=args)

    # Parse command-line arguments
    linear_velocity = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    angular_velocity = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    node = SendMessage(linear_velocity, angular_velocity)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
