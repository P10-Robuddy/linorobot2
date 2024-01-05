#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SendMessage(Node):

    def __init__(self):
        super().__init__("send_message_continuous")

        # Initalise node
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info("Send Message Continuous Node Started")

    # Give a heartbeat and show elapsed time
    def send_velocity_command(self, linear_velocity, angular_velocity):
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = angular_velocity
        self.cmd_vel_pub.publish(msg)

        self.get_logger().info("Sent command: linear velocity: " + str(linear_velocity) + ", angular velocity: " + str(angular_velocity))

def main(args=None):
    rclpy.init(args=args)
    node = SendMessage()

    while True:
        try:
            # Get user input for both linear and angular velocities
            input_str = input("Enter linear and angular velocities (e.g., 2.0 1.0): ")
            velocities = [float(val) for val in input_str.split()]

            if len(velocities) == 2:
                # Send the velocity command
                node.send_velocity_command(velocities[0], velocities[1])
            else:
                print("Invalid input. Please enter two numeric values separated by a space.")

        except ValueError:
            print("Invalid input. Please enter numeric values.")

if __name__ == '__main__':
    main()
