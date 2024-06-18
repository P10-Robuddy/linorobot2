#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import sys

class SendMessage(Node):

    def __init__(self):
        super().__init__("send_message")

        # Initalise node
        self.bool_pub = self.create_publisher(Bool, "exploration_listener", 10)
        self.timer = self.create_timer(1, self.send_bool_command)
        #self.send_bool_command
        self.get_logger().info("Send Message Node Started")
        
        # Set linear and angular velocities
        # self.linear_velocity = linear_velocity
        # self.angular_velocity = angular_velocity

    # Give a heartbeat and show elapsed time
    def send_bool_command(self):
        # msg = Twist()
        # msg.linear.x = self.linear_velocity
        # msg.angular.z = self.angular_velocity
        msg = Bool()
        msg.data = True
        self.bool_pub.publish(msg)

        self.get_logger().info("Sent command: " + str(msg.data))

def main(args=None):
    rclpy.init(args=args)

    # Parse command-line arguments
    # linear_velocity = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    # angular_velocity = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    node = SendMessage()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
