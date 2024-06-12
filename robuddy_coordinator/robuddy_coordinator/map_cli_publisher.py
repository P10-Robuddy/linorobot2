# my_publisher_pkg/my_publisher_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class MapCliPublisher(Node):
    def __init__(self):
        super().__init__('string_publisher')
        self.publisher_ = self.create_publisher(String, 'map_polygonization', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Publisher node started')

    def timer_callback(self):
        if len(sys.argv) > 1:
            msg = String()
            msg.data = ' '.join(sys.argv[1:])
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')
        else:
            self.get_logger().warn('Waiting for map string to be sent')

def main(args=None):
    rclpy.init(args=args)
    node = MapCliPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
