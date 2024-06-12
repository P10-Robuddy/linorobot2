# my_publisher_pkg/my_publisher_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class MapCliPublisher(Node):
    def __init__(self, message):
        super().__init__('map_cli_publisher')
        self.publisher_ = self.create_publisher(String, 'map_polygonization', 10)
        self.publish_message(message)

    def publish_message(self, message):
        msg = String()
        msg.data = message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) > 1:
        message = ' '.join(sys.argv[1:])
        node = MapCliPublisher(message)
    else:
        print("No string provided in CLI arguments.")
        return

    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
