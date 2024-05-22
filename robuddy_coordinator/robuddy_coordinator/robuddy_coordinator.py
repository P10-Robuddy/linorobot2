import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import datetime
import os
import subprocess

#todo, make call to robuddy_coordinater from explore.cpp when exploration is done
#then call exploration listener to save exploration map  (or maybe just let explore.cpp handle this?) and call coordinater afterwards?

#then coordinater should initiate python scripts 

#how about multiple robots?
#Should it distribute routes to each robot? 

#Should the coordinator start an exploration?

class robuddy_coordinator(Node):
    def __init__(self):
        super().__init__('rb_coordinator')
        self.get_logger().info('Robuddy Coordinator Node has been started.')

def main(args=None):
    rclpy.init(args=args)
    node = robuddy_coordinator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
