import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import datetime
import os
import subprocess

#todo, make call to robuddy_coordinater from explore.cpp when exploration is done
#then call exploration listener to save exploration map  (or maybe just let explore.cpp handle this?) and call coordinater afterwards?

#then coordinater should initiate python scripts 

#how about multiple robots?
#Should it distribute routes to each robot? 

#Should the coordinator start an exploration?

#step 1) create publisher in the exploration_listener node

#step 2) pseodo logic to start python scripts

class robuddy_coordinator(Node):
    def __init__(self):
        super().__init__('robuddy_coordinator') #gives the node its name
        self.get_logger().info('Robuddy Coordinator Node has been started.')
        self.subscriber = self.create_subscription(String, "robuddy_coordinator", self.callback_coordinator ,10)

    def callback_coordinator(self, msg):
        self.get_logger().info('Message recieved!: "%s"' % msg.data)
        #start executing python scripts


def main(args=None):
    rclpy.init(args=args)
    node = robuddy_coordinator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
