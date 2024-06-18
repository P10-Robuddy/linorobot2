import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import scripts.goal_pose_provider as goal_provider
import subprocess
import os

class patrolling_publisher(Node):
    map_dir = ''
    cycles = 2
    def __init__(self):
        super().__init__('patrolling_publisher') #gives the node its name
        self.declare_parameter('cycles', 2)
        patrolling_publisher.cycles = self.get_parameter('cycles').get_parameter_value().integer_value
        self.get_logger().info('Patrolling Publisher Node has been started.')
        self.subscriber = self.create_subscription(Bool, "patrolling_publisher", self.callback_patrolling_publisher ,10)
        self.mapSubscriber = self.create_subscription(String, "map_polygonization", self.callback_mapSubscriber ,10)
        self.endpublisher = self.create_publisher(Bool, "End_experiment",10)

    def callback_patrolling_publisher(self, msg):
        self.get_logger().info("Message date is: " + str(msg.data))
        gp = goal_provider.goal_provider()
        #subprocess.Popen(['ros2', 'launch','robuddy_coordinator', 'waypoint_publisher.launch.py'])
        if msg.data:
            self.get_logger().info("Starting Patrolling waypoints")
            self.get_logger().info("Providing map dir: " + patrolling_publisher.map_dir)
            gp.main(patrolling_publisher.map_dir, patrolling_publisher.cycles)
            msg = Bool()
            msg.data = True
            self.endpublisher.publish(msg)


    def callback_mapSubscriber(self,msg):
        patrolling_publisher.map_dir = os.path.dirname(msg.data)
        

            


def main(args=None):
    rclpy.init(args=args)
    node = patrolling_publisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
