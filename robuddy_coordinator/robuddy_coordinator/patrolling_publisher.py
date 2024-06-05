import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import scripts.goal_pose_provider as goal_provider
import subprocess

class patrolling_publisher(Node):
    def __init__(self):
        super().__init__('patrolling_publisher') #gives the node its name
        self.get_logger().info('Patrolling Publisher Node has been started.')
        self.subscriber = self.create_subscription(Bool, "patrolling_publisher", self.callback_patrolling_publisher ,10)

    def callback_patrolling_publisher(self, msg):
        self.get_logger().info("Message date is: " + str(msg.data))
        gp = goal_provider.goal_provider()
        subprocess.Popen(['ros2', 'launch','robuddy_coordinator', 'waypoint_publisher.launch.py'])
        if msg.data:
            self.get_logger().info("Starting Patrolling waypoints")
            gp.main()
        

            


def main(args=None):
    rclpy.init(args=args)
    node = patrolling_publisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
