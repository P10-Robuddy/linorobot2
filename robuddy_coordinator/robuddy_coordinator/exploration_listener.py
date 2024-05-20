import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import datetime
import os
import subprocess


class exploration_listener(Node):

    def __init__(self):
        super().__init__("coordination_listener")
        self.subscriber = self.create_subscription(Bool, "exploration_listener", self.callback_exploration ,10) #todo, publisher topic name
        self.get_logger().info("Waiting for exploration to be finished")

    def callback_exploration(self, message):
        if message.data:
            self.get_logger().info("Exploration is completed! Map processing is initialized")
            #send map to python script <3
            current_time = datetime.datetime.now().strftime("%d-%H-%M-%S")
            map_name = "generated_map" + current_time
            map_filepath = os.path.join("/home/polybotdesktop/linorobot2_ws/src/linorobot2/linorobot2_navigation/maps/", map_name)

            subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_filepath, '--ros-args', '-p', 'save_map_timeout:=10000.'])
            self.get_logger().info("Map saved succesfully as: " + map_name)
        else:
            self.get_logger().info("Oh no, exploration no completo!")



def main(args=None):
    rclpy.init(args=args)
    node = exploration_listener()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

        

