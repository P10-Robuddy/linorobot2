import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import datetime
import os
import subprocess

start_time = datetime.datetime.now()

class exploration_listener(Node):

    def __init__(self):
        super().__init__("exploration_listener")
        self.subscriber = self.create_subscription(Bool, "exploration_listener", self.callback_exploration ,10)
        self.get_logger().info("Waiting for exploration to be finished")
        #create publisher here, to publish to robuddy_coordinator
        self.publisher = self.create_publisher(String, 'map_polygonization', 10)
   

    def callback_exploration(self, message):
        if message.data:
            self.get_logger().info("Exploration is completed! Map processing is initialized")
            
            current_time = datetime.datetime.now().strftime("%d-%H-%M-%S")
            end_time = datetime.datetime.now()
            minutes, seconds = divmod((end_time-start_time).total_seconds(), 60)
            self.get_logger().info(str(minutes) + "minutes" + str(seconds) + "second(s)")


            #create map and store location
            map_name = "generated_map" + current_time
            map_filepath = os.path.join("/home/polybotdesktop/linorobot2_ws/src/linorobot2/robuddy_coordinator/robuddy_coordinator/generated_files", map_name)
            subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_filepath, '--ros-args', '-p', 'save_map_timeout:=10000.'])
            self.get_logger().info("Map saved succesfully as: " + map_name)
            #todo, publish filepath to robuddy coordinator
            self.get_logger().info("Saved file at: " + current_time)
            msg = String()
            msg.data = map_filepath
            self.publisher.publish(msg)
            self.get_logger().info("Filepath: " +  msg.data + "Sent to robuddy coordinator")

        else:
            startTime = datetime.datetime.now()
            self.get_logger().info("Exploration started")



def main(args=None):
    rclpy.init(args=args)
    node = exploration_listener()
    rclpy.spin(node)

if __name__ == "__main__":
    main()

        

