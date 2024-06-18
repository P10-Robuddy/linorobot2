import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import datetime
import os
import subprocess
import csv


class exploration_listener(Node):
    start_time = datetime.datetime.now()


    def __init__(self):
        super().__init__("exploration_listener")
        self.subscriber = self.create_subscription(Bool, "exploration_listener", self.callback_exploration ,10)
        self.get_logger().info("Waiting for exploration to be finished")
        #create publisher here, to publish directly to the polygonization node, to force the pipeline direction
        self.publisher = self.create_publisher(String, 'map_polygonization', 10)
   

    def callback_exploration(self, message):
        if message.data:
            self.get_logger().info("Exploration is completed! Map processing is initialized")
            
            current_time = datetime.datetime.now().strftime("%d-%H-%M-%S")
            end_time = datetime.datetime.now()
            timedelta = (end_time-exploration_listener.start_time).total_seconds()
            minutes, seconds = divmod(timedelta, 60)
            self.get_logger().info(str(minutes) + "minutes" + str(seconds) + "second(s)")


            #create map and store location
            map_name = "generated_map"
            map_dir = os.path.join("/home/polybotdesktop/linorobot2_ws/src/linorobot2/robuddy_coordinator/robuddy_coordinator/generated_files", current_time)
            map_filepath = os.path.join(map_dir, map_name)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)


            saveExplorationTime(map_dir, timedelta)
            subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_filepath, '--ros-args', '-p', 'save_map_timeout:=10000.'])
            #todo, publish filepath to robuddy coordinator
            self.get_logger().info("Saved file at: " + current_time)
            msg = String()
            msg.data = map_filepath
            self.publisher.publish(msg)
            self.get_logger().info("Filepath: " +  msg.data + "Sent to robuddy coordinator")

        else:
            exploration_listener.startTime = datetime.datetime.now()
            self.get_logger().info("Exploration started")


def saveExplorationTime(path, time):
    with open(os.path.join(path ,'explorationTime.csv'), mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['ExplorationTime'])
        writer.writerow([time])
        

    pass



def main(args=None):
    rclpy.init(args=args)
    node = exploration_listener()
    rclpy.spin(node)

if __name__ == "__main__":
    main()

        

