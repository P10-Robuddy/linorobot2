import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import subprocess
import threading
import os
import time

experiments = [{'name':'exp1','map': 'room1','gazebo_world':'/home/polybotdesktop/linorobot2_ws/src/linorobot2/linorobot2_gazebo/worlds/experiment_rooms/worlds/room1/world.model','cycles':10,'partitions':1,'ended': False},
                
                ]
class AutomaticTestSuite(Node):
    
    def __init__(self):
        super().__init__('automatic_test_suite')
        self.processes = {}

        # Subscription to stop commands
        self.subscription = self.create_subscription(
            Bool,
            'End_experiment',
            self.experiment_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            'map_polygonization',
            10
        )
    
    
    def experiment_callback(self, msg):
        current_experiment = self.change_experiment_state()
        next_experiment = self.get_next_experiment()
        
                
        self.stop_node(current_experiment['name'])
        self.get_logger().info(f"Ended experiment: {current_experiment['name']}")
        
        if msg.data:
            self.start_node(next_experiment)
        else:
            self.get_logger().info("False was passed to the callback, no method implemented for this")
    
    def change_experiment_state(self):
        for experiment in experiments:
            if not experiment['ended']:
                experiment['ended'] = True
                return experiment

    def get_next_experiment(self):
        for experiment in experiments:
            if not experiment['ended']:
                return experiment
    
            
    def start_node(self, experiment_params):
        # Start the node using subprocess.Popen
        process = subprocess.Popen(['ros2', 'launch','robuddy_coordinator', "experiments_control.launch.py", 
                                    'world:=' + experiment_params['gazebo_world'], 'cycles:=' + str(experiment_params['cycles']),
                                    'partitions:=' + str(experiment_params['partitions'])])
        self.processes[experiment_params['name']] = process
        self.get_logger().info(f"{experiment_params['name']} started with PID {process.pid}")
        map_name = "generated_map"
        map_dir = os.path.join("/home/polybotdesktop/linorobot2_ws/src/linorobot2/robuddy_coordinator/robuddy_coordinator/generated_files/experiments",experiment_params['map'])
        map_filepath = os.path.join(map_dir, map_name)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        msg = String()
        msg.data = map_filepath
        # Wait for subscribers to be available
        while self.publisher.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscribers to be available...")
            time.sleep(0.1)
        
        self.get_logger().info(f"Publishing message: {msg.data}")
        self.publisher.publish(msg)

    def stop_node(self, node_name):
        # Stop the node by killing the process
        if node_name in self.processes:
            process = self.processes[node_name]
            process.terminate()  # Graceful termination
            try:
                process.wait(timeout=5)  # Wait for process to terminate
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if it doesn't terminate
            self.get_logger().info(f'{node_name} stopped')
            del self.processes[node_name]
        else:
            self.get_logger().warning(f'Node {node_name} not found')

    def shutdown(self):
        # Stop all nodes
        for node_name in list(self.processes.keys()):
            self.stop_node(node_name)

def main(args=None):
    rclpy.init(args=args)

    test_suite = AutomaticTestSuite()

    # Starting nodes in a separate thread to keep the main thread responsive
    def start_node():
        test_suite.start_node(test_suite.get_next_experiment())

    thread = threading.Thread(target=start_node)
    thread.start()

    try:
        # Spin to process callbacks (e.g., stop commands)
        rclpy.spin(test_suite)
    except KeyboardInterrupt:
        pass
    finally:
        test_suite.shutdown()
        test_suite.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
