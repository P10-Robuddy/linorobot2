import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import threading

class SlamNodeManager(Node):
    def __init__(self):
        super().__init__('slam_node_manager')
        self.processes = {}

        # Subscription to stop commands
        self.subscription = self.create_subscription(
            Bool,
            'stop_node',
            self.slam_node_callback,
            10
        )

    def start_node(self, node_name, mode="mapping"):
        # Start the node using subprocess.Popen
        process = subprocess.Popen(['ros2', 'launch','linorobot2_navigation', "slam.launch.py", 'sim:=true', 'map_mode:='+mode])
        self.processes[node_name] = process
        self.get_logger().info(f'{node_name} started with PID {process.pid}')

    def slam_node_callback(self, msg):
        self.stop_node("slam_mapping")
        self.get_logger().info("Slam_Mapping killed, will now start localization")
        if msg.data:
            self.start_node("slam_localization", "localization")
        else:
            self.get_logger().info("False was passed to the callback, no method implemented for this")

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

    node_manager = SlamNodeManager()

    # Starting nodes in a separate thread to keep the main thread responsive
    def start_node():
        node_manager.start_node('slam_mapping')

    thread = threading.Thread(target=start_node)
    thread.start()

    try:
        # Spin to process callbacks (e.g., stop commands)
        rclpy.spin(node_manager)
    except KeyboardInterrupt:
        pass
    finally:
        node_manager.shutdown()
        node_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
