import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import os
import sys
import cv2
import scripts.polygonization as poly
import pathlib

class map_polygonization(Node):
    num_partitions = 1 # set default to be 1 
    def __init__(self):
        super().__init__('map_polygonization') #gives the node its name
        self.get_logger().info('Map Polygonization Node has been started.')
        self.declare_parameter('partitions', 1)
        map_polygonization.num_partitions = self.get_parameter('partitions').get_parameter_value().integer_value
        self.subscriber = self.create_subscription(String, "map_polygonization", self.callback_polygonization ,10)
        self.publisher = self.create_publisher(Bool, 'patrolling_publisher', 10)
        self.slamPub = self.create_publisher(Bool, "stop_node", 10)

    def callback_polygonization(self, msg):
        self.get_logger().info('Message recieved!: "%s"' % msg.data)
        mp = poly.MapProcessing()
        mv = poly.MapVisualization()
        

        map_path = msg.data + ".pgm"
        print("map path: " + map_path)
        yaml_path = os.path.splitext(msg.data)[0] + ".yaml"
        print("yaml path: " + yaml_path)
        export_path = os.path.join(os.path.dirname(msg.data), 'waypoints.csv')

        # Load the PGM file
        mapImage = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        # Polygonize the image
        polygons = mp.polygonizeImage(mapImage)

        # Triangulate the polygons
        triangles, all_points = mp.triangulatePolygons(polygons)

        # Calculate waypoints (centers of triangles)
        waypoints = mp.calculateWaypoints(triangles, all_points)

        # Create the waypoint graph
        G = mp.createWaypointGraph(waypoints, polygons)

        # Partition graph
        partitions = mp.partitionGraph(G, map_polygonization.num_partitions)

        # Create closed paths for each subgraph
        closed_paths = mp.createClosedPaths(partitions)
        
        # Load the YAML file
        yaml_data = mp.readYaml(yaml_path)

        # Remove duplicates from closed paths
        closed_paths = [list(dict.fromkeys(path)) for path in closed_paths]
        print("Closed paths after removing duplicates:", closed_paths)

        # Export waypoints and closed paths to CSV
        mp.exportWaypointsToCSV(G, closed_paths, mapImage.shape, yaml_data, filename=export_path)
        
        mv.visualizeClosedPathsOnMap(mapImage,G, closed_paths, os.path.dirname(msg.data))
        
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)
        self.slamPub.publish(msg)
        self.get_logger().info("Sent change slam mode flag!")
        self.get_logger().info("Sent starting flag to patrolling publisher!")
        


def main(args=None):
    rclpy.init(args=args)
    node = map_polygonization()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
