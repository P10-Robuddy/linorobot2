import numpy as np
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import Delaunay
import cv2
import os
import csv
import yaml
from networkx.algorithms.community import girvan_newman
from itertools import islice

class MapProcessing:

    def polygonizeImage(self, image, blur_ksize=(5, 5), area_threshold=10000) -> list:
        """
        This function preprocesses the image by applying Gaussian blur and then thresholding it to create a binary image.
        It finds contours in the binary image and filters them based on a given area threshold.
        The filtered contours are then approximated to polygons.

        Parameters:
        - image: Input grayscale image.
        - blur_ksize: Kernel size for Gaussian blur (default is (5, 5)).
        - area_threshold: Minimum area for a contour to be considered (default is 10000).

        Returns:
        - List of polygons (each polygon is an array of points).
        """

        # Preprocess the image
        blurredImage = cv2.GaussianBlur(image, blur_ksize, 0)
        _, binary_image = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                filtered_contours.append(contour)

        # Simplify contours to polygons
        polygons = []
        for contour in filtered_contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(polygon)

        return polygons

    def triangulatePolygons(self, polygons):
        """
        This function performs Delaunay triangulation on the set of points from the input polygons.

        Parameters:
        - polygons: List of polygons.

        Returns:
        - Triangles (indices of the points forming triangles).
        - Points (coordinates of the points).
        """

        all_points = np.concatenate(polygons)[:, 0, :]

        # Perform Delaunay triangulation
        triangulation = Delaunay(all_points)

        return triangulation.simplices, all_points

    def calculateWaypoints(self, triangles, points):
        """
        This function calculates the waypoints by computing the centroids of the Delaunay triangles.

        Parameters:
        - triangles: Indices of the points forming triangles.
        - points: Coordinates of the points.

        Returns:
        - Waypoints (centroids of the triangles).
        """

        # Calculate the centroid of each triangle
        waypoints = []
        for triangle_indices in triangles:
            # Calculate centroid (waypoint)
            centroid = np.mean(points[triangle_indices], axis=0, dtype=np.int32)
            waypoints.append(centroid)

        return waypoints

    def createWaypointGraph(self, waypoints, polygons):
        """
        This function creates a graph where nodes are waypoints and edges represent possible paths between waypoints that do not intersect polygons.
        Waypoints outside the polygons are removed from the graph.

        Parameters:
        - waypoints: Centroids of the triangles.
        - polygons: List of polygons for the walls.

        Returns:
        - Graph with waypoints as nodes and valid paths as edges.
        """

        G = nx.Graph()

        # Remove waypoints outside polygons
        waypoints_inside_polygons = []
        for waypoint in waypoints:
            # Convert the waypoint to tuple of integers
            waypoint_tuple = tuple(map(int, waypoint))
            if any(cv2.pointPolygonTest(polygon, waypoint_tuple, False) >= 0 for polygon in polygons):
                waypoints_inside_polygons.append(waypoint_tuple)

        # Add nodes (waypoints) to the graph
        for index, waypoint in enumerate(waypoints_inside_polygons):
            G.add_node(index, pos=waypoint)  # Store position as node attribute

        # Function to check if a line segment intersects with any polygon
        def intersects_with_polygon(p1, p2):
            for polygon in polygons:
                for i in range(len(polygon)):
                    p3 = polygon[i][0]
                    p4 = polygon[(i + 1) % len(polygon)][0]
                    # Check if the line segment intersects with any edge of the polygon
                    if segments_intersect(p1, p2, p3, p4):
                        return True
            return False

        # Function to check if two line segments intersect
        def segments_intersect(p1, p2, p3, p4):
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        discarded_edges = []

        # Add edges based on nearest neighbors, avoiding walls
        for i in range(len(waypoints_inside_polygons)):
            # Find indices of two nearest neighbors
            neighbor_indices = [idx for idx in range(len(waypoints_inside_polygons)) if idx != i]
            nearest_neighbors = sorted(neighbor_indices, key=lambda idx: np.linalg.norm(np.array(waypoints_inside_polygons[idx]) - np.array(waypoints_inside_polygons[i])))[:2]

            for neighbor in nearest_neighbors:
                if not intersects_with_polygon(waypoints_inside_polygons[i], waypoints_inside_polygons[neighbor]):
                    G.add_edge(i, neighbor)
                else:
                    discarded_edges.append((i, neighbor))
                    print(f"Edge ({i}, {neighbor}) intersects with a polygon and is not added.")

        # Find connected components (islands) in the graph
        islands = list(nx.connected_components(G))
        print("Discarded edges:", discarded_edges)

        # If there are more than one island, connect them with the closest discarded edge
        if len(islands) > 1:
            print("Islands:", islands)
            closest_edge = None
            min_distance = float('inf')
            for edge in discarded_edges:
                island1, island2 = None, None
                for island in islands:
                    if edge[0] in island:
                        island1 = island
                    if edge[1] in island:
                        island2 = island
                if island1 and island2:
                    distance = np.linalg.norm(np.array(waypoints_inside_polygons[edge[0]]) - np.array(waypoints_inside_polygons[edge[1]]))
                    if distance < min_distance:
                        min_distance = distance
                        closest_edge = edge
            if closest_edge:
                G.add_edge(closest_edge[0], closest_edge[1])
                print(f"Connecting islands with edge {closest_edge}")

        islands = list(nx.connected_components(G))
        print("Islands:", islands)

        # If there are still islands, connect the two closest islands using the shortest edge between them
        while len(islands) > 1:
            print("Islands:", islands)
            closest_edge = None
            min_distance = float('inf')
            for island1 in islands:
                for island2 in islands:
                    if island1 != island2:
                        for node1 in island1:
                            for node2 in island2:
                                distance = np.linalg.norm(np.array(waypoints_inside_polygons[node1]) - np.array(waypoints_inside_polygons[node2]))
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_edge = (node1, node2)
            if closest_edge:
                G.add_edge(closest_edge[0], closest_edge[1])
                print(f"Connecting islands with edge {closest_edge}")

            islands = list(nx.connected_components(G))

        return G


    def partitionGraph(self, Graph, num_divisions):
        """
        This function partitions the graph into `num_divisions` connected sections using the Girvan-Newman algorithm.

        Parameters:
        - Graph: The input graph to be partitioned.
        - num_divisions: The number of partitions to create.

        Returns:
        - List of subgraphs (each subgraph is a partition).
        """

        if num_divisions < 2:
            return [Graph]

        # Use the Girvan-Newman algorithm to find communities
        communities_generator = nx.algorithms.community.girvan_newman(Graph)
        limited_communities = list(islice(communities_generator, num_divisions - 1))

        # The last set of communities
        partition = limited_communities[-1]

        subgraphs = [Graph.subgraph(community).copy() for community in partition]

        return subgraphs

    def createClosedPaths(self, subgraphs):
        """
        This function creates closed paths for each subgraph using the Traveling Salesman Problem.

        Parameters:
        - subgraphs: List of subgraphs.

        Returns:
        - List of closed paths (each path is a list of nodes in the order they are visited).
        """

        closed_paths = []

        for subgraph in subgraphs:
            # Find the optimal path using TSP
            tsp_path = traveling_salesman_problem(subgraph, cycle=True)
            closed_paths.append(tsp_path)

        return closed_paths

    def readYaml(self, yaml_file):
        """
        This function reads the YAML file and extracts the necessary information.

        Parameters:
        - yaml_file: Path to the YAML file.

        Returns:
        - Dictionary with image metadata.
        """

        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def exportWaypointsToCSV(self, G, closed_paths, image_shape, yaml_data, filename='waypoints.csv'):
        """
        This function exports the waypoints and closed paths to a CSV file, with coordinates adjusted based on the YAML file.

        Parameters:
        - G: Graph with waypoints as nodes and valid paths as edges.
        - closed_paths: List of closed paths (each path is a list of waypoint indices).
        - image_shape: Shape of the image (height, width).
        - yaml_data: Metadata from the YAML file.
        - filename: Name of the CSV file (default is 'waypoints.csv').
        """

        height, width = image_shape
        resolution = yaml_data['resolution']
        origin = np.array(yaml_data['origin'][:2])  # We only need the first two values (x and y)

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Waypoint Index', 'X (meters)', 'Y (meters)', 'Path Index'])

                for path_index, path in enumerate(closed_paths):
                    # Write waypoints for each path
                    for waypoint_index in path:
                        # Convert image coordinates
                        image_coordinate = np.array(G.nodes[waypoint_index]['pos'])
                        x = image_coordinate[0] * resolution + origin[0]
                        y = image_coordinate[1] * resolution + origin[1]
                        writer.writerow([waypoint_index, x, y, path_index])

        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

class MapVisualization:

    def visualizeTriangles(self, image, triangles, points, waypoints):
        """
        This function visualizes the Delaunay triangles and their waypoints on the image.

        Parameters:
        - image: Input grayscale image.
        - triangles: Indices of the points forming triangles.
        - points: Coordinates of the points.
        - waypoints: Centroids of the triangles.
        """

        # Create a copy of the image to draw triangles, waypoints, and coordinates on
        image_with_triangles_and_waypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw each triangle on the image
        for triangle_indices in triangles:
            triangle_vertices = points[triangle_indices]

            # Draw the triangle
            cv2.polylines(image_with_triangles_and_waypoints, [triangle_vertices], True, (0, 255, 0), thickness=2)

        # Draw waypoints (centers of triangles) on the image and display coordinates
        for waypoint in waypoints:
            # Draw waypoint
            cv2.circle(image_with_triangles_and_waypoints, tuple(waypoint), 3, (0, 0, 255), -1)

            # Display coordinates of the waypoint
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            font_color = (212, 103, 113)
            line_type = 1
            cv2.putText(image_with_triangles_and_waypoints,
                        f"({waypoint[0]}, {waypoint[1]})",
                        (waypoint[0] + 5, waypoint[1] - 5),
                        font,
                        font_scale,
                        font_color,
                        line_type)

        # Display the image with the triangles, waypoints, and coordinates
        cv2.imshow('Triangles with Waypoints and Coordinates', image_with_triangles_and_waypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the image with the triangles, waypoints, and coordinates
        cv2.imwrite('triangles_with_waypoints.png', image_with_triangles_and_waypoints)

    def visualizeWaypointGraph(self, mapImage, Graph):
        """
        This function visualizes the waypoint graph on the image.

        Parameters:
        - mapImage: Input grayscale image.
        - G: Graph with waypoints as nodes and valid paths as edges.
        """

        # Convert the grayscale image to RGB
        map_with_graph = cv2.cvtColor(mapImage, cv2.COLOR_GRAY2BGR)

        # Get node positions from the graph
        pos = nx.get_node_attributes(Graph, 'pos')

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(map_with_graph)

        # Draw the graph on the axis
        nx.draw(Graph, pos, node_size=10, node_color='r', with_labels=False, ax=ax)

        # Save the figure
        plt.savefig('waypoint_graph.png')

        # Optionally show the figure
        plt.show()

    def visualizeClosedPathsOnMap(self, mapImage, Graph, closed_paths):
        """
        Visualizes the closed paths on the map image.

        Parameters:
        - mapImage: Input grayscale image.
        - Graph: The original graph with waypoints.
        - closed_paths: List of closed paths (each path is a list of nodes).
        """

        # Convert the grayscale image to RGB
        map_with_paths = cv2.cvtColor(mapImage, cv2.COLOR_GRAY2BGR)

        # Get node positions from the graph
        pos = nx.get_node_attributes(Graph, 'pos')
        colors = cm.rainbow(np.linspace(0, 1, len(closed_paths)))

        for i, path in enumerate(closed_paths):
            path_color = tuple([int(c * 255) for c in colors[i][:3]])
            # Draw the closed path
            for j in range(len(path)):
                p1 = pos[path[j]]
                p2 = pos[path[(j + 1) % len(path)]]
                cv2.line(map_with_paths, p1, p2, path_color, 2)
                # Optionally draw waypoints
                cv2.circle(map_with_paths, p1, 5, (0, 0, 255), -1)

        # Display the image with closed paths
        cv2.imshow('Closed Paths on Map', map_with_paths)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the image with closed paths
        cv2.imwrite('closed_paths_on_map.png', map_with_paths)

    def visualizeWalls(self, image, polygons):
        """
        This function visualizes the detected walls (polygons) on the image.

        Parameters:
        - image: Input grayscale image.
        - polygons: List of polygons.
        """

        # Convert the grayscale image to a color image
        image_with_walls = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Fill the polygons
        for polygon in polygons:
            cv2.fillPoly(image_with_walls, [polygon], (255, 0, 0))

        # Optionally, you can also draw the polygon outlines to make the edges more visible
        for polygon in polygons:
            cv2.polylines(image_with_walls, [polygon], True, (0, 255, 0), thickness=2)

        # Display the image with the filled polygons
        cv2.imshow('Walls', image_with_walls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the image with the filled polygons
        cv2.imwrite('walls.png', image_with_walls)

def simulationMain(visualize):
    # Load the PGM file
    mapImage = cv2.imread('linorobot2_gazebo/worlds/test/generated_map.pgm', cv2.IMREAD_GRAYSCALE)

    # Polygonize the image
    MP = MapProcessing()
    polygons = MP.polygonizeImage(mapImage)

    for polygon in polygons:
        print(f"Polygon: {polygon}")

    # Triangulate the polygons
    triangles, all_points = MP.triangulatePolygons(polygons)

    # Calculate waypoints (centers of triangles)
    waypoints = MP.calculateWaypoints(triangles, all_points)

    # Display and access the calculated waypoints
    for index, waypoint in enumerate(waypoints):
        print("Waypoint", index, ":", waypoint)

    if (visualize):
        # Visualize the triangles with waypoints and coordinates
        MV = MapVisualization()
        MV.visualizeTriangles(mapImage, triangles, all_points, waypoints)

    # Create the waypoint graph
    G = MP.createWaypointGraph(waypoints, polygons)

    # Is there any vertex in the graph that is not connected to any other vertex?
    isolated_vertices = [node for node, degree in G.degree if degree == 0]
    print("Isolated vertices:", isolated_vertices)

    if (visualize):
        # Visualize the waypoint graph
        MV.visualizeWaypointGraph(mapImage, G)

        # Visualize the walls
        MV.visualizeWalls(mapImage, polygons)

    # Partition the graph into n sections and create closed paths within those partitions
    num_partitions = 3
    partitions = MP.partitionGraph(G, num_partitions)

    # Create closed paths for each subgraph
    closed_paths = MP.createClosedPaths(partitions)

    # Print closed paths for verification
    for i, path in enumerate(closed_paths):
        print(f"Closed Path for Subgraph {i+1}: {path}")

    if (visualize):
        # Visualize the closed paths on the map image
        MV.visualizeClosedPathsOnMap(mapImage, G, closed_paths)

    # Load the YAML file
    yaml_data = MP.readYaml('linorobot2_gazebo/worlds/test/generated_map.yaml')

    # Remove duplicates from closed paths
    closed_paths = [list(dict.fromkeys(path)) for path in closed_paths]
    print("Closed paths after removing duplicates:", closed_paths)

    # Export waypoints and closed paths to CSV
    MP.exportWaypointsToCSV(G, closed_paths, mapImage.shape, yaml_data, filename='waypoints.csv')

# If environment variable robuddy_dev is set to true, run the following code
if os.getenv('robuddy_dev') == 'true':
    simulationMain(False)
