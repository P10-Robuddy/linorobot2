import numpy as np
import cv2
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.algorithms.approximation import christofides
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import csv

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

        # If there are more than one island, connect them with a single edge using discarded edges
        if len(islands) > 1:
            print("Islands:", islands)
            for edge in discarded_edges:
                island1, island2 = None, None
                for island in islands:
                    if edge[0] in island:
                        island1 = island
                    if edge[1] in island:
                        island2 = island
                if island1 and island2:
                    G.add_edge(edge[0], edge[1])
                    print(f"Connecting islands with edge {edge}")
                    break

        islands = list(nx.connected_components(G))
        print("Islands:", islands)
        return G

    def visualizeWaypointGraph(self, mapImage, Graph):
        """
        This function visualizes the waypoint graph on the image.

        Parameters:
        - mapImage: Input grayscale image.
        - G: Graph with waypoints as nodes and valid paths as edges.
        """

        pos = nx.get_node_attributes(G, 'pos')
        map_with_graph = cv2.cvtColor(mapImage, cv2.COLOR_GRAY2BGR)

        nx.draw(Graph, pos, node_size=10, node_color='r', with_labels=False)

        # Display the image with the waypoint graph
        plt.imshow(map_with_graph)
        plt.show()

    def partitionGraph(self, G, num_divisions):
        """
        This function partitions the graph into n sections and creates closed paths within those partitions
        using the Christofides algorithm.

        Parameters:
        - G: Graph with waypoints as nodes and valid paths as edges.
        - n: Number of sections to partition the graph into.

        Returns:
        - List of closed paths within each partition.
        """
        graph = nx.complete_graph(G)

        # Get the number of vertices in the graph
        num_vertices = graph.number_of_nodes()

        # Calculate how many vertices each subgraph should have approximately
        vertices_per_subgraph = num_vertices // num_divisions

        # Initialize a list to store the paths
        closed_paths = []

        # Iterate over each division
        for i in range(num_divisions):
            # Determine the range of vertices for the current subgraph
            start_vertex = i * vertices_per_subgraph
            end_vertex = start_vertex + vertices_per_subgraph if i < num_divisions - 1 else num_vertices

            # Extract the subgraph
            subgraph_nodes = list(graph.nodes())[start_vertex:end_vertex]
            subgraph = graph.subgraph(subgraph_nodes)

            # Apply Christofides algorithm to find an approximate solution to the TSP for the subgraph
            tsp_path = christofides(subgraph)

            # Append the TSP path to the list of paths
            closed_paths.append(tsp_path)

        return closed_paths

        # Apply Christofides algorithm to find an approximate solution to the TSP
        graph = nx.complete_graph(G)
        closed_paths = christofides(graph)

        return closed_paths

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

    def exportWaypointsToCSV(self, waypoints, filename='waypoints.csv'):
        """
        This function exports the waypoints to a CSV file.

        Parameters:
        - waypoints: Centroids of the triangles.
        - filename: Name of the CSV file (default is 'waypoints.csv').
        """
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Waypoint Index', 'X', 'Y'])

                for index, waypoint in enumerate(waypoints):
                    writer.writerow([index, waypoint[0], waypoint[1]])
        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

# Load the PGM file
mapImage = cv2.imread('linorobot2_gazebo/worlds/fishbot room/room.pgm', cv2.IMREAD_GRAYSCALE)

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

# Visualize the triangles with waypoints and coordinates
MP.visualizeTriangles(mapImage, triangles, all_points, waypoints)

# Create the waypoint graph
G = MP.createWaypointGraph(waypoints, polygons)

# Visualize the waypoint graph
MP.visualizeWaypointGraph(mapImage, G)

# Visualize the walls
MP.visualizeWalls(mapImage, polygons)

# Partition the graph into n sections and create closed paths within those partitions
closed_paths = MP.partitionGraph(G, num_divisions=3)
print("Closed paths:", closed_paths)

# Export waypoints to CSV
# mapProcessing.exportWaypointsToCSV(waypoints, filename='waypoints.csv')
