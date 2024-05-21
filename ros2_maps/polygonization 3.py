import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import csv
from shapely.geometry import LineString, Polygon

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

        Parameters:
        - waypoints: Centroids of the triangles.
        - polygons: List of polygons.

        Returns:
        - Graph with waypoints as nodes and valid paths as edges.
        """

        G = nx.Graph()
        shapely_polygons = [Polygon(polygon[:, 0, :]) for polygon in polygons]

        for index, waypoint in enumerate(waypoints):
            G.add_node(index, pos=tuple(waypoint))

        for i in range(len(waypoints)):
            for j in range(i + 1, len(waypoints)):
                line = LineString([waypoints[i], waypoints[j]])
                intersects = any(line.intersects(polygon) for polygon in shapely_polygons)

                if not intersects:
                    G.add_edge(i, j)
                else:
                    print(f"Edge ({i}, {j}) intersects with a polygon and is not added.")

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

    def visualizeWalls(self, image, polygons):
        """
        This function visualizes the detected walls (polygons) on the image.

        Parameters:
        - image: Input grayscale image.
        - polygons: List of polygons.
        """
        image_with_walls = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for polygon in polygons:
            cv2.polylines(image_with_walls, [polygon], True, (255, 0, 0), thickness=2)

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

# Export waypoints to CSV
# mapProcessing.exportWaypointsToCSV(waypoints, filename='waypoints.csv')
