import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import csv

class MapProcessing:

    def polygonizeImage(self, image) -> list:
        # Preprocessing
        blurredImage = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Set the area threshold to distinguish between rooms and obstacles
        area_threshold = 10000  # Adjust as needed based on the size of your rooms and obstacles

        # Filter contours based on area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                filtered_contours.append(contour)

        # Simplify contours to polygons
        polygons = []
        for contour in filtered_contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(polygon)

        return polygons

    def triangulatePolygons(self, polygons):
        all_points = np.concatenate(polygons)[:, 0, :]

        # Perform Delaunay triangulation
        triangulation = Delaunay(all_points)

        return triangulation.simplices, all_points

    def visualizeTriangles(self, image, triangles, points, waypoints):
        # Create a copy of the image to draw triangles, waypoints, and coordinates on
        image_with_triangles_and_waypoints = image.copy()

        # Draw each triangle on the image
        for triangle_indices in triangles:
            # Get the vertices of the triangle
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
            cv2.putText(image_with_triangles_and_waypoints, f"({waypoint[0]}, {waypoint[1]})",
                        (waypoint[0] + 5, waypoint[1] - 5), font, font_scale, font_color, line_type)

        # Display the image with the triangles, waypoints, and coordinates
        cv2.imshow('Triangles with Waypoints and Coordinates', image_with_triangles_and_waypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculateWaypoints(self, triangles, points):
        waypoints = []

        # Calculate the centroid of each triangle
        for triangle_indices in triangles:
            # Get the vertices of the triangle
            triangle_vertices = points[triangle_indices]

            # Calculate the centroid
            centroid = np.mean(triangle_vertices, axis=0, dtype=np.int32)

            # Append the centroid to the list of waypoints
            waypoints.append(centroid)

        return waypoints

    def createWaypointGraph(self, waypoints):
        # Create an empty undirected graph
        G = nx.Graph()

        # Add nodes (waypoints) to the graph
        for index, waypoint in enumerate(waypoints):
            G.add_node(index, pos=waypoint)  # Store position as node attribute

        # Add edges (connections between waypoints) to the graph
        # You can define edges based on your navigation requirements.
        # For example, you might connect each waypoint to its neighboring waypoints.
        # Here's a simple example connecting each waypoint to its two nearest neighbors:
        for i in range(len(waypoints)):
            # Find indices of two nearest neighbors
            neighbor_indices = [idx for idx in range(len(waypoints)) if idx != i]
            nearest_neighbors = sorted(neighbor_indices, key=lambda idx: np.linalg.norm(np.array(waypoints[idx]) - np.array(waypoints[i])))[:2]

            # Add edges between the waypoint and its two nearest neighbors
            G.add_edge(i, nearest_neighbors[0])
            G.add_edge(i, nearest_neighbors[1])

        return G

    def visualizeWaypointGraph(self, mapImage, G):
        # Create a blank image with the same size as the original map image
        map_with_graph = mapImage.copy()

        # Draw the graph on the image
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size=10, node_color='r', with_labels=False, ax=plt.gca())

        # Display the image with the graph
        plt.imshow(map_with_graph, cmap='gray')
        plt.show()

    def visualizeWalls(self, image, polygons):
        # Create a copy of the image to draw the walls on
        image_with_walls = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw each polygon (wall) on the image in blue color
        for polygon in polygons:
            cv2.polylines(image_with_walls, [polygon], True, (255, 0, 0), thickness=2)

        # Display the image with the walls
        cv2.imshow('Walls', image_with_walls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def exportWaypointsToCSV(self, waypoints, filename='waypoints.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Waypoint Index', 'X', 'Y'])
            for index, waypoint in enumerate(waypoints):
                writer.writerow([index, waypoint[0], waypoint[1]])

# Load the PGM file
mapImage = cv2.imread('linorobot2_gazebo/worlds/fishbot room/room.pgm', cv2.IMREAD_GRAYSCALE)

# Polygonize the image
mapProcessing = MapProcessing()
polygons = mapProcessing.polygonizeImage(mapImage)

# Triangulate the polygons
triangles, all_points = mapProcessing.triangulatePolygons(polygons)

# Calculate waypoints (centers of triangles)
waypoints = mapProcessing.calculateWaypoints(triangles, all_points)

# Visualize the triangles with waypoints and coordinates
mapProcessing.visualizeTriangles(mapImage, triangles, all_points, waypoints)

# Display and access the calculated waypoints
for index, waypoint in enumerate(waypoints):
    print("Waypoint", index, ":", waypoint)

# Create the waypoint graph
G = mapProcessing.createWaypointGraph(waypoints)

# Visualize the waypoint graph
mapProcessing.visualizeWaypointGraph(mapImage, G)

# Visualize the walls
mapProcessing.visualizeWalls(mapImage, polygons)

# # Export waypoints to CSV
# mapProcessing.exportWaypointsToCSV(waypoints, 'waypoints.csv')
