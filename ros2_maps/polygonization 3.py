import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import csv

class MapProcessing:

    def polygonizeImage(self, image, blur_ksize=(5, 5), area_threshold=10000) -> list:
        # Preprocessing
        blurredImage = cv2.GaussianBlur(image, blur_ksize, 0)
        _, binary_image = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold]

        # Simplify contours to polygons
        polygons = [cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) for contour in filtered_contours]

        return polygons

    def triangulatePolygons(self, polygons):
        all_points = np.vstack(polygons)[:, 0, :]

        # Perform Delaunay triangulation
        triangulation = Delaunay(all_points)

        return triangulation.simplices, all_points

    def calculateWaypoints(self, triangles, points):
        waypoints = [np.mean(points[triangle_indices], axis=0, dtype=np.int32) for triangle_indices in triangles]
        return waypoints

    def visualizeTriangles(self, image, triangles, points, waypoints):
        image_with_triangles_and_waypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw each triangle on the image
        for triangle_indices in triangles:
            triangle_vertices = points[triangle_indices]
            cv2.polylines(image_with_triangles_and_waypoints, [triangle_vertices], True, (0, 255, 0), thickness=2)

        # Draw waypoints (centers of triangles) on the image and display coordinates
        for waypoint in waypoints:
            cv2.circle(image_with_triangles_and_waypoints, tuple(waypoint), 3, (0, 0, 255), -1)
            cv2.putText(image_with_triangles_and_waypoints, f"({waypoint[0]}, {waypoint[1]})", (waypoint[0] + 5, waypoint[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (212, 103, 113), 1)

        # Display the image with the triangles, waypoints, and coordinates
        cv2.imshow('Triangles with Waypoints and Coordinates', image_with_triangles_and_waypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        pos = nx.get_node_attributes(G, 'pos')
        map_with_graph = cv2.cvtColor(mapImage, cv2.COLOR_GRAY2BGR)

        nx.draw(G, pos, node_size=10, node_color='r', with_labels=False)
        plt.imshow(map_with_graph)
        plt.show()

    def visualizeWalls(self, image, polygons):
        image_with_walls = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for polygon in polygons:
            cv2.polylines(image_with_walls, [polygon], True, (255, 0, 0), thickness=2)
        cv2.imshow('Walls', image_with_walls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def exportWaypointsToCSV(self, waypoints, filename='waypoints.csv'):
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Waypoint Index', 'X', 'Y'])
                for index, waypoint in enumerate(waypoints):
                    writer.writerow([index, waypoint[0], waypoint[1]])
        except IOError as e:
            print(f"Error writing to file {filename}: {e}")

# Load the PGM file
mapImage = cv2.imread('ros2_maps/fishbot room/room.pgm', cv2.IMREAD_GRAYSCALE)

# Polygonize the image
mapProcessing = MapProcessing()
polygons = mapProcessing.polygonizeImage(mapImage)

# Triangulate the polygons
triangles, all_points = mapProcessing.triangulatePolygons(polygons)

# Calculate waypoints (centers of triangles)
waypoints = mapProcessing.calculateWaypoints(triangles, all_points)

# Visualize the triangles with waypoints and coordinates
mapProcessing.visualizeTriangles(mapImage, triangles, all_points, waypoints)

# Create the waypoint graph
G = mapProcessing.createWaypointGraph(waypoints)

# Visualize the waypoint graph
mapProcessing.visualizeWaypointGraph(mapImage, G)

# Visualize the walls
mapProcessing.visualizeWalls(mapImage, polygons)

# # Export waypoints to CSV
# mapProcessing.exportWaypointsToCSV(waypoints, 'waypoints.csv')
