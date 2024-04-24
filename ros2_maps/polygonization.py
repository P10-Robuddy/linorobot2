import numpy as np
import cv2

# Load the PGM file
image = cv2.imread('C:/Users/magnu/GitHub/linorobot2/ros2_maps/fishbot room1/room1_map.pgm', cv2.IMREAD_GRAYSCALE)

# Preprocessing
image = cv2.GaussianBlur(image, (5, 5), 0)
_, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Simplify contours to polygons
polygons = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Filter out small contours
        epsilon = 0.01 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(polygon)

# Create a blank image with the same size as the original image
blank_image = np.zeros_like(image)

# Draw each polygon on the blank image
for polygon in polygons:
    cv2.fillPoly(blank_image, [polygon], (255, 255, 255))

# Display the image with the filled polygons
cv2.imshow('Polygons', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

for polygon in polygons:
    # Get the vertices of the polygon
    vertices = polygon.reshape(-1, 2)  # Reshape to Nx2 array of (x, y) coordinates

    # Calculate diagonals between vertices
    num_vertices = len(vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            vertex1 = vertices[i]
            vertex2 = vertices[j]
            # Calculate the distance between vertex1 and vertex2 (diagonal)
            diagonal_length = np.linalg.norm(vertex1 - vertex2)
            print(f"Diagonal between vertices {i} and {j}: {diagonal_length}")
