import numpy as np
import cv2

# Load the PGM file
image = cv2.imread('C:/Users/magnu/GitHub/linorobot2/ros2_maps/ctcv/ctcv.pgm', cv2.IMREAD_GRAYSCALE)

# Preprocessing
image = cv2.GaussianBlur(image, (5, 5), 0)
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


# Create a blank image with the same size as the original image
blank_image = np.zeros_like(image)

# Draw each polygon on the blank image
for polygon in polygons:
    cv2.fillPoly(blank_image, [polygon], (255, 255, 255))

# Display the image with the filled polygons
cv2.imshow('Polygons', blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Iterate over the polygons representing the boundaries of the rooms
for room_polygon in polygons:
    # Get the vertices of the room polygon
    room_vertices = room_polygon.reshape(-1, 2)

    # Calculate diagonals between vertices
    num_vertices = len(room_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            vertex1 = room_vertices[i]
            vertex2 = room_vertices[j]
            # Calculate the distance between vertex1 and vertex2 (diagonal)
            diagonal_length = np.linalg.norm(vertex1 - vertex2)
            print(f"Diagonal between vertices {i} and {j}: {diagonal_length}")
