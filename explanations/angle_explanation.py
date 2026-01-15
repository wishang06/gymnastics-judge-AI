"""
Visual explanation of the angle calculation function
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle_explained(point_a, point_b, point_c):
    """
    Calculate the angle at point_b formed by points a-b-c
    Returns angle in degrees
    
    VISUAL EXAMPLE:
    
        a
         \
          \
           b ---- c
           
    We want the angle ∠abc (angle at point b)
    """
    
    # Step 1: Convert MediaPipe landmarks to numpy arrays
    # MediaPipe gives us objects with .x and .y attributes (normalized 0-1)
    # We convert them to [x, y] arrays for vector math
    a = np.array([point_a.x, point_a.y])  # First point: [x_coord, y_coord]
    b = np.array([point_b.x, point_b.y])  # Vertex (middle point): [x_coord, y_coord]
    c = np.array([point_c.x, point_c.y])  # Third point: [x_coord, y_coord]
    
    # Step 2: Calculate vectors FROM point b TO points a and c
    # Vector ba = a - b (direction from b to a)
    
    # Vector bc = c - b (direction from b to c)
    
    # Step 3: Calculate angles of these vectors using arctan2
    # arctan2(y, x) gives the angle of a vector in radians
    # It handles all 4 quadrants correctly (unlike regular arctan)
    
    # Angle of vector from b to c
    angle_bc = np.arctan2(c[1] - b[1], c[0] - b[0])
    #                    ↑ delta_y    ↑ delta_x
    # This is: "What angle does the line b→c make with horizontal?"
    
    # Angle of vector from b to a
    angle_ba = np.arctan2(a[1] - b[1], a[0] - b[0])
    #                    ↑ delta_y    ↑ delta_x
    # This is: "What angle does the line b→a make with horizontal?"
    
    # Step 4: The difference between these angles gives us the angle between the vectors
    radians = angle_bc - angle_ba
    
    # Step 5: Convert radians to degrees
    # 180 degrees = π radians, so: degrees = radians * (180 / π)
    angle = np.abs(radians * 180.0 / np.pi)
    
    # np.abs() ensures we get a positive angle (0-180 degrees)
    # Without abs(), we might get negative angles or angles > 180
    
    return angle


# Alternative method using dot product (more intuitive for some)
def calculate_angle_dot_product(point_a, point_b, point_c):
    """
    Alternative method using dot product formula:
    cos(θ) = (a · b) / (|a| * |b|)
    
    This might be easier to understand conceptually
    """
    # Convert to vectors
    a = np.array([point_a.x, point_a.y])
    b = np.array([point_b.x, point_b.y])
    c = np.array([point_c.x, point_c.y])
    
    # Create vectors FROM vertex b TO points a and c
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c
    
    # Calculate cosine of angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clamp to [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert to angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle


# Example usage with dummy data
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Example: Right angle (90 degrees)
# Point a at (0, 1), vertex b at (0, 0), point c at (1, 0)
point_a = Point(0.0, 1.0)
point_b = Point(0.0, 0.0)  # Vertex
point_c = Point(1.0, 0.0)

angle1 = calculate_angle_explained(point_a, point_b, point_c)
angle2 = calculate_angle_dot_product(point_a, point_b, point_c)

print(f"Angle using arctan2 method: {angle1:.2f}°")
print(f"Angle using dot product method: {angle2:.2f}°")
print(f"Expected: 90.00°")
