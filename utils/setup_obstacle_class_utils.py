#!/usr/bin/env python3

import cv2 
import pdb 
import numpy as np 
import matplotlib.pyplot as plt  
from shapely.geometry import Point, Polygon  
from matplotlib.patches import RegularPolygon  
from matplotlib.collections import PatchCollection     

from colorama import Fore, Back, Style 




"""====================================================================================================="""  
"""Utils functions for "SetupObstacles class" in obstacle_main.py]"""   
"""====================================================================================================="""  

"""update_obstacle_info method"""
def compute_x_and_y(angle_degrees, distance, initial_point, initial_heading): 
    """
    Compute the dx and dy at a point robot turns 90 degrees for dynamic grid generation.
    """

    x, y = initial_point

    poses = [[x, y, initial_heading]]   
    delta = [[0, 0, 0]] 

    while True:    
        new_heading = poses[-1][-1] + angle_degrees
        angle_diff = abs(new_heading - initial_heading) 
        if angle_diff > 90:  
            break

        x = x + distance * np.cos(np.radians(new_heading))  
        y = y + distance * np.sin(np.radians(new_heading))   

        poses.append([round(x, 2), round(y,2), new_heading]) 

        delta_x = 2 * abs(round(x, 2))
        delta_y = 2 * abs(round(y, 2)) 
        delta.append([delta_x, delta_y, angle_diff])   

    return poses, delta   



"""obtain_grid_for_all_obstacles method"""

def is_interior_angle(v1, v2, v3):
    # Calculate the vectors relative to v3
    u = v1 - v3
    v = v2 - v3
    # Calculate the cross product to determine orientation 
    cross_product = np.cross(u, v)

    # Determine if the angle is an interior angle (counterclockwise) or exterior angle (clockwise)
    return (cross_product > 0)

def signed_angle_between_vectors(v1, v2, v3):
    # Calculate the vectors relative to v3
    u = v1 - v2
    v = v3 - v2 

    # Calculate the dot product of u and v
    dot_product = np.dot(u, v)

    # Calculate the magnitudes of u and v
    u_magnitude = np.linalg.norm(u)
    v_magnitude = np.linalg.norm(v)

    # Calculate the cosine of the angle between u and v
    cos_angle = dot_product / (u_magnitude * v_magnitude)

    # Ensure cos_angle is within the valid range [-1, 1] to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the signed angle in radians using arccos
    angle_rad = np.arccos(cos_angle)

    # Determine the sign of the angle based on the orientation using the cross product
    cross_product = np.cross(u, v)
    sign = np.sign(cross_product)  # Positive if counterclockwise, negative if clockwise or collinear 

    # Convert angle to degrees and apply the sign
    angle_deg = np.degrees(angle_rad) * sign

    # Determine if the angle is an interior or exterior angle
    is_interior = is_interior_angle(v1, v2, v3)

    if not is_interior:
        # Convert exterior angle to interior angle
        angle_deg = 360 - angle_deg

    return angle_deg  

def calculate_exterior_angles(vertices):
    n = len(vertices)
    exterior_angles = []

    for i in range(n):
        # Get vertices for the current vertex and its neighbors
        prev_vertex = vertices[i - 1]
        current_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % n]  # Wrap around to the first vertex for the last edge

        # Calculate the exterior angle using signed_angle_between_vectors
        angle = signed_angle_between_vectors(prev_vertex, current_vertex, next_vertex)

        # Append the exterior angle to the list
        exterior_angles.append(angle)

        positive_exterior_angles = [abs(num) for num in exterior_angles]  

    return positive_exterior_angles  



def bisector_angle(vertices, positive_exterior_angles): 

    """
    Compute the bisector angle for each vertex in a polygon to move that vertex.
    """

    vertex_orientations_towards_next_vertex = []
    for i in range(len(vertices)): 
        # Calculate the orientation angle in radians (counterclockwise from positive x-axis)
        vertex = vertices[i]
        next_vertex = vertices[(i + 1) % len(vertices)]  # Wrap around to the first vertex for the last edge  
        orientation_angle = np.rad2deg(np.arctan2(next_vertex[1] - vertex[1], next_vertex[0] - vertex[0])) 
        orientation_angle = (orientation_angle ) % 360  # Convert to range [0, 360)
        vertex_orientations_towards_next_vertex.append(orientation_angle)

    new_orientation_list = []
    for i in range(len(vertices)):
        new_orientation = vertex_orientations_towards_next_vertex[i] + (positive_exterior_angles[i])/2 
        new_orientation = (new_orientation-180 )% 360
        new_orientation_list.append(new_orientation)

    return new_orientation_list

def move_vertices_by_angles(vertices, angles, distance):
    """
    Move each vertex of a polygon by a specified angle and distance.

    Parameters:
        vertices (list of tuples): List of vertex coordinates (each vertex as a tuple of (x, y)).
        angles (list of float): List of angles (in degrees) corresponding to each vertex.
        distance (float): Distance to move each vertex along its specified angle (default is 1).

    Returns:
        list of tuples: List of new vertex coordinates after applying the angle transformations.
    """
    moved_vertices = []

    for i, vertex in enumerate(vertices):
        # Retrieve the angle for the current vertex
        angle_degrees = angles[i]

        # Convert angle from degrees to radians
        angle_radians = np.radians(angle_degrees)

        # Calculate the change in x and y coordinates based on the angle and distance
        delta_x = distance * np.cos(angle_radians)
        delta_y = distance * np.sin(angle_radians)

        # Calculate the new coordinates of the vertex after moving along the angle
        new_vertex = (vertex[0] + delta_x, vertex[1] + delta_y)

        # Append the new vertex coordinates to the list
        moved_vertices.append(new_vertex)

    return moved_vertices


def obtain_extended_polygon_vertices(polygon_vertices_list, distance): 
    """
    Obtain the vertices of a polygon after extending each vertex by a specified distance.
    """
    # Convert vertices to numpy arrays 
    polygon_vertices = [np.array(vertex) for vertex in polygon_vertices_list] 

    # Calculate exterior angles for each vertex in the polygon
    positive_exterior_angles = calculate_exterior_angles(polygon_vertices) 

    # Compute the bisector angle for each vertex 
    new_orientation_list = bisector_angle(polygon_vertices, positive_exterior_angles)

    # Move each vertex by its specified angle and distance   
    moved_vertices = move_vertices_by_angles(polygon_vertices, new_orientation_list, distance)  # actual 0.848  (25cm diag allowance) (actual allowance = 20cm, robot is 60cm)

    # ====================================================================================================         
    move_vertices_np = np.array(moved_vertices)
    # find the min and max of the x and y coordinates to make a rectangle
    min_x, min_y = np.min(move_vertices_np, axis=0)
    max_x, max_y = np.max(move_vertices_np, axis=0)
    obstacle_extended_rect = (min_x, min_y, max_x, max_y) 


    return moved_vertices, obstacle_extended_rect

def compute_grid_size(obstacle_size, delta_x, y_list1, total_area):

    """
    Compute the grid size for a given obstacle size and grid cell dimensions dynamically.
    """   
         
    dx_obstacle = obstacle_size[2]-obstacle_size[0]
    dy_obstacle = obstacle_size[3]-obstacle_size[1]   

    # calculate obstacle center from dx and dy
    obstacle_center = (obstacle_size[0] + dx_obstacle/2, obstacle_size[1] + dy_obstacle/2) 

    if dx_obstacle > delta_x[-1]:  
        y_offset = 0.5*y_list1[-1] # y_offset = 0.5*y_list1[-1] 
        grid_y_half = y_list1[-1] + y_offset + (dy_obstacle/2)  
    else:
        for i, delta in enumerate(delta_x): 
            if dx_obstacle < delta: 
                y_offset = 0.5*y_list1[i] # y_offset = 0.5*y_list1[i]  
                grid_y_half = y_list1[i] + y_offset + (dy_obstacle/2)  

                break
        

    if dy_obstacle > delta_x[-1]: 
        x_offset = 0.5*y_list1[-1] # x_offset = 0.5*y_list1[-1]
        grid_x_half = y_list1[-1] + x_offset + (dx_obstacle/2) 

    else:
        for i, delta in enumerate(delta_x): 
            if dy_obstacle < delta: 
                x_offset = 0.5*y_list1[i] # 0.5*y_list1[i] 
                grid_x_half = y_list1[i] + x_offset + (dx_obstacle/2) 

                break

    grid_lower_left_corner  = (max((obstacle_center[0] - grid_x_half), total_area[0][0]), max((obstacle_center[1] - grid_y_half), total_area[0][1]))
    grid_lower_right_corner = (min((obstacle_center[0] + grid_x_half), total_area[1][0]), max((obstacle_center[1] - grid_y_half), total_area[0][1]))
    grid_upper_right_corner = (min((obstacle_center[0] + grid_x_half), total_area[1][0]), min((obstacle_center[1] + grid_y_half), total_area[1][1]))
    grid_upper_left_corner  = (max((obstacle_center[0] - grid_x_half), total_area[0][0]), min((obstacle_center[1] + grid_y_half), total_area[1][1]))

    grid = [grid_lower_left_corner, grid_lower_right_corner, grid_upper_right_corner, grid_upper_left_corner] 

    grid_rect = [grid[0][0], grid[0][1], grid[2][0], grid[2][1]]

    grid_instance = Polygon(grid)

    return grid_rect, grid_instance





"""obtain_points_for_all_grids method""" 
def count_row_col_in_an_square_area(grid_size, rect):
    min_x, min_y, max_x, max_y = rect
    # Calculate the width and height of the rectangle
    rectangle_width = max_x - min_x
    rectangle_height = max_y - min_y 
    
    # Calculate the number of grid cells that can fit along the width and height
    num_cols_np = int(np.floor(rectangle_width / grid_size)) 
    num_rows_np = int(np.floor(rectangle_height / grid_size))
    
    ac_grid_x = num_cols_np
    ac_grid_y = num_rows_np

    return ac_grid_x, ac_grid_y

def calculate_square_grid_centers(num_rows, num_cols, grid_size, min_x, min_y):
    """
    Calculate grid centers within a specified grid layout starting from (min_x, min_y).
    """
    
    # Generate row indices and column indices for all grid points
    row_indices = np.arange(num_rows)
    col_indices = np.arange(num_cols)  
    
    # Create meshgrid of column indices and row indices 
    col_mesh, row_mesh = np.meshgrid(col_indices, row_indices, indexing='xy')
    
    # Calculate x-coordinates of all grid points relative to min_x
    x_centers = col_mesh * grid_size
    
    # Shift x-coordinates to start from min_x
    x_centers += min_x
    
    # Calculate y-coordinates of all grid points relative to min_y
    y_centers = row_mesh * grid_size
    
    # Shift y-coordinates to start from min_y
    y_centers += min_y
    
    # Combine x-coordinates and y-coordinates to obtain center points
    centers = np.column_stack((x_centers.flatten(), y_centers.flatten()))
    
    return centers

def generate_free_grid(num_rows, num_cols):
    """
    Generate a free grid with all cells initialized to 0 (free).
    """
    occ_grid = np.zeros((num_rows, num_cols), dtype=np.uint8)
    occ_grid = np.flipud(occ_grid)  # upside down.
    return occ_grid

def create_grid(ac_grid_x, ac_grid_y): # grid_x = cols, grid_y = rows
    """
    Create an occupancy grid with all cells initialized to 0 (free).
    """
    grid = np.zeros((ac_grid_y, ac_grid_x), dtype=np.uint8)
    return grid 

def obtain_grid_centers(grid_size, rect):    
    """
    Obtain grid centers for a specified grid size within a given rectangle.
    """
    
    ac_grid_x, ac_grid_y = count_row_col_in_an_square_area(grid_size, rect)

    grid_centers = calculate_square_grid_centers(ac_grid_y, ac_grid_x, grid_size, rect[0], rect[1])  
    # occ_grid = generate_free_grid(num_rows, num_cols) 
    occ_grid = create_grid(ac_grid_x, ac_grid_y)  
    # print(grid_shape(occ_grid))
    return grid_centers, ac_grid_x, ac_grid_y, occ_grid      


def check_points_outside_polygon(points, polygon_vertices):
    """Check which points are outside a polygon defined by vertices.""" 
    polygon = Polygon(polygon_vertices) 
    
    # Convert points to Shapely Point objects
    shapely_points = [Point(point) for point in points]
    
    # Check which points are outside the polygon using vectorized operation
    outside_mask = np.array([not polygon.contains(shapely_point) for shapely_point in shapely_points])
    
    # Filter points based on the outside mask
    points_outside_polygon = points[outside_mask]
    
    return points_outside_polygon  


def check_points_inside_polygon(points, polygon_vertices):
    """Check which points are outside a polygon defined by vertices.""" 
    polygon = Polygon(polygon_vertices)  
    
    # Convert points to Shapely Point objects
    shapely_points = [Point(point) for point in points]
    
    # Check which points are outside the polygon using vectorized operation
    outside_mask = np.array([polygon.contains(shapely_point) for shapely_point in shapely_points])
    
    # Filter points based on the outside mask
    points_outside_polygon = points[outside_mask]
    
    return points_outside_polygon   


def obtain_points_outside_obstacles(grid_dia, grid_rect, moved_vertices):
    """Obtain points that lie outside the obstacles for grid generation."""
    centers, ac_grid_x, ac_grid_y, occ_grid = obtain_grid_centers(grid_dia, grid_rect)  

    """ Check which points are inside the polygon """   
    points_outside = check_points_outside_polygon(centers, moved_vertices) 
    points_inside  = check_points_inside_polygon(centers, moved_vertices)   
    
    return points_outside, points_inside, ac_grid_x, ac_grid_y, occ_grid  


def find_points_within_buffer(points, polygon_vertices, buffer_distance): 
    """
    Find points that lie within a buffer around a polygon. 

    Parameters:
        points (np.ndarray): Array of shape (N, 2) representing the points to check.
        polygon_vertices (np.ndarray): Array of shape (M, 2) representing the vertices of the polygon.
        buffer_distance (float): Distance of the buffer around the polygon.

    Returns:
        np.ndarray: Array of shape (K, 2) representing the points that lie within the buffer.
    """ 
    # Create a Shapely Polygon object from polygon vertices 
    polygon = Polygon(polygon_vertices)
    
    # Create a buffer around the polygon with the specified distance
    buffered_polygon = polygon.buffer(buffer_distance)
    
    # Initialize an empty list to collect points within the buffer
    points_within_buffer = []
    
    # Iterate over each point and check if it lies within the buffered polygon
    for point in points:
        shapely_point = Point(point)
        if shapely_point.within(buffered_polygon):
            points_within_buffer.append(point)
    
    # Convert list of points to numpy array
    points_within_buffer = np.array(points_within_buffer)
    
    return points_within_buffer, polygon



"""====================================================================================================="""
"""====================================================================================================="""

