#!/usr/bin/env python3

import pdb
import time
import math
import numpy as np 
import networkx as nx  
import matplotlib.pyplot as plt   
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point, Polygon 

from utils.print_utils import * 




"""====================================================================================================="""  
"""Utils functions for "CheckObstacle class" in obstacle_main.py]"""   
"""====================================================================================================="""  


"""check_path_and_give_grid_and_salient_points method """ 

"""obs_ID_if_obs_in_path function."""
def check_if_obstacle_in_path(start_point, end_point, poly_obs_instance_list):
    """
    Check if an obstacle is in the path from start_point to end_point.
    """
    path_line = LineString([start_point, end_point])
    closest_obstacle_idx = None
    min_distance = float('inf')

    for i, poly_obs_instance in enumerate(poly_obs_instance_list):
        # pdb.set_trace()
        if path_line.intersects(poly_obs_instance):
            xx, yy = poly_obs_instance.exterior.coords.xy
            xy_np = np.array([xx, yy]).T
            distance = np.min(np.linalg.norm(xy_np - start_point, axis=1)) 
            if distance < min_distance:
                min_distance = distance
                closest_obstacle_idx = i

    if closest_obstacle_idx is not None:
        return True, closest_obstacle_idx  # True if an obstacle is in the path, and return the closest one. 
    return False, None  # False if no obstacle in path.

def check_if_robot_is_in_the_grid(robot_position, grid_instance):  
    """
    Check if the robot is in the grid.
    """
    
    point = Point(robot_position)
    is_robot_in_grid = grid_instance.contains(point)
    if is_robot_in_grid:
        return True  # True if robot in grid.
    return False     # False if robot not in grid.  


def obs_ID_if_obs_in_path(start_point, end_point, grid_instance_list, poly_obs_instance_list):           # start_point, end_point: np.array([x, y])     
    """
    Check if an obstacle is in the path from start_point to end_point and extract obstacle ID.
    """

    is_obs_i_in_path, obstacle_idx = check_if_obstacle_in_path(start_point, end_point, poly_obs_instance_list) # start_point = robot_position.       

    if is_obs_i_in_path: # obstacle in path.   
        """ Check if the robot is in the grid."""
        is_robot_in_grid = check_if_robot_is_in_the_grid(start_point, grid_instance_list[obstacle_idx])      

        if is_robot_in_grid:    # Check whether the robot is in any grid. 
                # pdb.set_trace()  
                return True, True, obstacle_idx     # True = obstacle in path, True = robot in grid, obstacle_idx = obstacle index.
        else:
            return True, False, obstacle_idx        # True = obstacle in path, False = robot not in grid, obstacle_idx = obstacle index.
            
    else:
        return False, None, None  # False = no obstacle in path, None = robot position does not matter, None = no obstacle index.     




       
"""extract_salient_points_to_generate_graph_no_quad function."""    

def find_max_dist_points_on_both_sides(projected_points, actual_points): 
    """
    Find the farthest points on both sides of the projected points from the vector along the current orientation.
    """

    # Calculate vector from projected points to actual points 
    vector_to_actual = actual_points - projected_points
    
    direction_vector = None
    # Calculate dot products with the direction vector for the first point 
    for i in range(len(vector_to_actual)):
        if (actual_points[i][0] != projected_points[i][0]) and (actual_points[i][1] != projected_points[i][1]):
            direction_vector = actual_points[i] - projected_points[i]  
            break

    if direction_vector is None:
         direction_vector = actual_points[1] - projected_points[1] 
        
    dot_products = np.sum(vector_to_actual * direction_vector, axis=1) 
    
    # Calculate distances between projected and actual points
    distances = np.linalg.norm(vector_to_actual, axis=1)

    # Determine indices for same and opposite direction points
    same_direction_indices      = dot_products >= 0
    opposite_direction_indices  = ~same_direction_indices

    # Initialize farthest points as None
    farthest_point_same_dir         = None
    farthest_point_opposite_dir     = None  

    if np.any(same_direction_indices): # if points in oppo
        # Find the farthest point in the same direction
        max_distance_same_dir   = np.argmax(distances[same_direction_indices])
        farthest_point_same_dir = actual_points[same_direction_indices][max_distance_same_dir]
    else:
        # If no points in the same direction, find the two closest points in the opposite direction
        closest_indices     = np.argsort(distances[opposite_direction_indices])[:2]
        closest_points      = actual_points[opposite_direction_indices][closest_indices]
        if len(closest_points) > 1:
            farthest_point_same_dir         = closest_points[0]
            farthest_point_opposite_dir     = closest_points[1]
        else:
            farthest_point_same_dir         = closest_points[0]
            farthest_point_opposite_dir     = closest_points[0]


    if np.any(opposite_direction_indices): 
        # Find the farthest point in the opposite direction
        max_distance_opposite_dir       = np.argmax(distances[opposite_direction_indices])
        farthest_point_opposite_dir     = actual_points[opposite_direction_indices][max_distance_opposite_dir]
    else:
        # If no points in the opposite direction, find the two closest points in the same direction
        closest_indices     = np.argsort(distances[same_direction_indices])[:2]
        closest_points      = actual_points[same_direction_indices][closest_indices]
        if len(closest_points) > 1:
            farthest_point_opposite_dir     = closest_points[0]
            farthest_point_same_dir         = closest_points[1]
        else:
            farthest_point_opposite_dir     = closest_points[0]
            farthest_point_same_dir         = closest_points[0]


    return farthest_point_same_dir, farthest_point_opposite_dir  


def project_point_onto_line_segment(points_to_project, line_start, line_end):
    """
    Project points onto the line segment defined by line_start and line_end.
    """

    # Vector representing the line segment
    line_vector = line_end - line_start
    
    # Vector from line start to all points
    point_vectors = points_to_project - line_start
    
    # Calculate dot products
    dot_product_line = np.dot(point_vectors, line_vector)
    dot_product_line_segment = np.dot(line_vector, line_vector)
    
    # Calculate projection parameters (t) for all points
    t = dot_product_line / dot_product_line_segment
    
    # Projected points on the infinite line
    projected_points = line_start + np.outer(t, line_vector)
    
    # Filter points based on projection parameter (t) within [0, 1]  
    mask = (t >= 0) & (t <= 1)
    projected_points = projected_points[mask] 
    points_list = points_to_project[mask] 
    
    return projected_points, points_list

def find_end_point_along_robot_heading(points, robot_position, robot_orientation, path_end_point, distance_threshold):
    """
    Find the farthest point from the buffer points along the robot's orientation from the robot position.
    """

    # Convert robot orientation from degrees to radians 
    angle_rad = robot_orientation
    
    # Calculate direction vector along robot's orientation
    direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Vector from robot position to all points
    vector_to_points = points - robot_position
    
    # Project vector_to_points onto the direction_vector
    projection_lengths = np.dot(vector_to_points, direction_vector)
    projected_points = robot_position + np.outer(projection_lengths, direction_vector)
    
    # Calculate perpendicular distances for all points
    perpendicular_distances = np.linalg.norm(points - projected_points, axis=1)
    
    # Calculate distances from robot position to all points
    distances_from_robot = np.linalg.norm(vector_to_points, axis=1)
    
    # Filter points based on perpendicular distance within the threshold
    mask = perpendicular_distances < distance_threshold
    filtered_points = points[mask]
    filtered_distances_from_robot = distances_from_robot[mask]
    
    # Find the farthest point on each side 
    if len(filtered_points) > 0:    # robot orientation is inside the obstacle.
        max_distance_index = np.argmax(filtered_distances_from_robot)
        farthest_point = filtered_points[max_distance_index]
    else:
        farthest_point = None       # robot orientation is facing outside the obstacle.
 
    if farthest_point is None: # If None, then change robot orientation to the path end point. Then, find the farthest point in the similar manner. 
        # print_red_text("Robot orientation is facing outside the obstacle. Changing robot orientation to the path end point.")
        # Calculate the orientation from the robot position to the path end point. 
        orientation_to_path_end = np.arctan2(path_end_point[1] - robot_position[1], path_end_point[0] - robot_position[0])
        new_robot_orientation = orientation_to_path_end
        farthest_point = find_end_point_along_robot_heading(points, robot_position, new_robot_orientation, path_end_point, distance_threshold) 

    return farthest_point 


def compute_salient_points(points_in_buffer, robot_position, robot_orientation, path_end_point, distance_threshold=1.0): 
    """
    Compute the salient points for generating the graph.
    """
    line_start  = robot_position 

    line_end    = find_end_point_along_robot_heading(points_in_buffer, robot_position, robot_orientation, path_end_point, distance_threshold) 
     
    projected_points, points_list = project_point_onto_line_segment(points_in_buffer, line_start, line_end)   
            
    # same_direction_points, opposite_direction_points, same_projected_points, opposite_projected_points = points_on_either_side(projected_points, points_list) 
    salient_point1, salient_point2 = find_max_dist_points_on_both_sides(projected_points, points_list)  
    
    salient_points = np.array([salient_point1, salient_point2])

    return salient_points


def extract_salient_points_to_generate_graph(idx, robot_position, robot_orientation, path_end_point, points_in_buffer_list, poly_obs_instance_list, line_buffer_dist): 
    """
    Extract salient points to generate the graph for the idx-th obstacle.
    """
    points_in_buffer        = points_in_buffer_list[idx] 
    poly_obs_instance_list  = poly_obs_instance_list[idx] 

    """0.033 seconds for one obstacle encounter."""       
    salient_points = compute_salient_points(points_in_buffer, robot_position, robot_orientation, path_end_point, line_buffer_dist)          

    return salient_points   




"""world_to_grid function""" 
def world_to_grid(world_coords, grid_init_np, grid_size):     
    """
    Convert world coordinates to grid coordinates.
    """

    if not isinstance(grid_init_np, np.ndarray): 
        grid_init_np = np.array(grid_init_np) 
    # Check if the variable is a NumPy array
    if not isinstance(world_coords, np.ndarray): 
        world_coords = (np.array(world_coords))
    # make sure that the world_coords is of shape (n, 2)
    
    world_coords = world_coords.reshape(-1, 2)

    grid_coord = ((world_coords - grid_init_np)/grid_size) 
    grid_max_x = np.round(np.max(grid_coord[:, 0]), 0).astype(int)
    grid_max_y = np.round(np.max(grid_coord[:, 1]), 0).astype(int) 

    # grid_coord = grid_coord.astype(int)  
    grid_coord_rounded = np.round(grid_coord, 0).astype(int) 

    # Conditionally adjust coordinates based on boundary limits
    adjusted_grid_coord = grid_coord.copy()

    # Clip x-coordinates if they exactly match the boundary limits
    adjusted_grid_coord[:, 0] = np.where(grid_coord_rounded[:, 0] == grid_max_x, grid_max_x - 1, adjusted_grid_coord[:, 0])

    # Clip y-coordinates if they exactly match the boundary limits 
    adjusted_grid_coord[:, 1] = np.where(grid_coord_rounded[:, 1] == grid_max_y, grid_max_y - 1, adjusted_grid_coord[:, 1])

    adjusted_grid_coord = np.round(adjusted_grid_coord, 0).astype(int) 

    return adjusted_grid_coord      

















"""a_star_search method"""   


def heuristic(node1, node2):
    return np.linalg.norm(np.array(node1) - np.array(node2))

def a_star_search(robot_position_grid, salient_points_grid, G, match_1_bool, match_2_bool, show_graph): 
    """
    Use the A* algorithm to find the shortest path from the robot's position to the salient points using the directed graph.
    """
    
    # Specify start and goal nodes    
    start_node      = (robot_position_grid[0,0], robot_position_grid[0,1])  
    salient_1_node  = (salient_points_grid[0, 0], salient_points_grid[0, 1])   
    salient_2_node  = (salient_points_grid[1, 0], salient_points_grid[1, 1])   

    try: 
        # Use A* algorithm to find the shortest path (based on edge weights and heuristic)       

        if match_1_bool:
            shortest_path_1 = nx.astar_path(G, source=start_node, target=salient_1_node, weight='weight', heuristic=heuristic)

        if match_2_bool:
            shortest_path_2 = nx.astar_path(G, source=start_node, target=salient_2_node, weight='weight', heuristic=heuristic)

        if match_1_bool:
            shortest_path = shortest_path_1
            goal_node = salient_1_node
        else:
            shortest_path = shortest_path_2 
            goal_node = salient_2_node
        


        if show_graph:
            """
            Visualize the directed graph with the shortest path from the start to the goal node.
            """
            # Create positions for nodes (to visualize on a 2D plane)
            pos = {node[:2]: node[:2] for node in G.nodes()}

            # Draw the graph
            plt.figure(figsize=(8, 8)) 
            nx.draw_networkx(G, pos=pos, with_labels=False, node_size=10, node_color='skyblue', font_size=10, font_color='black')

            # Highlight the nodes in the shortest path 
            path_edges = list(zip(shortest_path, shortest_path[1:]))
            nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges, edge_color='red', width=2)

            # Mark the start and goal nodes
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[start_node], node_color='green', node_size=400) 
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[goal_node], node_color='orange', node_size=400)

            # Display labels for start and goal nodes
            labels = {start_node: 'Start', goal_node: 'Goal'} 
            nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=10, font_color='black', font_weight='bold')

            plt.gca().set_aspect('equal')  # Set equal aspect ratio for the plot
            plt.axis('on')  # Ensure that axis ticks and labels are visible


            plt.title("Directed Graph with Shortest Path (A*)")
            plt.show() 

    except nx.NetworkXNoPath: 
        print_red_text("No path found from start to goal.") 
    
    return shortest_path   



"""extract_world_poses method"""   

def wrap_0_to_360(angle):      
    return angle % 360

def calculate_orientation(p1, p2):
    # Calculate orientation (angle) from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    orientation = math.atan2(dy, dx)
    return orientation

def find_poses_from_path(path): 
    total_points = len(path)//3
    four_points = [list(path[0]), list(path[total_points]), list(path[2*total_points]), list(path[-1])]  
    poses = []
    for i in range(len(four_points)-1):
        orientation = calculate_orientation(four_points[i], four_points[i+1]) 
        poses.append([four_points[i+1][0], four_points[i+1][1] , wrap_0_to_360(math.degrees(orientation))])    

    # modifying the last orientation.     
    orientation = wrap_0_to_360(math.degrees(calculate_orientation(path[-2], path[-1])))   
    poses[-1][2] = orientation 
    return poses   

def grid_to_world_list_form(grid_coords, grid_init, grid_size):
    world_coords_list = []
    for i, grid_coords in enumerate(grid_coords):
        world_coords_list.append([grid_init[0] + grid_size*grid_coords[0], grid_init[1] + grid_size*grid_coords[1], grid_coords[2]]) 
    return world_coords_list

 
def extract_world_poses(shortest_path, curr_grid_init, grid_dia):      
        """
        Extract the world poses from the shortest path.
        """
        poses = find_poses_from_path(shortest_path) 
        poses_in_world = grid_to_world_list_form(poses, curr_grid_init, grid_dia)    

        return poses_in_world  













"""generate_graph_and_goal_reach_status method """  

def update_grid(grid, array_int, ac_grid_y, value=255):  #  x_max = max_cols 
    array_int = np.array(array_int).reshape(-1, 2)
    grid[ac_grid_y-1-array_int[:,1], array_int[:, 0]] = value  
    return grid     

def check_if_cell_is_empty(grid, ac_grid_x, ac_grid_y, ac_grid_y_max):
    grid_values = grid[ac_grid_y_max-1-ac_grid_y, ac_grid_x] 
    return grid_values


def grid_shape(grid):
    return grid.shape[1], grid.shape[0]

       

def change_length_and_run(G, salient_points_reshaped_1, salient_points_reshaped_2, match_1_bool, match_2_bool, num_of_generation, default_num_children, grid, 
              robot_position, robot_orientation, line_lengths_np, angle_deviation_np):
    """
    Change the line lengths and run the graph generation process if the salient points are not found from the previous run.
    """

    G.clear()

    # Initialize the current generation with the robot's initial pose       
    previous_generation = np.array([robot_position[0, 0], robot_position[0, 1], robot_orientation]).reshape(-1, 3)   
    root_pose = (previous_generation[0, 0], previous_generation[0,1]) 
    G.add_node(root_pose, generation=0) 

    for gen in range(1, num_of_generation+1):  # Iterate over generations         

        endpoints_list = []   

        # Extract previous generation x, y, and orientation
        prev_x, prev_y, prev_orientation_rad = np.split(previous_generation, 3, axis=1)

        # Compute all new orientations at once for each deviation
        new_orientation_rad = prev_orientation_rad[:, None] + angle_deviation_np 

        # Compute all endpoint positions at once for each length and deviation 
        end_x = np.floor((prev_x[:, None] + line_lengths_np * np.cos(new_orientation_rad))).astype(np.int16)
        end_y = np.floor((prev_y[:, None] + line_lengths_np * np.sin(new_orientation_rad))).astype(np.int16)  


        # Stack filtered endpoints and orientations into a single array   
        endpoints_s = np.stack((end_x, end_y, new_orientation_rad), axis=-1).reshape(-1, 3)   
        
        # Extract x and y coordinates from the points array 
        xy_coordinates = endpoints_s[:, :2].astype(np.int16)  # Convert xy_coordinates to integers

        # check if the xy_ccordinates are outside the boundary limit if so then keep it to the last boundary limit.
        clipped_coordinates = np.clip(xy_coordinates, 0, np.array(grid_shape(grid)) - 1) # grid.shape 

        # # Create a boolean mask to identify valid indices  
        valid_indices = (
            np.all((xy_coordinates >= np.array([0, 0])) & (xy_coordinates < np.array(grid_shape(grid))), axis=1) &    
            (check_if_cell_is_empty(grid, clipped_coordinates[:, 0], clipped_coordinates[:, 1], grid_shape(grid)[1])==0) &
            (np.any(xy_coordinates == clipped_coordinates, axis=1))  
        )
        # Reshape the boolean array to have shape (-1, 4) where each row represents groups of four values        
        bool_reshaped = valid_indices.reshape(-1, default_num_children) 

        # Sum along the rows (axis=1) to count the number of True values in each group of four   
        num_children_np = np.sum(bool_reshaped, axis=1) 

        # Filter the points based on valid indices (within grid boundaries)   
        endpoints = endpoints_s[valid_indices] 

        if len(endpoints) == 0:   
            break 
        else:
            # Append filtered endpoints to the endpoints_list        
            endpoints_list.append(endpoints)   



        """uncomment this for regular behavior"""  
        current_generation = np.concatenate(endpoints_list, axis=0)  
        current_generation_e = current_generation[:, :2] 

        G.add_nodes_from(map(tuple, current_generation_e), generation=gen)  
        parent_nodes = np.repeat(previous_generation[:, :2], num_children_np, axis=0) 
        
        # compute the Euclidean distance between the nodes 
        egde_length = np.linalg.norm(parent_nodes - current_generation_e, axis=1)   

        # # Create edge tuples efficiently using zip with index  
        edges = [(tuple(parent), tuple(child), {'weight': egde_length[idx]}) for idx, (parent, child) in enumerate(zip(parent_nodes, current_generation_e))] 

        G.add_edges_from(edges)  
        
        # Update the previous generation with the current one           
        previous_generation = current_generation   

        # print("current generation: ", gen, current_generation.shape)    

        """ Check if all rows in salient_points are present in current_generation """   
        if not match_1_bool: 
            match_1 = np.all(current_generation[:, :2] == salient_points_reshaped_1, axis=2)
            if np.all(np.any(match_1, axis=1)):
                # print("First salient point found")
                match_1_bool = True 
                break
        
        if not match_2_bool:
            match_2 = np.all(current_generation[:, :2] == salient_points_reshaped_2, axis=2) 
            if np.all(np.any(match_2, axis=1)):
                # print("Second salient point found")
                match_2_bool = True 
                break


    return match_1_bool, match_2_bool, G 



def get_grid_with_graph_np_with_salient(G, salient_points, num_of_generation, grid, robot_position, robot_orientation, line_lengths, graph_length_reduction_iter, angle_deviation):  # faster one. 
    """
    Generate the graph and check if the salient points are present in the graph. If not, reduce the line lengths and run the graph generation process again.
    """

    # Reshape salient_points to enable broadcasting      
    salient_points_reshaped_1 = (salient_points[0]).reshape(-1, 2)[:, None, :]
    salient_points_reshaped_2 = (salient_points[1]).reshape(-1, 2)[:, None, :] 

    match_1_bool = False   
    match_2_bool = False 
  
    default_num_children = len(line_lengths) * len(angle_deviation)       


    angle_deviation_np  = np.array(angle_deviation)[None, :] 
    line_lengths_np     = np.array(line_lengths)[None, :]      

 
    for i in range(graph_length_reduction_iter): # 3   
        
        (match_1_bool, match_2_bool, G) = change_length_and_run(G, salient_points_reshaped_1, salient_points_reshaped_2, match_1_bool, match_2_bool, num_of_generation, 
                                                                                    default_num_children, grid, robot_position, robot_orientation, line_lengths_np, angle_deviation_np)

        if match_1_bool or match_2_bool:  
            break
        else:
            # print(f"{i} ==========================line_lengths_np: {line_lengths_np}")   
            line_lengths_np -= 0.4   




    return G, match_1_bool, match_2_bool     