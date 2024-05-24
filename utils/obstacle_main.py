#!/usr/bin/env python3

import pdb
import copy
import numpy as np
from typing import List, Tuple, Any
 
from utils.check_obstacle_class_utils import *      
from utils.setup_obstacle_class_utils import * 
 

class SetupObstacles:
    def __init__(self, angle_range_d: float, min_dist_threshold: float, total_area: List[List[float]], distance_to_extende_polygons: float, grid_dia: float, buffer_distance: float):
        """
        SetupObstacles class handles the initialization of obstacle-related parameters and settings for the path planning.

        Attributes
        ----------
        angle_range_d : float
            The angle range in degrees for obstacle avoidance.
        min_dist_threshold : float
            The minimum distance threshold for considering obstacles.
        total_area : List[List[float]]
            The total area for the operation, defined by a list of coordinates.
        distance_to_extende_polygons : float
            The distance to extend the polygons around obstacles.
        grid_dia : float
            The diameter of the grid cells for the occupancy grid.
        buffer_distance : float
            The buffer distance around obstacles for safe navigation.
        inital_point : Tuple[int, int]
            The initial point of the robot, fixed at (0, 0).
        initial_heading : int
            The initial heading of the robot, fixed at 90 degrees.
        """
        self.angle_range_d                      = angle_range_d
        self.min_dist_threshold                 = min_dist_threshold
        self.total_area                         = total_area
        self.distance_to_extende_polygons       = distance_to_extende_polygons 
        self.grid_dia                           = grid_dia
        self.buffer_distance                    = buffer_distance 

        """Robot specifications""" 
        self.inital_point                       = (0, 0)  # don't change this. 
        self.initial_heading                    = 90      # don't change this. 


    def obtain_grid_for_all_obstacles(self, polygon_vertices_list_list, delta_x, y_list1):
        """
        Obtain grid information for all obstacles.

        This method iterates over a list of polygon vertices lists representing obstacles. 
        For each obstacle, it computes the extended vertices, grid rectangle, and grid instance, based on parameters such as delta_x and y_list1.

        """
        moved_vertices_list = []
        grid_rect_list = []
        grid_instance_list = []


        for polygon_vertices_list in polygon_vertices_list_list: 
            """For each obstacles"""  # rectangles are represented as (min_x, min_y, max_x, max_y).   
            moved_vertices, obstacle_extended_rect      = obtain_extended_polygon_vertices(polygon_vertices_list, self.distance_to_extende_polygons)

            """grid is [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)] and obstacle_center is (x, y) and grid_rect is [min_x, min_y, max_x, max_y]"""
            grid_rect, grid_instance                    = compute_grid_size(obstacle_extended_rect, delta_x, y_list1, self.total_area)      

            moved_vertices_list.append(moved_vertices) 
            grid_rect_list.append(grid_rect)
            grid_instance_list.append(grid_instance)
            
        return moved_vertices_list, grid_rect_list, grid_instance_list 


    def obtain_points_for_all_grids(self, moved_vertices_list, grid_rect_list):
        """
        Obtain points inside and within buffer zones for all grids.

        This method iterates over the moved vertices list and corresponding grid rectangle list to obtain points inside each grid and within buffer zones. 
        It returns lists containing points inside each grid, points within buffer zones, polygon instances for obstacle buffers, empty occupancy grids, and grid sizes.

        Parameters:
        - moved_vertices_list (List[List[Tuple[float, float]]]): A list of lists containing the vertices of moved polygons.
        - grid_rect_list (List[List[float]]): A list of lists containing the rectangle coordinates of grid regions.

        Returns:
        - points_inside_list (List[np.ndarray]): A list of arrays containing points inside each grid region.
        - points_in_buffer_list (List[np.ndarray]): A list of lists containing points within buffer zones for each grid.
        - poly_obs_instance_list (List[Polygons]): A list of lists containing polygon instances representing obstacle buffers for each grid.
        - empty_occ_grid_list (List[np.ndarray]): A list of arrays containing empty occupancy grids.
        - grid_size_list (List[List[int]]): A list of lists containing grid sizes.
        """
        points_inside_list      = []
        points_in_buffer_list   = [] 
        poly_obs_instance_list  = []
        empty_occ_grid_list = []
        grid_size_list = []

        for i in range(len(moved_vertices_list)):   
            
            points_outside, points_inside, ac_grid_x, ac_grid_y, empty_occ_grid     = obtain_points_outside_obstacles(self.grid_dia, grid_rect_list[i], moved_vertices_list[i]) 
            points_in_buffer, poly_obs_instance                                     = find_points_within_buffer(points_outside, moved_vertices_list[i], self.buffer_distance)               # points_to_check, polygon_vertices, buffer_distance=1.0  

            points_inside_list.append(points_inside)
            points_in_buffer_list.append(points_in_buffer)
            poly_obs_instance_list.append(poly_obs_instance)
            empty_occ_grid_list.append(empty_occ_grid)  
            grid_size_list.append([ac_grid_x, ac_grid_y])  


        return points_inside_list, points_in_buffer_list, poly_obs_instance_list, empty_occ_grid_list, grid_size_list

    def get_grid_init(self, grid_rect_list):
        
        """
        Get the initial coordinates of the grids.

        Extracts the lower-left corner coordinates of each grid rectangle and rounds them to one decimal place.
        """
        
        grid_init_list = [rect[:2].copy() for rect in grid_rect_list]

        # Apply offset to each corner of the rectangle in actual_region   
        for rect in grid_init_list:
            
            # Modify the lower-left corner 
            rect[0] = round((rect[0]), 1) 
            rect[1] = round((rect[1]), 1)    
            
        return grid_init_list 

    def put_obs_in_empty_grid(self, grid_init_list, empty_occ_grid_list, points_inside_obs_list_for_grid, grid_size_list):
        """
        Populate empty occupancy grids with obstacle points.

        Converts world coordinates of obstacles to grid coordinates and updates the empty occupancy grids with these obstacles.
        """
        
        for i, empty_occ_grid in enumerate(empty_occ_grid_list):
            grid_coords     = world_to_grid(points_inside_obs_list_for_grid[i], grid_init_list[i], self.grid_dia)   
            empty_occ_grid  = update_grid(empty_occ_grid, grid_coords, grid_size_list[i][1])     
        return empty_occ_grid_list  

    def create_grid_polygon_instance(self, grid_rect_list):

        """
        Create polygon instances for each grid based on the provided rectangle vertices.

        Constructs Shapely polygons from the lower-left and upper-right corners of each rectangle in the grid,
        storing these polygons and their vertices for further processing.
        """
        
        # List to store Shapely polygons           
        poly_grid_instance_list = []
        grid_vertices_list = []

        # Iterate over each sublist in grid_rect_list     
        for rect in grid_rect_list:
            # Extract lower-left and upper-right corners 
            ll = (rect[0], rect[1])  # Lower-left corner 
            ur = (rect[2], rect[3])  # Upper-right corner 
            
            # Construct the polygon
            grid_vertices_list.append([ll, (ur[0], ll[1]), ur, (ll[0], ur[1])])
            polygon = Polygon([ll, (ur[0], ll[1]), ur, (ll[0], ur[1])])
            
            # Append the polygon to the list    
            poly_grid_instance_list.append(polygon)
        
        return poly_grid_instance_list, grid_vertices_list

    def update_obstacle_info(self, polygon_vertices_list_list: List[List[Tuple[float, float]]]):      
        """
        Update obstacle information and compute necessary details for path planning.

        Parameters
        ----------
        polygon_vertices_list_list : List[List[Tuple[float, float]]]
            List of polygons representing obstacles, where each polygon is defined by a list of vertices.

        Returns
        -------
        moved_vertices_list : List[List[Tuple[float, float]]]
            List of moved vertices for each obstacle.
        points_in_buffer_list : List[Tuple[float, float]]
            List of points in the buffer area around obstacles.
        grid_rect_list : List[List[Tuple[float, float]]]
            List of grid rectangles for each obstacle.
        poly_obs_instance_list : List[Polygon]
            List of polygon obstacle instances.
        grid_instance_list : List[Polygon]
            List of grid instances.
        grid_init_list : List[List[float]]
            List of initial grid values.
        occ_grid_list : List[np.ndarray]
            List of occupancy grids with obstacles marked.
        poly_grid_instance_list : List[Polygon]
            List of grid polygon instances.
        grid_vertices_list : List[List[Tuple[float, float]]]
            List of vertices for each grid.
        """

        """ Data for dynamic grid generation."""
        poses, delta_p  = compute_x_and_y(self.angle_range_d, self.min_dist_threshold, self.inital_point, self.initial_heading)   
        delta_x         = [delta[0] for delta in delta_p]
        y_list1         = [pose[1] for pose in poses]   

    # ==============================================================================================================================

        """Obtain grid for all obstacles"""
        (moved_vertices_list, grid_rect_list, grid_instance_list) = self.obtain_grid_for_all_obstacles(polygon_vertices_list_list, delta_x, y_list1)  


    # ==============================================================================================================================   
        
        (points_inside_list, points_in_buffer_list, poly_obs_instance_list, empty_occ_grid_list, grid_size_list) = self.obtain_points_for_all_grids(moved_vertices_list, grid_rect_list)  
    
        grid_init_list = self.get_grid_init(grid_rect_list) 

        points_inside_obs_list_for_grid = [] 
        for points_inside_obs in points_inside_list:
            points_inside_obs_list_for_grid.append(np.round(points_inside_obs, decimals=1))  
        
        """Update the occupancy grid with obstacles"""      
        occ_grid_list = self.put_obs_in_empty_grid(grid_init_list, empty_occ_grid_list, points_inside_obs_list_for_grid, grid_size_list)      

        """Create grid polygon instances"""
        poly_grid_instance_list, grid_vertices_list = self.create_grid_polygon_instance(grid_rect_list)

        return (moved_vertices_list, points_in_buffer_list, grid_rect_list, poly_obs_instance_list, grid_instance_list, grid_init_list, occ_grid_list, poly_grid_instance_list, grid_vertices_list) 



























class CheckObstacle: 
    """
    Class for checking obstacles and performing related computations.

    Attributes:
    - graph_length_reduction_iter (float)   : The iteration count for graph length reduction.
    - angle_deviation (float)               : The angle deviation value.
    - max_generation (int)                  : The maximum generation value.
    - grid_dia (float)                      : The diameter of the grid.
    - step_length_interp_instance (function): The interpolation function for step length.
    - line_buffer_dist (float)              : The buffer distance for lines.
    """

    def __init__(self, graph_length_reduction_iter, angle_deviation, max_generation, grid_dia, step_length_interp_instance, line_buffer_dist): 
        
        self.graph_length_reduction_iter     = graph_length_reduction_iter
        self.angle_deviation                 = angle_deviation
        self.max_generation                   = max_generation
        self.grid_dia                        = grid_dia
        self.step_length_interp_instance     = step_length_interp_instance
        self.line_buffer_dist                = line_buffer_dist 

    def retrive_length(self, distances_to_test):     
        """
        Retrieve the step length based on the given distances.
        """
        length = self.step_length_interp_instance(distances_to_test)     
        if length < 2.15:
            length = 2.15  
        return [length]     


    def check_path_and_give_grid_and_salient_points(self, robot_position, path_end_point, robot_orientation, grid_instance_list, points_in_buffer_list, poly_obs_instance_list, occ_grid_list, grid_init_list, poly_grid_instance_list):

        """
        Check the path for obstacles and return grid and salient points.
        """

        """Only uses start and end point for collision detection"""    
        is_obs_i_in_path, is_robot_in_grid, obstacle_idx = obs_ID_if_obs_in_path(robot_position, path_end_point, grid_instance_list, poly_obs_instance_list)       

        if is_obs_i_in_path: # True if obstacle in path.  

            obstacle_present = True  
            """Extract salient points to generate graph. It uses robot position and orientation. Not end point."""  
            (salient_points_world) = extract_salient_points_to_generate_graph(obstacle_idx, robot_position, robot_orientation, path_end_point, points_in_buffer_list, poly_obs_instance_list, self.line_buffer_dist) 
            
            """If robot is not in grid, then move it to the edge of the grid based on intersection."""
            if not is_robot_in_grid:
                grid_poly               = poly_grid_instance_list[obstacle_idx]
                line                    = LineString([robot_position, path_end_point]) 
                intersection            = grid_poly.intersection(line)
                robot_position          = list(intersection.coords[0])   
                robot_position[0]       = robot_position[0] + np.cos(robot_orientation)
                robot_position[1]       = robot_position[1] + np.sin(robot_orientation) 
    

            """parameters required for further processing"""     
            obstacle_idx                        = obstacle_idx 
            occ_grid_list                       = occ_grid_list
            salient_points_world                = salient_points_world
            robot_position                      = robot_position
            grid_init_list                      = grid_init_list 


            curr_occ_grid                           = occ_grid_list[obstacle_idx]    
            curr_grid_init                          = grid_init_list[obstacle_idx]   
            curr_grid_init_np                       = np.array(curr_grid_init).reshape(-1, 2)

            """Convert all the points to grid space"""          
            robot_position_grid                     = world_to_grid(robot_position, curr_grid_init_np, self.grid_dia)  
            salient_points_grid                     = world_to_grid(salient_points_world, curr_grid_init_np, self.grid_dia)


            return obstacle_present, obstacle_idx, salient_points_grid, curr_occ_grid, curr_grid_init, robot_position_grid  

        else:  
            obstacle_absent = False 
            return obstacle_absent, None, None, None, None, None



    def generate_graph_and_goal_reach_status(self, salient_points_grid, robot_position_grid, curr_occ_grid, robot_orientation):
        """
        Generate a graph and determine the goal reach status.
        """
        
        diff = np.linalg.norm(salient_points_grid - robot_position_grid, axis=1)
        min_diff = np.min(diff) 
    
        line_lengths = self.retrive_length(min_diff)   

        """Generate graph"""     
        G = nx.DiGraph()         
        """Obtain graph connecting start and goal points to avoid obstacles."""
        G, match_1_bool, match_2_bool = get_grid_with_graph_np_with_salient(G, salient_points_grid, self.max_generation, curr_occ_grid, robot_position_grid, robot_orientation, line_lengths, self.graph_length_reduction_iter, self.angle_deviation) 

        return G, match_1_bool, match_2_bool    


    def find_free_path_if_obs_intersect(self, robot_position, path_end_point, robot_orientation, default_grid_list, grid_instance_list, points_in_buffer_list, poly_obs_instance_list, occ_grid_list, grid_init_list, poly_grid_instance_list):
        """
        Find a free path if an obstacle intersects.

        This method checks if an obstacle intersects the path from the robot's current position to the path end point. 
        If an obstacle is present, it generates a graph and attempts to find the shortest path around the obstacle. 
        If no obstacle is present, it returns None.
        """

        (is_obstacle_present, obstacle_idx, salient_points_grid, curr_occ_grid, curr_grid_init, robot_position_grid   
         ) = self.check_path_and_give_grid_and_salient_points(robot_position, path_end_point, robot_orientation, grid_instance_list, points_in_buffer_list, poly_obs_instance_list, occ_grid_list, grid_init_list, poly_grid_instance_list)

        
        if is_obstacle_present: # True if obstacle is present in the path.                   

            curr_occ_grid = copy.deepcopy(default_grid_list[obstacle_idx])    
            
            G, match_1_bool, match_2_bool = self.generate_graph_and_goal_reach_status(salient_points_grid, robot_position_grid, curr_occ_grid, robot_orientation)

            if not match_1_bool and not match_2_bool:    
                print_red_text("No salient points found in the graph.")      
                return []  # empty showing no path found.  
        
            """Find the shortest path in the graph using A* search."""
            shortest_path  = a_star_search(robot_position_grid, salient_points_grid, G, match_1_bool, match_2_bool, show_graph=False)  

            """Extract intermediate poses from the shortest path."""
            poses_in_world = extract_world_poses(shortest_path, curr_grid_init, self.grid_dia)   

            return poses_in_world          

        else:       
            return None # Nonw: No obstacle in path.   
        

