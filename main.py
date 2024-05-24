#!/usr/bin/env python3 

import pdb 
import yaml 
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Tuple
from scipy.interpolate import interp1d
from shapely.geometry import Polygon    

from dubins import dubins   
from utils.obstacle_main import SetupObstacles, CheckObstacle         
from utils.print_utils import * 
        

def wrap_0_to_360(angle: float) -> float:      
    return angle % 360

class PathPlanning:          

    def __init__(self, robot_initial_pose: np.ndarray, goal_pos: np.ndarray, heading: float, search_angle: float, turning_radius: float,
                 min_dist_threshold_local_grid: float, distance_to_extende_polygons: float, buffer_distance: float,
                 square_dia: float, max_generation: int, line_buffer_dist: float, dist_moved_from_salient_points: float, total_area: float,
                 polygon_vertices_list_list: List[List[Tuple[float, float]]], square_offset: float, grid_dia: float, angle_deviation: List[float],
                 graph_length_reduction_iter: int, strips: List[List[Tuple[float, float]]]) -> None:

        """General parameters."""
        self.robot_pos              = robot_initial_pose 
        self.goal_pos               = goal_pos
        self.current_heading        = wrap_0_to_360(heading)
        self.previous_heading       = wrap_0_to_360(heading)
        self.search_angle           = search_angle
        self.turning_radius         = turning_radius
        self.final_pose_list        = [[self.robot_pos[0], self.robot_pos[1], self.current_heading]] 
        self.dubins_path            = [] 
        self.last_goal_point        = False 



        """Obstacle avoidance parameters."""
        self.strips                          = strips  
        # Extract the first point of each sublist using list comprehension 
        first_points                         = [strip[0] for strip in strips]
        self.strips_np                       = np.array(first_points).reshape(-1, 2)
        self.min_dist_threshold_local_grid   = min_dist_threshold_local_grid 
        self.distance_to_extende_polygons    = distance_to_extende_polygons
        self.buffer_distance                 = buffer_distance
        self.square_dia                      = square_dia
        self.max_generation                  = max_generation
        self.line_buffer_dist                = line_buffer_dist
        self.dist_moved_from_salient_points  = dist_moved_from_salient_points
        self.total_area                      = total_area
        self.polygon_vertices_list_list      = polygon_vertices_list_list
        self.square_offset                   = square_offset
        self.grid_dia                        = grid_dia 
        self.angle_deviation                 = angle_deviation
        self.graph_length_reduction_iter     = graph_length_reduction_iter
        self.step_length_interp_instance = self.retrive_length_fitted_line()           

        
        self.setup_obstacles    = SetupObstacles(self.search_angle, self.min_dist_threshold_local_grid, self.total_area, self.distance_to_extende_polygons, self.grid_dia, self.buffer_distance)
        self.check_obstacle     = CheckObstacle(self.graph_length_reduction_iter, self.angle_deviation, self.max_generation, self.grid_dia, self.step_length_interp_instance, self.line_buffer_dist)

        """Setup the obstacles."""
        (self.moved_vertices_list, self.points_in_buffer_list, self.grid_rect_list, self.poly_obs_instance_list, self.grid_instance_list, self.grid_init_list, self.occ_grid_list, self.poly_grid_instance_list, 
        self.grid_vertices_list) = self.setup_obstacles.update_obstacle_info(self.polygon_vertices_list_list)     

        """Store the default grid list."""
        self.default_grid_list = copy.deepcopy(self.occ_grid_list) 

        self.generate_path()  
    
    def retrive_length_fitted_line(self):  

        """Fit a line to the data points."""
        distances = np.array([11.66, 14.14, 18, 25, 32, 40.3, 45.3, 50, 55, 107.6]) 
        step_lengths = np.array([2.15, 2.4, 2.7, 2.9, 3.1, 3.46, 3.60, 4.0, 4.6, 8.1])   

        # Create an interpolation function to predict step length based on distance  
        step_length_interp_instance = interp1d(distances, step_lengths, kind='linear', fill_value='extrapolate')    

        return step_length_interp_instance 



    def plot_polygon_obs(self, moved_vertices_list: List[List[Tuple[float, float]]], boundary_color: str = 'b-', fill_color: str = 'lightblue', alpha: float = 0.5, label: str = 'Polygon') -> None:
        
        """Plot the polygon obstacles."""
        # append the first point to the last point for all lists to close the polygon
        for i in range(len(moved_vertices_list)):
            moved_vertices_list[i].append(moved_vertices_list[i][0])
        
        # for moved_vertices_np in moved_vertices_list:
        for i, moved_vertices_np in enumerate(moved_vertices_list):
            moved_vertices_np = np.array(moved_vertices_np).reshape(-1, 2)
            if i == 0:
                plt.plot(moved_vertices_np[:, 0], moved_vertices_np[:, 1], boundary_color, label=label) 
                plt.fill(moved_vertices_np[:, 0], moved_vertices_np[:, 1], color=fill_color, alpha=alpha)

            plt.plot(moved_vertices_np[:, 0], moved_vertices_np[:, 1], boundary_color)
            plt.fill(moved_vertices_np[:, 0], moved_vertices_np[:, 1], color=fill_color, alpha=alpha) 

    def plot_data(self, fig_num: int, path: np.ndarray, path_label: str, title: str) -> None:

        f1 = plt.figure(fig_num)
        self.plot_polygon_obs(self.moved_vertices_list, label='Extended Obstacles') 
        self.plot_polygon_obs(self.polygon_vertices_list_list, boundary_color='r-', fill_color='red', label='Actual Obstacles')  
        self.plot_polygon_obs(self.grid_vertices_list, boundary_color='m-', fill_color='blue', alpha=0.2, label='Grid')  
        plt.plot(path[:, 0], path[:, 1], 'g-', label=path_label)   
        plt.title(title)   
        plt.xlabel('X-axis in meters')  
        plt.ylabel('Y-axis in meters')   
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)   
        plt.legend()  

    def convert_line_to_area(self, line: List[Tuple[float, float]], extension_length: float = 1.0, perpendicular_distance: float = 1.0) -> List[Tuple[float, float]]:
        
        """
        Convert a line into a rectangular area by extending it.

        Parameters
        ----------
        line : List[Tuple[float, float]]
            A list containing two tuples, each representing the coordinates of the endpoints of the line.
        extension_length : float, optional
            The length by which to extend the line in both directions (default is 1.0).
        perpendicular_distance : float, optional
            The distance by which to extend the rectangle perpendicular to the line (default is 1.0).

        Returns
        -------
        List[Tuple[float, float]]
            A list of tuples representing the four corners of the rectangle.
        """
        
        # Convert the line to a numpy array 
        line = np.array(line)

        # Calculate the angle of the line
        delta = line[1] - line[0]

        # Calculate the unit direction vector of the line
        direction = delta / np.linalg.norm(delta)

        # Extend the line in both directions
        extended_start = line[0] - extension_length * direction
        extended_end = line[1] + extension_length * direction

        # Calculate perpendicular vectors
        perpendicular_vector = np.array([-direction[1], direction[0]])

        # Find the four corner points of the rectangle 
        p1 = list(extended_start + perpendicular_distance * perpendicular_vector)
        p2 = list(extended_start - perpendicular_distance * perpendicular_vector)
        p3 = list(extended_end + perpendicular_distance * perpendicular_vector)
        p4 = list(extended_end - perpendicular_distance * perpendicular_vector)

        # Create a rectangle polygon
        rectangle = [p1, p2, p4, p3]
        return rectangle


    def generate_dubins_path(self, pose_list: List[List[float]], turning_radius: float) -> Tuple[np.ndarray, float]:

        """
        Generate a Dubins path given a list of poses and a turning radius.

        Parameters
        ----------
        pose_list : List[List[float]]
            A list of poses, where each pose is a list containing [x, y, yaw].
            Yaw is given in degrees.
        turning_radius : float
            The minimum turning radius for the Dubins path.

        Returns
        -------
        path : np.ndarray
            An array of points representing the Dubins path.
        """

        curvature = 1 / turning_radius  # Low curvature, more turning radius; high curvature, less turning radius. 
        
        path_segments = []   
        
        for i in range(len(pose_list) - 1):   
            start_pose = pose_list[i]
            end_pose = pose_list[i + 1]

            # Convert start and end yaws from degrees to radians
            start_yaw = np.deg2rad(start_pose[2])
            end_yaw = np.deg2rad(end_pose[2])

            # Plan Dubins path for the current segment
            path_x, path_y, path_yaw, mode, lengths = dubins.plan_dubins_path(
                start_pose[0], start_pose[1], start_yaw,
                end_pose[0], end_pose[1], end_yaw,
                curvature, step_size=0.1 
            )
            
            # Store the current segment for later concatenation 
            path_segments.append(np.column_stack((path_x, path_y)))

        # Concatenate all segments to form the complete path  
        path = np.concatenate(path_segments, axis=0)  
        return path   


    def compute_least_distant_point(self, curr_position: np.ndarray, strips_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Compute the least distant point from the current position within a set of strips.

        Parameters
        ----------
        curr_position : np.ndarray
            The current position of the robot as a numpy array [x, y].
        strips_np : np.ndarray
            Numpy array containing points in the strips.

        Returns
        -------
        least_distant_point : np.ndarray
            The point in strips_np that is closest to the current position.
        other_edge : np.ndarray
            The corresponding other edge of the strip containing the least distant point.
        """
        diff = strips_np - curr_position
        dist = np.linalg.norm(diff, axis=1)
        min_idx = np.argmin(dist)
        least_distant_point = strips_np[min_idx]  

        other_edge = None  
        for strip in self.strips: 
            if list(least_distant_point) == strip[0]:
                other_edge = strip[1]
                break

        return least_distant_point, other_edge

    def extract_extended_positions(self):
        """
        Extract the extended position from the current position and heading.

        This function calculates a new position by moving a certain distance 
        in the direction of the current heading to avoid being in the created obstacle.

        Parameters
        ----------
        None

        Returns
        -------
        curr_position : np.ndarray
            The new extended position as a numpy array [x, y].
        """
        curr_position   = np.array([self.final_pose_list[-1][0], self.final_pose_list[-1][1]]) 
        angle_to_extend = np.radians(self.final_pose_list[-1][2])

        # move this heading some units in the direction of angle_to_extend. 
        curr_position[0] += 5*np.cos(angle_to_extend)
        curr_position[1] += 5*np.sin(angle_to_extend)  

        return curr_position 
    

    def generate_path(self)-> None:   

        """
        Generate a path for striping while avoiding obstacles and aiming towards the goal position in the end.

        This function computes a series of waypoints that the robot should follow to reach its goal while avoiding obstacles.
        It uses Dubins path planning for smooth transitions and orientation management.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no path can be found to avoid obstacles.
        """
        
        curr_position = self.robot_pos  

        while True:      
            
            if not self.last_goal_point:     
                least_distant_point, other_edge = self.compute_least_distant_point(curr_position, self.strips_np)    
            else:
                least_distant_point = self.goal_pos 
                other_edge = None
            print_yellow_text(f"Least Distant Point: {least_distant_point}")

            # compute the current heading from current position to least_distant_point.  
            curr_heading_rad = np.arctan2(least_distant_point[1] - curr_position[1], least_distant_point[0] - curr_position[0]) 

            """Orientations are in degrees in poses """ 
            poses = self.check_obstacle.find_free_path_if_obs_intersect(curr_position, least_distant_point, curr_heading_rad, self.default_grid_list, self.grid_instance_list, self.points_in_buffer_list, 
                                                                        self.poly_obs_instance_list, self.occ_grid_list, self.grid_init_list, self.poly_grid_instance_list)

            if poses is None: # No Obstacle in path.   
                if self.last_goal_point:
                    self.final_pose_list.append([least_distant_point[0], least_distant_point[1], self.final_pose_list[-2][2]])  
                    print_green_text("Goal point reached.")  
                    break  

                print_yellow_text(f"No Obstacle in the path. point {least_distant_point} reached")
                print_green("Striping the line from", [least_distant_point, other_edge])

                angle_degrees = wrap_0_to_360(np.degrees(np.arctan2(other_edge[1] - least_distant_point[1], other_edge[0] - least_distant_point[0])))

                # Add the least_distant_point and other edge to the final_pose_list."""
                self.final_pose_list.append([least_distant_point[0], least_distant_point[1], angle_degrees]) 
                self.final_pose_list.append([other_edge[0], other_edge[1], angle_degrees])

                # remove least_distant_point and other edge from the self.strips_np and strips list.
                self.strips.remove([list(least_distant_point), list(other_edge)])
                self.strips.remove([list(other_edge), list(least_distant_point)])    

                
                # Adding new area to the polygon_vertices_list_list 
                new_area = self.convert_line_to_area([least_distant_point, other_edge])  

                # Add the new area to the polygon_vertices_list_list and update the obstacle information.
                self.polygon_vertices_list_list += [new_area]   
                (self.moved_vertices_list, self.points_in_buffer_list, self.grid_rect_list, self.poly_obs_instance_list, self.grid_instance_list, self.grid_init_list, self.occ_grid_list, self.poly_grid_instance_list, 
                self.grid_vertices_list) = self.setup_obstacles.update_obstacle_info(self.polygon_vertices_list_list)  
                
                self.default_grid_list = copy.deepcopy(self.occ_grid_list) 

                # Obtain extended position.
                curr_position = self.extract_extended_positions() 
                
                if len(self.strips) == 0:  
                    self.last_goal_point = True  
                    
                else: 
                    first_points                         = [strip[0] for strip in self.strips] 
                    self.strips_np                       = np.array(first_points).reshape(-1, 2)    


            elif len(poses) > 0: # Obstacle in path and another path found.     
                print_red("Obstacle in path. New path found.", poses)
                self.final_pose_list += poses
                curr_position = np.array([self.final_pose_list[-1][0], self.final_pose_list[-1][1]])

            else:   # Obstacle in path and no path found. 
                print_red_text("Obstacle in path. No path found.")
                raise RuntimeError("Cannot find any path. Exiting.")    
            
            print_magenta_line() 


        """Generate Dubins path""" 
        self.dubins_path = self.generate_dubins_path(self.final_pose_list, self.turning_radius)       


        """Plotting the data."""  
        final_pose_np = np.array(self.final_pose_list)[:, :2]   
        self.plot_data(fig_num=1, path=final_pose_np, path_label='Straight Path', title='Obstacles free straight Path')
        self.plot_data(fig_num=2, path=self.dubins_path, path_label='Dubins Path', title='Obstacles free Dubins Path')
        plt.show()  

  















     
                
        

   



# Constants
YAML_FILE_PATH: str = "./param.yaml"

def load_params(file_path: str) -> Any:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def main() -> None:
    # Load parameters
    param = load_params(YAML_FILE_PATH)
    if param is None:
        return

    # Tunable Hyperparameters 
    robot_init_pose: np.ndarray             = np.array(param['tunable_params']['start_position'])
    goal_pos: np.ndarray                    = np.array(param['tunable_params']['goal_position'])
    init_heading: int                       = param['tunable_params']['init_heading']
    search_angle: int                       = param['tunable_params']['search_angle']
    turning_radius: float                   = param['tunable_params']['turning_radius']

    # Obstacle parameters
    strips: List[List[List[int]]]           = param['obstacle_params']['strips']
    min_dist_threshold_local_grid: float    = param['obstacle_params']['min_dist_threshold_local_grid']
    distance_to_extend_polygons: float      = param['obstacle_params']['distance_to_extende_polygons']
    buffer_distance: float                  = param['obstacle_params']['buffer_distance']
    square_dia: float                       = param['obstacle_params']['square_dia']
    max_generation: int                     = param['obstacle_params']['max_generation']
    line_buffer_dist: float                 = param['obstacle_params']['line_buffer_dist']
    dist_moved_from_salient_points: float   = param['obstacle_params']['dist_moved_from_salient_points']
    total_area: List[List[int]]             = param['obstacle_params']['total_area']
    polygon_1: List[List[int]]              = param['obstacle_params']['polygon_1']
    polygon_2: List[List[int]]              = param['obstacle_params']['polygon_2']
    polygon_3: List[List[int]]              = param['obstacle_params']['polygon_3']

    polygon_vertices_list_list: List[List[List[int]]] = [polygon_1, polygon_2, polygon_3]  


    graph_length_reduction_iter: int        = param['obstacle_params']['graph_length_reduction_iter']

    # Grid parameters
    square_offset: float                    = square_dia / 2
    grid_dia: float                         = square_dia

    # Define lengths of the lines (in meters) and angle deviations (in degrees)
    angle_deviation: List[float]            = [np.radians(-search_angle), np.radians(search_angle)]

    # Main algorithm
    new_path = PathPlanning(
        robot_init_pose, goal_pos, init_heading, search_angle,
        turning_radius, min_dist_threshold_local_grid,
        distance_to_extend_polygons, buffer_distance, square_dia, max_generation,
        line_buffer_dist, dist_moved_from_salient_points, total_area,
        polygon_vertices_list_list, square_offset, grid_dia, angle_deviation,
        graph_length_reduction_iter, strips
    )

if __name__ == "__main__":
    main() 
     