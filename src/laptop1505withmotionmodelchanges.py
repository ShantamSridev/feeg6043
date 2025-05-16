# Copyright (c) 2023 The uos_sess6072_build Authors.
# Authors: Miquel Massot, Blair Thornton, Sam Fenton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# See LICENSE.md file in the project root for full license information.
# """

import numpy as np
import traceback 
import argparse
from datetime import datetime
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation, t2v, v2t, polar2cartesian
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control
from model_feeg6043 import graphslam_frontend, lidar_scan, graphslam_backend 
from classifier import GPC_input_output, load_model
from plot_feeg6043 import plot_2dframe, sigma_contour, show_information,  plot_graph
import copy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import joblib
import csv
import os

class LaptopPilot:
    """
    This class controls a differential drive robot using visual feedback from ArUco markers
    and wheel encoder measurements. It implements trajectory following with feedback control.
    """

    def __init__(self, simulation):
        """
        Initialize the robot controller.

        Args:
            simulation (bool): Whether running in simulation mode or on real hardware
        """

        # ============ NETWORK AND COMMUNICATION SETUP ============
        # ArUco marker detection parameters - these markers are visual tags the robot uses to determine its position
        aruco_params = {
            "port": 50000,      # Network port to receive ArUco marker data (DO NOT CHANGE)
            "marker_id": 24,    # ID of the specific marker to track (CHANGE THIS to your marker ID)
        }
        self.corner_pose_northings = None
        self.corner_pose_eastings = None
        # Set the robot's IP address for communication
        self.robot_ip = "192.168.90.1"  # Default IP for real robot

        # ============ SIMULATION MODE CONFIGURATION ============
        # When running in simulation, we need different settings
        self.sim_time_offset = 0      # Used to convert between simulator time and real time
        self.sim_init = False         # Flag to track if we've initialized simulation time
        self.simulation = simulation  # Store whether we're in simulation mode

        if self.simulation:
            self.robot_ip = "127.0.0.1"      # Local IP for simulation
            aruco_params['marker_id'] = 0    # Simulation uses marker ID 0
            self.sim_init = True             # Need to initialize simulation time

        print("Connecting to robot with IP", self.robot_ip)

        # Create the ArUco driver that will receive position updates
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)


        # ============ ROBOT PHYSICAL PARAMETERS ============
        # These define the physical dimensions of the robot
        wheel_distance =  0.163/2  # Distance between left and right wheels in meters
        wheel_diameter = 0.065  # Diameter of each wheel in meters

        # Create differential drive configuration object
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)


        # ============ TRAJECTORY PARAMETERS ============
        # These control how the robot moves along its path
        self.velocity = 0.1              # Desired forward velocity in m/s
        self.acceleration = self.velocity/3 # How quickly to reach desired velocity
        self.turning_radius = 0.2         # Minimum turning radius in meters
        self.acceptance_radius = 0.2 

        # ============ CONTROL PARAMETERS ============
        # These values control how aggressively the robot corrects errors
        self.tau_s = 1  # Time constant for removing along-track error (seconds)
        self.L = 0.2      # Distance constant for removing cross-track and angular error (meters)

        # Control gains - these determine how strongly the robot responds to errors
        self.k_s = 1/self.tau_s  # Along-track gain (how strongly to correct forward/backward error)

        # Velocity and turning rate limits for safety
        self.v_max = 0.15        # Maximum forward/backward speed in m/s
        self.w_max = np.deg2rad(30)  # Maximum turning rate in rad/s (15 degrees/s)

        # Initialization flags
        self.initialise_control = True  # Will be set to False after first control update
        self.initialise_pose = True     # Will be set to False after receiving first position


        # ============ PATH WAYPOINTS ============
        # Define the path the robot should follow as a series of points
        # Each point has a northing (y) and easting (x) coordinate
        self.northings_path_single = [0.2, 1.2, 1.2, 0.2]
        self.eastings_path_single = [0.0, 0.0, 1.1, 1.1]
        self.northings_path = self.northings_path_single + self.northings_path_single + [0.0]
        self.eastings_path = self.eastings_path_single + self.eastings_path_single + [0.0]  
        self.relative_path = True  # If True, path is relative to robot's starting position


        # ============ ROBOT POSE ESTIMATES ============
        # These store the robot's estimated position and orientation
        self.est_pose_northings_m = None  # Estimated north position in meters
        self.est_pose_eastings_m = None   # Estimated east position in meters
        self.est_pose_yaw_rad = None      # Estimated heading angle in radians


        # ============ SENSOR MEASUREMENTS ============
        # Store the latest measurements from various sensors

        # Visual position measurements from ArUco markers
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # Wheel speed commands sent to the robot
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None

        # Actual wheel speeds measured by encoders
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None


        # ============ LIDAR SENSOR SETUP ============
        # LIDAR measures distances to obstacles around the robot
        self.lidar_timestamp_s = None
        self.lidar_data = None

        # LIDAR sensor position relative to robot center
        self.lidar_xb = 0.07  # 7cm forward of robot center
        self.lidar_yb = 0.0   # Centered left-right
        self.lidar = RangeAngleKinematics(self.lidar_xb, self.lidar_yb)


        # ============ STATE VECTOR FOR SIMPLE TRACKING ============
        # The state vector contains all variables we're tracking:
        # [North position, East position, Heading angle, Forward velocity, Angular velocity]

        # Index names for the state vector (makes code more readable)
        self.N = 0      # North position index
        self.E = 1      # East position index
        self.G = 2      # Heading (gamma) angle index
        self.DOTX = 3   # Forward velocity index
        self.DOTG = 4   # Angular velocity index

        # Initialize the state vector with 5 elements
        self.state = Vector(3)
        self.state[self.N] = 0.0  # North position
        self.state[self.E] = 0.0
        self.state[self.G] = 0.0  # Heading angle

        self.p_gt_path = []  # Store ground truth path for plotting


        # ============ LOGGING AND DEBUGGING ============
        self.aruco_count = 0  # Count of ArUco measurements received
        self.loop_count = 0   # Count of control loops executed

        # Data logger to save sensor data for later analysis
        self.datalog = DataLogger(log_dir="logs")


        # ============ ROS-STYLE COMMUNICATION SETUP ============
        # Publishers send commands to the robot
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )


        
         # ============ SLAM SETUP ============

        #Motion model linear noise due to v and w
        # self.sigma_motion=Matrix(3,2)
        # self.sigma_motion[0,0]= 0.1**2 # impact of v linear velocity on x           #Task
        # self.sigma_motion[0,1]= np.deg2rad(1)**2 # impact of w angular velocity on x
        # self.sigma_motion[1,0]=0.1**2 # impact of v linear velocity on y
        # self.sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
        # self.sigma_motion[2,0]=0.1**2 # impact of v linear velocity on gamma
        # self.sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma



        # self.sigma_motion=Matrix(3,2)
        # self.sigma_motion[0,0]= 0.5**2 # impact of v linear velocity on x           #Task
        # self.sigma_motion[0,1]= np.deg2rad(5)**2 # impact of w angular velocity on x
        # self.sigma_motion[1,0]=0.5**2 # impact of v linear velocity on y
        # self.sigma_motion[1,1]=np.deg2rad(1)**2 # impact of w angular velocity on y
        # self.sigma_motion[2,0]=0.5**2 # impact of v linear velocity on gamma
        # self.sigma_motion[2,1]=np.deg2rad(1)**2 # impact of w angular velocity on gamma



        self.sigma_motion=Matrix(3,2)
        self.sigma_motion[0,0]= 0.15**2 # impact of v linear velocity on x           #Task
        self.sigma_motion[0,1]= np.deg2rad(1)**2 # impact of w angular velocity on x
        self.sigma_motion[1,0]=0.15**2 # impact of v linear velocity on y
        self.sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
        self.sigma_motion[2,0]=0.15**2 # impact of v linear velocity on gamma
        self.sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma


        print('3x2 motion noise model:\n', self.sigma_motion, '\n')

        # Observation model linear noise with range
        self.sigma_observe = Matrix(2, 2)
        self.sigma_observe[0, 0] = 0.25**2  # 10% of range
        self.sigma_observe[0, 1] = 0
        self.sigma_observe[1, 0] = np.deg2rad(30)**2  # 5 degree per metre range
        self.sigma_observe[1, 1] = 0
        print('2x2 measurement noise model:\n', self.sigma_observe, '\n')


        # anchor constraint, matrix must be invertable

        self.sigma = Matrix(3,3) 
        self.sigma[0,0]=0.1
        self.sigma[0,1]=0.01
        self.sigma[1,0]=0.01
        self.sigma[1,1]=0.1
        self.sigma[0,2]=0.01
        self.sigma[1,2]=0.01
        self.sigma[2,0]=0.01
        self.sigma[2,1]=0.01
        self.sigma[2,2]=0.1

        ################ initialise graph ################
        print('Start graph data association')
        self.graph = graphslam_frontend()
        self.graph.anchor(self.sigma)

        self.completed_loop = False
        self.corner_detected = False
        self.optimisation_statement = False
        self.d_p_eb = Vector(3)
        self.d_p_eb[0] = 0
        self.d_p_eb[1] = 0
        self.d_p_eb[2] = 0

        self.prob_thresh = 0.78           # min classifier confidence
        self.max_corner_dist = 2       # meters



        self.last_landmark_id = 0  # Track the last landmark we visited
        self.landmark_id = 0
        self.landmark_visits = {0: 0, 1: 0, 2: 0, 3: 0}  # Count visits to each landmark
        self.last_landmark_timestamp = None
        self.landmark_locations = np.array([
            [0.0, 0.0],  # Landmark 0
            [0.0, 2.0],  # Landmark 1
            [2.0, 2.0],  # Landmark 2
            [2.0, 0.0]   # Landmark 3
        ])
        self.observation_counter = 0
        self.optimised = False

        # ——— NEW: open CSV for corner detections ———
        out_dir = os.path.join("logs", "corners")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"corner_detections_{ts}.csv"
        self._corner_log = open(filename, "w", newline="")
        self._corner_writer = csv.writer(self._corner_log)
        self._corner_writer.writerow([
            "timestamp_iso8601", "northing_m", "easting_m", "landmark_id"
        ])
        print(f"Logging corners to {filename}")

        # Subscribers receive data from the robot
        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds", Vector3Stamped,
            self.true_wheel_speeds_callback, ip=self.robot_ip
        )

        self.lidar_sub = Subscriber(
            "/lidar", LaserScan,
            self.lidar_callback, ip=self.robot_ip
        )

        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose,
            self.groundtruth_callback, ip=self.robot_ip
        )



    def true_wheel_speeds_callback(self, msg):
        """
        Callback function that receives actual wheel speed measurements from encoders.

        Args:
            msg: Message containing right wheel speed (x) and left wheel speed (y) in rad/s
        """
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y

        # Log the data for later analysis
        self.datalog.log(msg, topic_name="/true_wheel_speeds")


    def lidar_callback(self, msg):
        """
        Callback function that receives LIDAR scan data.
        Converts the raw range/angle measurements to map coordinates.

        Args:
            msg: LaserScan message containing ranges and angles to obstacles
        """
        # Handle time synchronization for simulation
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset

        # Store the timestamp of this LIDAR scan
        self.lidar_timestamp_s = msg.header.stamp

        # Safety check - only process LIDAR if we have a valid pose estimate
        if (self.est_pose_northings_m is None or
            self.est_pose_eastings_m is None or
            self.est_pose_yaw_rad is None):
            return  # Skip processing if pose not initialized

        # Create array to store converted LIDAR data

        self.lidar_data = np.zeros((len(msg.ranges), 2))

        # Current robot pose in earth frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m  # North position
        p_eb[1] = self.est_pose_eastings_m   # East position
        p_eb[2] = self.est_pose_yaw_rad      # Heading angle

        # Convert each LIDAR measurement from range/angle to map coordinates
        z_lm = Vector(2)
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]  # Distance to obstacle
            z_lm[1] = msg.angles[i]  # Angle to obstacle

            # Transform from robot frame to earth/map frame
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            # Store the converted coordinates
            self.lidar_data[i,0] = t_em[0]  # North coordinate
            self.lidar_data[i,1] = t_em[1]  # East coordinate

        # Remove any invalid measurements (NaN values)
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

        # Log the data
        self.datalog.log(msg, topic_name="/lidar")


    def groundtruth_callback(self, msg):
        """
        Callback that receives ground truth position from simulator.
        This is only available in simulation for debugging/evaluation.

        Args:
            msg: Pose message with true robot position
        """
        self.datalog.log(msg, topic_name="/groundtruth")




    

    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]

        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.utcnow().timestamp()-msg[0]
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset

            time_stamp = msg[0] + self.sim_time_offset                

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else: quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat        
        return pose_msg

    def generate_trajectory(self):
        """
        Generate a smooth trajectory from the waypoint list.
        This converts discrete waypoints into a continuous path with velocity profiles.
        """
        # If using relative path, add current position to all waypoints
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.measured_pose_northings_m
                self.eastings_path[i] += self.measured_pose_eastings_m

        # Convert waypoint lists to matrix format
        C = l2m([self.northings_path, self.eastings_path])

        # Create trajectory generator object
        self.path = TrajectoryGenerate(C[:,0], C[:,1])

        # Set trajectory parameters
        self.path.path_to_trajectory(self.velocity, self.acceleration)
        self.path.turning_arcs(self.acceptance_radius)
        self.path.wp_id = 0  # Start at first waypoint

    def find_corner(self, corner, threshold=0.001):
        # identify the reference coordinate as the inflection point

        # Step 1: Compute slope
        slope = np.gradient(corner.data[:, 0])

        # Step 2: Compute the second derivative (curvature)
        curvature = np.gradient(slope)

        # Step 3: Check if criteria is more than threshold
        # print('Max inflection value is ',np.nanmax(abs(np.gradient(np.gradient(curvature)))), ': Threshold ',threshold)
        if np.nanmax(abs(np.gradient(np.gradient(curvature)))) > threshold:
            # compute index of inflection point
            largest_inflection_idx = np.nanargmax(
                abs(np.gradient(np.gradient(curvature)))
            )

            r = corner.data[
                largest_inflection_idx, 0
            ]  # Radial distance at the largest curvature
            theta = corner.data[
                largest_inflection_idx, 1
            ]  # Angle at the largest curvature
            return r, theta, largest_inflection_idx

        else:
            return None, None, None  # No inflection points found

    class GPC_input_output:
        def __init__(self, data, label):
            """
            Initializes an observation with data and a label.

            Parameters:
            data (matrix): The observation data (e.g., a matrix).
            data_filled (matrix): The observation data after zero offset and making nan's mean
            label (str): The label associated with the observation.
            ne_representative: representative northings and eastings location
            """
            self.data = data
            self.data_filled = self._fill_nan(data)
            self.label = label
            self.ne_representative = None
            # make filled and zero offset version

        def _fill_nan(self, data):
            data_filled = np.copy(data)
            mean = np.nanmean(data[:, 0])
            for i in range(len(data[:, 1])):
                if np.isnan(data[i, 0]):
                    data_filled[i, 0] = 0
                else:
                    data_filled[i, 0] = data[i, 0] - mean
            return data_filled


    
    def run(self, time_to_run=-1):
        """
        Main execution loop of the robot controller.

        Args:
            time_to_run: How long to run in seconds (-1 for infinite)
        """
        self.start_time = datetime.utcnow().timestamp()

        try:
            # Create rate limiter for 10 Hz control loop
            r = Rate(10.0)

            while True:
                current_time = datetime.utcnow().timestamp()

                # Check if we should stop
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break

                # Execute one control cycle
                self.infinite_loop()

                # Sleep to maintain 10 Hz rate
                r.sleep()
    
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print(f"An exception of type {type(e).__name__} occurred.")
            print(f"Exception message: {str(e)}")
            print("Detailed traceback:")
            traceback.print_exc()
        finally:
            # Clean up subscribers
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()
            self._corner_log.close()



    def graph_optimisation_solve(self, graph_opt):
        # Initialize variables
        initial_residual=100
        residual_threshold=1E-12
        delta_threshold=1/1000
        lim_iterations=40
        n_iterations = 0
        delta_residual = initial_residual
        residual = initial_residual

        iteration_continue = True 
        residual_continue = True
        converge_continue = True

        # Start timer for solver performance measurement
        #cpu_start_solver = datetime.now()

        while iteration_continue and residual_continue and converge_continue:    
            graph_opt.solve()
            
            prev_residual = residual
            residual = graph_opt.residual

            delta_residual = abs((prev_residual - residual) /prev_residual)
            n_iterations += 1

            print('**************  Residual = ',residual,' ***************')        
            residual_continue = (residual > residual_threshold)
            print('Residual above threshold?',residual_continue)    
            
            print('************** Iteration = ',n_iterations,' ***************')
            iteration_continue = (n_iterations <= lim_iterations)
            print('Iterations below limit?',iteration_continue)
            
            print('********* Delta Residual = ',delta_residual,' ***************')
            converge_continue = (delta_residual > delta_threshold)
            print('Residual still changing?',converge_continue)
            
            #reconstruct the graph with these nodes
            graph_opt = graphslam_frontend(graph_opt)   # Task
            graph_opt.construct_graph() # Task
            graph_opt = graphslam_backend(graph_opt)    # Task
            
        
            #cpu_end_solver = datetime.now()
            #delta =  cpu_end_solver - cpu_start_solver       
            #print('********* Final solution took:',(delta.total_seconds()),'s ***************')   


        return graph_opt

    def stop_robot(self):
        """
        Stops the robot by setting wheel speeds to zero.
        """
        # Create wheel speed command message with zero speeds
        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = 0.0  # Right wheel speed
        wheel_speed_msg.vector.y = 0.0  # Left wheel speed

        # Store commands for logging
        self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        self.cmd_wheelrate_left = wheel_speed_msg.vector.y

        # Send stop command to robot
        self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")


    def evaluate_graphslam_performance(self, initial_pose, optimised_pose, ground_truth_pose):
        print("Before Optimisation Pose:", initial_pose)
        print("Optimised Pose:", optimised_pose)
        print("Actual Pose:", ground_truth_pose)
        
        # Calculate error metrics
        position_error = self.calculate_position_error(optimised_pose, ground_truth_pose)
        
        print(f"Position Error: {position_error:.4f} meters")
        
        return position_error

    def calculate_position_error(self, pose1, pose2):
        return np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])

    def get_closest_landmark_id(self, corner_position):
        corner_x = float(corner_position[0])
        corner_y = float(corner_position[1])
        
        distances = [
            np.sqrt((corner_x - x)**2 + (corner_y - y)**2) 
            for x, y in self.landmark_locations
        ]
        
        closest_id = np.argmin(distances)
        min_distance = distances[closest_id]
        
        # Use distance threshold to validate landmarks
        distance_threshold = 0.5 
        valid = min_distance < distance_threshold
        
   
        # if self.observation_counter == 0:
        #     expected_id = closest_id
        # elif closest_id == self.last_landmark_id:
        #     expected_id = closest_id
        # elif self.last_landmark_id == 3:
        #     expected_id = 0
        # else:
        #     expected_id = self.last_landmark_id + 1
        
        print('Closest landmark ID:', closest_id)
        # print('Last landmark ID:', self.last_landmark_id)
        # print('Distance to closest landmark:', min_distance)
        # print('Expected landmark ID:', expected_id)
        # print('Valid:', valid)
        valid = True
        return valid, closest_id

    def infinite_loop(self):
        """
        Main control loop that runs continuously.
        This function:
        1. Gets sensor measurements
        2. Updates state estimate
        3. Computes control commands
        4. Sends commands to robot
        """

        p_ = None
        sigma_ = None
        # ============ SENSING PHASE ============
        # Get the latest position measurement from ArUco markers
        aruco_pose = self.aruco_driver.read()

        if aruco_pose is not None:
            # Parse and store the ArUco measurement
            msg = self.pose_parse(aruco_pose, aruco=True)

            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()

            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2)

            p_gt = Vector(3)
            p_gt[0] = self.measured_pose_northings_m
            p_gt[1] = self.measured_pose_eastings_m
            p_gt[2] = self.measured_pose_yaw_rad 
            self.p_gt_path.append(p_gt)
            # Log the measurement
            self.datalog.log(msg, topic_name="/aruco")
            self.aruco_count += 1



        # ============ INITIALIZATION PHASE ============
        # Wait for first sensor reading before starting control
        if self.initialise_pose == True and aruco_pose is not None:
            print('Initializing robot pose...')

            # Initialize state vector with measured position
            self.state[self.N] = self.measured_pose_northings_m
            self.state[self.E] = self.measured_pose_eastings_m
            self.state[self.G] = self.measured_pose_yaw_rad
         
            print('Initial state:')
            print(self.state)

            # Initialize pose estimates
            self.est_pose_northings_m = self.measured_pose_northings_m
            self.est_pose_eastings_m = self.measured_pose_eastings_m
            self.est_pose_yaw_rad = self.measured_pose_yaw_rad

            # Initialize timing
            self.t_prev = datetime.utcnow().timestamp()
            self.t = 0  # Elapsed time since start

            # Wait briefly before starting
            time.sleep(0.1)

            # Generate trajectory based on starting position
            self.generate_trajectory()

            
            self.gpc_corner = joblib.load('gpc_model.pkl')



            self.initialise_pose = False


        # ============ MAIN CONTROL PHASE ============
        # Only run control if initialized and have wheel speed measurements
        if (self.initialise_pose != True and
            self.measured_wheelrate_right is not None and
            self.measured_wheelrate_left is not None and self.completed_loop == False):

            # -------- Motion Model Update --------
            # Convert wheel speeds to robot velocity
            q = Vector(2)
            q[0] = self.measured_wheelrate_right  # Right wheel speed
            q[1] = self.measured_wheelrate_left   # Left wheel speed

            # Calculate forward and angular velocity
            u = self.ddrive.fwd_kinematics(q)
            p_gt = Vector(3)
            p_gt[0] = self.measured_pose_northings_m
            p_gt[1] = self.measured_pose_eastings_m
            p_gt[2] = self.measured_pose_yaw_rad 
            self.p_gt_path.append(p_gt)

            # Calculate time step
            t_now = datetime.utcnow().timestamp()
            dt = t_now - self.t_prev
            self.t += dt
            self.t_prev = t_now

            if dt != 0:
                # self.state, self.covariance = extended_kalman_filter_predict(
                #     self.state, self.covariance, u, motion_model, self.R, dt
                # )
                p_=copy.copy(self.state)
                sigma_=copy.copy(self.sigma)  
                #print(self.sigma)
                self.state, self.sigma, self.d_p_eb, p_gt =  rigid_body_kinematics(self.state,u,dt=dt,mu_gt=p_gt,sigma_motion=self.sigma_motion,sigma_xy=self.sigma)
                #print(self.sigma)

            # p_=copy.copy(self.state)
            # sigma_=copy.copy(self.sigma)  
            #print(self.sigma)
            # THE ERROR IS HERE AND IDK YWHY
            # sigma = Matrix(3,3) 
            # sigma[0,0]=0.1
            # sigma[0,1]=0.01
            # sigma[1,0]=0.01
            # sigma[1,1]=0.1
            # sigma[0,2]=0.01
            # sigma[1,2]=0.01
            # sigma[2,0]=0.01
            # sigma[2,1]=0.01
            # sigma[2,2]=0.1

            # #Motion model linear noise due to v and w
            # sigma_motion=Matrix(3,2)
            # sigma_motion[0,0]=0.1**2 # impact of v linear velocity on x           #Task
            # sigma_motion[0,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on x
            # sigma_motion[1,0]=0.1**2 # impact of v linear velocity on y
            # sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
            # sigma_motion[2,0]=0.1**2 # impact of v linear velocity on gamma
            # sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma

            # self.state, self.sigma, self.d_p_eb, p_gt =  rigid_body_kinematics(self.state,u,dt=dt,mu_gt=p_gt,sigma_motion=self.sigma_motion,sigma_xy=self.sigma)

            # Extract pose estimates from state
            self.est_pose_northings_m = self.state[self.N, 0]
            self.est_pose_eastings_m = self.state[self.E, 0]
            self.est_pose_yaw_rad = self.state[self.G, 0]

            p_eb = Vector(3); 
            p_eb[0] = self.est_pose_northings_m  # North position
            p_eb[1] = self.est_pose_eastings_m   # East position
            p_eb[2] = self.est_pose_yaw_rad % (2 * np.pi)     # Heading angle


            # Log estimated pose
            est_msg = self.pose_parse([datetime.utcnow().timestamp(), 
                                     self.est_pose_northings_m, 
                                     self.est_pose_eastings_m, 
                                     0, 0, 0, 
                                     self.est_pose_yaw_rad])
            
            self.datalog.log(est_msg, topic_name="/est_pose")


            H_eb = HomogeneousTransformation(p_eb[0:2], p_eb[2])

            # -------- Trajectory Following --------
            if hasattr(self, 'path'):
                # Check progress along path and update waypoint if needed

                # print('Current waypoint ID:', self.path.wp_id)



                self.path.wp_progress(self.t, self.state[:3], self.turning_radius)
                                # Get reference position and velocity at current time
                p_ref, u_ref = self.path.p_u_sample(self.t)
                # print('Completed is:', self.completed)
                # -------- Feedback Control --------
                # Calculate position error in earth frame
                dp = Vector(3)
                dp = p_ref - self.state[:3]

                # Wrap heading error to [-π, π]
                dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi

                # Transform error to robot body frame
                H_eb = HomogeneousTransformation(self.state[:3][0:2], self.state[:3][2])
                ds = Inverse(H_eb.H_R) @ dp

                # Initialize control gains on first iteration
                if self.initialise_control == True:
                    self.k_n = (2 * u_ref[0]) / (self.L**2)  # Cross-track gain
                    self.k_g = u_ref[0] / self.L             # Heading gain
                    self.initialise_control = False

                # Calculate feedback control correction
                du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

                # Combine feedforward and feedback control
                u = u_ref + du

                # Apply velocity limits for safety
                if u[0] > self.v_max: u[0] = self.v_max
                if u[0] < -self.v_max: u[0] = -self.v_max
                if u[1] > self.w_max: u[1] = self.w_max
                if u[1] < -self.w_max: u[1] = -self.w_max

                # Update control gains for next iteration
                self.k_n = (2 * u[0]) / (self.L**2)
                self.k_g = u[0] / self.L


                # -------- Send Commands to Robot --------
                # Convert desired velocity to wheel speeds
                q = self.ddrive.inv_kinematics(u)

                # Create wheel speed command message
                wheel_speed_msg = Vector3Stamped()
                wheel_speed_msg.vector.x = q[0,0]  # Right wheel speed
                wheel_speed_msg.vector.y = q[1,0]  # Left wheel speed

                # Store commands for logging
                self.cmd_wheelrate_right = wheel_speed_msg.vector.x
                self.cmd_wheelrate_left = wheel_speed_msg.vector.y

                # Send commands to robot
                self.wheel_speed_pub.publish(wheel_speed_msg)
                self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")



            t_em = Vector(2)
            t_em[0] = 0
            t_em[1] = 0
            if self.lidar_data is not None:
                observation, _ = lidar_scan(
                    p_eb, self.lidar_data, self.lidar, self.sigma_observe
                )
                if (observation is not None or not np.isnan(observation.data_filled[:, 0]).any()):
                    new_observation = self.GPC_input_output(observation, None)
                    feats = new_observation.data_filled[:, 0].reshape(1, -1)

                    # get both label and probability
                    probs = self.gpc_corner.predict_proba(feats)[0]
                    max_p = np.max(probs)
                    label = self.gpc_corner.classes_[np.argmax(probs)]
                    new_observation.label = label
                    
                    if new_observation.label == "corner" and max_p >= self.prob_thresh:
                        # print(f"Detected corner with probability {max_p:.2f} at {new_observation.data_filled[:, 0]}")
                        r, theta, idx = self.find_corner(new_observation)
                        # print('r, theta, idx', r, theta, idx)
                        if r is not None:
                            x_l, y_l = polar2cartesian(r, theta)
                            H_eb = HomogeneousTransformation(p_eb[0:2], p_eb[2]) 
                            t_em = t2v(H_eb.H@self.lidar.H_bl.H@v2t([x_l, y_l]))
                        
                            self.corner_detected = True
                            self.corner_pose_northings = t_em[0]
                            self.corner_pose_eastings = t_em[1]

                            print(f"#######################\n\n CORNER DETECTED at: [{t_em[0]:.3f}, {t_em[1]:.3f}] \n\n#######################")
                            
                            ts = datetime.utcnow().isoformat()
                            self._corner_writer.writerow([
                                ts,
                                float(t_em[0]),
                                float(t_em[1]),
                                self.landmark_id
                            ])
                            self._corner_log.flush()
                            
                #t_em, flag = self.run_classifier(p_eb)
    
    
                # Get the current pose in the graph
    

            #CONDITION FOR WHEN A LANDMARK IS OBSERVED
            if self.corner_detected == True:
                self.observation_counter += 1
                _, _, t_lm, sigma_xy = self.lidar.loc_to_rangeangle( p_eb, t_em, sigma_observe=self.sigma_observe) 
                   

                if hasattr(t_em, 'shape') and t_em.shape == (2,):
                    t_em = t_em.reshape(2, 1)
                    
                if hasattr(t_lm, 'shape') and t_lm.shape == (2,):
                    t_lm = t_lm.reshape(2, 1)

                # print('t_em', t_em)
                # print('t_lm', t_lm)
                

                valid, id = self.get_closest_landmark_id(t_em)
                if valid:
                    self.landmark_id = id



                current_time = datetime.utcnow().timestamp()

                # if self.last_landmark_timestamp is None or (current_time - self.last_landmark_timestamp) > 3.0:            # Update landmark tracking
                #     self.landmark_visits[self.landmark_id] += 1
                #     print(f'Observation of Landmark ID {self.landmark_id} (Visit #{self.landmark_visits[self.landmark_id]})')
                    
                # Check for loop completion: transitioning from landmark 3 back to 0
                if self.last_landmark_id == 1 and self.landmark_id == 0:
                    self.loop_count += 1
                    print("LOOP", self.loop_count ,"COMPLETED: Detected transition from landmark 3 back to landmark 0")


                if (not np.isnan(t_lm[0]) and not np.isnan(t_lm[1]) and 
                    not np.isnan(t_em[0]) and not np.isnan(t_em[1]) and valid == True):

                    self.graph.observation(t_em, sigma_xy,self.landmark_id, t_lm)   
                    print('t_lm', t_lm) 
                    print( '(',self.observation_counter,')', 'Observation of Landmark ID',self.landmark_id)
                    self.last_landmark_id = self.landmark_id
                    self.last_landmark_timestamp = current_time

                self.corner_detected = False

            else:
                p_=copy.copy(self.state)
                sigma_=copy.copy(self.sigma)
                # adds to the graph as a motion
                self.graph.motion(p_, sigma_, self.d_p_eb, final=False)
                # print('Underwent Motion')


        # should it be completed one lap  and then it optimises then it exits the optimise and does the motion model again?

        if self.loop_count >= 2 and self.optimised == False:
            self.stop_robot()
            # self.loop_count += 1
            p_=copy.copy(self.state)
            sigma_=copy.copy(self.sigma)

            # completes the motion
            self.graph.motion(p_, sigma_, Vector(3), final=True)
            print('Finish graph data association')
            print('*************************************************')
            
            ########### BACKEND ###############
            print('Before graph construction:')
            self.graph.construct_graph()

            graph_init = copy.deepcopy(self.graph)
            graph_opt = graphslam_backend(graph_init)

            print('after backend')

            self.graph_optimisation_solve(graph_opt)


            optimised_graph_pose = graph_opt.reduce2pose()



            em = Vector(2)
            H_em = HomogeneousTransformation(em, 0)
            
            m_e0 = Vector(2)
            m_e0[0] = 0
            m_e0[1] = 0
            
            m_e1 = Vector(2)
            m_e1[0] = 2
            m_e1[1] = 0
            
            m_e2 = Vector(2)
            m_e2[0] = 2
            m_e2[1] = 2
            
            m_e3 = Vector(2)
            m_e3[0] = 0
            m_e3[1] = 2


            

            map_ground_truth = [m_e0, m_e1, m_e2, m_e3]
            map_labels = ['c0', 'c1', 'c2','c3']
            plot_graph(graph_init, self.p_gt_path, H_em, map_ground_truth, map_labels)
            plot_graph(graph_opt, self.p_gt_path, H_em, map_ground_truth, map_labels)
            plot_graph(optimised_graph_pose, self.p_gt_path, H_em, map_ground_truth, map_labels)

            update_pose = next((p for p in reversed(optimised_graph_pose.pose) if np.any(p != 0)), None)
            update_pose = np.array(update_pose).flatten()

            print('Unoptimised state:', self.state)
            print('Optimised Pose:', update_pose)
            print('Actual state:', p_gt)


            self.state[self.N] = update_pose[self.N]
            self.state[self.E] = update_pose[self.E]
            self.state[self.G] = update_pose[self.G]

            update_covariance= next((p for p in reversed(optimised_graph_pose.pose_covariance) if np.any(p != 0)), None)

            self.sigma = update_covariance

            print('After graph construction')
            self.completed_loop = False
            self.optimised = True



            
            self.graph.construct_graph(visualise_flag=True)



if __name__ == "__main__":
    """
    Main entry point of the program.
    Parses command line arguments and starts the robot controller.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add command line arguments
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever."
    )

    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False"
    )

    # Parse arguments
    args = parser.parse_args()
    args.simulation = True
    # Create and run the robot controller
    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
