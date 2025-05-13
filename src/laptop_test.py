"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Miquel Massot, Blair Thornton, Sam Fenton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
from datetime import datetime
import time
import copy
import json
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from drivers.aruco_udp_driver import ArUcoUDPDriver

from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

from math_feeg6043 import Vector, Matrix, Identity, l2m, Inverse, HomogeneousTransformation
from model_feeg6043 import rigid_body_kinematics, feedback_control, extended_kalman_filter_predict, extended_kalman_filter_update
from model_feeg6043 import ActuatorConfiguration, RangeAngleKinematics, TrajectoryGenerate

from model_feeg6043 import graphslam_frontend, graphslam_backend
from math_feeg6043 import wrapped_mean, wrapped_std,v2t,t2v

from plot_feeg6043 import plot_graph, show_observation

# State indexing labels for convience
N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4

def optimize_graph(graph_opt, initial_residual=100, residual_threshold=1E-12, delta_threshold=1/1000, lim_iterations=20, visualise_flag=False):
    """
    Optimizes a pose graph using the provided graph optimization object.

    Parameters:
    - graph_opt: The graph optimization object to be optimized.
    - initial_residual: Initial residual value to avoid triggering convergence on large residuals (default is 100).
    - residual_threshold: Threshold for residual to stop the optimization (default is 1E-12).
    - delta_threshold: Threshold for delta residual to stop the optimization (default is 1/1000).
    - lim_iterations: Maximum number of iterations (default is 20).
    - visualise_flag: Whether to visualize the optimization progress (default is False).

    Returns:
    - graph_opt: Optimized graph object.
    """

    # Initialize variables
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


def motion_model(state, u, dt):

    """
    A function to simulate robot dynamics using rigidbody kinematics.

    Args:
        state (array): current state vector.
        u (array): input vector.
        dt (float): time step

    Returns:
        state (array): new state vector
        F (matrix): associated jacobian matrix for the new state
    """

    # Get current state infomation
    N_k_1 = state[N]
    E_k_1 = state[E]
    G_k_1 = state[G]
    DOTX_k_1 = state[DOTX]
    DOTG_k_1 = state[DOTG]

    # Create initial pose variable 
    p = Vector(3)
    p[0] = N_k_1
    p[1] = E_k_1
    p[2] = G_k_1

    p_gt = p
    

    sigma = Matrix(3,3) 
    sigma[0,0]=0.1
    sigma[0,1]=0.01
    sigma[1,0]=0.01
    sigma[1,1]=0.1
    sigma[0,2]=0.01
    sigma[1,2]=0.01
    sigma[2,0]=0.01
    sigma[2,1]=0.01
    sigma[2,2]=0.1

    #Motion model linear noise due to v and w
    sigma_motion=Matrix(3,2)
    sigma_motion[0,0]=0.1**2 # impact of v linear velocity on x           #Task
    sigma_motion[0,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on x
    sigma_motion[1,0]=0.1**2 # impact of v linear velocity on y
    sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
    sigma_motion[2,0]=0.1**2 # impact of v linear velocity on gamma
    sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma

    # update pose infomation with current pose, current input and the time step
    p, sigma, dp, p_gt =  rigid_body_kinematics(p,u,dt=dt,mu_gt=p_gt,sigma_motion=sigma_motion,sigma_xy=sigma)

    # vertically join pose and input to create new state info
    state = np.vstack((p, u))
    
    # populate new state variables
    N_k = state[N]
    E_k = state[E]
    G_k = state[G]
    DOTX_k = state[DOTX]
    DOTG_k = state[DOTG]
    
    # ##### Compute its jacobian #####

    F = Identity(5)    # init jacobian matrix
    
    # Formulate simple jacobian when DOTG_k is very small ~0 (avoid numerical instability)
    if abs(DOTG_k) <1E-2: 
        F[N, G] = -DOTG_k * dt * np.sin(G_k_1)
        F[N, DOTX] = dt * np.cos(G_k_1)
        F[E, G] = DOTX_k * dt * np.cos(G_k_1)
        F[E, DOTX] = dt * np.sin(G_k_1)
        F[G, DOTG] = dt  

    # else calculates full jacobian   
    else:
        F[N, G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
        F[N, DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[N, DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
        F[E, G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[E, DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
        F[E, DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
        F[G, DOTG] = dt

    #Return state infomation and jacobian matrix
    return state, F
    

def pose_observation(x):

    """
    Formulates an arteficial state observation for ekf updates

    Args:
        x (array): state vector.

    Returns:
        z (array): state observation vector
        H (matrix): state uncertainty matrix
    """

    # Create state observation (assignment means perfect knowledge)
    z = Vector(5)
    z[N] = x[N]
    z[E] = x[E]
    z[G] = x[G]

    # Create the associated observation uncertainty matrix (identity means perfect certainty)
    H = Matrix(5,5)
    H[N,N] = 1
    H[E,E] = 1
    H[G,G] = 1

    # Return both
    return z, H                 


def calculate_curvature(points, k=5):
    """
    Calculate the curvature for each point based on its local neighborhood
    using PCA (principal component analysis).
    
    Parameters:
    - points: ndarray (n_points, 2) - LIDAR data points.
    - k: int - number of nearest neighbors to consider for curvature estimation.
    
    Returns:
    - curvature: ndarray (n_points,) - Curvature values for each point.
    """
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Initialize an array to store curvature values
    curvature = np.zeros(points.shape[0])
    
    for i, idx in enumerate(indices):
        # Get the local neighborhood of the point
        neighborhood = points[idx]
        
        # Perform PCA on the neighborhood
        pca = PCA(n_components=2)
        pca.fit(neighborhood)
        
        # The curvature is related to the smallest eigenvalue of the covariance matrix
        # The smaller the eigenvalue, the higher the curvature (indicating a bend or corner)
        curvature[i] = pca.explained_variance_ratio_[1]  # Smallest eigenvalue (curvature)
    
    return curvature


def is_corner(lidar_scan, threshold):
    curvature = calculate_curvature(lidar_scan, k=5)
    curvature = np.nan_to_num(curvature, nan=0.0)

    max_idx = np.argmax(curvature)
    max_curv = curvature[max_idx]
    max_coord = lidar_scan[max_idx, :2] 
    
    if max_curv > threshold:
        return np.array(max_coord)
    else:
        return None

class LaptopPilot:

    """
    Handles the complete sensing/actuation of the robot as well as the runtime loop  we must program within.

    Attributes:
        self (LaptopPilot): this
        simulation (Simulation?): variable associated with the connection to the webots simulation enviroment
    """

    def __init__(self, simulation):

        ##### Network Setup for Sensed Pose #####

        # Parameters for ArUco marker detection via UDP
        aruco_params = {
            "port": 50000,  # Port to listen on (DO NOT CHANGE)
            "marker_id": 24,  # Marker ID to track (CHANGE THIS to match your marker)
        }

        self.robot_ip = "192.168.90.1"  # Default IP address for physical robot

        # Simulation time handling parameters
        self.sim_time_offset = 0  # Offset to correct Webots timestamps (seconds)
        self.sim_init = False  # Flag to track if simulation time has been initialized
        self.simulation = simulation  # Whether we are running in simulation mode

        if self.simulation:
            self.robot_ip = "127.0.0.1"  # Override IP address for local simulation
            aruco_params['marker_id'] = 0  # Override marker ID for simulation
            self.sim_init = True  # Initialize simulation time offset

        print("Connecting to robot with IP", self.robot_ip)

        # Initialize ArUco UDP driver for receiving pose data
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)


        ############# INITIALISE ATTRIBUTES ##########

        ##### Robot Path Parameters #####

        # Waypoint northing (Y) coordinates (meters)
        self.northings_path = [0, 1.2, 1.2, 0, 0,1.2, 1.2, 0,0]

        # Waypoint easting (X) coordinates (meters)
        self.eastings_path = [0, 0, 1.2, 1.2, 0,0, 1.2, 1.2,0]

        self.relative_path = True  # Use relative coordinates (True) or absolute (False)
        self.velocity = 0.1  # Desired robot velocity (m/s)
        self.acceleration = 0.1  # Desired robot acceleration (m/s²)
        self.arc_radius = 0.2  # Minimum allowed arc radius (m)
        self.acceptance_radius = 0.1  # Radius within which a waypoint is considered "reached" (m)

        ##### Robot Control Parameters #####

        self.tau_s = 1  # Time constant to remove along-track error (s)
        self.L = 0.2  # Lookahead distance to remove lateral and angular error (m)
        self.v_max = 0.1  # Maximum linear speed (m/s)
        self.w_max = np.deg2rad(30)  # Maximum angular speed (rad/s)

        self.kn = 0  # Normal error gain
        self.kg = 0  # Heading (angular) error gain
        self.ks = 0  # Along-track error gain

        self.initialise_control = True  # True during control initialization; False once gains are set

        ##### EKF (Extended Kalman Filter) Variables #####

        # Measurement noise covariance matrix
        self.R = Identity(5)
        self.R[N, N] = 0.0**2  # Northing measurement noise
        self.R[E, E] = 0.0**2  # Easting measurement noise
        self.R[G, G] = np.deg2rad(0.0)**2  # Heading measurement noise
        self.R[DOTX, DOTX] = 0.01**2  # Forward velocity measurement noise
        self.R[DOTG, DOTG] = np.deg2rad(0.05)**2  # Angular velocity measurement noise

        self.Q = Identity(5)  # Process noise covariance matrix

        self.groundtruth_updated = False  # Flag indicating if groundtruth has been received

        # Initial EKF state vector
        self.init_state = Vector(5)
        self.init_state[N] = 0.3
        self.init_state[E] = 0.3
        self.init_state[G] = 0
        self.init_state[DOTX] = 0.1
        self.init_state[DOTG] = 0

        # Initial EKF covariance matrix
        self.init_covariance = Identity(5)
        self.init_covariance[N, N] = 0.1**2
        self.init_covariance[E, E] = 0.1**2
        self.init_covariance[G, G] = 0.1**2
        self.init_covariance[DOTX, DOTX] = 0.01**2
        self.init_covariance[DOTG, DOTG] = np.deg2rad(0)**2

        self.state = self.init_state  # Current EKF state
        self.covariance = self.init_covariance  # Current EKF covariance

        ##### Robot Graph - SLAM Parameters #####
        self.graph = graphslam_frontend()
        print("graph created")

        self.planned_landmarks = [[0,0],[0,2],[2,2],[2,0]]

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

        #Motion model linear noise due to v and w
        self.sigma_motion=Matrix(3,2)
        self.sigma_motion[0,0]=0.1**2 # impact of v linear velocity on x           #Task
        self.sigma_motion[0,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on x
        self.sigma_motion[1,0]=0.1**2 # impact of v linear velocity on y
        self.sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
        self.sigma_motion[2,0]=0.1**2 # impact of v linear velocity on gamma
        self.sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma

        # Anchor the first pose using a strong prior
        self.graph.anchor(self.sigma)
        print("Graph anchored successfully.")

        self.sigma_observe = Matrix(2,2)
        self.sigma_observe[0,0] = 0.1**2                          
        self.sigma_observe[0,1] = 0.01 **2                                          
        self.sigma_observe[1,0] = np.deg2rad(5)**2 #5 degree per metre range               
        self.sigma_observe[1,1] = 0.1**2


        sigma_motion=Matrix(3,2)
        sigma_motion[0,0]=0.1**2 # impact of v linear velocity on x           #Task
        sigma_motion[0,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on x
        sigma_motion[1,0]=0.1**2 # impact of v linear velocity on y
        sigma_motion[1,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on y
        sigma_motion[2,0]=0.1**2 # impact of v linear velocity on gamma
        sigma_motion[2,1]=np.deg2rad(0.3)**2 # impact of w angular velocity on gamma

        self.observation_dist = 0.4
        self.n_observations = 2
        self.p_gt_path = []
        self.calibrating_graph = True
        self.constructed_graph = False
        self.update_landmark = False
        self.corner_landmark = None
        self.corner_lm = None
        self.landmark_index = 1
        self.motion_interval = 2
        self.motion_timer = self.motion_interval

        em = Vector(2)
        self.H_em = HomogeneousTransformation(em,0)
        print("graph setup complete")

        self.gpc = joblib.load('gaussian_process_classifier.joblib')

        ##### Robot Modelling Parameters #####
        self.pose_prev = Vector(3)

        self.est_pose_northings_m = 0.3  # Estimated northing position (m)
        self.est_pose_eastings_m = 0.3  # Estimated easting position (m)
        self.est_pose_yaw_rad = 0  # Estimated yaw angle (rad)

        # Differential drive parameters
        wheel_distance = 0.085  # Distance between wheels (m)
        wheel_diameter = 0.073  # Diameter of wheels (m)
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)

        self.initialise_pose = True  # True until pose is initialized

        # Measured robot pose
        self.measured_pose_timestamp_s = None  # Timestamp of measured pose (s)
        self.measured_pose_northings_m = None  # Measured northing (m)
        self.measured_pose_eastings_m = None  # Measured easting (m)
        self.measured_pose_yaw_rad =None  # Measured yaw angle (rad)

        # Wheel speed commands
        self.cmd_wheelrate_right = None  # Commanded right wheel speed (rad/s)
        self.cmd_wheelrate_left = None  # Commanded left wheel speed (rad/s)

        # Measured wheel speeds
        self.measured_wheelrate_right = None  # Actual right wheel speed (rad/s)
        self.measured_wheelrate_left = None  # Actual left wheel speed (rad/s)

        # Lidar sensor setup
        self.lidar_timestamp_s = None  # Timestamp of last lidar scan (s)
        self.lidar_data = None  # Lidar data (range measurements)

        ######################## Observation model ##################
        # locate lidar on robot (keep it simple)
        x_bl = 0; y_bl = 0.09 # Task
        t_bl = Vector(2)
        t_bl[0] = x_bl
        t_bl[1] = y_bl
        self.H_bl = HomogeneousTransformation(t_bl,0)   # Task
        self.lidar = RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 2], scan_fov = np.deg2rad(120))    # Task
        self.laserOutput = None

        #############################################################################

        # Data logger
        self.datalog = DataLogger(log_dir="logs")

        ##### Communication Setup #####

        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )  # Publisher for wheel speed commands

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds", Vector3Stamped, self.true_wheel_speeds_callback, ip=self.robot_ip
        )  # Subscriber for true wheel speeds

        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )  # Subscriber for lidar data

        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )  # Subscriber for groundtruth pose

                            
    def true_wheel_speeds_callback(self, msg):
        """
        Callback function for receiving true wheel speeds.

        Updates the measured right and left wheel rates based on incoming
        sensor data, and logs the received message for later analysis.

        Args:
            msg (Vector3Stamped): Message containing right (x) and left (y) wheel speeds.
        """
        #print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)

        # Update measured wheel rates from message
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y

        # Log the wheel speed message
        self.datalog.log(msg, topic_name="/true_wheel_speeds")


    def groundtruth_callback(self, msg):
        """
        Callback function for receiving ground truth odometry from the simulator.

        Logs the received ground truth pose message and sets a flag indicating
        that the ground truth has been updated.

        Args:
            msg (Pose): Incoming ground truth pose message.
        """
        # Log the received ground truth message
        self.datalog.log(msg, topic_name="/groundtruth")

        # Set ground truth update flag
        self.groundtruth_updated = True


    def lidar_callback(self, msg):
        """
        Callback function for processing incoming lidar scan messages.

        Handles timestamp correction for simulation, extracts and transforms lidar data
        from sensor frame to the environment frame (E-frame), and logs the received message.

        Args:
            msg (LaserScan): Incoming lidar scan message containing ranges and angles.
        """
        #print("Received lidar message", msg.header.seq)
        self.laserOutput = msg

        # Apply simulation time correction on first message
        if self.sim_init:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        # Correct message timestamp
        msg.header.stamp += self.sim_time_offset

        # Save lidar measurement timestamp
        self.lidar_timestamp_s = msg.header.stamp

        # Prepare lidar data array (columns: northings, eastings)
        self.lidar_data = np.zeros((len(msg.ranges), 2))

        # Use ranges and angles as initial placeholders
        self.lidar_data[:, 0] = msg.ranges
        self.lidar_data[:, 1] = msg.angles

        self.lidar_data = np.nan_to_num(self.lidar_data, nan=0.0)

        # Robot pose in environment frame (northings, eastings, yaw)
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m  # Estimated northings
        p_eb[1] = self.est_pose_eastings_m   # Estimated eastings
        p_eb[2] = self.est_pose_yaw_rad      # Estimated yaw

        ## Check if there are valid points to predict
        if self.lidar_data.shape[0] > 0:
            scan = self.lidar_data

            input = []
            input.append(scan)
            input = np.array(input)
            input = input.reshape(input.shape[0], -1)

            corner_confidence = self.gpc.predict_proba(input)[0][1]

            if corner_confidence > 0.7:
                scan = np.nan_to_num(scan, nan=0.0)
                curvature = np.nan_to_num(calculate_curvature(scan, k=5), nan=0.0)
                max_coord = np.array(scan[np.argmax(curvature), :2])

                corner = Vector(2)
                corner[0] = max_coord[0]
                corner[1] = max_coord[1]

                self.corner_landmark = corner

                #self.corner_lm = np.array([corner[0],corner[1]]).flatten()
                #print(self.corner_lm)

                #H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
                #corner = t2v(H_eb.H@self.H_bl.H@v2t(corner))

                #self.corner_landmark = np.array([corner[1],corner[0]]).flatten()
                self.update_landmark = True

        else:
            print("No valid lidar data for prediction")

        # Log the raw lidar message
        self.datalog.log(msg, topic_name="/lidar")

        # Recompute lidar data: transform each scan point from lidar frame to environment frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))  # Reset array

        z_lm = Vector(2)  # Measurement vector (range, angle)

        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            # Transform from lidar to environment coordinates
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            self.lidar_data[i, 0] = t_em[0]  # Northing (meters)
            self.lidar_data[i, 1] = t_em[1]  # Easting (meters)

        # Filter out any invalid (NaN) lidar measurements
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

        
    def pose_parse(self, msg, aruco=False):
        """
        Parse incoming pose data into a standardized PoseStamped format for logging.

        Handles timestamp correction if the data comes from an ArUco sensor,
        adjusts for simulation time offset when necessary, and converts the 
        heading angle to a quaternion orientation.

        Args:
            msg (list): Incoming pose data [timestamp, x, y, ..., heading].
            aruco (bool, optional): Flag indicating if the pose is from an ArUco sensor. Defaults to False.

        Returns:
            PoseStamped: Parsed and formatted pose message.
        """
        time_stamp = msg[0]

        if aruco:
            if self.sim_init:
                # Correct simulation time offset on first message
                self.sim_time_offset = datetime.utcnow().timestamp() - msg[0]
                self.sim_init = False

            # Print delay information (useful for debugging time issues)
            #print(
            #    "Received position update from",
            #    datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset,
            #    "seconds ago",
            #)

            # Adjust timestamp for simulation
            time_stamp = msg[0] + self.sim_time_offset

        # Create a PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp

        # Set position fields
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0  # Assume flat ground (z = 0)

        # Create quaternion orientation from heading angle
        quat = Quaternion()
        if not self.simulation and aruco:
            quat.from_euler(0, 0, np.deg2rad(msg[6]))  # Convert degrees to radians
        else:
            quat.from_euler(0, 0, msg[6])  # Already in radians

        pose_msg.pose.orientation = quat

        return pose_msg


    def generate_trajectory(self):
        """
        Generate a trajectory for the robot based on predefined waypoints.

        If the path is relative, offsets the waypoints by the current measured pose.
        Then creates a trajectory instance from the waypoints, and configures
        velocity, acceleration, and turning arcs.

        """
        # Offset waypoints if using relative path
        if self.relative_path:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += 0.3  # Offset northings
                self.eastings_path[i] += 0.3    # Offset eastings

        # Convert path points to a matrix format
        C = l2m([self.northings_path, self.eastings_path])

        # Create a TrajectoryGenerate instance from waypoints
        self.path = TrajectoryGenerate(C[:, 0], C[:, 1])

        # Set trajectory parameters
        self.path.path_to_trajectory(self.velocity, self.acceleration)  # Apply velocity and acceleration
        self.path.turning_arcs(self.arc_radius)                         # Apply turning radius
        self.path.wp_id = 0                                             # Initialize next waypoint index


    def run(self, time_to_run=-1):
        """
        Main control loop for running the robot.

        Repeatedly calls the infinite_loop method at a fixed rate.
        Optionally stops after a specified duration or on keyboard interrupt.

        Args:
            time_to_run (float, optional): Duration (in seconds) to run the loop.
                                            If negative (default), runs indefinitely.
        """
        self.start_time = datetime.utcnow().timestamp()

        try:
            r = Rate(10.0)  # Set loop rate to 10 Hz
            while True:
                current_time = datetime.utcnow().timestamp()

                # Stop the loop if the specified run time is exceeded
                if time_to_run > 0 and (current_time - self.start_time) > time_to_run:
                    print("Time is up, stopping…")
                    break

                # Main loop logic
                self.infinite_loop()
                r.sleep()

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")

        except Exception as e:
            print("Exception:", e)

        finally:
            # Always stop all subscribers when exiting
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()


    def infinite_loop(self):
        """
        Main robot logic loop: Sense → Think → Act cycle.

        - Senses the latest measurements (pose, wheel speeds, lidar).
        - Estimates robot's pose via motion and measurement models.
        - Updates trajectory following and control.
        - Publishes actuator commands.
        """
        # > Sense < #
        # Read latest position measurement
        aruco_pose = self.aruco_driver.read()

        # > Think < #
        if aruco_pose is not None:
            # Parse ArUco pose to PoseStamped message
            msg = self.pose_parse(aruco_pose, aruco=True)

            # Update measured pose values
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()
            self.measured_pose_yaw_rad %= (2 * np.pi)  # Handle angle wrapping
            p_gt = Vector(3)
            p_gt[0] = self.measured_pose_northings_m
            p_gt[1] = self.measured_pose_eastings_m
            p_gt[2] = self.measured_pose_yaw_rad 
            
            # Log the sensed pose
            self.datalog.log(msg, topic_name="/aruco")

            # Initialize estimated pose and generate trajectory if not yet done
            if self.initialise_pose:
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad
                

                self.generate_trajectory()

                # Initialize timing variables
                self.t_prev = datetime.utcnow().timestamp()
                self.t = 0
                time.sleep(0.1)  # Allow system to stabilize

                self.initialise_pose = False

        if not self.initialise_pose:
            # > Motion Model < #
            # Compute robot twist from wheel rates
            q = Vector(2)
            q[0] = self.measured_wheelrate_right
            q[1] = self.measured_wheelrate_left
            u = self.ddrive.fwd_kinematics(q)

            p_gt = Vector(3)
            p_gt[0] = self.measured_pose_northings_m
            p_gt[1] = self.measured_pose_eastings_m
            p_gt[2] = self.measured_pose_yaw_rad 

            # Compute timestep
            t_now = datetime.utcnow().timestamp()
            dt = t_now - self.t_prev
            self.t += dt
            self.t_prev = t_now
   
            # > Measurement Update < #
            if self.groundtruth_updated:
                z = Vector(5)
                z[N] = self.measured_pose_northings_m
                z[E] = self.measured_pose_eastings_m
                z[G] = self.measured_pose_yaw_rad

                # self.state, self.covariance = extended_kalman_filter_update(
                #     self.state, self.covariance, z, pose_observation, self.Q, wrap_index=G
                # )

                self.groundtruth_updated = False

            # > Motion Prediction < #
            if dt != 0:
                # self.state, self.covariance = extended_kalman_filter_predict(
                #     self.state, self.covariance, u, motion_model, self.R, dt
                # )
                p_=copy.copy(self.state)
                sigma_=copy.copy(self.sigma)  
                #print(self.sigma)
                self.state, self.sigma, dp, p_gt =  rigid_body_kinematics(self.state,u,dt=dt,mu_gt=p_gt,sigma_motion=self.sigma_motion,sigma_xy=self.sigma)
                #print(self.sigma)

            # Update estimated pose for visualization
            self.est_pose_northings_m = self.state[N][0]
            self.est_pose_eastings_m = self.state[E][0]
            self.est_pose_yaw_rad = self.state[G][0]

            # Build pose message for logging
            msg = self.pose_parse([
                datetime.utcnow().timestamp(),
                self.est_pose_northings_m,
                self.est_pose_eastings_m,
                0, 0, 0,
                self.est_pose_yaw_rad
            ])

            self.datalog.log(msg, topic_name="/est_pose")

            # > Trajectory Following < #

            p_robot = Vector(3)
            p_robot[0, 0] = self.est_pose_northings_m
            p_robot[1, 0] = self.est_pose_eastings_m
            p_robot[2, 0] = self.est_pose_yaw_rad % (2 * np.pi)
            
            self.p_gt_path.append(p_robot)

            self.path.wp_progress(self.t, p_robot, self.acceptance_radius)
            p_ref, u_ref = self.path.p_u_sample(self.t)

            # Compute pose error in robot frame
            dp = p_ref - p_robot
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi
            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp

            # > Feedback Control < #
            self.ks = 1 / self.tau_s
            if self.initialise_control:
                self.kn = 2 * u_ref[0] / (self.L ** 2)
                self.kg = u_ref[0] / self.L
                self.initialise_control = False

            # Compute corrective control
            du = feedback_control(ds, self.ks, self.kn, self.kg)

            # Combine feedforward and feedback controls
            u = u_ref + du

            # Update control gains
            self.kn = 2 * u[0] / (self.L ** 2)
            self.kg = u[0] / self.L

            # > Actuator Limits < #
            u[1] = np.clip(u[1], -self.w_max, self.w_max)
            u[0] = np.clip(u[0], -self.v_max, self.v_max)

            # > Act < #
            # Convert twist to wheel speeds
            q = self.ddrive.inv_kinematics(u)

            # Publish wheel speeds
            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0, 0]  # Right wheel speed (rad/s)
            wheel_speed_msg.vector.y = q[1, 0]  # Left wheel speed (rad/s)

            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y

            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")

            if  self.update_landmark and self.calibrating_graph:

                self.landmark_index = np.argmin(
                                        np.linalg.norm(
                                            self.planned_landmarks - np.asarray(t2v(H_eb.H @ self.H_bl.H @ v2t(self.corner_landmark))).reshape(1, -1), axis=1
                                        )
                                    )
                
                z_lm, sigma_rtheta, t_lm, sigma_xy = self.lidar.loc_to_rangeangle(p_robot, t2v(H_eb.H @ self.H_bl.H @ v2t(self.corner_landmark)), sigma_observe = self.sigma_observe)                     
                self.graph.observation(t2v(H_eb.H@self.H_bl.H@v2t(t_lm)),sigma_xy,self.landmark_index,t_lm) # Task

                print(f"Observation of Corner {self.landmark_index} at: {t2v(H_eb.H @ self.H_bl.H @ v2t(self.corner_landmark))}")
                self.update_landmark = False
            else: 
                self.motion_timer -= dt

                if self.motion_timer <= 0:
                    delta_pose = self.state - self.pose_prev
                    delta_pose[2] = (delta_pose[2] + np.pi) % (2 * np.pi ) - np.pi
                    self.pose_prev = copy.deepcopy(self.state) 

                    print("Delta Pose: \n", delta_pose)
                    self.graph.motion(p_, sigma_, delta_pose, final=False)

                    self.motion_timer = self.motion_interval
                    print("MOTION")

            if self.graph.observation_id >= self.n_observations and self.constructed_graph == False:

                self.graph.motion(p_, sigma_, Vector(3), final=True)

                print('Finish graph data association')
                print('*************************************************')

                print('Creating graph...')

                self.graph.construct_graph()
                print('Frontend Finished')

                graph_init = copy.deepcopy(self.graph)

                graph_opt = graphslam_backend(graph_init)
                print('Backend Created')

                optimize_graph(graph_opt)
                print('Graph Optimised')

                self.calibrating_graph = False
                self.constructed_graph = True


if __name__ == "__main__":

    """
    Entry point for running the LaptopPilot.

    Parses command-line arguments for:
        --time: Duration to run the experiment.
        --simulation: Flag to enable simulation mode.

    Creates a LaptopPilot instance and starts the main run loop.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Time to run the experiment
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run the experiment (in seconds). If negative, runs indefinitely.",
    )

    # Simulation mode flag
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create and run the LaptopPilot
    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)