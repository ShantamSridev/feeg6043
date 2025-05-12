# Copyright (c) 2023 The uos_sess6072_build Authors.
# Authors: Miquel Massot, Blair Thornton, Sam Fenton
# All rights reserved.
# Licensed under the BSD 3-Clause License.
# See LICENSE.md file in the project root for full license information.
# """

import numpy as np
import argparse
from datetime import datetime
import time
import copy
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control
import g2o
import matplotlib.pyplot as plt


class G2OGraphSLAM:
    """
    This class implements Graph SLAM using the g2o-python library.
    
    It optimizes robot poses and landmark positions based on odometry and observation constraints.
    While the tutorial in the PDF uses a custom implementation, this class leverages g2o-python
    which is more efficient for large-scale SLAM problems.
    """
    
    def __init__(self):
        """Initialize the Graph SLAM system with g2o optimizer"""
        # Create a g2o optimizer
        self.optimizer = g2o.SparseOptimizer()
        
        # Set up the solver
        solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)
        
        # Bookkeeping
        self.pose_vertices = {}  # Store pose vertices by ID
        self.landmark_vertices = {}  # Store landmark vertices by ID
        self.next_pose_id = 0
        self.next_landmark_id = 10000  # Start landmark IDs at a high number to distinguish from poses
        self.landmark_id_mapping = {}  # Map external landmark IDs to internal g2o IDs
        
    def add_pose(self, pose, is_fixed=False):
        """
        Add a new robot pose to the graph.
        
        Args:
            pose: Vector(3) containing [x, y, theta]
            is_fixed: Whether to fix this pose (e.g., for the initial pose)
            
        Returns:
            The pose ID in the graph
        """
        # Create a pose vertex (SE2)
        v_pose = g2o.VertexSE2()
        
        # Set the ID and initial estimate
        pose_id = self.next_pose_id
        v_pose.set_id(pose_id)
        
        # Convert pose to g2o SE2 format
        pose_g2o = g2o.SE2(pose[0], pose[1], pose[2])
        v_pose.set_estimate(pose_g2o)
        
        # First pose is usually fixed as reference
        v_pose.set_fixed(is_fixed)
        
        # Add to optimizer and bookkeeping
        self.optimizer.add_vertex(v_pose)
        self.pose_vertices[pose_id] = v_pose
        
        # Increment ID counter
        self.next_pose_id += 1
        
        return pose_id
    
    def add_landmark(self, landmark_pos, external_id=None):
        """
        Add a landmark to the graph.
        
        Args:
            landmark_pos: Vector(2) containing [x, y]
            external_id: Optional external ID for bookkeeping
            
        Returns:
            The landmark ID in the graph
        """
        # Create a landmark vertex (point)
        v_landmark = g2o.VertexPointXY()
        
        # Set the ID and initial estimate
        landmark_id = self.next_landmark_id
        v_landmark.set_id(landmark_id)
        
        # Convert landmark to g2o Point format
        landmark_g2o = g2o.Vector2d(landmark_pos[0], landmark_pos[1])
        v_landmark.set_estimate(landmark_g2o)
        
        # Add to optimizer and bookkeeping
        self.optimizer.add_vertex(v_landmark)
        self.landmark_vertices[landmark_id] = v_landmark
        
        # Map external ID if provided
        if external_id is not None:
            self.landmark_id_mapping[external_id] = landmark_id
        
        # Increment ID counter
        self.next_landmark_id += 1
        
        return landmark_id
    
    def add_pose_pose_constraint(self, pose_id1, pose_id2, relative_pose, information_matrix):
        """
        Add a constraint between two poses (odometry).
        
        Args:
            pose_id1: ID of the first pose
            pose_id2: ID of the second pose
            relative_pose: Vector(3) containing the relative transform [dx, dy, dtheta]
            information_matrix: 3x3 information matrix (inverse of covariance)
        """
        # Create a g2o edge for the constraint
        edge = g2o.EdgeSE2()
        
        # Connect vertices
        edge.set_vertex(0, self.pose_vertices[pose_id1])
        edge.set_vertex(1, self.pose_vertices[pose_id2])
        
        # Set relative transformation measurement
        measurement = g2o.SE2(relative_pose[0], relative_pose[1], relative_pose[2])
        edge.set_measurement(measurement)
        
        # Set information matrix (inverse of covariance)
        edge.set_information(information_matrix)
        
        # Add robust kernel to handle outliers (optional)
        kernel = g2o.RobustKernelHuber()
        edge.set_robust_kernel(kernel)
        
        # Add to optimizer
        self.optimizer.add_edge(edge)
    
    def add_pose_landmark_constraint(self, pose_id, landmark_id, measurement, information_matrix):
        """
        Add a constraint between a pose and a landmark (observation).
        
        Args:
            pose_id: ID of the pose
            landmark_id: ID of the landmark (can be external ID)
            measurement: Vector(2) containing the relative observation [x, y] in robot frame
            information_matrix: 2x2 information matrix (inverse of covariance)
        """
        # Handle external landmark IDs
        if landmark_id in self.landmark_id_mapping:
            internal_landmark_id = self.landmark_id_mapping[landmark_id]
        else:
            internal_landmark_id = landmark_id
        
        # Create a g2o edge for the constraint
        edge = g2o.EdgeSE2PointXY()
        
        # Connect vertices
        edge.set_vertex(0, self.pose_vertices[pose_id])
        edge.set_vertex(1, self.landmark_vertices[internal_landmark_id])
        
        # Set relative measurement
        edge.set_measurement(g2o.Vector2d(measurement[0], measurement[1]))
        
        # Set information matrix (inverse of covariance)
        edge.set_information(information_matrix)
        
        # Add robust kernel to handle outliers (optional)
        kernel = g2o.RobustKernelHuber()
        edge.set_robust_kernel(kernel)
        
        # Add to optimizer
        self.optimizer.add_edge(edge)
    
    def optimize(self, max_iterations=20):
        """
        Run the graph optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Final chi2 error
        """
        print(f"Starting optimization with {len(self.pose_vertices)} poses and {len(self.landmark_vertices)} landmarks")
        
        # Initialize optimizer
        self.optimizer.initialize_optimization()
        
        # Run optimization
        self.optimizer.compute_active_errors()
        initial_chi2 = self.optimizer.chi2()
        print(f"Initial chi2 error: {initial_chi2}")
        
        t_start = datetime.now()
        
        self.optimizer.optimize(max_iterations)
        
        t_end = datetime.now()
        final_chi2 = self.optimizer.chi2()
        
        print(f"Final chi2 error: {final_chi2}")
        print(f"Optimization time: {(t_end - t_start).total_seconds() * 1000:.1f} ms")
        
        return final_chi2
    
    def get_pose_estimate(self, pose_id):
        """Get the optimized pose estimate."""
        v_pose = self.pose_vertices[pose_id]
        pose_g2o = v_pose.estimate()
        
        # Convert to Vector format
        pose = Vector(3)
        pose[0] = pose_g2o.translation().x()
        pose[1] = pose_g2o.translation().y()
        pose[2] = pose_g2o.rotation().angle()
        
        return pose
    
    def get_landmark_estimate(self, landmark_id):
        """Get the optimized landmark estimate."""
        if landmark_id in self.landmark_id_mapping:
            internal_landmark_id = self.landmark_id_mapping[landmark_id]
        else:
            internal_landmark_id = landmark_id
            
        v_landmark = self.landmark_vertices[internal_landmark_id]
        landmark_g2o = v_landmark.estimate()
        
        # Convert to Vector format
        landmark = Vector(2)
        landmark[0] = landmark_g2o.x()
        landmark[1] = landmark_g2o.y()
        
        return landmark
    
    def get_all_poses(self):
        """Get all optimized poses as a dictionary of ID -> pose_vector."""
        poses = {}
        for pose_id, v_pose in self.pose_vertices.items():
            pose_g2o = v_pose.estimate()
            
            pose = Vector(3)
            pose[0] = pose_g2o.translation().x()
            pose[1] = pose_g2o.translation().y()
            pose[2] = pose_g2o.rotation().angle()
            
            poses[pose_id] = pose
            
        return poses
    
    def get_all_landmarks(self):
        """Get all optimized landmarks as a dictionary of ID -> landmark_vector."""
        landmarks = {}
        for landmark_id, v_landmark in self.landmark_vertices.items():
            landmark_g2o = v_landmark.estimate()
            
            landmark = Vector(2)
            landmark[0] = landmark_g2o.x()
            landmark[1] = landmark_g2o.y()
            
            landmarks[landmark_id] = landmark
            
        return landmarks


class RobotSLAM:
    """
    This class adapts the LaptopPilot control system to perform SLAM with g2o.
    It uses motion estimates from wheel encoders and landmark observations
    to build and optimize a pose graph over time.
    """
    
    def __init__(self, laptop_pilot):
        """
        Initialize the SLAM system for the robot.
        
        Args:
            laptop_pilot: Reference to the LaptopPilot instance
        """
        self.robot = laptop_pilot
        self.slam = G2OGraphSLAM()
        
        # Initialize with robot's current pose
        self.current_pose_id = self.slam.add_pose(
            Vector([
                self.robot.est_pose_northings_m,
                self.robot.est_pose_eastings_m,
                self.robot.est_pose_yaw_rad
            ]), 
            is_fixed=True  # Fix first pose as reference
        )
        
        # Last update time for odometry
        self.last_update_time = datetime.utcnow().timestamp()
        
        # Flag for loop closure detection
        self.loop_closure_enabled = True
        self.optimize_interval = 10  # Optimize every N updates
        self.update_count = 0
        
        # Keep a history of poses for loop closure
        self.pose_history = []
        
        # Landmark tracking
        self.observed_landmarks = {}  # landmark_id -> position
    
    def update(self):
        """
        Main update function to be called periodically.
        This handles odometry updates and processes any landmark observations.
        """
        current_time = datetime.utcnow().timestamp()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if dt < 0.01:
            return  # Skip if time delta is too small
        
        # Check if robot's state is initialized
        if (self.robot.est_pose_northings_m is None or
            self.robot.est_pose_eastings_m is None or
            self.robot.est_pose_yaw_rad is None):
            return
        
        # Get current robot pose estimate
        current_pose = Vector([
            self.robot.est_pose_northings_m,
            self.robot.est_pose_eastings_m,
            self.robot.est_pose_yaw_rad
        ])
        
        # Only add a new pose if the robot has moved enough
        prev_pose = self.slam.get_pose_estimate(self.current_pose_id)
        
        # Calculate distance and angle difference
        dx = current_pose[0] - prev_pose[0]
        dy = current_pose[1] - prev_pose[1]
        dist = np.sqrt(dx*dx + dy*dy)
        angle_diff = abs((current_pose[2] - prev_pose[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # Add a new pose if moved enough
        if dist > 0.1 or angle_diff > np.deg2rad(5):
            # Create a new pose vertex
            new_pose_id = self.slam.add_pose(current_pose)
            
            # Create a relative pose constraint (odometry)
            relative_pose = Vector(3)
            
            # Calculate relative transform
            H_prev = HomogeneousTransformation(prev_pose[0:2], prev_pose[2])
            H_curr = HomogeneousTransformation(current_pose[0:2], current_pose[2])
            H_rel = Inverse(H_prev.H) @ H_curr.H
            
            # Extract relative pose
            relative_pose[0] = H_rel[0, 2]  # x
            relative_pose[1] = H_rel[1, 2]  # y
            
            # Calculate angle from rotation matrix
            relative_pose[2] = np.arctan2(H_rel[1, 0], H_rel[0, 0])
            
            # Create information matrix (inverse of covariance)
            # Higher values = more certain
            information_matrix = np.eye(3)
            information_matrix[0, 0] = 100.0  # x
            information_matrix[1, 1] = 100.0  # y
            information_matrix[2, 2] = 150.0  # theta
            
            # Add the constraint to the graph
            self.slam.add_pose_pose_constraint(
                self.current_pose_id, 
                new_pose_id, 
                relative_pose, 
                information_matrix
            )
            
            # Update current pose ID
            self.current_pose_id = new_pose_id
            
            # Add to pose history for loop closure
            self.pose_history.append(new_pose_id)
            
            # Process any visible landmarks
            self.process_landmarks()
            
            # Attempt loop closure
            if self.loop_closure_enabled and len(self.pose_history) > 10:
                self.attempt_loop_closure()
            
            # Optimize the graph periodically
            self.update_count += 1
            if self.update_count % self.optimize_interval == 0:
                self.slam.optimize()
                self.update_robot_pose()
    
    def process_landmarks(self):
        """Process any landmarks visible from the current robot pose."""
        # Skip if no lidar data available
        if self.robot.lidar_data is None:
            return
        
        # Process each lidar point that seems to represent a landmark
        for point_idx in range(len(self.robot.lidar_data)):
            # Get point coordinates in map frame
            point_x = self.robot.lidar_data[point_idx, 0]  # North
            point_y = self.robot.lidar_data[point_idx, 1]  # East
            
            # Simple landmark detection - this should be replaced with actual feature extraction
            # For now, just consider points as potential landmarks
            
            # Here we would typically run a landmark detection algorithm
            # For this example, we'll do a very simple filtering
            
            # Skip if too close to robot or too far
            current_pose = self.slam.get_pose_estimate(self.current_pose_id)
            dx = point_x - current_pose[0]
            dy = point_y - current_pose[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 0.5 or dist > 5.0:
                continue
            
            # Try to associate with existing landmarks
            landmark_id = self.associate_landmark(point_x, point_y)
            
            if landmark_id is None:
                # New landmark
                landmark_pos = Vector([point_x, point_y])
                landmark_id = self.slam.add_landmark(landmark_pos)
                self.observed_landmarks[landmark_id] = landmark_pos
            
            # Calculate measurement in robot frame
            H_pose = HomogeneousTransformation(current_pose[0:2], current_pose[2])
            point_global = np.array([point_x, point_y, 1]).reshape(3, 1)
            point_local = Inverse(H_pose.H) @ point_global
            
            measurement = Vector([point_local[0, 0], point_local[1, 0]])
            
            # Information matrix for landmark observation
            # Higher values = more certain
            information_matrix = np.eye(2)
            information_matrix[0, 0] = 100.0  # x
            information_matrix[1, 1] = 100.0  # y
            
            # Add the constraint to the graph
            self.slam.add_pose_landmark_constraint(
                self.current_pose_id,
                landmark_id,
                measurement,
                information_matrix
            )
    
    def associate_landmark(self, x, y, threshold=0.5):
        """
        Associate a detected point with an existing landmark.
        
        Args:
            x, y: Point coordinates
            threshold: Maximum distance for association
            
        Returns:
            Landmark ID if found, None otherwise
        """
        for landmark_id, landmark_pos in self.observed_landmarks.items():
            dx = x - landmark_pos[0]
            dy = y - landmark_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < threshold:
                return landmark_id
                
        return None
    
    def attempt_loop_closure(self):
        """Attempt to find and add loop closure constraints."""
        # Get current pose
        current_pose = self.slam.get_pose_estimate(self.current_pose_id)
        
        # Check against older poses (skip recent ones)
        for pose_id in self.pose_history[:-10]:
            old_pose = self.slam.get_pose_estimate(pose_id)
            
            # Calculate distance between poses
            dx = current_pose[0] - old_pose[0]
            dy = current_pose[1] - old_pose[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # If poses are close enough, add a loop closure constraint
            if dist < 0.8:
                print(f"Loop closure detected between poses {pose_id} and {self.current_pose_id}")
                
                # Calculate relative transform
                H_old = HomogeneousTransformation(old_pose[0:2], old_pose[2])
                H_curr = HomogeneousTransformation(current_pose[0:2], current_pose[2])
                H_rel = Inverse(H_old.H) @ H_curr.H
                
                # Extract relative pose
                relative_pose = Vector(3)
                relative_pose[0] = H_rel[0, 2]  # x
                relative_pose[1] = H_rel[1, 2]  # y
                relative_pose[2] = np.arctan2(H_rel[1, 0], H_rel[0, 0])
                
                # Loop closure information matrix - a bit less certain than odometry
                information_matrix = np.eye(3)
                information_matrix[0, 0] = 80.0  # x
                information_matrix[1, 1] = 80.0  # y
                information_matrix[2, 2] = 100.0  # theta
                
                # Add the constraint to the graph
                self.slam.add_pose_pose_constraint(
                    pose_id, 
                    self.current_pose_id, 
                    relative_pose, 
                    information_matrix
                )
                
                # Run optimization after loop closure
                self.slam.optimize()
                self.update_robot_pose()
                
                # Only use the first loop closure found
                break
    
    def update_robot_pose(self):
        """Update the robot's pose estimate from the SLAM system."""
        optimized_pose = self.slam.get_pose_estimate(self.current_pose_id)
        
        # Update the robot's estimated pose
        self.robot.est_pose_northings_m = optimized_pose[0]
        self.robot.est_pose_eastings_m = optimized_pose[1]
        self.robot.est_pose_yaw_rad = optimized_pose[2]


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
        wheel_distance = 0.165/2  # Distance between left and right wheels in meters
        wheel_diameter = 0.070  # Diameter of each wheel in meters

        # Create differential drive configuration object
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)


        # ============ TRAJECTORY PARAMETERS ============
        # These control how the robot moves along its path
        self.velocity = 0.08              # Desired forward velocity in m/s
        self.acceleration = self.velocity/3 # How quickly to reach desired velocity
        self.turning_radius = 0.3         # Minimum turning radius in meters


        # ============ CONTROL PARAMETERS ============
        # These values control how aggressively the robot corrects errors
        self.tau_s = 0.5  # Time constant for removing along-track error (seconds)
        self.L = 0.3      # Distance constant for removing cross-track and angular error (meters)

        # Control gains - these determine how strongly the robot responds to errors
        self.k_s = 1/self.tau_s  # Along-track gain (how strongly to correct forward/backward error)

        # Velocity and turning rate limits for safety
        self.v_max = 0.08          # Maximum forward/backward speed in m/s
        self.w_max = np.deg2rad(15)  # Maximum turning rate in rad/s (15 degrees/s)

        # Initialization flags
        self.initialise_control = True  # Will be set to False after first control update
        self.initialise_pose = True     # Will be set to False after receiving first position


        # ============ PATH WAYPOINTS ============
        # Define the path the robot should follow as a series of points
        # Each point has a northing (y) and easting (x) coordinate
        self.northings_path = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        self.eastings_path = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
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
        lidar_xb = 0.07  # 7cm forward of robot center
        lidar_yb = 0.0   # Centered left-right
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)


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
        self.state = Vector(5)


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
        
        # ============ SLAM SYSTEM ============
        # The SLAM system will be initialized in run() method
        self.slam = None  # Will be set up in run() method


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
        """
        Parser converts pose data to a standard format for logging.
        
        Args:
            msg: Raw pose data
            aruco: Whether this is from ArUco markers
            
        Returns:
            PoseStamped message
        """
        time_stamp = msg[0]

        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.utcnow().timestamp()-msg[0]
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse time
            print(
                "Received position update from",
                datetime.now().timestamp() - msg[0] - self.sim_time_offset,
                "seconds ago",
            )
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
    

def run(self, time_to_run=-1):
    """
    Main execution loop of the robot controller.

    Args:
        time_to_run: How long to run in seconds (-1 for infinite)
    """
    self.start_time = datetime.utcnow().timestamp()

    # Initialize the SLAM system
    self.slam = RobotSLAM(self)
    print("SLAM system initialized")

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

            # Update SLAM if initialized
            if hasattr(self, 'slam') and self.slam is not None:
                self.slam.update()

            # Sleep to maintain 10 Hz rate
            r.sleep()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping…")
    except Exception as e:
        print("Exception: ", e)
    finally:
        # Clean up subscribers
        self.lidar_sub.stop()
        self.groundtruth_sub.stop()
        self.true_wheel_speed_sub.stop()
        
        # If SLAM was used, print final results
        if hasattr(self, 'slam') and self.slam is not None:
            print("Final SLAM optimization...")
            self.slam.slam.optimize(max_iterations=50)
            
            # Get optimized poses and landmarks
            poses = self.slam.slam.get_all_poses()
            landmarks = self.slam.slam.get_all_landmarks()
            
            print(f"SLAM completed with {len(poses)} poses and {len(landmarks)} landmarks")
            
            # Save map visualization if matplotlib is available
            try:
                # Plot the final map
                plt.figure(figsize=(10, 8))
                
                # Plot poses
                pose_x = [poses[pose_id][1] for pose_id in sorted(poses.keys())]
                pose_y = [poses[pose_id][0] for pose_id in sorted(poses.keys())]
                plt.plot(pose_x, pose_y, 'b-', label='Robot Path')
                plt.plot(pose_x, pose_y, 'r.', markersize=3)
                
                # Plot landmarks
                landmark_x = [landmarks[landmark_id][1] for landmark_id in landmarks.keys()]
                landmark_y = [landmarks[landmark_id][0] for landmark_id in landmarks.keys()]
                plt.scatter(landmark_x, landmark_y, c='g', marker='x', label='Landmarks')
                
                plt.axis('equal')
                plt.grid(True)
                plt.legend()
                plt.title('Graph SLAM Map')
                plt.xlabel('East (m)')
                plt.ylabel('North (m)')
                
                # Save the figure
                plt.savefig('slam_map.png')
                print("Map saved as slam_map.png")
            except Exception as e:
                print(f"Could not save map visualization: {e}")


def visualize_slam_map(self):
    """
    Visualize the current SLAM map with robot trajectory and landmarks.
    This can be called separately to generate a visualization during operation.
    """
    if not hasattr(self, 'slam') or self.slam is None:
        print("SLAM system not initialized")
        return
    
    # Get current poses and landmarks
    poses = self.slam.slam.get_all_poses()
    landmarks = self.slam.slam.get_all_landmarks()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot poses
    pose_x = [poses[pose_id][1] for pose_id in sorted(poses.keys())]
    pose_y = [poses[pose_id][0] for pose_id in sorted(poses.keys())]
    plt.plot(pose_x, pose_y, 'b-', label='Robot Path')
    plt.plot(pose_x, pose_y, 'r.', markersize=3)
    
    # Plot landmarks
    landmark_x = [landmarks[landmark_id][1] for landmark_id in landmarks.keys()]
    landmark_y = [landmarks[landmark_id][0] for landmark_id in landmarks.keys()]
    plt.scatter(landmark_x, landmark_y, c='g', marker='x', label='Landmarks')
    
    # Plot current lidar scan if available
    if self.lidar_data is not None:
        plt.scatter(self.lidar_data[:, 1], self.lidar_data[:, 0], 
                   c='lightblue', s=2, alpha=0.5, label='Current Scan')
    
    # Mark current position
    if self.est_pose_northings_m is not None and self.est_pose_eastings_m is not None:
        plt.plot(self.est_pose_eastings_m, self.est_pose_northings_m, 'mo', 
                markersize=8, label='Current Position')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Graph SLAM Map')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    
    # Save the figure
    plt.savefig('current_slam_map.png')
    print("Current map saved as current_slam_map.png")


def infinite_loop(self):
    """
    Main control loop that runs continuously.
    This function:
    1. Gets sensor measurements
    2. Updates state estimate
    3. Computes control commands
    4. Sends commands to robot
    """

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

        # Wrap angle to [0, 2π] range
        self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2)

        # Log the measurement
        self.datalog.log(msg, topic_name="/aruco")
        self.aruco_count += 1

        # Update pose estimate with ArUco measurement (sensor fusion)
        if not self.initialise_pose and hasattr(self, 'state'):
            # Simple sensor fusion - update state with ArUco measurement
            # In a full implementation, this would use a proper filter (e.g., EKF)
            # Here we use a simple weighted average
            alpha = 0.7  # Weight for ArUco measurement (0.7 = 70% ArUco, 30% motion model)
            
            self.state[self.N] = alpha * self.measured_pose_northings_m + (1-alpha) * self.state[self.N]
            self.state[self.E] = alpha * self.measured_pose_eastings_m + (1-alpha) * self.state[self.E]
            
            # For heading, we need to handle the circular nature of angles
            # Get the difference between measured and estimated heading
            angle_diff = self.measured_pose_yaw_rad - self.state[self.G]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
            
            # Apply weighted update to heading
            self.state[self.G] = (self.state[self.G] + alpha * angle_diff) % (2 * np.pi)
            
            # Update estimated pose
            self.est_pose_northings_m = self.state[self.N, 0]
            self.est_pose_eastings_m = self.state[self.E, 0]
            self.est_pose_yaw_rad = self.state[self.G, 0]


    # ============ INITIALIZATION PHASE ============
    # Wait for first sensor reading before starting control
    if self.initialise_pose == True and aruco_pose is not None:
        print('Initializing robot pose...')

        # Initialize state vector with measured position
        self.state[self.N] = self.measured_pose_northings_m
        self.state[self.E] = self.measured_pose_eastings_m
        self.state[self.G] = self.measured_pose_yaw_rad
        self.state[self.DOTX] = 0.0  # Start with zero velocity
        self.state[self.DOTG] = 0.0  # Start with zero angular velocity

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

        # Mark initialization as complete
        self.initialise_pose = False


    # ============ MAIN CONTROL PHASE ============
    # Only run control if initialized and have wheel speed measurements
    if (self.initialise_pose != True and
        self.measured_wheelrate_right is not None and
        self.measured_wheelrate_left is not None):

        self.loop_count += 1

        # -------- Motion Model Update --------
        # Convert wheel speeds to robot velocity
        q = Vector(2)
        q[0] = self.measured_wheelrate_right  # Right wheel speed
        q[1] = self.measured_wheelrate_left   # Left wheel speed

        # Calculate forward and angular velocity
        u = self.ddrive.fwd_kinematics(q)

        # Calculate time step
        t_now = datetime.utcnow().timestamp()
        dt = t_now - self.t_prev
        self.t += dt
        self.t_prev = t_now

        # Update state estimate using motion model only (no ArUco correction)
        self.state = self.motion_model(self.state, u, dt)

        # Extract pose estimates from state
        self.est_pose_northings_m = self.state[self.N, 0]
        self.est_pose_eastings_m = self.state[self.E, 0]
        self.est_pose_yaw_rad = self.state[self.G, 0]

        # Log estimated pose
        est_msg = self.pose_parse([datetime.utcnow().timestamp(), 
                                 self.est_pose_northings_m, 
                                 self.est_pose_eastings_m, 
                                 0, 0, 0, 
                                 self.est_pose_yaw_rad])
        self.datalog.log(est_msg, topic_name="/est_pose")

        # -------- Trajectory Following --------
        if hasattr(self, 'path'):
            # Check progress along path and update waypoint if needed
            self.path.wp_progress(self.t, self.state[:3], self.turning_radius)

            # Get reference position and velocity at current time
            p_ref, u_ref = self.path.p_u_sample(self.t)

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
            
            
            # -------- Periodically visualize SLAM map (every 100 loops) --------
            if self.loop_count % 100 == 0 and hasattr(self, 'slam') and self.slam is not None:
                self.visualize_slam_map()


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
    
    parser.add_argument(
        "--disable_slam",
        action="store_true",
        help="Disable SLAM functionality. Defaults to False"
    )
    
    parser.add_argument(
        "--disable_loop_closure",
        action="store_true",
        help="Disable loop closure detection in SLAM. Defaults to False"
    )
    
    parser.add_argument(
        "--optimize_interval",
        type=int,
        default=10,
        help="Number of SLAM updates between graph optimizations"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create and run the robot controller
    laptop_pilot = LaptopPilot(args.simulation)
    
    # Run the robot controller
    laptop_pilot.run(args.time)