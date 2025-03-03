import numpy as np
import argparse
from datetime import datetime
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control, kalman_filter_predict, kalman_filter_update


class LaptopPilot:
    """
    Main class for robot navigation, localization, and control.
    
    This class implements a complete robot navigation system including:
    - Extended Kalman Filter (EKF) for robot localization
    - Trajectory generation and following
    - Sensor data processing (ArUco markers, wheel encoders, LIDAR)
    - Control commands to the differential drive robot
    
    The system uses a combination of wheel odometry and visual marker detection
    for accurate localization, and implements a feedback control loop to follow
    predefined trajectories.
    """
    def __init__(self, simulation):
        """
        Initialize the LaptopPilot with all necessary parameters and components.
        
        Args:
            simulation (bool): Flag indicating if we're running in simulation mode
        """
        # ----- NETWORK SETUP -----
        # ArUco marker parameters for visual localization
        # ArUco markers are visual patterns that can be detected by a camera
        # and used to estimate the robot's position and orientation
        aruco_params = {
            "port": 50000,  # UDP port to listen to (DO NOT CHANGE)
            "marker_id": 24,  # Marker ID to track (CHANGE THIS to your marker ID)            
        }
        # Default IP address for the physical robot
        self.robot_ip = "192.168.90.1"
        
        # ----- SIMULATION HANDLING -----
        # Configure different parameters for simulation vs. real robot
        self.sim_time_offset = 0  # Used to handle timestamp differences in simulation
        self.sim_init = False     # Flag to initialize simulation timestamp handling
        self.simulation = simulation  # Store simulation flag
        if self.simulation:
            # In simulation mode, use localhost IP and special ArUco marker ID
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  # Override ArUco marker ID for simulation
            self.sim_init = True  # Mark simulation timestamp handling for initialization

        print("Connecting to robot with IP", self.robot_ip)
        # Initialize ArUco driver for visual marker detection
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        # ----- ROBOT PHYSICAL PARAMETERS -----       
        # Differential drive robot physical dimensions
        wheel_distance = 0.165  # Distance between wheels in meters
        wheel_diameter = 0.065  # Wheel diameter in meters
        
        # ----- TRAJECTORY PARAMETERS -----
        # Define how fast the robot should move and accelerate
        self.velocity = 0.1      # Target linear velocity in m/s
        self.acceleration = self.velocity/3  # Acceleration rate in m/s²
        
        # Minimum turning radius in meters
        # This constrains how sharply the robot can turn
        self.turning_radius = 0.3  
        
        # ----- CONTROL PARAMETERS -----
        # Gain variables for feedback control
        self.tau_s = 0.5  # Time constant (seconds) to remove along-track error
        self.L = 0.3      # Distance (meters) to remove normal and angular error

        # Compute control gains (initial values)
        self.k_s = 1/self.tau_s  # Along-track gain 

        # Maximum speed limits for safety
        self.v_max = 0.2               # Maximum linear velocity (m/s)
        self.w_max = np.deg2rad(15)    # Maximum angular velocity (rad/s)

        # ----- INITIALIZATION FLAGS -----
        # These flags track whether certain components are initialized
        self.initialise_control = True  # Flag for control gains initialization
        self.initialise_pose = True     # Flag for pose initialization

        # ----- DIFFERENTIAL DRIVE MODEL -----
        # Create the differential drive model using the physical parameters
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) 

        # ----- PATH DEFINITION -----
        # Define a sequence of waypoints for the robot to follow
        # These points define a square path that repeats multiple times
        # [0,0] -> [1,0] -> [1,1] -> [0,1] -> [0,0] and so on
        self.northings_path = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        self.eastings_path = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        
        # If True, waypoints are relative to starting position
        # If False, waypoints are in absolute coordinates
        self.relative_path = True  

        # ----- STATE ESTIMATION VARIABLES -----
        # Variables to store the robot's estimated state
        self.est_pose_northings_m = None  # Estimated north position (m)
        self.est_pose_eastings_m = None   # Estimated east position (m)
        self.est_pose_yaw_rad = None      # Estimated heading angle (rad)

        # ----- MEASUREMENT VARIABLES -----
        # Variables to store raw measurements from sensors
        self.measured_pose_timestamp_s = None   # Timestamp of pose measurement
        self.measured_pose_northings_m = None   # Measured north position (m)
        self.measured_pose_eastings_m = None    # Measured east position (m)
        self.measured_pose_yaw_rad = None       # Measured heading angle (rad)

        # ----- CONTROL COMMAND VARIABLES -----
        # Variables to store wheel speed commands
        self.cmd_wheelrate_right = None  # Command for right wheel (rad/s)
        self.cmd_wheelrate_left = None   # Command for left wheel (rad/s)

        # ----- ENCODER FEEDBACK VARIABLES -----
        # Variables to store actual wheel speeds from encoders
        self.measured_wheelrate_right = None  # Measured right wheel speed (rad/s)
        self.measured_wheelrate_left = None   # Measured left wheel speed (rad/s) 

        # ----- LIDAR SETUP -----
        # Variables for LIDAR data and configuration
        self.lidar_timestamp_s = None  # Timestamp of LIDAR measurement
        self.lidar_data = None         # Processed LIDAR data
        # Position of LIDAR sensor in the robot's body frame
        lidar_xb = 0.07  # LIDAR x-position in body frame (m)
        lidar_yb = 0.0   # LIDAR y-position in body frame (m)
        # Initialize LIDAR model with its position
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)    

        # ----- EXTENDED KALMAN FILTER (EKF) SETUP -----
        # Define state vector indices for easy reference
        self.N = 0      # North position index
        self.E = 1      # East position index
        self.G = 2      # Heading (yaw) angle index
        self.DOTX = 3   # Linear velocity index
        self.DOTG = 4   # Angular velocity index

        # ----- NOISE PARAMETERS -----
        # Standard deviations for process and measurement noise
        # Position noise parameters (north, east)
        self.n_std = [1.0]               # North position std dev (m)
        self.e_std = [1.0]               # East position std dev (m)
        self.g_std = [np.deg2rad(1.0)]   # Heading angle std dev (rad)
        
        # Convert standard deviations to matrix format
        self.G_std = l2m(self.g_std)           # Heading std dev as matrix
        self.NE_std = l2m([self.n_std,self.e_std])  # Position std dev as matrix

        # Standard deviations for process noise (R matrix)
        self.dot_x_R_std = l2m([0.02])               # Velocity noise (m/s)
        self.dot_g_R_std = l2m([np.deg2rad(0.01)])   # Angular velocity noise (rad/s)
        
        # Standard deviations for measurement noise (Q matrix)
        self.NE_Q_std = l2m([[0.1],[0.1]])        # Position measurement noise (m)
        self.g_Q_std = l2m([np.deg2rad(1)])       # Heading measurement noise (rad)

        # ----- EKF STATE AND COVARIANCE INITIALIZATION -----
        # Initialize state vector (5 elements) and covariance matrix (5x5)
        self.state = Vector(5)           # State vector [N, E, G, DOTX, DOTG]
        self.covariance = Identity(5)    # Initial covariance matrix (identity)
        self.R = Identity(5)             # Process noise covariance matrix
        
        # Set initial covariance values (variances on the diagonal)
        # Higher values indicate more initial uncertainty
        self.covariance[self.N, self.N] = self.NE_std[0,0]**2        # North position variance
        self.covariance[self.E, self.E] = self.NE_std[0,1]**2        # East position variance
        self.covariance[self.G, self.G] = self.G_std[0]**2           # Heading variance
        self.covariance[self.DOTX, self.DOTX] = 0.0**2               # Velocity variance (initially 0)
        self.covariance[self.DOTG, self.DOTG] = np.deg2rad(0)**2     # Angular velocity variance (initially 0)
        
        # Set process noise covariance (R matrix)
        self.R[self.N, self.N] = 0.1**2                      # North position process noise
        self.R[self.E, self.E] = 0.1**2                      # East position process noise
        self.R[self.G, self.G] = np.deg2rad(5)**2            # Heading process noise
        self.R[self.DOTX, self.DOTX] = self.dot_x_R_std**2   # Velocity process noise
        self.R[self.DOTG, self.DOTG] = np.deg2rad(self.dot_g_R_std)**2  # Angular velocity process noise
        
        # ----- COUNTERS -----
        # Counters for tracking ArUco and loop iterations
        self.aruco_count = 0   # Count of ArUco marker detections
        self.loop_count = 0    # Count of main loop iterations
        
        # ----- DATA LOGGING -----
        # Initialize data logger for recording measurements and commands
        self.datalog = DataLogger(log_dir="logs")

        # ----- COMMUNICATION SETUP -----
        # Initialize publishers and subscribers for robot communication
        
        # Publisher for wheel speed commands
        # Wheel speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        # Subscriber for actual wheel speeds from encoders
        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds", Vector3Stamped, self.true_wheel_speeds_callback, ip=self.robot_ip,
        )
        
        # Subscriber for LIDAR data
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        
        # Subscriber for ground truth pose (used in simulation)
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )
                    
    def true_wheel_speeds_callback(self, msg):
        """
        Callback function for wheel encoder measurements.
        
        This is triggered whenever new wheel speed data is received from the robot.
        It extracts the wheel speeds and stores them in class variables.
        
        Args:
            msg (Vector3Stamped): Message containing wheel speed data
                - msg.vector.x: Right wheel speed (rad/s)
                - msg.vector.y: Left wheel speed (rad/s)
        """
        # Store the measured wheel speeds from encoders
        self.measured_wheelrate_right = msg.vector.x  # Right wheel speed in rad/s
        self.measured_wheelrate_left = msg.vector.y   # Left wheel speed in rad/s
        
        # Log the wheel speed data for later analysis
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        """
        Callback function for LIDAR sensor data.
        
        This is triggered whenever new LIDAR data is received from the robot.
        It processes the range and angle data to create a map of detected points
        in the world coordinate frame.
        
        Args:
            msg (LaserScan): Message containing LIDAR scan data
                - msg.ranges: Array of distance measurements
                - msg.angles: Array of corresponding angles
        """
        # ----- HANDLE SIMULATION TIMESTAMPS -----
        # In simulation, adjust the timestamp to match real-world time
        if self.sim_init == True:
            # Calculate offset between simulation time and real time
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False     

        # Apply the time offset to the message timestamp
        msg.header.stamp += self.sim_time_offset

        # Store the LIDAR measurement timestamp
        self.lidar_timestamp_s = msg.header.stamp
        
        # ----- PROCESS LIDAR DATA -----
        # Initialize array to store processed LIDAR points (x,y coordinates)
        self.lidar_data = np.zeros((len(msg.ranges), 2))
        
        # Skip processing if we don't have a pose estimate yet
        if self.est_pose_northings_m is None:
            return
            
        # Create vector for robot's current pose in earth frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m  # North position
        p_eb[1] = self.est_pose_eastings_m   # East position
        p_eb[2] = self.est_pose_yaw_rad      # Heading angle
        
        # Initialize array for earth-frame LIDAR points
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                    
        # Create a temporary vector for each measurement
        z_lm = Vector(2)        
        
        # Process each LIDAR measurement
        for i in range(len(msg.ranges)):
            # Store range and angle measurement
            z_lm[0] = msg.ranges[i]   # Distance to detected point
            z_lm[1] = msg.angles[i]   # Angle to detected point
            
            # Transform range-angle measurement to earth-frame coordinates
            # This converts polar coordinates (range, angle) to cartesian coordinates (x, y)
            # while accounting for the robot's current pose
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            # Store the earth-frame coordinates
            self.lidar_data[i, 0] = t_em[0]  # North coordinate
            self.lidar_data[i, 1] = t_em[1]  # East coordinate

        # Filter out any invalid measurements (NaN values)
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        
        # Log the LIDAR data
        self.datalog.log(msg, topic_name="/lidar")

    def groundtruth_callback(self, msg):
        """
        Callback function for ground truth pose data.
        
        This is primarily used in simulation to record the actual robot pose
        for later analysis and comparison with estimated pose.
        
        Args:
            msg (Pose): Message containing ground truth pose data
        """
        # Log the ground truth pose data
        self.datalog.log(msg, topic_name="/groundtruth")

    def pose_parse(self, msg, aruco=False):
        """
        Parse pose data into a standard format.
        
        This converts raw pose data (either from simulation or ArUco markers)
        into a standardized PoseStamped message format for consistent processing.
        
        Args:
            msg (list): Raw pose data [timestamp, x, y, z, roll, pitch, yaw]
            aruco (bool): Flag indicating if this is from ArUco markers
            
        Returns:
            pose_msg (PoseStamped): Standardized pose message
        """
        # Extract timestamp from the message
        time_stamp = msg[0]

        # Handle ArUco marker timestamp adjustments for simulation
        if aruco == True:
            if self.sim_init == True:
                # Initialize simulation time offset
                self.sim_time_offset = datetime.utcnow().timestamp() - msg[0]
                self.sim_init = False                                         
                
            # Apply simulation time offset (will be 0 for real robot)
            time_stamp = msg[0] + self.sim_time_offset                

        # Create standardized pose message
        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        
        # Set position (x,y,z coordinates)
        pose_msg.pose.position.x = msg[1]  # North position
        pose_msg.pose.position.y = msg[2]  # East position
        pose_msg.pose.position.z = 0       # Height (assumed flat ground)

        # Set orientation (as quaternion)
        quat = Quaternion()        
        if self.simulation == False and aruco == True:
            # ArUco marker sends yaw in degrees for real robot
            quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else:
            # Simulation provides yaw in radians
            quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat        
        
        return pose_msg

    def generate_trajectory(self):
        """
        Generate a smooth trajectory from the defined waypoints.
        
        This converts the sequence of waypoints into a continuous trajectory
        with proper velocity profiles, acceleration limits, and turning arcs.
        """
        # ----- ADJUST WAYPOINTS IF RELATIVE -----
        # If waypoints are relative to starting position, adjust them
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                # Offset each waypoint by the current position
                self.northings_path[i] += self.measured_pose_northings_m  # Offset north
                self.eastings_path[i] += self.measured_pose_eastings_m    # Offset east

            # ----- CREATE TRAJECTORY -----
            # Convert waypoint lists to a matrix
            C = l2m([self.northings_path, self.eastings_path])
            
            # Create trajectory generator with the waypoints        
            self.path = TrajectoryGenerate(C[:, 0], C[:, 1])        
            
            # ----- SET TRAJECTORY PARAMETERS -----
            # Configure velocity, acceleration, and turning radius constraints
            self.path.path_to_trajectory(self.velocity, self.acceleration)  # Set velocity and acceleration
            self.path.turning_arcs(self.turning_radius)                     # Set turning radius
            self.path.wp_id = 0                                            # Initialize waypoint index

    def run(self, time_to_run=-1):
        """
        Main execution method for the robot controller.
        
        This method runs the control loop at a fixed rate until either the
        specified time elapses or the program is interrupted.
        
        Args:
            time_to_run (float): Time in seconds to run (-1 for indefinite)
        """
        # Record start time for time limit tracking
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            # Create a rate controller for 10Hz operation
            r = Rate(10.0)  # 10 iterations per second
            
            # Main control loop
            while True:
                # Check if time limit has elapsed
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                    
                # Run one iteration of the control loop
                self.infinite_loop()
                
                # Sleep to maintain the desired rate
                r.sleep()
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("KeyboardInterrupt received, stopping…")
            
        except Exception as e:
            # Handle other exceptions
            print("Exception2: ", e)
            
        finally:
            # Clean up by stopping all subscribers
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()
            
    # NOTE: The following methods are commented out but would normally be part of the EKF implementation
    # These have been replaced by imported functions from model_feeg6043
    
    def extended_kalman_filter_predict(self, mu, Sigma, u, f, R, dt):
        # (1) Project the state forward
        # f is the rigid body motion model
        pred_mu, F = f(mu, u, dt)
        print("123")
        # (2) Project the error forward: 
        pred_Sigma = (F @ Sigma @ F.T) + R
        
        # Return the predicted state and the covariance
        return pred_mu, pred_Sigma

    def extended_kalman_filter_update(self, mu, Sigma, z, h, Q, wrap_index = None):
        
        # Prepare the estimated measurement
        pred_z, H = h(mu)
    
        # (3) Compute the Kalman gain
        K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
        
        # (4) Compute the updated state estimate
        delta_z = z- pred_z        
        if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
        cor_mu = mu + K @ (delta_z)
    
        # (5) Compute the updated state covariance
        cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
        
        # Return the state and the covariance
        return cor_mu, cor_Sigma


    def motion_model(self, state, u, dt):
        """
        Non-linear motion model for the Extended Kalman Filter.
        
        This implements the robot's motion dynamics based on differential drive kinematics.
        It projects the state forward in time given the control inputs and calculates
        the Jacobian matrix for EKF linearization.
        
        Args:
            state (Vector): Current state [N, E, G, DOTX, DOTG]
            u (Vector): Control input [linear_velocity, angular_velocity]
            dt (float): Time step in seconds
            
        Returns:
            state (Vector): Predicted next state
            F (Matrix): Jacobian matrix of the motion model
        """
        # ----- EXTRACT CURRENT STATE COMPONENTS -----
        # Extract each element from the current state vector
        N_k_1 = state[self.N]      # Previous north position
        E_k_1 = state[self.E]      # Previous east position
        G_k_1 = state[self.G]      # Previous heading angle
        DOTX_k_1 = state[self.DOTX]  # Previous linear velocity
        DOTG_k_1 = state[self.DOTG]  # Previous angular velocity

        # ----- PROPAGATE POSITION USING RIGID BODY KINEMATICS -----
        # Create position vector [N, E, G]
        p = Vector(3)
        p[self.N] = N_k_1
        p[self.E] = E_k_1
        p[self.G] = G_k_1
        
        # Apply rigid body motion model to get new position
        # This handles the non-linear motion equations including special cases
        p = rigid_body_kinematics(p, u, dt)    
        
        # ----- COMBINE POSITION AND CONTROL INTO STATE -----
        # Create new state vector with updated position and current control
        # Essentially, the new state uses the propagated position and
        # assumes the control inputs become the new velocities
        state = np.vstack((p, u))
        
        # Extract components of the new state for Jacobian calculation
        N_k = state[self.N]
        E_k = state[self.E]
        G_k = state[self.G]
        DOTX_k = state[self.DOTX]
        DOTG_k = state[self.DOTG]
        
        # ----- CALCULATE JACOBIAN MATRIX -----
        # Initialize Jacobian as identity matrix
        F = Identity(5)    

        # Calculate Jacobian elements based on motion equations
        # Handle special case for (almost) zero angular velocity
        if abs(DOTG_k) < 1E-2:  # Near-zero angular velocity case
            # Simplified Jacobian for straight-line motion
            F[self.N, self.G] = -DOTX_k * dt * np.sin(G_k_1)
            F[self.N, self.DOTX] = dt * np.cos(G_k_1)
            F[self.E, self.G] = DOTX_k * dt * np.cos(G_k_1)
            F[self.E, self.DOTX] = dt * np.sin(G_k_1)
            F[self.G, self.DOTG] = dt     
            
        else:  # Normal case with non-zero angular velocity
            # Full Jacobian for curved motion
            F[self.N, self.G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
            F[self.N, self.DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[self.N, self.DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
            F[self.E, self.G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[self.E, self.DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
            F[self.E, self.DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
            F[self.G, self.DOTG] = dt

        return state, F


    def h_g_update(self, x):
        """
        Measurement model for heading updates.
        
        This defines how the heading angle state relates to measurements.
        It's used in the EKF update step when a heading measurement is available.
        
        Args:
            x (Vector): Current state estimate
            
        Returns:
            z (Vector): Predicted measurement
            H (Matrix): Jacobian matrix of the measurement model
        """
        # Create measurement vector (same size as state)
        z = Vector(5)
        
        # Only the heading component (G) is measured
        z[self.G] = x[self.G]
        
        # Initialize measurement Jacobian
        H = Matrix(5, 5)
        
        # Only heading is observed (1-to-1 relationship)
        H[self.G, self.G] = 1
        
        return z, H

    def h_ne_update(self, x):
        """
        Measurement model for position updates.
        
        This defines how the position state relates to measurements.
        It's used in the EKF update step when position measurements are available.
        
        Args:
            x (Vector): Current state estimate
            
        Returns:
            z (Vector): Predicted measurement
            H (Matrix): Jacobian matrix of the measurement model
        """
        # Create measurement vector (same size as state)
        z = Vector(5)
        
        # Position components (N, E) are measured
        z[self.N] = x[self.N]
        z[self.E] = x[self.E]
        
        # Initialize measurement Jacobian
        H = Matrix(5, 5)
        
        # Position components are directly observed (1-to-1 relationship)
        H[self.N, self.N] = 1
        H[self.E, self.E] = 1
        
        return z, H

    def infinite_loop(self):
        """
        Main control loop that runs continuously.
        
        This method implements the complete sense-plan-act cycle including:
        1. Processing sensor data (ArUco, wheel encoders)
        2. Updating the state estimate using Extended Kalman Filter
        3. Following the trajectory with feedback control
        4. Sending commands to the robot
        """
        # ----- SENSE: GET POSE MEASUREMENTS -----
        # Read the latest ArUco marker detection
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            # Parse the ArUco pose data into standard format
            msg = self.pose_parse(aruco_pose, aruco=True)
            
            # Extract and store the measured pose
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            
            # Handle angle wrapping (keep angle between 0 and 2π)
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2)
            
            # Log the ArUco pose data
            self.datalog.log(msg, topic_name="/aruco")
            self.aruco_count += 1
        
        # ----- INITIALIZATION PHASE -----
        # Wait for the first pose measurement to initialize the state
        if self.initialise_pose == True and aruco_pose is not None:
            # Initialize state with first measurement
            self.state[self.N] = self.measured_pose_northings_m
            self.state[self.E] = self.measured_pose_eastings_m
            self.state[self.G] = self.measured_pose_yaw_rad

            # Initialize estimated pose
            self.est_pose_northings_m = self.measured_pose_northings_m
            self.est_pose_eastings_m = self.measured_pose_eastings_m
            self.est_pose_yaw_rad = self.measured_pose_yaw_rad

            # Initialize time tracking
            self.t_prev = datetime.utcnow().timestamp()  # Initial timestamp
            self.t = 0  # Elapsed time counter
            time.sleep(0.1)  # Brief pause to ensure time difference
            
            # Generate trajectory based on initial pose
            self.generate_trajectory()
            
            # Mark pose as initialized
            self.initialise_pose = False 
        
        # ----- MAIN CONTROL CYCLE -----
        # Only proceed if pose is initialized and wheel speeds are available
        if (self.initialise_pose != True and 
            self.measured_wheelrate_right is not None and 
            self.measured_wheelrate_left is not None):
            
            # Increment loop counter
            self.loop_count += 1
            
            # ----- MOTION MODEL UPDATE (PREDICTION STEP) -----
            # Convert measured wheel speeds to twist (linear and angular velocity)
            q = Vector(2)            
            q[0] = self.measured_wheelrate_right  # Right wheel speed (rad/s)
            q[1] = self.measured_wheelrate_left   # Left wheel speed (rad/s)
            
            # Apply forward kinematics to get robot velocity
            u = self.ddrive.fwd_kinematics(q) 
            
            # Calculate time step since last iteration
            t_now = datetime.utcnow().timestamp()        
            dt = t_now - self.t_prev  # Time step (s)
            self.t += dt              # Update elapsed time
            self.t_prev = t_now       # Store current time for next iteration
            
            # ----- EXTENDED KALMAN FILTER PREDICTION -----
            # Predict next state and covariance based on motion model
            self.state, self.covariance = kalman_filter_predict(
                self.state, self.covariance, u, self.motion_model, self.R, dt)

            # ----- MEASUREMENT UPDATE -----
            # Prepare for measurement update if ArUco data is available
            z = Vector(5)
            Q = Identity(5)
            
            # Use position measurement model
            h = self.h_ne_update
            
            # Set measurement values from ArUco data
            z[self.N] = self.measured_pose_northings_m
            z[self.E] = self.measured_pose_eastings_m
            z[self.G] = self.measured_pose_yaw_rad

            # Set measurement noise covariance
            Q[self.N, self.N] = self.NE_Q_std[0, 0] ** 2    # North position variance
            Q[self.E, self.E] = self.NE_Q_std[0, 1] ** 2    # East position variance
            Q[self.G, self.G] = self.g_Q_std[0] ** 2        # Heading variance
            
            # Perform measurement update if ArUco data is available
            if aruco_pose is not None:
                self.state, self.covariance = kalman_filter_update(
                    self.state, self.covariance, z, h, Q)

            # ----- UPDATE ESTIMATED POSE -----
            # Extract current pose estimate from state
            self.est_pose_northings_m = self.state[self.N, 0]
            self.est_pose_eastings_m = self.state[self.E, 0]
            self.est_pose_yaw_rad = self.state[self.G, 0]
            
            # ----- TRAJECTORY FOLLOWING -----
            # Check if trajectory has been initialized
            if hasattr(self, 'path'):
                # ----- FEEDFORWARD CONTROL -----
                # Update waypoint progress based on current time and position
                self.path.wp_progress(self.t, self.state[:3], self.turning_radius)
                
                # Sample the reference trajectory at current time
                p_ref, u_ref = self.path.p_u_sample(self.t)
                
                # ----- FEEDBACK CONTROL -----
                # Calculate pose difference between reference and actual
                dp = Vector(3)  # Vector for pose difference in earth frame
                dp = p_ref - self.state[:3]  # Calculate difference
                
                # Handle angle wrapping for heading difference
                dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi
                
                # Transform difference to body frame for control
                H_eb = HomogeneousTransformation(self.state[:3][0:2], self.state[:3][2])
                ds = Inverse(H_eb.H_R) @ dp  # Pose error in body frame
                
                # ----- COMPUTE CONTROL GAINS -----
                # Initialize gains if this is the first control iteration
                if self.initialise_control == True:
                    # Initial gains when starting from rest
                    self.k_n = (2 * (u_ref[0])) / (self.L**2)  # Cross-track gain
                    self.k_g = u_ref[0] / self.L              # Heading gain
                    self.initialise_control = False            # Mark as initialized
                
                # ----- COMPUTE CONTROL COMMANDS -----
                # Calculate feedback control correction
                du = feedback_control(ds, self.k_s, self.k_n, self.k_g)
                
                # Combine feedforward and feedback control
                u = u_ref + du
                
                # ----- APPLY CONTROL LIMITS -----
                # Ensure control commands are within limits
                if u[0] > self.v_max: u[0] = self.v_max        # Limit forward velocity
                if u[0] < -self.v_max: u[0] = -self.v_max      # Limit reverse velocity
                if u[1] > self.w_max: u[1] = self.w_max        # Limit turning rate (CCW)
                if u[1] < -self.w_max: u[1] = -self.w_max      # Limit turning rate (CW)
                
                # ----- UPDATE CONTROL GAINS FOR NEXT ITERATION -----
                # Adjust gains based on current velocity
                self.k_n = (2 * u[0]) / (self.L**2)  # Cross-track gain
                self.k_g = u[0] / self.L            # Heading gain
                
                # ----- COMPUTE WHEEL SPEED COMMANDS -----
                # Convert twist (v, ω) to individual wheel speeds
                q = self.ddrive.inv_kinematics(u)            
                
                # Prepare wheel speed message
                wheel_speed_msg = Vector3Stamped()
                wheel_speed_msg.vector.x = q[0, 0]  # Right wheel speed (rad/s)
                wheel_speed_msg.vector.y = q[1, 0]  # Left wheel speed (rad/s)
                
                # Store commanded wheel speeds
                self.cmd_wheelrate_right = wheel_speed_msg.vector.x
                self.cmd_wheelrate_left = wheel_speed_msg.vector.y
                
                # ----- ACT: SEND COMMANDS TO ROBOT -----
                # Publish wheel speed commands
                self.wheel_speed_pub.publish(wheel_speed_msg)
                
                # Log the commanded wheel speeds
                self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")


# ----- MAIN ENTRY POINT -----
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    # Create and run the LaptopPilot
    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
