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
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control


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
        wheel_diameter = 0.065  # Diameter of each wheel in meters

        # Create differential drive configuration object
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)


        # ============ TRAJECTORY PARAMETERS ============
        # These control how the robot moves along its path
        self.velocity = 0.08              # Desired forward velocity in m/s
        self.acceleration = self.velocity/3 # How quickly to reach desired velocity
        self.turning_radius = 0.3         # Minimum turning radius in meters


        # ============ CONTROL PARAMETERS ============
        # These values control how aggressively the robot corrects errors
        self.tau_s = 0.8  # Time constant for removing along-track error (seconds)
        self.L = 0.2      # Distance constant for removing cross-track and angular error (meters)

        # Control gains - these determine how strongly the robot responds to errors
        self.k_s = 1/self.tau_s  # Along-track gain (how strongly to correct forward/backward error)

        # Velocity and turning rate limits for safety
        self.v_max = 0.15          # Maximum forward/backward speed in m/s
        self.w_max = np.deg2rad(15)  # Maximum turning rate in rad/s (15 degrees/s)

        # Initialization flags
        self.initialise_control = True  # Will be set to False after first control update
        self.initialise_pose = True     # Will be set to False after receiving first position


        # ============ PATH WAYPOINTS ============
        # Define the path the robot should follow as a series of points
        # Each point has a northing (y) and easting (x) coordinate
        self.northings_path = [0.0, 1.0, 1.0, 0.0, 0.0]
        self.eastings_path = [0.0, 0.0, 1.0, 1.0, 0.0]
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
        self.path.turning_arcs(self.turning_radius)
        self.path.wp_id = 0  # Start at first waypoint


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
            print("Exception: ", e)
        finally:
            # Clean up subscribers
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()


    def motion_model(self, state, u, dt):
        """
        Predict how the robot's state changes based on control inputs.
        This implements the differential drive motion model.

        Args:
            state: Current state vector [N, E, G, DOTX, DOTG]
            u: Control input [forward velocity, angular velocity]
            dt: Time step in seconds

        Returns:
            Updated state vector
        """
        # Extract current state values
        N_k_1 = state[self.N]      # Current north position
        E_k_1 = state[self.E]      # Current east position
        G_k_1 = state[self.G]      # Current heading
        DOTX_k_1 = state[self.DOTX]  # Current forward velocity
        DOTG_k_1 = state[self.DOTG]  # Current angular velocity

        # Create position vector
        p = Vector(3)
        p[self.N] = N_k_1
        p[self.E] = E_k_1
        p[self.G] = G_k_1

        # Update position using rigid body kinematics
        # This handles the special case when angular velocity is zero
        # p = rigid_body_kinematics(p, u, dt)
        p, _, _, _ = rigid_body_kinematics(p, u, dt=0.1)  

        # Create new state vector with updated position and velocities
        new_state = np.vstack((p, u))
        new_state[0:3] = p    # Updated position
        new_state[3:5] = u    # New velocities from control input

        return new_state


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

    # Create and run the robot controller
    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)