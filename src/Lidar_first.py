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
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control
from graphslam_demo import GraphSLAM2D

N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4


class LaptopPilot:
    def __init__(self, simulation):
        
        ######################## Fixed Parameters #############################
        
        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 24,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1"
        
        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        self.simulation = simulation
        if self.simulation:
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  #Ovewrites Aruco marker ID to 0 (needed for simulation)
            self.sim_init = True #used to deal with webots timestamps

        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)
        
        self.datalog = DataLogger(log_dir="logs")

        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )

        ###################### Motion Model Parameters ########################

        wheel_distance = 0.165
        wheel_diameter = 0.065
        
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) 
        
        ###################### Path & Control Parameters ######################
        
        self.velocity = 0.15
        self.acceleration = self.velocity/3
        self.turning_radius = 0.3  # meters - minimum turning radius   

        #GAIN VARIABLES
        self.tau_s = 0.3 # s to remove along track error
        self.L = 0.7 # m distance to remove normal and angular error

        # compute control gains for the initial condition (where robot is stationary)
        self.k_s = 1/self.tau_s  # along track gain
        
        self.v_max = 0.3 # fastest the robot can go
        self.w_max = np.deg2rad(30) # fastest the robot can turn

        self.initialise_control = True # False once control gains is initialised 
        
        self.initialise_pose = True # False once the pose is initialised 


        # path
        self.northings_path = [0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0] # create a list of waypoints
        self.eastings_path = [0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0] # create a list of waypoints
        
        self.relative_path = True # False if you want it to be absolute
        
        ################# Initial State & Measurements ########################
        
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None


        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None 

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None   

        ################# Lidar Parameters ####################################
        
        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0.07 # location of lidar centre in b-frame primary axis
        lidar_yb = 0.0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb)
        
        self.graph_slam = GraphSLAM2D()
        
        self.prev_pose = None

        ###################### EKF Parameters #################################

        self.n_std = [1.0]
        self.e_std = [1.0]
        self.g_std = [np.deg2rad(1.0)]
        self.G_std = l2m(self.g_std)
        self.NE_std = l2m([self.n_std,self.e_std])
        


        self.dot_x_R_std = l2m([0.02])
        self.dot_g_R_std = l2m([np.deg2rad(0.01)])
        self.NE_Q_std = l2m([[0.1],[0.1]])
        self.g_Q_std = l2m([np.deg2rad(1)])

        self.state = Vector(5)
        self.covariance = Identity(5)
        self.R = Identity(5) 
        
        self.covariance[N,N] = self.NE_std[0,0]**2
        self.covariance[E, E] = self.NE_std[0,1]**2
        self.covariance[G, G] = self.G_std[0]**2
        self.covariance[DOTX, DOTX] = 0.0**2
        self.covariance[DOTG, DOTG] = np.deg2rad(0)**2
        
        self.R[N, N] = 0.1**2
        self.R[E, E] = 0.1**2
        self.R[G, G] = np.deg2rad(5)**2
        self.R[DOTX, DOTX] = self.dot_x_R_std**2
        self.R[DOTG, DOTG] = np.deg2rad(self.dot_g_R_std)**2
        
        self.aruco_count = 0
        self.loop_count = 0
                    
    def true_wheel_speeds_callback(self, msg):
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")
        
    def find_corners_from_lidar(self, scan_data, robot_pose, lidar, threshold=0.001):
        """
        Given LIDAR scan data, detect corners and return their position in the world frame.
        
        Parameters:
            scan_data (np.array): LIDAR scan data, an Nx2 array where each row is [distance, angle].
            robot_pose (np.array): The robot's pose in the world frame, [x, y, yaw].
            lidar: LIDAR object that provides the rangeangle_to_loc function for converting polar to world coordinates.
            threshold (float): Threshold to detect corner sharpness.
    
        Returns:
            np.array: The world coordinates of the detected corners.
        """
        distances = scan_data[:, 0]
        angles = scan_data[:, 1]
        
        corners_world = []
    
        # Simple corner detection based on sharp changes in angle and distance
        for i in range(2, len(distances)-2):  # avoid boundary issues
            diff1 = distances[i] - distances[i-1]  # difference in distances
            diff2 = distances[i+1] - distances[i]  # next difference
            
            angle_diff = np.abs(angles[i+1] - angles[i])  # angle change
            
            # Detect corner based on sharp angle change and large distance change
            if angle_diff > threshold and (np.abs(diff1) > threshold or np.abs(diff2) > threshold):
                # Convert the detected corner to world frame using rangeangle_to_loc
                z_lm = Vector(2)
                z_lm[0] = distances[i]
                z_lm[1] = angles[i]
                
                # Robot pose in world frame [x, y, yaw]
                p_eb = Vector(3)
                p_eb[N] = robot_pose[N]  # Northing
                p_eb[E] = robot_pose[E]  # Easting
                p_eb[G] = robot_pose[G]  # Yaw
                
                # Convert the LIDAR scan to world coordinates
                corner_world = lidar.rangeangle_to_loc(p_eb, z_lm)
                corners_world.append(corner_world)
    
        return np.array(corners_world)

    def lidar_callback(self, msg):
        # Process LIDAR and robot pose
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m  # Robot position
        p_eb[1] = self.est_pose_eastings_m
        p_eb[2] = self.est_pose_yaw_rad

        # Convert LIDAR data to world coordinates
        self.lidar_data = np.zeros((len(msg.ranges), 2))
        z_lm = Vector(2)

        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)  # Convert to world frame
            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]
        
        # Detect corners in the LIDAR scan (Optional for landmarks)
        corners_world = self.find_corners_from_lidar(self.lidar_data, [self.est_pose_northings_m,
                                                                       self.est_pose_eastings_m,
                                                                       self.est_pose_yaw_rad],
                                                     self.lidar)

        # Assuming you are also using odometry to update poses
        current_pose = np.array([self.est_pose_northings_m, self.est_pose_eastings_m, self.est_pose_yaw_rad])
        
        if self.prev_pose is not None:
            # Create odometry measurement (just an example)
            odometry = np.array([current_pose[0] - self.prev_pose[0],
                                 current_pose[1] - self.prev_pose[1],
                                 current_pose[2] - self.prev_pose[2]])

            # Add the current pose to the GraphSLAM
            v = self.graph_slam.add_pose(current_pose, fixed=False)

            # Add an odometry edge between the previous pose and the current pose
            self.graph_slam.add_odometry_edge(self.prev_pose_id, v.id(), odometry)
        
        # Add landmarks from detected corners
        for corner in corners_world:
            # Assuming corners are stored with a unique landmark ID and position (corner)
            # For simplicity, assume we add the corner as a new landmark with a unique ID.
            landmark_id = len(self.graph_slam.landmark_vertices)  # Unique ID based on the number of landmarks
            self.graph_slam.add_landmark(landmark_id, np.array(corner))

            # Add an edge between the current pose and the detected landmark
            self.graph_slam.add_landmark_edge(v.id(), landmark_id, corner)

        # Optimize the graph periodically (every second in this case)

        if self.prev_time is None or (datetime.utcnow().timestamp() - self.prev_time > 1):
            self.graph_slam.optimize()
            self.prev_time = datetime.utcnow().timestamp()

        # Set the current pose as the previous pose for the next loop
        self.prev_pose = current_pose
        self.prev_pose_id = v.id()

        # Log or print the current optimized pose (for debugging)
        optimized_pose = self.graph_slam.get_poses()
        print(f"Optimized Pose: {optimized_pose}")

        self.datalog.log(msg, topic_name="/lidar")


    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.datalog.log(msg, topic_name="/groundtruth")

    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]

        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.utcnow().timestamp()-msg[0]
                self.sim_init = False                                         
                
            time_stamp = msg[0] + self.sim_time_offset                

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True:
            quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else:
            quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat        
        
        return pose_msg

    # TRAJECTORY GENERATION
    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.measured_pose_northings_m  # offset by current northings
                self.eastings_path[i] += self.measured_pose_eastings_m  # offset by current eastings

            # convert path to matrix and create a trajectory class instance
            C = l2m([self.northings_path, self.eastings_path])        
            self.path = TrajectoryGenerate(C[:,0], C[:,1])        
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.velocity, self.acceleration)  # velocity and acceleration
            self.path.turning_arcs(self.turning_radius)  # turning radius
            self.path.wp_id = 0  # initialises the next waypoint

    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception2: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()
        
    def extended_kalman_filter_predict(self, mu, Sigma, u, f, R, dt):
        # (1) Project the state forward
        # f is the rigid body motion model
        pred_mu, F = f(mu, u, dt)
        
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
            
        N_k_1 = state[N]
        E_k_1 = state[E]
        G_k_1 = state[G]
        DOTX_k_1 = state[DOTX]
        DOTG_k_1 = state[DOTG]

        p = Vector(3)
        p[N] = N_k_1
        p[E] = E_k_1
        p[G] = G_k_1
        
        # note rigid_body_kinematics already handles the exception dynamics of w=0
        p = rigid_body_kinematics(p,u,dt)    

        # vertically joins two vectors together
        state = np.vstack((p, u))
        
        N_k =  state[N]
        E_k = state[E]
        G_k = state[G]
        DOTX_k = state[DOTX]
        DOTG_k =  state[DOTG]

        # Compute its jacobian
        F = Identity(5)    

        if abs(DOTG_k) <1E-2: # caters for zero angular rate, but uses a threshold to avoid numerical instability
            F[N, G] = -DOTX_k * dt *np.sin(G_k_1)
            F[N, DOTX] = dt * np.cos(G_k_1)
            F[E, G] = DOTX_k * dt * np.cos(G_k_1)
            F[E, DOTX] = dt * np.sin(G_k_1)
            F[G, DOTG] = dt     
            
        else:
            F[N, G] = (DOTG_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
            F[N, DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[N, DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
            F[E, G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[E, DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
            F[E, DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
            F[G, DOTG] = dt

        return state, F


    def h_g_update(self,x):
        z = Vector(5)
        z[G] = x[G]
        H = Matrix(5,5)
        H[G,G] = 1
        return z, H

    def h_ne_update(self,x):
        z = Vector(5)
        z[N] = x[N]
        z[E] = x[E]
        H = Matrix(5,5)
        H[N,N] = 1
        H[E,E] = 1
        return z, H

    def infinite_loop(self):
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            # reads sensed pose for local use 
            msg = self.pose_parse(aruco_pose, aruco = True)
            
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrapping

            # logs the data            
            self.datalog.log(msg, topic_name="/aruco")
            self.aruco_count += 1


        ###### wait for the first sensor info to initialize the pose ######
        if self.initialise_pose == True and aruco_pose is not None:
            self.state[N] = self.measured_pose_northings_m
            self.state[E] = self.measured_pose_eastings_m
            self.state[G] = self.measured_pose_yaw_rad

            self.state[N,0] = self.measured_pose_northings_m
            self.state[E,0] = self.measured_pose_eastings_m
            self.state[G,0] = self.measured_pose_yaw_rad

            # get current time and determine timestep
            self.t_prev = datetime.utcnow().timestamp() #initialise the time
            self.t = 0 #elapsed time
            time.sleep(0.1) #wait for approx a timestep before proceeding
            
            # Generate trajectory after initializing pose
            self.generate_trajectory()
            # path and trajectory are initialised
            self.initialise_pose = False 

        if self.initialise_pose != True and self.measured_wheelrate_right is not None and self.measured_wheelrate_left is not None:  
            self.loop_count += 1
            ################### Motion Model ##############################
            # convert true wheel speeds in to twist
            q = Vector(2)            
            q[0] = self.measured_wheelrate_right # wheel rate rad/s (measured)
            q[1] = self.measured_wheelrate_left # wheel rate rad/s (measured)
            u = self.ddrive.fwd_kinematics(q) 
            #determine the time step
            t_now = datetime.utcnow().timestamp()        
        
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop
            
            self.state , self.covariance  = self.extended_kalman_filter_predict(self.state, self.covariance, u, self.motion_model, self.R, dt)

            if aruco_pose is not None:
                
                z = Vector(5)
                Q = Identity(5)
    
                h = self.h_ne_update
                z[N] = self.measured_pose_northings_m
                z[E] = self.measured_pose_eastings_m
                z[G] = self.measured_pose_yaw_rad
    
                Q[N, N] = self.NE_Q_std[0,0] ** 2
                Q[E, E] = self.NE_Q_std[0,1] ** 2
                Q[G, G] = self.g_Q_std[0] ** 2
                
                self.state , self.covariance  = self.extended_kalman_filter_update(self.state, self.covariance, z, h, Q)

            # take current pose estimate and update by twist
            ##################### SLAM ########################################
            
            
            
            ##################### SLAM ########################################
            
            #################### Trajectory sample #################################    
            #if hasattr(self, 'path'):
            # feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(self.t, self.state[:3], self.turning_radius)  # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(self.t)  # sample the path at the current elapsetime (i.e., seconds from start of motion modelling)


            # feedback control: get pose change to desired trajectory from body
            dp = Vector(3)  # Create vector for pose difference in e-frame
            dp = p_ref - self.state[:3]  # Northings difference
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi  # handle angle wrapping for yaw

            # Transform difference to body frame
            H_eb = HomogeneousTransformation(self.state[:3][0:2],self.state[:3][2])  # body to earth transform
            ds = Inverse(H_eb.H_R) @ dp 
            if self.initialise_control == True:
                # Initial gains when starting from rest
                self.k_n = (2*(u_ref[0]))/(self.L**2)
                self.k_g = u_ref[0]/self.L # heading gain
                self.initialise_control = False  # maths changes after first iteration

            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control - combine feedback and feedforward
            u = u_ref + du
            # ensure within performance limitations
            if u[0] > self.v_max: u[0] = self.v_max
            if u[0] < -self.v_max: u[0] = -self.v_max
            if u[1] > self.w_max: u[1] = self.w_max
            if u[1] < -self.w_max: u[1] = -self.w_max

            # update control gains for next timestep
            self.k_n = (2*u[0])/(self.L**2) # cross track gain
            self.k_g = u[0]/self.L  # heading gain

            # actuator commands                 
            q = self.ddrive.inv_kinematics(u)            
            #print(f"q: {q}")
            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0,0]  # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1,0]  # Left wheelspeed rad/s

            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
    ################################################################################

            # > Act < #
            # Send commands to the robot        
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
            
            self.est_pose_northings_m = self.state[N,0]
            self.est_pose_eastings_m = self.state[E,0]
            self.est_pose_yaw_rad = self.state[G,0]


if __name__ == "__main__":
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

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)