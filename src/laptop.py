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
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation, t2v, v2t, polar2cartesian
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control
from model_feeg6043 import graphslam_frontend, lidar_scan, graphslam_backend 
from classifier import GPC_input_output, load_model
from plot_feeg6043 import plot_2dframe, sigma_contour, show_information
import copy
import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class LaptopPilot:
    def __init__(self, simulation):
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

        ############# INITIALISE ATTRIBUTES ##########       
        wheel_distance = 0.09
        wheel_diameter = 0.07
        

        # Trajectory parameters
        self.velocity = 0.1
        self.acceleration = self.velocity/3
        
        self.turning_radius = 0.35  # meters - minimum turning radius
        # control parameters        

        #GAIN VARIABLES
        self.tau_s = 0.55 # s to remove along track error
        self.L = 0.35 # m distance to remove normal and angular error

        # compute control gains for the initial condition (where robot is stationary)
        self.k_s = 1/self.tau_s  # along track gain
        
        self.v_max = 0.2 # fastest the robot can go
        self.w_max = np.deg2rad(15) # fastest the robot can turn




        self.initialise_control = True # False once control gains is initialised 
        
        self.initialise_pose = True # False once the pose is initialised 

        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) 

        # path
        pathx = [0.0, 1.5, 1.5, 0.0]
        pathy = [0.0, 0.0, 1.5, 1.5]
        self.northings_path = pathx + pathx + [0.0] # create a list of waypoints
        self.eastings_path = pathy + pathy + [0.0] # create a list of waypoints
        
        
        self.relative_path = True # False if you want it to be absolute

        # model pose
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

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0.07 # location of lidar centre in b-frame primary axis
        lidar_yb = 0.0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb)    


        # EKF 
        # Easy names for indexing
        self.N = 0
        self.E = 1
        self.G = 2
        self.DOTX = 3
        self.DOTG = 4

        #INITIAL STD DEVIATIONS ############################################
        self.n_std = [1.0]
        self.e_std = [1.0]
        self.g_std = [np.deg2rad(1.0)]


        self.G_std = l2m(self.g_std)
        self.NE_std = l2m([self.n_std,self.e_std])
        
        

        #MEASUREMENT NOISES ############################################
        #From ARUCO
        self.NE_Q_std = l2m([[0.1],[0.1]]) # Standard deviation of the northings and eastings noise
        self.g_Q_std = l2m([np.deg2rad(1)])  # Standard deviation of the yaw noise

        self.state = Vector(5)
        self.covariance = Identity(5)
        self.R = Identity(5) 
        
        self.covariance[self.N,self.N] = self.NE_std[0,0]**2
        self.covariance[self.E, self.E] = self.NE_std[0,1]**2
        self.covariance[self.G, self.G] = self.G_std[0]**2
        self.covariance[self.DOTX, self.DOTX] = 0.0**2
        self.covariance[self.DOTG, self.DOTG] = np.deg2rad(0)**2
        
        #PROCESS NOISES ############################################
        #From the motion model
        self.R_N = 0.1 # Standard deviation of the northings noise
        self.R_E = 0.1 # Standard deviation of the eastings noise
        self.R_G = np.deg2rad(5) # Standard deviation of the yaw noise
        self.dot_x_R_std = l2m([0.02]) # Standard deviation of the velocity noise
        self.dot_g_R_std = l2m([np.deg2rad(0.01)]) # Standard deviation of the angular rate noise


        self.R[self.N, self.N] = self.R_N**2
        self.R[self.E, self.E] = self.R_E**2
        self.R[self.G, self.G] = self.R_G**2
        self.R[self.DOTX, self.DOTX] = self.dot_x_R_std**2
        self.R[self.DOTG, self.DOTG] = self.dot_g_R_std**2

        # ============ SLAM SETUP ============

        # Create a GraphSLAM2D object for SLAM
        graph = graphslam_frontend()
        self.sigma_xy = Matrix(3, 3)
        self.sigma_xy[0, 0] = 0.1  # Small non-zero value
        self.sigma_xy[1, 1] = 0.1  # Small non-zero value
        self.sigma_xy[2, 2] = 0.1  # Small non-zero value



        # Motion model linear noise due to v and w
        self.sigma_motion = Matrix(3, 2)
        self.sigma_motion[0, 0] = 0.1*2    # impact of v linear velocity on x
        self.sigma_motion[0, 1] = np.deg2rad(0.1)**2  # impact of w angular velocity on x
        self.sigma_motion[1, 0] = 0.3**2   # impact of v linear velocity on y
        self.sigma_motion[1, 1] = np.deg2rad(0.3)**2  # impact of w angular velocity on y
        self.sigma_motion[2, 0] = 0.1**2   # impact of v linear velocity on gamma
        self.sigma_motion[2, 1] = np.deg2rad(0.3)**2  # impact of w angular velocity on gamma
        print('3x2 motion noise model:\n', self.sigma_motion, '\n')

        # Observation model linear noise with range
        self.sigma_observe = Matrix(2, 2)
        self.sigma_observe[0, 0] = 0.1**2  # 10% of range
        self.sigma_observe[0, 1] = 0
        self.sigma_observe[1, 0] = np.deg2rad(5)**2  # 5 degree per metre range
        self.sigma_observe[1, 1] = 0
        print('2x2 measurement noise model:\n', self.sigma_observe, '\n')


        # anchor constraint, matrix must be invertable
        self.sigma = Matrix(3, 3)
        self.sigma[0, 0] = 0.1
        self.sigma[0, 1] = 0.01
        self.sigma[1, 0] = 0.01
        self.sigma[1, 1] = 0.1
        self.sigma[0, 2] = 0.01
        self.sigma[1, 2] = 0.01
        self.sigma[2, 0] = 0.01
        self.sigma[2, 1] = 0.01
        self.sigma[2, 2] = 0.1


        # SLAM tracking variables
        self.last_slam_update_time = None
        self.slam_update_frequency = 1.0  # Update SLAM every 1 second
        self.landmarks = {}  # Dictionary to track observed landmarks
        self.wall_point_threshold = 0.1  # Distance threshold to consider points part of same wall

        # SLAM intensity values
        self.node_intensity = 3
        self.motion_intensity = 1
        self.observation_intensity = 2

        # SLAM observations list [type, poses, type, landmarks] 
        self.slam_observations = ['pose', [], 'landmark', []]
        self.p_gt = Vector(3)


        print('Start graph data association')
        self.graph = graphslam_frontend()
        self.graph.anchor(self.sigma)

        self.completed = False


        self.d_p_eb = Vector(3)
        self.d_p_eb[0] = 0
        self.d_p_eb[1] = 0
        self.d_p_eb[2] = 0

        self.gpc_corner = None
        
        self.aruco_count = 0
        self.loop_count = 0
        
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
                    
    def true_wheel_speeds_callback(self, msg):
        #print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        #print("Received lidar message", msg.header.seq)
            
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = msg.header.stamp #we want the lidar measurement timestamp here
        
        self.lidar_data = np.zeros((len(msg.ranges), 2)) #specify length of the lidar data
        #self.lidar_data[:,0] = msg. # use ranges as a placeholder, workout northings in Task 4
        #self.lidar_data[:,1] = msg. # use angles as a placeholder, workout eastings in Task 4
        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m #robot pose northings (see Task 3)
        p_eb[1] = self.est_pose_eastings_m #robot pose eastings (see Task 3)
        p_eb[2] = self.est_pose_yaw_rad #robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                    
        z_lm = Vector(2)        
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
                
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm) # see tutotial

            self.lidar_data[i,0] = t_em[0]
            self.lidar_data[i,1] = t_em[1]

        # this filters out any 
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
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
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            # print(
            #     "Received position update from",
            #     datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset,
            #     "seconds ago",
            # )
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

    def find_corner(self, corner, threshold=0.01):
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
        
    def motion_model(self, state, u, dt):
            
        N_k_1 = state[self.N]
        E_k_1 = state[self.E]
        G_k_1 = state[self.G]
        DOTX_k_1 = state[self.DOTX]
        DOTG_k_1 = state[self.DOTG]

        p = Vector(3)
        p[self.N] = N_k_1
        p[self.E] = E_k_1
        p[self.G] = G_k_1
        
        # note rigid_body_kinematics already handles the exception dynamics of w=0
        # m = Matrix[3,2]
        # m_ugt = None
        # s_xy = Matrix[3,3]
        # p = rigid_body_kinematics(p,u,dt,m_ugt, m, s_xy )    

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
        sigma_motion = Matrix(3,2)  # Create a new 3x2 matrix instance
        p, _, _, _ = rigid_body_kinematics(p, u, dt=dt, sigma_motion=sigma_motion, sigma_xy=sigma)

        # vertically joins two vectors together
        state = np.vstack((p, u))
        
        N_k =  state[self.N]
        E_k = state[self.E]
        G_k = state[self.G]
        DOTX_k = state[self.DOTX]
        DOTG_k =  state[self.DOTG]

        # Compute its jacobian
        F = Identity(5)    

        if abs(DOTG_k) <1E-2: # caters for zero angular rate, but uses a threshold to avoid numerical instability
            F[self.N, self.G] = -DOTX_k * dt *np.sin(G_k_1)
            F[self.N, self.DOTX] = dt * np.cos(G_k_1)
            F[self.E, self.G] = DOTX_k * dt * np.cos(G_k_1)
            F[self.E, self.DOTX] = dt * np.sin(G_k_1)
            F[self.G, self.DOTG] = dt     
            
        else:
            F[self.N, self.G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
            F[self.N, self.DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[self.N, self.DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
            F[self.E, self.G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
            F[self.E, self.DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
            F[self.E, self.DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
            F[self.G, self.DOTG] = dt

        return state, F


    def h_g_update(self,x):
        z = Vector(5)
        z[self.G] = x[self.G]
        H = Matrix(5,5)
        H[self.G,self.G] = 1
        return z, H

    def h_ne_update(self,x):
        z = Vector(5)
        z[self.N] = x[self.N]
        z[self.E] = x[self.E]
        H = Matrix(5,5)
        H[self.N,self.N] = 1
        H[self.E,self.E] = 1
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
            #print(f"aruco={self.aruco_count}")
            self.aruco_count += 1


        ###### wait for the first sensor info to initialize the pose ######
        if self.initialise_pose == True and aruco_pose is not None:
            print('hello')
            self.state[self.N] = self.measured_pose_northings_m
            self.state[self.E] = self.measured_pose_eastings_m
            self.state[self.G] = self.measured_pose_yaw_rad

            print('State:')
            print(self.state)

            print('\nProcess noise covariance:')

            self.est_pose_northings_m = self.measured_pose_northings_m
            self.est_pose_eastings_m = self.measured_pose_eastings_m
            self.est_pose_yaw_rad = self.measured_pose_yaw_rad

            p_eb = Vector(3); 
            p_eb[0] = self.est_pose_northings_m  # North position
            p_eb[1] = self.est_pose_eastings_m   # East position
            p_eb[2] = self.est_pose_yaw_rad      # Heading angle

            print('3x1 state vector:\n', p_eb, '\n')
            H_eb = HomogeneousTransformation(p_eb[0:2], p_eb[2])

            # get current time and determine timestep
            self.t_prev = datetime.utcnow().timestamp() #initialise the time
            self.t = 0 #elapsed time
            time.sleep(0.1) #wait for approx a timestep before proceeding
            
            # Generate trajectory after initializing pose
            self.generate_trajectory()
            # path and trajectory are initialised

            ###################################SETUP JOBLB CLASSIFIER########   
            self.gpc_corner = joblib.load("gaussian_process_classifier.joblib")

            self.initialise_pose = False 

        if self.initialise_pose != True and self.measured_wheelrate_right is not None and self.measured_wheelrate_left is not None:  
            #print(self.measured_pose_northings_m)
            #print(f"loop_={self.loop_count}")
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

            z = Vector(5)
            Q = Identity(5)

            h = self.h_ne_update
            z[self.N] = self.measured_pose_northings_m
            z[self.E] = self.measured_pose_eastings_m
            z[self.G] = self.measured_pose_yaw_rad
            #print("4")

            Q[self.N, self.N] = self.NE_Q_std[0,0] ** 2
            Q[self.E, self.E] = self.NE_Q_std[0,1] ** 2
            Q[self.G, self.G] = self.g_Q_std[0] ** 2
            #print(z)
            
            if aruco_pose is not None:
                self.state , self.covariance  = self.extended_kalman_filter_update(self.state, self.covariance, z, h, Q)

            # take current pose estimate and update by twist
            
            self.est_pose_northings_m = self.state[self.N,0]
            self.est_pose_eastings_m = self.state[self.E,0]
            self.est_pose_yaw_rad = self.state[self.G,0]
            
            msg = self.pose_parse([datetime.utcnow().timestamp(), self.est_pose_northings_m, self.est_pose_eastings_m, 0, 0, 0, self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/est_pose")
            #################### Trajectory sample #################################

            # feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(self.t, self.state[:3], self.turning_radius)  # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(self.t)  # sample the path at the current elapsetime (i.e., seconds from start of motion modelling)


            #print("5")

            # feedback control: get pose change to desired trajectory from body
            dp = Vector(3)  # Create vector for pose difference in e-frame
            dp = p_ref - self.state[:3]  # Northings difference
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi  # handle angle wrapping for yaw

            # Transform difference to body frame
            H_eb_l = HomogeneousTransformation(self.state[:3][0:2],self.state[:3][2])  # body to earth transform
            ds = Inverse(H_eb_l.H_R) @ dp 
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

            p_eb = Vector(3); 
            p_eb[0] = self.est_pose_northings_m  # North position
            p_eb[1] = self.est_pose_eastings_m   # East position
            p_eb[2] = self.est_pose_yaw_rad      # Heading angle

            print('3x1 state vector:\n', p_eb, '\n')
            H_eb = HomogeneousTransformation(p_eb[0:2], p_eb[2])



    ################################################################################

            # > Act < #
            # Send commands to the robot        
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")

            ######################ADDING GRAPH SLAM HERE###########################
           # IF CORNER DETECTED, UPDATE THE GRAPH
            print("Before calling run_classifier")
            print("p_eb:", p_eb)
            flag = False
            t_em = Vector(2)
            t_em[0] = 0
            t_em[1] = 0
            observation, _ = lidar_scan(
                p_eb, self.lidar_data, self.lidar, self.sigma_observe
            )
            if (
                observation is not None
                or not np.isnan(observation.data_filled[:, 0]).any()
            ):
                new_observation = self.GPC_input_output(observation, None)
                # Ensure the input data has the correct shape (240 features)
                input_data = np.zeros(240)  # Create array of expected size
                actual_data = new_observation.data_filled[:, 0]
                # Copy available data, pad with zeros if needed
                input_data[:len(actual_data)] = actual_data
                
                new_observation.label = self.gpc_corner.classes_[
                    np.argmax(
                        self.gpc_corner.predict_proba(
                            [input_data]  # Use the properly shaped input
                        )
                    )
                ]
                if new_observation.label == "corner":
                    flag = True
                    r, theta, idx = self.find_corner(new_observation)
                    if r is not None:
                        # Convert polar coordinates to cartesian in sensor frame
                        x_l, y_l = polar2cartesian(r, theta)
                        
                        # Convert to environment frame using current robot pose
                        H_eb = HomogeneousTransformation(p_eb[0:2], p_eb[2]) 
                        t_em = t2v(H_eb.H@self.lidar.H_bl.H@v2t([x_l, y_l]))
                        
                        print(f"#######################\n\n CORNER DETECTED at: [{t_em[0]:.3f}, {t_em[1]:.3f}] \n\n#######################")
            #t_em, flag = self.run_classifier(p_eb)
            print("After run_classifier, flag:", flag)

            
            if flag == True:
                print("POINT1")
                _, _, t_lm, sigma_xy_lm = self.lidar.loc_to_rangeangle( p_eb, t_em, sigma_observe=self.sigma_observe) 

                self.sigma_xy[0,0] = sigma_xy_lm[0,0]
                self.sigma_xy[1,1] = sigma_xy_lm[1,1]

                print("POINT2")

                if (p_eb[0] < 1 and p_eb[1] < 1):
                    landmark_id = 0

                elif (p_eb[0] > 1 and p_eb[0] < 2 ) and  (p_eb[1] > 0 and p_eb[1] < 1 ):
                    landmark_id = 1               

                elif (p_eb[0] > 1 and p_eb[0] < 2 ) and  (p_eb[1] > 1 and p_eb[1] < 2 ):
                    landmark_id = 2

                elif (p_eb[0] > 0 and p_eb[0] < 1 ) and  (p_eb[1] > 1 and p_eb[1] < 2 ):
                    landmark_id = 3 
                # adds to the graph as a landmark observation together with its ID
                self.graph.observation(t_em, sigma_xy_lm, landmark_id, t_lm)  # Task
                print('Observation of Landmark ID', landmark_id)

            else:
      
                print('Motion')
                # store current pose and covaariance
                p_ = copy.copy(p_eb)
                sigma_ = copy.copy(self.sigma_xy)

                # progress pose through motion model

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
                p_eb, self.sigma_xy, self.d_p_eb, _ = rigid_body_kinematics(p_eb, u, dt=dt, sigma_motion=self.sigma_motion, sigma_xy=sigma)


                # adds to the graph as a motion
                self.graph.motion(p_, sigma_, self.d_p_eb, final=False)


            print("s3")
            ################################## BACKEND ######################################

            if self.completed == True:
    
                # completes the motion
                self.graph.motion(p_eb, self.sigma_xy, Vector(3), final=True)
                print('Finish graph data association')
                print('*************************************************')

                self.graph.construct_graph()

                

                initial_residual = 100  # just needs to be a big number to avoid triggering convergence if the first iteration has large residuals
                initial_flag = True

                residual_threshold = 1E-12  # if result changes by <1
                delta_threshold = 1/10  # if result changes by <1
                lim_iterations = 20

                n_iterations = 0
                delta_residual = initial_residual
                residual = initial_residual

                visualise_flag = False
                iteration_continue = True
                residual_continue = True
                converge_continue = True

                # if any of the conditions become false, then while loop will exit
                cpu_start_solver = datetime.now()


                graph_init = copy.deepcopy(self.graph)
                graph_validate = copy.deepcopy(self.graph)
                graph_opt = graphslam_backend(self.graph)

                while iteration_continue and residual_continue and converge_continue:
                    graph_opt.solve()

                    prev_residual = residual
                    residual = graph_opt.residual

                    delta_residual = abs((prev_residual - residual) / prev_residual)
                    n_iterations += 1

                    print('**************  Residual = ', residual, ' ***************')
                    residual_continue = (residual > residual_threshold)
                    print('Residual above threshold?', residual_continue)

                    print('************** Iteration = ', n_iterations, ' ***************')
                    iteration_continue = (n_iterations <= lim_iterations)
                    print('Iterations below limit?', iteration_continue)

                    print('********* Delta Residual = ', delta_residual, ' ***************')
                    converge_continue = (delta_residual > delta_threshold)
                    print('Residual still changing?', converge_continue)

                    # reconstruct the graph with these nodes
                    graph_opt = graphslam_frontend(graph_opt)   # Task
                    graph_opt.construct_graph()  # Task
                    graph_opt = graphslam_backend(graph_opt)    # Task

                cpu_end_solver = datetime.now()
                delta = cpu_end_solver - cpu_start_solver
                print('********* Final solution took:', (delta.total_seconds()), 's ***************')



                #show the original graph
                graph_opt = graphslam_backend(graph_init)
                print('Original graph has:')
                print('Poses',graph_init.n)
                print('Landmarks',graph_init.m)
                print('Edges',graph_init.e)


                #show the reduced form
                pose_graph = graph_opt.reduce2pose()
                print('Reduced graph has:')
                print('Poses',pose_graph.n)
                print('Landmarks',pose_graph.m)
                print('Edges',pose_graph.e)

                # shows information vector and matrix
                visualise_flag = True 
                pose_graph = graph_opt.reduce2pose(visualise_flag)
                print('Grey cells indicate information that has been modified through the graph reduction')



                initial_residual = 100 #just needs to be a big number to avoid triggering convergence if the first iteration has large residuals
                initial_flag = True

                residual_threshold = 1E-12 #if result changes by <1
                delta_threshold = 1/1000 #if result changes by <1
                lim_iterations = 20

                n_iterations = 0
                delta_residual = initial_residual
                residual = initial_residual

                visualise_flag = False
                iteration_continue = True 
                residual_continue = True
                converge_continue = True

                # reset the pose graph (full_graph_ should be unaffected by previous calculations)
                pose_graph = graph_opt.reduce2pose(visualise_flag)
                l = graph_opt.n*3

                # if any of the conditions become false, then while loop will exit
                cpu_start_solver = datetime.now()

                while iteration_continue and residual_continue and converge_continue:
                    
                    pose_graph.solve()    
                    
                    prev_residual = residual
                    residual = pose_graph.residual

                    delta_residual = abs((prev_residual - residual) /prev_residual)
                    n_iterations += 1

                    print('**************  Residual = ',residual,' ***************')        
                    residual_continue = (residual > residual_threshold)
                    print('Residual above threshold?',residual_continue)    
                    
                    print('************** Iteration = ',n_iterations,' ***************')
                    iteration_continue = (n_iterations <= lim_iterations)
                    print('Iterations below limit?',iteration_continue)

                    print('********* Delta Residual = ', delta_residual, ' ***************')
                    converge_continue = (delta_residual > delta_threshold)
                    print('Residual still changing?', converge_continue)

                    #reconstruct the graph with these nodes
                    graph_opt.state_vector[0:l] = pose_graph.state_vector # Task
                    graph_opt.state_vector[l:] = Inverse(graph_opt.H[l:,l:])@(graph_opt.b[l:]+graph_opt.H[l:,0:l]@graph_opt.state_vector[0:l])
                    graph_opt = graphslam_frontend(graph_opt)
                    graph_opt.construct_graph()
                    graph_opt = graphslam_backend(graph_opt) 
                    
                    pose_graph = graph_opt.reduce2pose(visualise_flag) # Task

                cpu_end_solver = datetime.now()

                show_information(pose_graph.H,pose_graph.n,3,pose_graph.m,2, matrix_compare = graph_init.H[0:l,0:l])
                    

                delta =  cpu_end_solver - cpu_start_solver
                print('********* Final solution took:',(delta.total_seconds()),'s ***************')                                                                

################################## BACKEND ######################################
 


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