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
from joblib import load
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control, t2v, v2t, lidar_scan


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

        self.corner_clf = load("corner_classifier.joblib")
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
        self.northings_path = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] # create a list of waypoints
        self.eastings_path = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0] # create a list of waypoints
        
        
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
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb, n_beams=360)    


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

        
        #################### Noise Attributes #########################

        # position uncertainty
        self.sigma_xy = Matrix(3, 3)

        # motion model linear noise due to v and w
        self.sigma_motion = Matrix(3, 2)
        self.sigma_motion[0, 0] = 0.1**2  # impact of v linear velocity on x
        self.sigma_motion[0, 1] = (
            np.deg2rad(0.1) ** 2
        )  # impact of w angular velocity on x

        self.sigma_motion[1, 0] = 0.1**2  # impact of v linear velocity on y
        self.sigma_motion[1, 1] = (
            np.deg2rad(0.1) ** 2
        )  # impact of w angular velocity on y

        self.sigma_motion[2, 0] = 0.1**2  # impact of v linear velocity on gamma
        self.sigma_motion[2, 1] = (
            np.deg2rad(0.1) ** 2
        )  # impact of w angular velocity on gamma

        # observation model linear noise with range
        self.sigma_observe = Matrix(2, 2)
        self.sigma_observe[0, 0] = 0.1**2  # 10% of range
        self.sigma_observe[0, 1] = 0
        self.sigma_observe[1, 0] = np.deg2rad(0.1) ** 2  # 0.1 degree per metre range
        self.sigma_observe[1, 1] = 0

        # anchor constraint, matrix must be invertable
        self.sigma_anchor = Matrix(3, 3)
        self.sigma_anchor[0, 0] = 0.1
        self.sigma_anchor[0, 1] = 0.01
        self.sigma_anchor[1, 0] = 0.01
        self.sigma_anchor[1, 1] = 0.1
        self.sigma_anchor[0, 2] = 0.01
        self.sigma_anchor[1, 2] = 0.01
        self.sigma_anchor[2, 0] = 0.01
        self.sigma_anchor[2, 1] = 0.01
        self.sigma_anchor[2, 2] = 0.1

        ######################## Graph SLAM ###########################
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

    # def lidar_scan(self, p_eb, environment_map, lidar, sigma_observe):
    #     """ Gets observations from the robot pose and the map.
        
    #     Parameters
    #     -----------
    #     p_eb = [3 x 1] Vector of floats
    #         The robot pose in the e frame as
    #             [[x
    #             y
    #             gamma]]
    #     environment_map = [l x 2] Matrix of floats
    #         A list observable element in the robots surroundings
    #         where each element is a transposed 2x1 vector of x, y coordinates in the environment frame
    #             [[x
    #             y]].T
    #     lidar = RangeAngleKinematics class
    #         A member of the RangeAngleKinematics that determines sensor properties
    #         range [min, max] in metres, and scan_fov in radians
    #         e.g., RangeAngleKinematics(x_bl, y_bl, distance_range = [0.1, 1], scan_fov = np.deg2rad(90)) 
            
    #     sigma_observe = [2 x 2] Matrix of floats
    #         Observation model for linear noise with range
    #         [[sigma_xr sigma_xtheta]
    #         [sigma_yr sigma_ytheta]]
            
    #     Returns
    #     -------
    #     observations: [n_beams x 2] Matrix of floats
    #         The observations as an array of (range, bearing) points. 
    #     observations_std: [n_beams x 1] Matrix of floats
    #         The observations set as an array of (range_std, bearing) points.         
    #     """

    #     m_range = [] # range and bearing to map elements
    #     m_bearing = []    
        
    #     z_range = [] # range and bearing measurements (resamples at resolution and with some noise)
    #     z_range_std = []
    #     z_bearing = []        
        
    #     for i in range(len(environment_map)):
    #         t_em = environment_map[[i],:].T
    #         z_lm = lidar.loc_to_rangeangle(p_eb, t_em)
    #         m_range.append(z_lm[0])
    #         m_bearing.append(z_lm[1])   
        
    #     #sampling the map
    #     bearing_resolution = np.linspace(-lidar.scan_fov/2,lidar.scan_fov/2,lidar.n_beams)

    #     #picks the nearest map entity and adds range based noise. Bearing noise is simulated to be half the beam width (i.e., (fov/n_beams)) 
    #     for theta in bearing_resolution:        
    #         if ~np.all(np.isnan(m_bearing)):
    #             i = np.nanargmin(abs((theta -m_bearing)))
            
    #             if abs(theta - m_bearing[i]) < lidar.scan_fov/(2*lidar.n_beams):
    #                 z_range.append(m_range[i]+np.random.normal(0,m_range[i]*sigma_observe[0,0],1))
    #                 z_range_std.append(abs(m_range[i]*sigma_observe[0,0]))
    #                 z_bearing.append(theta)
    #             else:
    #                 z_range.append(np.nan)
    #                 z_range_std.append(np.nan)
    #                 z_bearing.append(theta)
    #         else:
    #             z_range.append(np.nan)
    #             z_range_std.append(np.nan)
    #             z_bearing.append(theta)
    
    #     observations = l2m([z_range,z_bearing]) 
    #     observations_std = l2m([z_range_std, z_bearing]) 
        
    #     return observations, observations_std


    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received
        # print("Received lidar message", msg.header.seq)
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = (
            msg.header.stamp
        )  # we want the lidar measurement timestamp here

        self.lidar_data = np.zeros(
            (len(msg.ranges), 2)
        )  # specify length of the lidar data
        self.lidar_data[:, 0] = (
            msg.ranges
        )  # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:, 1] = (
            msg.angles
        )  # use angles as a placeholder, workout eastings in Task 4
        self.datalog.log(msg, topic_name="/lidar")

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.measured_pose_northings_m
        p_eb[1] = self.measured_pose_eastings_m
        p_eb[2] = self.measured_pose_yaw_rad

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))

        z_lm = Vector(2)
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]

        # this filters out any NaN
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]


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

    def create_training_data(self, env_map, lidar, sigma_observe):
        # decide some random position and angular offsets to make sure the training data is varied
        pos_noise_std = 0.1
        heading_noise_std = 10

        # create a containor to store the GPC training data
        corner_training = []
        p = Vector(3)
        z_lm = Vector(2)

        for dist in np.arange(0.1, 0.5, 0.2):
            for i in range(40):
                # determine basic pose for each corner
                if i <= 10:  # southwest corner
                    p[0] = 0.0 + dist
                    p[1] = 0.0 + dist
                    p[2] = np.deg2rad(225)
                elif i <= 20:  # northwest corner
                    p[0] = 2.0 - dist
                    p[1] = 0.0 + dist
                    p[2] = np.deg2rad(315)
                elif i <= 30:  # northeast corner
                    p[0] = 2.0 - dist
                    p[1] = 2.0 - dist
                    p[2] = np.deg2rad(45)
                else:
                    p[0] = 0.0 + dist
                    p[1] = 2.0 - dist
                    p[2] = np.deg2rad(135)

                # add random offsets
                p[0] += np.random.normal(-pos_noise_std, pos_noise_std)
                p[1] += np.random.normal(-pos_noise_std, pos_noise_std)
                p[2] += np.deg2rad(
                    np.random.normal(-heading_noise_std, heading_noise_std)
                )

                # compute observations with noise
                observation, _ = lidar_scan(p, env_map, lidar, sigma_observe)
                if (
                    observation is not None
                    or not np.isnan(observation.data_filled[:, 0]).any()
                ):
                    # check if it is a corner with the inflection point
                    new_observation = self.GPC_input_output(observation, None)

                    threshold = 0.001  # can reduce to make less conservative
                    z_lm[0], z_lm[1], loc = self.find_corner(new_observation, threshold)

                    # if the bepoke model says returns a location, add to training data
                    if loc is not None:
                        # label corner and add to corner training set
                        new_observation.label = "corner"
                        new_observation.ne_representative = z_lm
                        # print('Map observation made at, Northings = ',new_observation.ne_representative[0],'m, Eastings =',new_observation.ne_representative[1],'m')
                        corner_training.append(new_observation)

        # decide some random position and angular offsets to make sure the training data is varied
        for i in range(40):
            # determine basic pose for each wall
            if i <= 10:  # west wall
                p[0] = 0.8
                p[1] = 0.4
                p[2] = np.deg2rad(0)
            elif i <= 20:  # north wall
                p[0] = 1.6
                p[1] = 0.8
                p[2] = np.deg2rad(90)
            elif i <= 30:  # east
                p[0] = 1.2
                p[1] = 1.6
                p[2] = np.deg2rad(180)
            else:
                p[0] = 0.4
                p[1] = 1.2
                p[2] = np.deg2rad(270)

            # add random offsets
            p[0] += np.random.normal(-pos_noise_std, pos_noise_std)
            p[1] += np.random.normal(-pos_noise_std, pos_noise_std)
            p[2] += np.deg2rad(np.random.normal(-heading_noise_std, heading_noise_std))

            # compute observations with noise
            observation, _ = lidar_scan(p, env_map, lidar, sigma_observe)

            if (
                observation is not None
                or not np.isnan(observation.data_filled[:, 0]).any()
            ):
                # check if it is a corner with the inflection point
                new_observation = self.GPC_input_output(observation, None)
                threshold = 0.01  # can reduce to make less conservative
                _, _, loc = self.find_corner(new_observation, threshold)

                # if no corner is found, register as a not corner for the training
                if loc is None:
                    new_observation.label = "not corner"
                    corner_training.append(new_observation)
        return corner_training

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
        p = rigid_body_kinematics(p,u,dt)    

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

            # get current time and determine timestep
            self.t_prev = datetime.utcnow().timestamp() #initialise the time
            self.t = 0 #elapsed time
            time.sleep(0.1) #wait for approx a timestep before proceeding
            
            # Generate trajectory after initializing pose
            self.generate_trajectory()


            #TRAINING LIDAR CLASSIFIER HERE
            # train Gaussian Process Classifier
            m_x = []
            m_y = []
            for x in np.arange(0, 2, 0.01):
                m_x.append(x)
                m_y.append(0)  # west wall
            for x in np.arange(0, 2, 0.01):
                m_x.append(x)
                m_y.append(2)  # east wall
            for y in np.arange(0, 2, 0.01):
                m_x.append(0)
                m_y.append(y)  # south wall
            for y in np.arange(0, 2, 0.01):
                m_x.append(2)
                m_y.append(y)  # north wall

            environment_map = l2m([m_x, m_y])
            corner_training = self.create_training_data(
                environment_map, self.lidar, self.sigma_observe
            )
            # for i in range(len(corner_training)):
            #     print(
            #         "Entry:",
            #         i,
            #         ", Class",
            #         corner_training[i].label,
            #         ", Size",
            #         corner_training[i].data_filled[:, 0].size,
            #     )
            #     print("Data", corner_training[i].data_filled[:, 0])
            # preallocate memory for the training data, inputs are each scan, outputs are the class
            X_train = np.full(
                (len(corner_training), corner_training[0].data_filled[:, 0].size),
                None,
            )
            y_train = np.full(len(corner_training), None, dtype=object)

            # populate with the training data
            for i in range(len(corner_training)):
                X_train[i, :] = corner_training[i].data_filled[:, 0]
                y_train[i] = corner_training[i].label

            # train the classifier
            kernel = 1.0 * RBF(1.0)
            self.gpc_corner = GaussianProcessClassifier(
                kernel=kernel, random_state=0
            ).fit(X_train, y_train)

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
            observation, _ = lidar_scan(
                self.state, self.lidar_data, self.lidar, self.sigma_observe
            )
            if (
                observation is not None
                or not np.isnan(observation.data_filled[:, 0]).any()
            ):
                new_observation = self.GPC_input_output(observation, None)
                new_observation.label = self.gpc_corner.classes_[
                    np.argmax(
                        self.gpc_corner.predict_proba(
                            [new_observation.data_filled[:, 0]]
                        )
                    )
                ]
                if new_observation.label == "corner":
                    print(
                        f"#######################\n\n CORNER DETECTED at {self.state}  \n\n#######################"
                    )
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