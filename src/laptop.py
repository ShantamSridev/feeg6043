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
from math_feeg6043 import Vector, Matrix, Identity, Inverse, eigsorted, gaussian, l2m, HomogeneousTransformation, polar2cartesian
from model_feeg6043 import ActuatorConfiguration, rigid_body_kinematics, RangeAngleKinematics, TrajectoryGenerate, feedback_control, lidar_scan, graphslam_frontend, graphslam_backend
from plot_feeg6043 import plot_graph

from model_feeg6043 import (
    ActuatorConfiguration,
    rigid_body_kinematics,
    RangeAngleKinematics,
    TrajectoryGenerate,
    feedback_control,
    lidar_scan,
    t2v, v2t
)
from math_feeg6043 import Vector, Matrix, l2m, Inverse, HomogeneousTransformation, polar2cartesian
import joblib
import copy


N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4


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

        ####################### General Parameters ############################

        self.initialise_control = True # False once control gains is initialised 
        
        self.initialise_pose = True # False once the pose is initialised 
        
        self.relative_path = True # False if you want it to be absolute
        
        self.aruco_count = 0
        self.loop_count = 0
        
        self.t = -1
        
        ####################### Motion Model Parameters #######################       
        
        wheel_distance = 0.09
        wheel_diameter = 0.07
        
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) 
        
        ########################## Control Parameters #########################
        
        # Trajectory parameters
        self.velocity = 0.05
        self.acceleration = self.velocity/3
        
        self.turning_radius = 0.35  # meters - minimum turning radius
        # control parameters        

                             ####### Control Gains ######
        
        self.tau_s = 1 # s to remove along track error
        self.L = 0.4 # m distance to remove normal and angular error

        # compute control gains for the initial condition (where robot is stationary)
        self.k_s = 1/self.tau_s  # along track gain
        
        self.v_max = 0.2 # fastest the robot can go
        self.w_max = np.deg2rad(15) # fastest the robot can turn

        ############################## Waypoints ##############################

        #self.northings_path = [0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0] # create a list of waypoints
        #self.eastings_path = [0.0, -0.1, 1.4, 1.5, 0.0, -0.1, 1.4, 1.5, 0.0, -0.1, 1.4, 1.5, 0.0] # create a list of waypoints
        
        self.northings_path = [0.0, 1.5, 1.5, 0.0, 0.0] # create a list of waypoints
        self.eastings_path = [0.0, -0.1, 1.4, 1.5, 0.0] # create a list of waypoints
        
        ########################### Storage Variables #########################

        # Estimated pose - sent to show_laptop
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None

        # measured pose (Aruco)
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
        
        self.state = Vector(5)
        
        ######################## Lidar Parameters #############################
        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0.07 # location of lidar centre in b-frame primary axis
        lidar_yb = 0.0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb)    

        self.corner_pose_northings = None
        self.corner_pose_eastings = None  
        
        #################### Corner Detection Model ###########################
        
        self.gpc_corner = joblib.load('gpc_model.pkl')
        
        ############################ Noises ###################################
        
                        ###### lidar measurement noise ######
        self.sigma_observe = Matrix(2, 2) 
        self.sigma_observe[0, 0] = 0.1**2  # 10% of range
        self.sigma_observe[0, 1] = 0
        self.sigma_observe[1, 0] = np.deg2rad(0.1) ** 2  # 0.1 degree per metre range00
        self.sigma_observe[1, 1] = 0
        

                    ###### Motion model expected noise ######

        self.sigma = Matrix(3,3) #noise in motion model for uncertainty
        self.sigma[0,0]=0.1
        self.sigma[0,1]=0.01
        self.sigma[1,0]=0.01
        self.sigma[1,1]=0.1
        self.sigma[0,2]=0.01
        self.sigma[1,2]=0.01
        self.sigma[2,0]=0.01
        self.sigma[2,1]=0.01
        self.sigma[2,2]=0.1
        
        self.sigma_motion = Matrix(3, 2)
        self.sigma_motion[0, 0] = 0.1**2  # impact of v linear velocity on x           # Task
        self.sigma_motion[0, 1] = np.deg2rad(0.1)**2  # impact of w angular velocity on x
        self.sigma_motion[1, 0] = 0.1**2  # impact of v linear velocity on y
        self.sigma_motion[1, 1] = np.deg2rad(0.3)**2  # impact of w angular velocity on y
        self.sigma_motion[2, 0] = 0.1**2  # impact of v linear velocity on gamma
        self.sigma_motion[2, 1] = np.deg2rad(0.3)**2  # impact of w angular velocity on gamma
                
        self.dp = Vector(3)
        
        ################################ SLAM #################################
        
        self.graph = graphslam_frontend()
        self.graph.anchor(self.sigma/100)
        
        self.p_gt_path = []
        
        ################ Random things that need to be started ################
        
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###############################################################################    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  


    # def find_corner(self, corner, threshold=0.01):
    #     # identify the reference coordinate as the inflection point

    #     # Step 1: Compute slope
    #     slope = np.gradient(corner.data[:, 0])

    #     # Step 2: Compute the second derivative (curvature)
    #     curvature = np.gradient(slope)

    #     # Step 3: Check if criteria is more than threshold
    #     #if np.nanmax(abs(np.gradient(np.gradient(curvature)))) > threshold:
    #     if not np.all(np.isnan(curvature)) and np.nanmax(np.abs(curvature)) > threshold:
    #         # compute index of inflection point
    #         largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))

    #         r = corner.data[largest_inflection_idx, 0]  # Radial distance at the largest curvature
    #         theta = corner.data[largest_inflection_idx, 1]  # Angle at the largest curvature
            
    #         return r, theta, largest_inflection_idx

    #     else:
    #         return None, None, None  # No inflection points found
        
    def find_corner(self, corner, threshold=0.01):
        
        # --- guard against too few points for a second derivative ---
        n = corner.data.shape[0]
        if n < 3:
            # not enough points to compute two gradients
            return None, None, None
        
        y = corner.data[:,0]
        # need at least a few valid points to take two derivatives
        if y.size < 3 or np.isnan(y).all():
            return None, None, None
    
        # only compute on the valid region
        valid = ~np.isnan(y)
        yv = y[valid]
    
        slope     = np.gradient(yv)
        curvature = np.gradient(slope)
        sec_der   = np.gradient(curvature)
    
        # if that second-derivative is still all NaN, bail out
        if np.isnan(sec_der).all():
            return None, None, None
    
        max_curv = np.nanmax(np.abs(sec_der))
        if max_curv > threshold:
            idx_valid = np.nanargmax(np.abs(sec_der))
            idx_orig  = np.where(valid)[0][idx_valid]
            return corner.data[idx_orig,0], corner.data[idx_orig,1], idx_orig
        else:
            return None, None, None
        


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

        # def _fill_nan(self, data):
        #     data_filled = np.copy(data)
        #     mean = np.nanmean(data[:, 0])
        #     for i in range(len(data[:, 1])):
        #         if np.isnan(data[i, 0]):
        #             data_filled[i, 0] = 0
        #         else:
        #             data_filled[i, 0] = data[i, 0] - mean
        #     return data_filled
        
        def _fill_nan(self, data):
            data_filled = np.copy(data)
            col0 = data[:, 0]
        
            # if everything is nan, define mean = 0 (or some default)
            if np.isnan(col0).all():
                mean = 0.0
            else:
                mean = np.nanmean(col0)
        
            for i in range(len(data[:, 1])):
                if np.isnan(data[i, 0]):
                    data_filled[i, 0] = 0
                else:
                    data_filled[i, 0] = data[i, 0] - mean
        
            return data_filled
         
    def true_wheel_speeds_callback(self, msg):
        #print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received
        # print("Received lidar message", msg.header.seq)
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = (msg.header.stamp)  # we want the lidar measurement timestamp here

        self.lidar_data = np.zeros((len(msg.ranges), 2))  # specify length of the lidar data
        self.lidar_data[:, 0] = (msg.ranges)  # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:, 1] = (msg.angles)  # use angles as a placeholder, workout eastings in Task 4
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

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)  # this gives it in northings and eastings

            self.lidar_data[i, 0] = t_em[0].item()  # list of all northings points of each lidar point
            self.lidar_data[i, 1] = t_em[1].item()  # list of all eastings points of each lidar point

        # this filters out any NaN
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

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
        time_to_run = 110
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            r = Rate(1)
            i=0
            while True:
                current_time = datetime.utcnow().timestamp()
                
                if time_to_run > 0 and (current_time - self.start_time) > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
            print('Made it out the loop, yay')
            
            p = Vector(3)
            p[0] = self.measured_pose_northings_m  # Northings
            p[1] = self.measured_pose_eastings_m  # Eastings
            p[2] = self.measured_pose_yaw_rad  # Heading (rad)
            self.p_gt_path.append(p)
            
            self.graph.motion(self.state[:3], self.sigma, Vector(3), final=True)
            print('did final motion bit of graph')
            
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
            print('plotting graph')
            plot_graph(self.graph, self.p_gt_path, H_em, map_ground_truth, map_labels)
            
            print('constructing graph')
            self.graph.construct_graph(visualise_flag=True)
            print('graph constructed')
            
            
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception2: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()

    
    def state_space_control(self):
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
        H_eb = HomogeneousTransformation(self.state[:2],self.state[2])  # body to earth transform
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
        return q
    
    
    def parse_aruco_pose(self,aruco_pose):
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
    
    
    def initialise_position(self):
        self.state[N] = self.measured_pose_northings_m
        self.state[E] = self.measured_pose_eastings_m
        self.state[G] = self.measured_pose_yaw_rad

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
        # path and trajectory are initialised
        
        self.initialise_pose = False
        
    
    def lidar2world_frame(self, r, theta):
        
        x_l, y_l = polar2cartesian(r, theta)
        
        # Convert to environment frame using current robot pose
        H_eb = HomogeneousTransformation(self.state[:2], self.state[2]) 
        t_em = t2v(H_eb.H@self.lidar.H_bl.H@v2t([x_l, y_l]))
        self.corner_pose_northings = t_em[0]
        self.corner_pose_eastings = t_em[1]
        msg = Vector3Stamped()
        msg.header = Header(stamp=time.time())  # or whatever timestamping you're using
        msg.vector.x = float(t_em[0])
        msg.vector.y = float(t_em[1])
        
        # Log the message
        self.datalog.log(msg, topic_name="/detected_corners")
        return t_em
    
    def which_corner(self, northings, eastings):
        # northings = t_em[0]
        # eastings = t_em[1]
        
        if northings < 0.5:
            if eastings < 0.5:
                return 0, True
            if eastings > 1.5:
                return 3, True
        elif northings > 1.5:
            if eastings < 0.5:
                return 1, True
            if eastings > 1.5:
                return 2, True
        return None, False
        
        

    def infinite_loop(self):
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            self.parse_aruco_pose(aruco_pose)

        ###### wait for the first sensor info to initialize the pose ##########
    
        if self.initialise_pose is True and aruco_pose is not None:
            self.initialise_position()

        if self.initialise_pose is not True and self.measured_wheelrate_right is not None and self.measured_wheelrate_left is not None:  
            self.loop_count += 1

            
            ################ Determine the time step ##########################
            t_now = datetime.utcnow().timestamp()        
        
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop
            
            ############## Get Wheel Speeds & Motion Model ####################
            
            q_measured = Vector(2)            
            q_measured[0] = self.measured_wheelrate_right # wheel rate rad/s (measured)
            q_measured[1] = self.measured_wheelrate_left # wheel rate rad/s (measured)
            u = self.ddrive.fwd_kinematics(q_measured)    
            
            H_eb = HomogeneousTransformation(self.state[:2],self.state[2])
            
            statecopy = copy.copy(self.state)
            sigmacopy = copy.copy(self.sigma)            
            
            self.state, self.sigma, self.dp, p_gt = rigid_body_kinematics(self.state[:3], u, dt=dt, sigma_motion=self.sigma_motion, sigma_xy=self.sigma)
            
            ######################## Lidar ####################################
            
            if self.lidar_data is not None:
                observation, _ = lidar_scan(self.state[:3], self.lidar_data, self.lidar, self.sigma_observe)
                
                if (observation is not None or not np.isnan(observation.data_filled[:, 0]).any()):
                    new_observation = self.GPC_input_output(observation, None)
                    new_observation.label = self.gpc_corner.classes_[np.argmax(self.gpc_corner.predict_proba([new_observation.data_filled[:, 0]]))]
                    
                    if new_observation.label == "corner":
                        r, theta, idx = self.find_corner(new_observation)
                        if r is not None:
                            # Convert polar coordinates to cartesian in sensor frame
                            # t_em = [[Northings_pose],[Eastings_pose]]
                            t_em = self.lidar2world_frame(r, theta)
                            t_lm = Vector(2)
                            t_lm[0], t_lm[1] = polar2cartesian(r, theta)
                            
                            landmark_id, useful = self.which_corner(t_em[0], t_em[1])
                            if useful:
                                self.graph.observation(t2v(H_eb.H@self.lidar.H_bl.H@v2t(t_lm)), self.sigma_observe, landmark_id, t_lm)# The sigma in this should be an error related to lidar position and distance
            ########################### Control ###############################
            
            self.graph.motion(statecopy[:3], sigmacopy, self.dp, final=False)
            
            p = Vector(3)
            p[0] = self.measured_pose_northings_m  # Northings
            p[1] = self.measured_pose_eastings_m  # Eastings
            p[2] = self.measured_pose_yaw_rad  # Heading (rad)
            self.p_gt_path.append(p)
            
            q_desired = self.state_space_control()
            
            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q_desired[0,0]  # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q_desired[1,0]  # Left wheelspeed rad/s

            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
            
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")            
            
            ###################################################################
            self.est_pose_northings_m = self.state[N,0]
            self.est_pose_eastings_m = self.state[E,0]
            self.est_pose_yaw_rad = self.state[G,0]

            # Save Values / send Commands       
            msg = self.pose_parse([datetime.utcnow().timestamp(), self.state[N,0], self.state[E,0], 0, 0, 0, self.state[G,0]])
            self.datalog.log(msg, topic_name="/est_pose")



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
    
    args.simulation = True
    
    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
