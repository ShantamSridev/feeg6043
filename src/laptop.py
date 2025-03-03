
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
# add more libraries here
from model_feeg6043 import ActuatorConfiguration
from math_feeg6043 import Vector
from model_feeg6043 import rigid_body_kinematics
from model_feeg6043 import RangeAngleKinematics
from model_feeg6043 import TrajectoryGenerate
from math_feeg6043 import l2m
from model_feeg6043 import feedback_control
from math_feeg6043 import Inverse, HomogeneousTransformation

# Particle Filter Imports
from model_feeg6043 import Particles, initialise_particle_distribution, discrete_motion_model, pf_measurement_probability, pf_normalise_weights, pf_update, neff, pf_resample, kde_probability, Measurement, systematic_resample

class LaptopPilot:
    def __init__(self, simulation):
        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 21,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
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
        # 
        self.initialise_pose = True #False
        # 
        # modelling parameters
        wheel_distance =  0.09# measure this 
        wheel_diameter =  0.07# 9easure this
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this     
 
        # waypoints
        self.northings_path = [0,1.25,1.25,0,0,1.25,1.25,0,0,1.25,1.25,0,0]
        self.eastings_path =  [0,0,1.25,1.25,0,0,1.25,1.25,0,0,1.25,1.25,0]
        # self.northings_path = [0,2]
        # self.eastings_path =  [0,0]
        self.relative_path = True # False if you want it absolute       

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
        lidar_xb = 0.1 # location of lidar centre in b-frame primary axis
        lidar_yb = 0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)

        # velocity, accelerations and turning radius
        self.v=0.1 #velocity #0.12
        self.a=0.1/3 #acceleration 
        self.turning_radius = 0.3 #turning radius #0.6

        # control parameters
        self.tau_s = 0.3 # s to remove along track error #1
        self.L = 0.3 # m distance to remove normal and angular error #0.2m
        self.v_max = 0.13 # fastest the robot can go #0.18
        self.w_max = np.deg2rad(20) # fastest the robot can turn

        self.initialise_control = True # False once control gains is initialised
        ###############################################################        

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

        # Particle Filter Initialization
        self.N_particles = 150  # Number of particles (tune as needed)
        self.particles = Particles(self.N_particles)
        initialise_particle_distribution(self.particles, centre=[0, 0], radius=0.25, heading=0) #initialize particles
        self.process_noise = [0.1**2, 0.1**2, np.deg2rad(5)**2]  # Process noise for [x, y, yaw] (tune as needed)
        self.measurement_noise = [0.01, 0.01, np.deg2rad(5)]  # Measurement noise for [x, y, yaw] (tune as needed)
        self.resample_threshold = 0.5 * self.N_particles  # Resampling threshold (tune as needed)
        self.jitter = 0.05
        self.sigma_resolution = 0.5  # KDE resolution (tune as needed)
        self.sampling_resolution = 100  # KDE sampling resolution (tune as needed)




                    
    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)

        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        #print("Starting lidar message")
        print("Received lidar message", msg.header.seq)        
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset
        self.lidar_timestamp_s = msg.header.stamp #msg.??.?? #we want the lidar measurement timestamp here

        self.lidar_data = np.zeros((len(msg.angles), 2)) #specify length of the lidar data
        self.lidar_data[:,0] = msg.angles # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:,1] = msg.ranges # use angles as a placeholder, workout eastings in Task 4

        self.datalog.log(msg, topic_name="/lidar")

        p_eb = Vector(3)
        # TODO replace with estimates
        #robot pose northings (see Task 3)
        p_eb[0] = self.est_pose_northings_m
        p_eb[1] = self.est_pose_eastings_m #robot pose eastings (see Task 3)
        p_eb[2] = self.est_pose_yaw_rad #robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                    
        z_lm = Vector(2)        
        # for each map measurement
        for i in range(len(msg.ranges)):
            #print( f"for loop" )
            #print( f"msg? {msg}" )
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
                
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm) # see tutotial

            self.lidar_data[i,0] = t_em[0]
            self.lidar_data[i,1] = t_em[1]
            #print( f"lidar result {t_em}" )
        # this filters out any 
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

        msg.header.stamp += self.sim_time_offset
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
            print(
                "Received position update from",
                datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset,
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
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.est_pose_northings_m #offset by current northings
                self.eastings_path[i] += self.est_pose_eastings_m #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            C = l2m([self.northings_path, self.eastings_path])        
            self.path = TrajectoryGenerate(C[:,0],C[:,1])        
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.v, self.a) #velocity and acceleration
            self.path.turning_arcs(self.turning_radius) #turning radius
            self.path.wp_id=0 #initialises the next waypoint

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
            print("Exception: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()

    def infinite_loop(self):

        ################### Motion Model ##############################
        # convert true wheel speeds in to twist
        q = Vector(2)    
        q[0] = self.measured_wheelrate_right
        q[1] = self.measured_wheelrate_left
        u = self.ddrive.fwd_kinematics(q)
       
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            
            # converts aruco date to zeroros PoseStamped format
            msg = self.pose_parse(aruco_pose, aruco = True)

            # reads sensed pose for local use 
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrappig

            # logs the data            
            self.datalog.log(msg, topic_name="/aruco")


            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True:
                # Initialize particle distribution with the first measured pose
                initialise_particle_distribution(self.particles,
                                                centre=[self.measured_pose_northings_m, self.measured_pose_eastings_m],
                                                radius=0.1,  # Reduce radius for better initial accuracy
                                                heading=self.measured_pose_yaw_rad)
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
                
                # generate the trajectory
                self.generate_trajectory()

                # path and tragectory are initialised
                self.initialise_pose = False 

        if self.initialise_pose != True:  

            #actuator forward kinematic to determine u

            
            #determine the time step
            t_now = datetime.utcnow().timestamp()        
                    
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            # Particle Filter Prediction
            discrete_motion_model(self.particles, self.est_pose_yaw_rad, u, dt, self.process_noise)
            
        

            if aruco_pose is not None: 
                # Measurement Update
                measurement = Measurement()
                measurement.northings = self.measured_pose_northings_m
                measurement.eastings = self.measured_pose_eastings_m
                measurement.northings_std = self.measurement_noise[0]  # Tune noise
                measurement.eastings_std = self.measurement_noise[1]  # Tune noise
                measurement.gamma = self.measured_pose_yaw_rad
                measurement.gamma_std = self.measurement_noise[2]  # Tune noise

                print("hello 1")

                pf_update(self.particles, measurement)

                print("hello 2")

                # Resampling
                if neff(self.particles) < self.resample_threshold:
                    print("hello 3")
                    pf_resample(self.particles, self.jitter)

                print("hello 4")

            # State Estimation
            est_northings, est_eastings = kde_probability(self.particles, sigma_resolution=self.sigma_resolution, sampling_resolution=self.sampling_resolution)
            self.est_pose_northings_m = est_northings
            self.est_pose_eastings_m = est_eastings
            self.est_pose_yaw_rad = np.arctan2(np.mean(np.sin(self.particles.gamma)), np.mean(np.cos(self.particles.gamma))) #wrapped mean

            # Recreate p_robot from the particle filter's estimated pose
            p_robot = Vector(3)
            p_robot[0, 0] = self.est_pose_northings_m
            p_robot[1, 0] = self.est_pose_eastings_m
            p_robot[2, 0] = self.est_pose_yaw_rad

            # logs the data            
            msg = self.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/est_pose")


            #################### Trajectory sample #################################    

            # feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(self.t, p_robot, self.turning_radius) # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)
        
            # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot #compute difference between reference and estimated pose in the $e$-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
            self.H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2]) #homogeneous transformation from $e$ to $b$ frame
            ds = Inverse(self.H_eb.H_R) @ dp # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)

            # compute control gains for the initial condition (where the robot is stationalry)
            self.k_s = 1/self.tau_s #ks
            if self.initialise_control == True:
                self.k_n = (2*u_ref[0])/(self.L**2) #kn
                self.k_g = u_ref[0]/self.L #kg
                self.initialise_control = False # maths changes a bit after the first iteration

            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = u_ref + du # combine feedback and feedforward control twist components

            # update control gains for the next timestep
            self.k_n = 2*(u[0])/(self.L**2) #kn
            self.k_g = u[0]/self.L #kg

            # ensure within performance limitations
            if u[0] > self.v_max: u[0] = self.v_max
            if u[0] < -self.v_max: u[0] = -self.v_max
            if u[1] > self.w_max: u[1] = self.w_max
            if u[1] < -self.w_max: u[1] = -self.w_max
            
            # actuator commands
            q = self.ddrive.inv_kinematics(u) #convert twist to wheel rates
        
        # > Think < #
        ################################################################################
        #  TODO: Implement your state estimation

        ################################################################################
        #  TODO: Implement your controller here                                        #

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0,0]  # Right wheel 1 rev/s = 1*pi rad/s
            wheel_speed_msg.vector.y = q[1,0]  # Left wheel 1 rev/s = 2*pi rad/s

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
