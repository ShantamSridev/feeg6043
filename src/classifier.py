import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import joblib
from model_feeg6043 import RangeAngleKinematics, lidar_scan
from math_feeg6043 import Vector, Matrix, l2m

import copy
import argparse
from datetime import datetime
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import (
    LaserScan,
    Vector3Stamped,
    Pose,
    PoseStamped,
    Header,
    Quaternion,
)
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

# add more libraries here
from model_feeg6043 import (
    ActuatorConfiguration,
    rigid_body_kinematics,
    RangeAngleKinematics,
    TrajectoryGenerate,
    feedback_control,
    lidar_scan,
    graphslam_frontend,
    graphslam_backend,
    t2v, v2t
)
from math_feeg6043 import Vector, Matrix, l2m, Inverse, HomogeneousTransformation, polar2cartesian

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

    @classmethod
    def create_training_data(cls, env_map, lidar, sigma_observe):
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
                    new_observation = cls(observation, None)

                    threshold = 0.001  # can reduce to make less conservative
                    z_lm[0], z_lm[1], loc = cls.find_corner(new_observation, threshold)

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
                new_observation = cls(observation, None)
                threshold = 0.01  # can reduce to make less conservative
                _, _, loc = cls.find_corner(new_observation, threshold)

                # if no corner is found, register as a not corner for the training
                if loc is None:
                    new_observation.label = "not corner"
                    corner_training.append(new_observation)
        return corner_training
    
    @staticmethod
    def find_corner(corner, threshold=0.01):
        # identify the reference coordinate as the inflection point

        # Step 1: Compute slope
        slope = np.gradient(corner.data[:, 0])

        # Step 2: Compute the second derivative (curvature)
        curvature = np.gradient(slope)

        # Step 3: Check if criteria is more than threshold
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

def train_and_save_model():
    # Setup lidar parameters
    lidar_xb = 0.07  # location of lidar centre in b-frame primary axis
    lidar_yb = 0  # location of lidar centre in b-frame secondary axis
    lidar = RangeAngleKinematics(
        lidar_xb,
        lidar_yb,
        distance_range=[0.1, 2],
        scan_fov=np.deg2rad(120),
        n_beams=30,
    )

    # Setup observation noise model
    sigma_observe = Matrix(2, 2)
    sigma_observe[0, 0] = 0.1**2  # 10% of range
    sigma_observe[0, 1] = 0
    sigma_observe[1, 0] = np.deg2rad(0.1) ** 2  # 0.1 degree per metre range
    sigma_observe[1, 1] = 0

    # Create environment map
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

    # Create training data
    corner_training = GPC_input_output.create_training_data(
        environment_map, lidar, sigma_observe
    )

    # Prepare training data
    X_train = np.full(
        (len(corner_training), corner_training[0].data_filled[:, 0].size),
        None,
    )
    y_train = np.full(len(corner_training), None, dtype=object)

    # Populate training data
    for i in range(len(corner_training)):
        X_train[i, :] = corner_training[i].data_filled[:, 0]
        y_train[i] = corner_training[i].label

    # Train the classifier
    kernel = 1.0 * RBF(1.0)
    gpc_corner = GaussianProcessClassifier(
        kernel=kernel, random_state=0
    ).fit(X_train, y_train)

    # Save the trained model
    joblib.dump(gpc_corner, 'corner_classifier.joblib')
    print("Model trained and saved successfully!")

def load_model():
    """Load the trained model from file"""
    try:
        return joblib.load('corner_classifier.joblib')
    except FileNotFoundError:
        print("No trained model found. Please run train_and_save_model() first.")
        return None

if __name__ == "__main__":
    train_and_save_model()