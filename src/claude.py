import numpy as np
import copy
from datetime import datetime
import time
import g2o
from math_feeg6043 import Vector, Matrix, Identity, Inverse, HomogeneousTransformation

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


# Example of how to integrate with LaptopPilot
def setup_slam_with_laptop_pilot(laptop_pilot):
    """
    Set up and return a SLAM system integrated with the LaptopPilot.
    
    Args:
        laptop_pilot: An instance of LaptopPilot
        
    Returns:
        An instance of RobotSLAM
    """
    # Create the SLAM system
    robot_slam = RobotSLAM(laptop_pilot)
    
    # Modify the LaptopPilot's infinite_loop method to update SLAM
    original_infinite_loop = laptop_pilot.infinite_loop
    
    def infinite_loop_with_slam():
        # Run the original control loop
        original_infinite_loop()
        
        # Update SLAM
        robot_slam.update()
    
    # Replace the method
    laptop_pilot.infinite_loop = infinite_loop_with_slam
    
    return robot_slam