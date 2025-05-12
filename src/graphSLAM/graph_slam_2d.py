class GraphSLAM2D:
    def __init__(self, verbose=False) -> None:
        '''
        GraphSLAM in 2D with G2O
        
        GraphSLAM is a graph-based SLAM (Simultaneous Localization and Mapping) algorithm.
        This implementation uses the g2o (General Graph Optimization) framework to solve the
        optimization problem in 2D space.
        
        Parameters:
            verbose (bool): Whether to print detailed information during operations
        '''
        # Initialize the sparse optimizer from g2o, which handles the graph optimization
        self.optimizer = g2o.SparseOptimizer()
        
        # Set up the solver - BlockSolverX is a general-purpose solver for SE2/SE3 problems
        # LinearSolverDenseX uses dense matrix operations for solving linear systems
        self.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        
        # Levenberg-Marquardt algorithm is used for non-linear optimization
        # It's a combination of gradient descent and Gauss-Newton methods
        self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver)
        
        # Attach the algorithm to the optimizer
        self.optimizer.set_algorithm(self.algorithm)

        # Counter for vertices (poses and landmarks) added to the graph
        self.vertex_count = 0
        
        # Counter for edges (constraints) added to the graph
        self.edge_count = 0
        
        # Flag to control verbosity of output
        self.verbose = verbose

    def vertex_pose(self, id):
        '''
        Get position of vertex by id
        
        Parameters:
            id (int): Vertex identifier
            
        Returns:
            g2o.SE2: The 2D pose (position and orientation) of the vertex
        '''
        return self.optimizer.vertex(id).estimate()

    def vertex(self, id):
        '''
        Get vertex by id
        
        Parameters:
            id (int): Vertex identifier
            
        Returns:
            g2o.VertexSE2 or g2o.VertexPointXY: The vertex object
        '''
        return self.optimizer.vertex(id)

    def edge(self, id):
        '''
        Get edge by id
        
        Parameters:
            id (int): Edge identifier
            
        Returns:
            g2o.EdgeSE2 or g2o.EdgeSE2PointXY: The edge object representing a constraint
        '''
        return self.optimizer.edge(id)

    def add_fixed_pose(self, pose, vertex_id=None):
        '''
        Add fixed pose to the graph
        
        This adds a pose vertex that will not be adjusted during optimization,
        typically used for the initial pose to anchor the map.
        
        Parameters:
            pose (g2o.SE2): Initial pose (x, y, theta)
            vertex_id (int, optional): Custom ID for the vertex. If None, use vertex_count
        '''
        # Create a new SE2 vertex (for 2D pose - x, y, orientation)
        v_se2 = g2o.VertexSE2()
        
        # Assign an ID to the vertex, either custom or next available
        if vertex_id is None:
            vertex_id = self.vertex_count
        v_se2.set_id(vertex_id)
        
        # Print debug info if verbose mode is enabled
        if self.verbose:
            print("Adding fixed pose vertex with ID", vertex_id)
            
        # Set the initial estimate for this vertex
        v_se2.set_estimate(pose)
        
        # Mark this vertex as fixed (won't be adjusted during optimization)
        v_se2.set_fixed(True)
        
        # Add the vertex to the graph optimizer
        self.optimizer.add_vertex(v_se2)
        
        # Increment vertex counter
        self.vertex_count += 1

    def add_odometry(self, northings, eastings, heading, information):
        '''
        Add odometry to the graph
        
        Creates a new pose vertex and connects it to the previous pose with an odometry edge,
        representing robot movement between two consecutive poses.
        
        Parameters:
            northings (float): X-coordinate in 2D space (forward direction)
            eastings (float): Y-coordinate in 2D space (right direction)
            heading (float): Orientation angle in radians
            information (numpy.ndarray): 3x3 information matrix representing measurement certainty
                                         (inverse of covariance matrix)
        '''
        # Get all vertices from the optimizer
        vertices = self.optimizer.vertices()
        
        # Find the last pose vertex ID to connect the new pose to it
        if len(vertices) > 0:
            # Filter vertices to get only SE2 type (pose vertices)
            last_id = [v for v in vertices if type(vertices[v]) == g2o.VertexSE2][0]
            print("Last id is", last_id)
        else:
            # Cannot add odometry without at least one existing pose
            raise ValueError("There is no previous pose, have you forgot to add a fixed initial pose?")
        
        # Create a new pose vertex
        v_se2 = g2o.VertexSE2()
        
        if self.verbose:
            print("Adding pose vertex", self.vertex_count)
            
        # Set the ID for the new vertex
        v_se2.set_id(self.vertex_count)
        
        # Create an SE2 transformation from the provided coordinates and heading
        pose = g2o.SE2(northings, eastings, heading)
        
        # Set the initial estimate for this vertex
        v_se2.set_estimate(pose)
        
        # Add the vertex to the graph optimizer
        self.optimizer.add_vertex(v_se2)
        
        # Create an edge to connect this pose to the previous one
        e_se2 = g2o.EdgeSE2()
        
        # Connect edge to the previous pose vertex (vertex 0 in the edge)
        e_se2.set_vertex(0, self.vertex(last_id))
        
        # Connect edge to the current pose vertex (vertex 1 in the edge)
        e_se2.set_vertex(1, self.vertex(self.vertex_count))
        
        # Set the relative transformation (measurement) between poses
        e_se2.set_measurement(pose)
        
        # Set the information matrix (certainty of the measurement)
        e_se2.set_information(information)
        
        # Add the edge to the graph optimizer
        self.optimizer.add_edge(e_se2)
        
        # Increment counters
        self.vertex_count += 1
        self.edge_count += 1
        
        if self.verbose:
            print("Adding SE2 edge between", last_id, self.vertex_count-1)

    def add_landmark(self, x, y, information, pose_id, landmark_id=None):
        '''
        Add landmark to the graph
        
        Creates a landmark vertex and connects it to a pose with an observation edge,
        representing a sensor measurement of a landmark from a specific pose.
        
        Parameters:
            x (float): Relative x-coordinate of landmark in pose's reference frame
            y (float): Relative y-coordinate of landmark in pose's reference frame
            information (numpy.ndarray): 2x2 information matrix for the measurement
            pose_id (int): ID of the pose vertex from which the landmark was observed
            landmark_id (int, optional): Custom ID for the landmark. If None, use vertex_count
        '''
        # Store the relative measurement from the pose to the landmark
        relative_measurement = np.array([x, y])
        
        # Verify that the provided pose_id corresponds to a valid pose vertex
        if type(self.optimizer.vertex(pose_id)) != g2o.VertexSE2:
            raise ValueError("The pose_id that you have provided does not correspond to a VertexSE2")
        
        # Get the global transformation of the pose
        trans0 = self.optimizer.vertex(pose_id).estimate()
        
        # Transform the relative measurement to global coordinates
        measurement = trans0 * relative_measurement
        
        print(relative_measurement, measurement)
        
        # Create a new landmark vertex if landmark_id is not provided
        if landmark_id is None:
            landmark_id = self.vertex_count
            # Create a 2D point vertex for the landmark
            v_pointxy = g2o.VertexPointXY()
            
            # Set the initial global position estimate for the landmark
            v_pointxy.set_estimate(measurement)
            
            # Set the ID for the landmark vertex
            v_pointxy.set_id(landmark_id)
            
            if self.verbose:
                print("Adding landmark vertex", landmark_id)
                
            # Add the landmark vertex to the graph optimizer
            self.optimizer.add_vertex(v_pointxy)
            
            # Increment vertex counter
            self.vertex_count += 1
            
        # Create an edge connecting the pose to the landmark
        e_pointxy = g2o.EdgeSE2PointXY()
        
        # Connect edge to the pose vertex (vertex 0 in the edge)
        e_pointxy.set_vertex(0, self.vertex(pose_id))
        
        # Connect edge to the landmark vertex (vertex 1 in the edge)
        e_pointxy.set_vertex(1, self.vertex(landmark_id))
        
        # Increment edge counter
        self.edge_count += 1
        
        # Set the relative measurement between pose and landmark
        e_pointxy.set_measurement(relative_measurement)
        
        # Set the information matrix (certainty of the measurement)
        e_pointxy.set_information(information)
        
        # Add the edge to the graph optimizer
        self.optimizer.add_edge(e_pointxy)
        
        if self.verbose:
            print("Adding landmark edge between", pose_id, landmark_id)

    def optimize(self, iterations=10, verbose=None):
        '''
        Optimize the graph
        
        Runs the optimization algorithm to find the best configuration of poses and landmarks
        that minimizes the overall error in the graph.
        
        Parameters:
            iterations (int): Maximum number of iterations for the optimizer
            verbose (bool, optional): Whether to print detailed optimization information.
                                     If None, use the class's verbose setting.
                                     
        Returns:
            float: The final chi-squared error after optimization
        '''
        # Initialize the optimization (sets up internal structures)
        self.optimizer.initialize_optimization()
        
        # Use class-level verbosity setting if not specified
        if verbose is None:
            verbose = self.verbose
            
        # Set verbosity level for the optimizer
        self.optimizer.set_verbose(verbose)
        
        # Run the optimization for the specified number of iterations
        self.optimizer.optimize(iterations)
        
        # Return the final error (chi-squared) after optimization
        return self.optimizer.chi2()