import numpy as np
from scipy.optimize import curve_fit
import json
from IPython.display import display, HTML
import random
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def jpshape(percent):
    display(HTML(f"<style>.container {{ width:{percent}% !important; }}</style>"))    

class GPC_input_output:
    def __init__(self, data, label):
        """
        a class to store lidar scans and labels for use in a gpc - this fills any nans and zero offsets about the mean 
        and can accomodate a representative location
        
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
        
    def _fill_nan(self,data):
        import copy
        data_filled = np.copy(data)
        mean=np.nanmean(data[:,0])  
        for i in range(len(data[:,1])):
            if np.isnan(data[i,0]):
                data_filled[i,0]=0
            else: 
                data_filled[i,0]=data[i,0]-mean
        return data_filled

def f(x, A, B):
    return A*x + B

def find_corner2(corner):#, threshold = 0.01):
    """ Identifies if there is a corner and calculates the inflexion point of the corner and the
    reference coordinate & index of the inflexion point
    input: corner (takes format from output of "GPC_input_output"), threshold = 0.01 (optional)
    
    output: r, theta, largest_inflection_idx, slopes = [[slope1, y intersept1], [slope2, y intersept2]], [x_intersect, y_intersect] ]
    """
    r = corner.data[:,0] # convert to cartesian
    a = corner.data[:,1]
    x = r*np.cos(a)
    y = r*np.sin(a)

    #curve fit the 1st and last 20 points
    popt1, pcov1 = curve_fit(f, x[2:22], y[2:22]) # your data x, y to fit - popt[0] = slope, popt[1] = intercept 
    popt2, pcov2 = curve_fit(f, x[-20:], y[-20:])
    
    x_intersect = (popt2[1]-popt1[1])/(popt1[0]-popt2[0])
    y_intersect = f(x_intersect, popt1[0], popt1[1])

    slopes = np.array([[popt1[0], popt1[1]], [popt2[0] , popt2[1]], [x_intersect, y_intersect] ] )
    # slopes = np.array([[popt1[0] %(2*np.pi), popt1[1]], [popt2[0] %(2*np.pi), popt2[1]], [x_intersect, y_intersect] ] )
    slope_angles = abs(np.rad2deg( np.arctan((popt1[0]-popt2[0])/(1+popt1[0]*popt2[0]))))


    if slope_angles>15 and slope_angles<80:    # reject due to wrong angle
        return None, None, 9999, slopes
    
    if slope_angles<100 and slope_angles>170:    # reject due to wrong angle
        return None, None, 9999, slopes #np.array([[0,0],[0,0], [0,0]])

    if  slope_angles>190:    # reject due to wrong angle
        return None, None, 9999, slopes

    
    if slope_angles>=83 and slope_angles<=97:
        # Step 1: Compute slope
        slope = np.gradient(corner.data[:, 0])
        # Step 2: Compute the second derivative (curvature)
        curvature = np.gradient(slope)
        # compute index of inflection point    
        largest_inflection_idx = np.nanargmax(abs(np.gradient(np.gradient(curvature))))
        r = corner.data[largest_inflection_idx, 0]  # Radial distance at the largest curvature
        theta = corner.data[largest_inflection_idx, 1]  # Angle at the largest curvature

        x_inlfex = r*np.cos(theta)
        y_inflex = r*np.sin(theta)
        dist_inflections = ((x_inlfex-x_intersect)**2 + (y_inflex-y_intersect)**2)**0.5  # distance between intersect and estimated inflextion point

        if abs(dist_inflections) > 0.10: # compare slope intersect with largest inflection estimate & reject if dist > 10cm
            return r, theta, 9999, slopes
        
        return r, theta, largest_inflection_idx, slopes
    
    else:
        return None, None, None, slopes  # No inflection points found -> assume its a wall

def parse_json_file(file_path):
    data_by_topic = {}  # Dictionary to store data organised by topic

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse each JSON object
                obj = json.loads(line.strip())

                # Organise data by topic
                obj_topic = obj.get("topic_name", "Unknown")
                if obj_topic not in data_by_topic:
                    data_by_topic[obj_topic] = []
                data_by_topic[obj_topic].append(obj)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    return data_by_topic

def fill_nan(row):
    # Calculate mean of non-NaN values
    mean = np.nanmean(row)
    # Replace NaNs with mean
    filled = np.where(np.isnan(row), mean, row)
    # Subtract the mean (centering)
    return filled - mean

def train_sim():
    f1='src/sim_walls.json'
    f2 ='src/sim_corners.json'
    data = parse_json_file(f1)
    data2 = parse_json_file(f2)
    lidar = data['/lidar']
    lidar2 = data2['/lidar']
    angles = []
    ranges = []
    data = []
    for i in range(len(lidar)):
        r = list(lidar[i]['message']['ranges'])
        a = list(lidar[i]['message']['angles'])
        angles.append(a)
        ranges.append(r)
    for i in range(len(lidar2)):
        r = list(lidar2[i]['message']['ranges'])
        a = list(lidar2[i]['message']['angles'])
        angles.append(a)
        ranges.append(r)
    for i in range(len(angles)):
        data.append(np.column_stack((ranges[i],angles[i])))
    # removes the zeros from the data originating from nans
    new_data = []
    for i in range(len(data)):
        t = data[i]
        w = np.where(t[:,0] <= 0.010)
        t = np.delete(t,w,0)
        new_data.append(t)

    new_data2 = []
    for i in range(len(new_data)):
        t = new_data[i]
        w = np.where(np.isnan(t[:,0]))
        t = np.delete(t,w,0)
        new_data2.append(t)

    # Create a container to store the GPC training data
    corner_training = []
    rejected_len = 0
    rejected_inflex = 0

    for i in range(len(new_data2)): # in range of number of readings for corners
        # compute observations 
        observation = new_data2[i]        # data filtered to remove nan for compatibility with find_corner2
        observation_with_nan = data[i]    # reference data which will be appended to learning model

        #only select data with len > 110
        if len(observation)<=110:
            rejected_len = rejected_len + 1

        else:        
            # check if it is a corner with the inflection point
            new_observation = GPC_input_output(observation, None)
            new_observation_with_nan = GPC_input_output(observation_with_nan, None)

            #threshold = 0.01 # can reduce to make less conservative
            z_lm = [0,0]
            z_lm[0], z_lm[1], loc, slopes = find_corner2(new_observation)#, threshold)

            # if the bespoke model says returns a location, add to training data
    #         if loc == 9999: # do not append to training set as the inflexction point is wrong or empty array

            if loc is not None:
                # label corner and add to corner training set
                new_observation.ne_representative = z_lm
                new_observation.label='corner'
                corner_training.append(new_observation)


            elif loc is None:        
                new_observation.label='not corner'        
                corner_training.append(new_observation)
                plot_title = "Not corner"


    num_samples = len(corner_training)
    max_length = max(len(sample.data_filled[:, 0]) for sample in corner_training)

    X_train = np.full((num_samples, max_length), np.nan)
    y_train = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        data = corner_training[i].data_filled[:, 0]
        X_train[i, :len(data)] = data
        y_train[i] = corner_training[i].label
    X_train = np.apply_along_axis(fill_nan, 1, X_train)

    kernel = 1.0 * RBF(1.0)
    gpc_corner = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)

    theta_1 = gpc_corner.kernel_.k2.get_params()['length_scale']
    theta_0 = np.sqrt(gpc_corner.kernel_.k1.get_params()['constant_value'])
    neg_log_likelihood = -gpc_corner.log_marginal_likelihood_value_
    return gpc_corner

def train(n):
    file_path = r'src/lidar_data1.json'
    data = parse_json_file(file_path)
    lidar = data['/lidar']
    angles = []
    ranges = []
    for i in range(len(lidar)):
        r = list(lidar[i]['message']['ranges'])
        a = list(lidar[i]['message']['angles'])
        angles.append(a)
        ranges.append(r)
    data = []
    for i in range(len(angles)):
        data.append(np.column_stack((ranges[i],angles[i])))

    # removes the zeros from the data originating from nans
    new_data = []
    for i in range(len(data)):
        t = data[i]
        w = np.where(t[:,0] <= 0.010)
        t = np.delete(t,w,0)
        new_data.append(t)

    new_data2 = []
    for i in range(len(new_data)):
        t = new_data[i]
        w = np.where(np.isnan(t[:,0]))
        t = np.delete(t,w,0)
        new_data2.append(t)

    # Create a container to store the GPC training data
    corner_training = []
    rejected_len = 0
    rejected_inflex = 0

    index = np.zeros(n)
    for i in range(n):
        index[i] = random.randint(0,len(new_data2))

    for j in range(n): # in range of number of readings for corners
        i = int(index[j])
        # compute observations 
        observation = new_data2[i]        # data filtered to remove nan for compatibility with find_corner2
        observation_with_nan = data[i]    # reference data which will be appended to learning model

        #only select data with len > 110
        if len(observation)<=110:
            rejected_len = rejected_len + 1

        else:        
            # check if it is a corner with the inflection point
            new_observation = GPC_input_output(observation, None)
            new_observation_with_nan = GPC_input_output(observation_with_nan, None)

            #threshold = 0.01 # can reduce to make less conservative
            z_lm = [0,0]
            z_lm[0], z_lm[1], loc, slopes = find_corner2(new_observation)#, threshold)

            # if the bespoke model says returns a location, add to training data
    #         if loc == 9999: # do not append to training set as the inflexction point is wrong or empty array

            if loc is not None:
                # label corner and add to corner training set
                new_observation.ne_representative = z_lm
                new_observation.label='corner'
                corner_training.append(new_observation)


            elif loc is None:        
                new_observation.label='not corner'        
                corner_training.append(new_observation)
                plot_title = "Not corner"


    num_samples = len(corner_training)
    max_length = max(len(sample.data_filled[:, 0]) for sample in corner_training)

    X_train = np.full((num_samples, max_length), np.nan)
    y_train = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        data = corner_training[i].data_filled[:, 0]
        X_train[i, :len(data)] = data
        y_train[i] = corner_training[i].label
    X_train = np.apply_along_axis(fill_nan, 1, X_train)

    kernel = 1.0 * RBF(1.0)
    gpc_corner = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)

    theta_1 = gpc_corner.kernel_.k2.get_params()['length_scale']
    theta_0 = np.sqrt(gpc_corner.kernel_.k1.get_params()['constant_value'])
    neg_log_likelihood = -gpc_corner.log_marginal_likelihood_value_
    return gpc_corner

def predict(ob,gpc_corner):
        z_lm = [0,0]
        corner = False
        if  (40 > len(ob) <= 189) == True:
            return False, False, False, False
        new_ob = GPC_input_output(ob,None)
        num_features = gpc_corner.n_features_in_
        data_length = new_ob.data_filled[:, 0].shape[0]
        if data_length < num_features:
            padded_data = np.pad(new_ob.data_filled[:, 0], (0, num_features - data_length), mode='constant')
        else:
            padded_data = new_ob.data_filled[:, 0]
        prediction = gpc_corner.predict_proba(padded_data.reshape(1, -1))
        if prediction[0][1] > 0.4:
            id = "corner"
            z_lm[0], z_lm[1], loc, slopes = find_corner2(new_ob)
            if z_lm[0] == None:
                return False, False, False, False
            corner = True
            return corner, id, z_lm[0], z_lm[1]
        else:
            return False, False, False, False

def predict_sim(ob,gpc_corner):
        z_lm = [0,0]
        corner = False
        t = ob
        w = np.where(t[:,0] <= 0.010)
        t = np.delete(t,w,0)
        w = np.where(np.isnan(t[:,0]))
        t = np.delete(t,w,0)
        ob = t
        if  (40 > len(ob) <= 189) == True:
            return False, False, False, False
        new_ob = GPC_input_output(ob,None)
        num_features = gpc_corner.n_features_in_
        data_length = new_ob.data_filled[:, 0].shape[0]
        if data_length < num_features:
            padded_data = np.pad(new_ob.data_filled[:, 0], (0, num_features - data_length), mode='constant')
        else:
            padded_data = new_ob.data_filled[:, 0]
        prediction = gpc_corner.predict_proba(padded_data.reshape(1, -1))
        if prediction[0][1] > 0.4:
            id = "corner"
            z_lm[0], z_lm[1], loc, slopes = find_corner2(new_ob)
            if z_lm[0] == None:
                return False, False, False, False
            corner = True
            return corner, id, z_lm[0], z_lm[1]
        else:
            return False, False, False, False


def get_data():
    file_path = r'src/lidar_data1.json'
    data = parse_json_file(file_path)
    lidar = data['/lidar']
    angles = []
    ranges = []
    for i in range(len(lidar)):
        r = list(lidar[i]['message']['ranges'])
        a = list(lidar[i]['message']['angles'])
        angles.append(a)
        ranges.append(r)
    data = []
    for i in range(len(angles)):
        data.append(np.column_stack((ranges[i],angles[i])))

    # removes the zeros from the data originating from nans
    new_data = []
    for i in range(len(data)):
        t = data[i]
        w = np.where(t[:,0] <= 0.010)
        t = np.delete(t,w,0)
        new_data.append(t)

    new_data2 = []
    for i in range(len(new_data)):
        t = new_data[i]
        w = np.where(np.isnan(t[:,0]))
        t = np.delete(t,w,0)
        new_data2.append(t)
    return new_data2