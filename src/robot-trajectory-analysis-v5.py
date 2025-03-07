


import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import sqrt

def parse_json_file(file_path):
    """
    Load and parse the robot's log file, organizing it by topic.
    
    Args:
        file_path (str): Path to the log file to analyze
        
    Returns:
        dict: Dictionary with topic names as keys and lists of entries as values
    """
    data_by_topic = {}  # Dictionary to store data organized by topic
    skipped_lines = 0
    
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                try:
                    # Parse each JSON object
                    obj = json.loads(line.strip())
                    
                    # Organize data by topic_name
                    obj_topic = obj.get("topic_name", "Unknown")
                    if obj_topic not in data_by_topic:
                        data_by_topic[obj_topic] = []
                    data_by_topic[obj_topic].append(obj)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    if skipped_lines <= 5:
                        print(f"Warning: Skipping invalid JSON at line {i+1}: {line[:50]}...")
                    elif skipped_lines == 6:
                        print("Additional invalid JSON lines found (not showing all warnings)...")
        
        if skipped_lines > 0:
            print(f"Total of {skipped_lines} invalid JSON lines skipped")
            
        return data_by_topic
    
    except Exception as e:
        print(f"Error loading log file: {e}")
        raise

def quaternion_to_euler(w, x, y, z):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    
    Args:
        w, x, y, z: Quaternion components
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def extract_data_from_topics(data_by_topic, aruco_path=None):
    """
    Extract trajectory data from different topics following the Jupyter notebook approach.
    
    Args:
        data_by_topic (dict): Data organized by topic
        aruco_path (str, optional): Path to the ArUco CSV file for ground truth
        
    Returns:
        tuple: (extracted_data, simulation_flag)
    """
    extracted_data = {}
    
    # Check if this is simulation or real robot data
    simulation_flag = '/groundtruth' in data_by_topic
    
    # Extract data for /est_pose (same structure as /aruco)
    if '/est_pose' in data_by_topic:
        est_pose_timestamps = [item['message']['header']['stamp'] for item in data_by_topic['/est_pose']]
        est_pose_northings = [item['message']['pose']['position']['x'] for item in data_by_topic['/est_pose']]
        est_pose_eastings = [item['message']['pose']['position']['y'] for item in data_by_topic['/est_pose']]
        est_pose_headings = [(quaternion_to_euler(item['message']['pose']['orientation']['w'], 
                                                item['message']['pose']['orientation']['x'], 
                                                item['message']['pose']['orientation']['y'], 
                                                item['message']['pose']['orientation']['z'])[2]) % (2*np.pi)
                            for item in data_by_topic['/est_pose']]
        
        extracted_data['est_pose'] = {
            'timestamps': est_pose_timestamps,
            'northings': est_pose_northings,
            'eastings': est_pose_eastings,
            'headings': est_pose_headings
        }
    
    # Extract ArUco data (reference for real robot)
    if '/aruco' in data_by_topic:
        aruco_timestamps = [item['message']['header']['stamp'] for item in data_by_topic['/aruco']]
        aruco_northings = [item['message']['pose']['position']['x'] for item in data_by_topic['/aruco']]
        aruco_eastings = [item['message']['pose']['position']['y'] for item in data_by_topic['/aruco']]
        aruco_headings = [(quaternion_to_euler(item['message']['pose']['orientation']['w'], 
                                              item['message']['pose']['orientation']['x'], 
                                              item['message']['pose']['orientation']['y'], 
                                              item['message']['pose']['orientation']['z'])[2]) % (2*np.pi)
                          for item in data_by_topic['/aruco']]
        
        extracted_data['aruco'] = {
            'timestamps': aruco_timestamps,
            'northings': aruco_northings,
            'eastings': aruco_eastings,
            'headings': aruco_headings
        }
    
    # Read ArUco data directly from CSV file
    if aruco_path and os.path.exists(aruco_path):
        # Read the ArUco CSV file
        df = pd.read_csv(aruco_path)
        
        if 'est_pose' in extracted_data:
            # Get time range from estimated pose data
            est_pose_timestamps = extracted_data['est_pose']['timestamps']
            start_time = np.min(est_pose_timestamps)
            end_time = np.max(est_pose_timestamps)
            
            # Filter data based on the est_pose timestamp range
            filtered_data = df.loc[(df['epoch [s]'] >= start_time) & (df['epoch [s]'] <= end_time)]
            
            # Extract columns
            aruco_csv_timestamps = filtered_data['epoch [s]'].tolist()
            aruco_csv_northings = filtered_data['x [m]'].tolist()
            aruco_csv_eastings = filtered_data['y [m]'].tolist()
            aruco_csv_headings = (np.deg2rad(filtered_data['yaw [deg]']) % (2*np.pi)).tolist()
            
            extracted_data['aruco_csv'] = {
                'timestamps': aruco_csv_timestamps,
                'northings': aruco_csv_northings,
                'eastings': aruco_csv_eastings,
                'headings': aruco_csv_headings
            }
    
    # Extract wheel speeds
    if '/true_wheel_speeds' in data_by_topic:
        true_wheel_timestamps = [float(item['timestamp']) for item in data_by_topic['/true_wheel_speeds']]
        true_wheel_speeds = [item['message']['vector']['x'] for item in data_by_topic['/true_wheel_speeds']]
        
        extracted_data['true_wheel_speeds'] = {
            'timestamps': true_wheel_timestamps,
            'speeds': true_wheel_speeds
        }
    
    if '/wheel_speeds_cmd' in data_by_topic:
        wheel_speeds_cmd_timestamps = [float(item['timestamp']) for item in data_by_topic['/wheel_speeds_cmd']]
        wheel_speeds_cmd = [item['message']['vector']['x'] for item in data_by_topic['/wheel_speeds_cmd']]
        
        extracted_data['wheel_speeds_cmd'] = {
            'timestamps': wheel_speeds_cmd_timestamps,
            'speeds': wheel_speeds_cmd
        }
    
    # Extract ground truth data
    if simulation_flag and '/groundtruth' in data_by_topic:
        # Simulation groundtruth
        groundtruth_timestamps = [float(item['timestamp']) for item in data_by_topic['/groundtruth']]
        groundtruth_northings = [item['message']['position']['x'] for item in data_by_topic['/groundtruth']]
        groundtruth_eastings = [item['message']['position']['y'] for item in data_by_topic['/groundtruth']]
        groundtruth_headings = [(quaternion_to_euler(item['message']['orientation']['w'], 
                                                    item['message']['orientation']['x'], 
                                                    item['message']['orientation']['y'], 
                                                    item['message']['orientation']['z'])[2]) % (2*np.pi)
                                for item in data_by_topic['/groundtruth']]
        
        extracted_data['groundtruth'] = {
            'timestamps': groundtruth_timestamps,
            'northings': groundtruth_northings,
            'eastings': groundtruth_eastings,
            'headings': groundtruth_headings
        }
    elif not simulation_flag and aruco_path and os.path.exists(aruco_path):
        # For real robot - use ArUco CSV as ground truth if available
        if 'aruco_csv' in extracted_data:
            # Use the already extracted ArUco CSV data for ground truth
            extracted_data['groundtruth'] = extracted_data['aruco_csv'].copy()
    
    return extracted_data, simulation_flag

def filter_data_by_time(data, start_time, end_time):
    """
    Filter data arrays by timestamp range.
    
    Args:
        data (dict): Dictionary with extracted data for a topic
        start_time (float): Start timestamp
        end_time (float): End timestamp
        
    Returns:
        dict: Filtered data dictionary
    """
    if 'timestamps' not in data:
        return data
    
    timestamps = data['timestamps']
    mask = [(t >= start_time) and (t <= end_time) for t in timestamps]
    
    filtered_data = {}
    for key, values in data.items():
        if len(values) == len(timestamps):
            filtered_data[key] = [v for v, m in zip(values, mask) if m]
        else:
            filtered_data[key] = values
    
    return filtered_data

def plot_trajectory(data, reference_topic, output_path=None):
    """
    Plot the robot's trajectory.
    
    Args:
        data (dict): Dictionary with extracted data for each topic
        reference_topic (str): Topic to use as reference ('groundtruth' or 'aruco')
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(12, 10))
    
    # # Plot the red square dots first (before everything else)
    # if 'est_pose' in data and 'northings' in data['est_pose'] and 'eastings' in data['est_pose']:
    #     if len(data['est_pose']['northings']) > 0 and len(data['est_pose']['eastings']) > 0:
    #         # Get the starting position of est_pose
    #         start_easting = data['est_pose']['eastings'][0]
    #         start_northing = data['est_pose']['northings'][0]
    #         
    #         # Calculate the four corners of a 1m square
    #         square_points = [
    #             (start_easting, start_northing),                # Starting point
    #             (start_easting + 1.0, start_northing),          # 1m east
    #             (start_easting + 1.0, start_northing + 1.0),    # 1m east, 1m north
    #             (start_easting, start_northing + 1.0)           # 1m north
    #         ]
    #         
    #         # Extract eastings and northings from the square points
    #         square_eastings = [p[0] for p in square_points]
    #         square_northings = [p[1] for p in square_points]
    #         
    #         # Plot the red dots with size 26
    #         plt.scatter(square_eastings, square_northings, color='red', s=26*10, alpha=0.7, zorder=1, label='Reference Square (1m)')
    
    # Define plot order for legend control - ArUco first
    plot_order = ['aruco', 'groundtruth', 'est_pose']
    
    # Plot each trajectory
    color_map = {
        'groundtruth': ('orange', 'Ground Truth'),
        'est_pose': ('green', 'Estimated Pose'),
        'aruco': ('blue', 'Aruco Markers')
    }
    
    # Plot all trajectories in the defined order
    for topic in plot_order:
        if topic in data and 'northings' in data[topic] and 'eastings' in data[topic]:
            topic_data = data[topic]
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            # Special case for aruco (blue): only plot markers, no line
            if topic == 'aruco':
                plt.scatter(eastings, northings, color=color, marker='x', label=label, zorder=7)
            
            # Special case for groundtruth (orange): plot line with alpha and markers with no alpha
            elif topic == 'groundtruth':
                # Plot line with transparency
                plt.plot(eastings, northings, '-', color=color, alpha=0.4, label=label, zorder=5)
                # Plot points with full opacity
                plt.scatter(eastings, northings, color=color, marker='.', s=20, zorder=6)
            
            # Default case for other topics
            else:
                if topic == reference_topic:
                    line_style = '-'
                    marker = '.'
                else:
                    line_style = '--' if topic == 'est_pose' else '-.'
                    marker = '.'
                    
                plt.plot(eastings, northings, line_style, label=label, color=color, marker=marker, zorder=5)
    
    # Now add start/end markers for non-aruco topics
    for topic in plot_order:
        if topic == 'aruco':
            continue  # Skip aruco markers
            
        if topic in data and 'northings' in data[topic] and 'eastings' in data[topic]:
            topic_data = data[topic]
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            if len(northings) > 0 and len(eastings) > 0:
                # Use marker='o' and set color separately
                plt.plot(eastings[0], northings[0], 'o', color=color, markersize=12, label=f"{label} Start", zorder=10)
                plt.plot(eastings[-1], northings[-1], 'x', color=color, markersize=16, label=f"{label} End", zorder=10)
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.axis('equal')
    
    # Create a clean legend (removing duplicates but preserving order)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    
    plt.legend(unique_handles, unique_labels, loc='best')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()

def plot_wheel_speeds(data, output_path=None):
    """
    Plot wheel speeds over time as in the Jupyter notebook.
    
    Args:
        data (dict): Dictionary with extracted data for each topic
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(10, 6))
    
    if 'true_wheel_speeds' in data:
        true_wheel_data = data['true_wheel_speeds']
        plt.plot(true_wheel_data['timestamps'], true_wheel_data['speeds'], 
                label="Measured Wheel Speeds", marker=".", color="orange")
    
    if 'wheel_speeds_cmd' in data:
        cmd_wheel_data = data['wheel_speeds_cmd']
        plt.plot(cmd_wheel_data['timestamps'], cmd_wheel_data['speeds'], 
                label="Commanded Wheel Speeds", marker=".", color="green")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Wheel Speed (rad/s)")
    plt.legend()
    plt.title("Measured and Commanded Wheel Speeds")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Wheel speeds plot saved to {output_path}")
    else:
        plt.show()

def plot_headings(data, output_path=None):
    """
    Plot headings over time as in the Jupyter notebook.
    
    Args:
        data (dict): Dictionary with extracted data for each topic
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(10, 6))
    
    # Define plot order for legend control - ArUco first
    plot_order = ['aruco', 'groundtruth', 'est_pose']
    
    color_map = {
        'groundtruth': ('orange', 'Ground Truth Heading'),
        'est_pose': ('green', 'Estimated Pose Heading'),
        'aruco': ('blue', 'Aruco Heading')
    }
    
    # Plot in specific order to control legend
    for topic in plot_order:
        if topic in data and 'timestamps' in data[topic] and 'headings' in data[topic]:
            topic_data = data[topic]
            color, label = color_map[topic]
            marker = 'x' if topic == 'aruco' else '.'
            plt.plot(topic_data['timestamps'], topic_data['headings'], 
                    label=label, marker=marker, color=color)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heading (radians)")
    plt.legend()
    plt.title("Headings Over Time")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Headings plot saved to {output_path}")
    else:
        plt.show()






def calculate_trajectory_errors(reference_data, est_pose_data):
    """
    Calculate error metrics between reference trajectory (ground truth) and estimated pose.
    
    Args:
        reference_data (dict): Dictionary with reference trajectory data
        est_pose_data (dict): Dictionary with estimated pose data
    
    Returns:
        dict: Dictionary with error metrics
    """
    if not reference_data or not est_pose_data:
        return None
    
    # Get data arrays
    ref_times = np.array(reference_data['timestamps'])
    ref_northings = np.array(reference_data['northings'])
    ref_eastings = np.array(reference_data['eastings'])
    ref_headings = np.array(reference_data['headings'])
    
    est_times = np.array(est_pose_data['timestamps'])
    est_northings = np.array(est_pose_data['northings'])
    est_eastings = np.array(est_pose_data['eastings'])
    est_headings = np.array(est_pose_data['headings'])
    
    # Find common time range
    start_time = max(ref_times.min(), est_times.min())
    end_time = min(ref_times.max(), est_times.max())
    
    # Filter estimated pose timestamps to the common range
    valid_est_indices = (est_times >= start_time) & (est_times <= end_time)
    valid_est_times = est_times[valid_est_indices]
    
    # Extract samples at estimated pose times
    errors = {
        'timestamps': valid_est_times.tolist(),
        'euclidean_distance': [],
        'heading_error': []
    }
    
    # Interpolate reference data at each estimated pose timestamp
    for t in valid_est_times:
        # Find nearest reference data points
        idx = np.argmin(np.abs(ref_times - t))
        
        # Get actual values at estimated pose time
        est_idx = np.where(est_times == t)[0][0]
        est_n = est_northings[est_idx]
        est_e = est_eastings[est_idx]
        est_h = est_headings[est_idx]
        
        # Get reference values
        ref_n = ref_northings[idx]
        ref_e = ref_eastings[idx]
        ref_h = ref_headings[idx]
        
        # Calculate Euclidean distance error
        euclidean_distance = np.sqrt((est_n - ref_n)**2 + (est_e - ref_e)**2)
        errors['euclidean_distance'].append(euclidean_distance)
        
        # Calculate heading error (accounting for angle wrapping)
        heading_error = min(abs(est_h - ref_h), 2*np.pi - abs(est_h - ref_h))
        errors['heading_error'].append(heading_error)
    
    return errors

def plot_trajectory_errors(errors, output_path=None):
    """
    Plot error metrics over time.
    
    Args:
        errors (dict): Dictionary with error data
        output_path (str, optional): Path to save the plot, or None to display it
    """
    if not errors:
        print("No error data available for plotting")
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Euclidean distance error
    ax1.plot(errors['timestamps'], errors['euclidean_distance'], 'r-', label='Position Error')
    ax1.set_ylabel('Euclidean Distance Error (m)')
    ax1.set_title('Position Error Over Time')
    ax1.grid(True)
    
    # Plot heading error
    ax2.plot(errors['timestamps'], np.rad2deg(errors['heading_error']), 'b-', label='Heading Error')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Heading Error (degrees)')
    ax2.set_title('Heading Error Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Error plots saved to {output_path}")
    else:
        plt.show()

def print_error_statistics(errors):
    """
    Calculate and print error statistics.
    
    Args:
        errors (dict): Dictionary with error data
    """
    if not errors:
        print("No error data available for statistics")
        return
    
    # Convert to numpy arrays for calculations
    euclidean_errors = np.array(errors['euclidean_distance'])
    heading_errors = np.array(errors['heading_error'])
    
    # Calculate statistics for Euclidean distance error
    mean_position_error = np.mean(euclidean_errors)
    median_position_error = np.median(euclidean_errors)
    max_position_error = np.max(euclidean_errors)
    std_position_error = np.std(euclidean_errors)
    rmse_position_error = np.sqrt(np.mean(euclidean_errors**2))
    
    # Calculate statistics for heading error (convert to degrees)
    mean_heading_error = np.mean(heading_errors) * 180 / np.pi
    median_heading_error = np.median(heading_errors) * 180 / np.pi
    max_heading_error = np.max(heading_errors) * 180 / np.pi
    std_heading_error = np.std(heading_errors) * 180 / np.pi
    rmse_heading_error = np.sqrt(np.mean(heading_errors**2)) * 180 / np.pi
    
    # Print results in a table format
    print("\n" + "="*80)
    print("Error Statistics".center(80))
    print("="*80)
    
    print("\nPosition Error (Euclidean Distance):")
    print(f"  Mean Error:     {mean_position_error:.4f} m")
    print(f"  Median Error:   {median_position_error:.4f} m")
    print(f"  Maximum Error:  {max_position_error:.4f} m")
    print(f"  Std Deviation:  {std_position_error:.4f} m")
    print(f"  RMSE:           {rmse_position_error:.4f} m")
    
    print("\nHeading Error:")
    print(f"  Mean Error:     {mean_heading_error:.4f} deg")
    print(f"  Median Error:   {median_heading_error:.4f} deg")
    print(f"  Maximum Error:  {max_heading_error:.4f} deg")
    print(f"  Std Deviation:  {std_heading_error:.4f} deg")
    print(f"  RMSE:           {rmse_heading_error:.4f} deg")
    
    print("="*80)


def get_log_by_number(log_dir, log_number):
    """
    Get a log file by its index number in the directory (sorted by name).
    
    Args:
        log_dir (str): Directory containing log files
        log_number (int): Index of the log file to select (1-indexed)
        
    Returns:
        str: Full path to the selected log file
    """
    # Get all log files in the directory with .json extension
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.json')])
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in directory: {log_dir}")
    
    # Adjust for 1-indexed input
    if log_number < 1 or log_number > len(log_files):
        raise ValueError(f"Log number must be between 1 and {len(log_files)}")
    
    selected_log = os.path.join(log_dir, log_files[log_number - 1])
    print(f"Selected log file {log_number} of {len(log_files)}: {os.path.basename(selected_log)}")
    
    return selected_log



def extract_lidar_data(data_by_topic):
    """
    Extract LIDAR data from the topics.
    
    Args:
        data_by_topic (dict): Data organized by topic
        
    Returns:
        list: List of dictionaries containing LIDAR scan data
    """
    lidar_data = []
    
    if '/lidar' in data_by_topic:
        for item in data_by_topic['/lidar']:
            # Extract scan data
            timestamp = item['message']['header']['stamp']
            angle_min = item['message']['angle_min']
            angle_max = item['message']['angle_max']
            angle_increment = item['message']['angle_increment']
            ranges = item['message']['ranges']
            intensities = item['message']['intensities']
            angles = item['message']['angles']
            
            # Store scan data
            scan = {
                'timestamp': timestamp,
                'angle_min': angle_min,
                'angle_max': angle_max,
                'angle_increment': angle_increment,
                'ranges': ranges,
                'intensities': intensities,
                'angles': angles
            }
            
            lidar_data.append(scan)
    
    return lidar_data

def plot_lidar_scan_polar(lidar_scan, output_path=None):
    """
    Plot a single LIDAR scan in polar coordinates.
    
    Args:
        lidar_scan (dict): Dictionary containing LIDAR scan data
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Extract data for plotting
    angles = np.array(lidar_scan['angles'])
    ranges = np.array(lidar_scan['ranges'])
    
    # Filter out invalid ranges (zeros)
    mask = ranges > 0
    valid_angles = angles[mask]
    valid_ranges = ranges[mask]
    
    # Create polar plot
    scatter = ax.scatter(valid_angles, valid_ranges, s=10, c=valid_ranges, cmap='viridis', alpha=0.7)
    
    # Set the limits and labels
    ax.set_theta_zero_location('N')  # 0 degrees at top (forward direction)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_rlabel_position(0)  # Move radius labels to 0 degrees
    ax.set_rticks([0.3, 0.6, 0.9])  # Set radial ticks
    
    # Set title with timestamp
    timestamp = lidar_scan['timestamp']
    plt.title(f"LIDAR Scan at t={timestamp:.3f}s")
    
    # Add colorbar
    plt.colorbar(scatter, label='Range (m)')
    
    if output_path:
        plt.savefig(output_path)
        print(f"LIDAR scan plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_lidar_scan_cartesian(lidar_scan, output_path=None):
    """
    Plot a single LIDAR scan in Cartesian coordinates.
    
    Args:
        lidar_scan (dict): Dictionary containing LIDAR scan data
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(10, 10))
    
    # Extract data for plotting
    angles = np.array(lidar_scan['angles'])
    ranges = np.array(lidar_scan['ranges'])
    
    # Filter out invalid ranges (zeros)
    mask = ranges > 0
    valid_angles = angles[mask]
    valid_ranges = ranges[mask]
    
    # Convert polar coordinates to Cartesian
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    
    # Create scatter plot
    scatter = plt.scatter(x, y, s=10, c=valid_ranges, cmap='viridis', alpha=0.7)
    
    # Add robot position
    plt.scatter(0, 0, s=100, c='red', marker='x', label='Robot')
    
    # Add axis lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    timestamp = lidar_scan['timestamp']
    plt.title(f"LIDAR Scan at t={timestamp:.3f}s (Cartesian)")
    
    # Add colorbar
    plt.colorbar(scatter, label='Range (m)')
    
    # Add grid and equal aspect ratio
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"LIDAR scan plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_lidar_data(data_by_topic, scan_index=0, output_path=None, plot_type='polar'):
    """
    Extract and plot LIDAR data from the logs.
    
    Args:
        data_by_topic (dict): Data organized by topic
        scan_index (int): Index of the scan to plot (0 for first scan)
        output_path (str, optional): Path to save the plot, or None to display it
        plot_type (str): Type of plot ('polar' or 'cartesian')
    """
    # Extract LIDAR data
    lidar_data = extract_lidar_data(data_by_topic)
    
    if not lidar_data:
        print("No LIDAR data found in the logs.")
        return
    
    print(f"Found {len(lidar_data)} LIDAR scans in the logs.")
    
    if scan_index < 0 or scan_index >= len(lidar_data):
        print(f"Invalid scan index. Using first scan (index 0).")
        scan_index = 0
    
    if plot_type == 'polar':
        print(f"Plotting LIDAR scan {scan_index} in polar coordinates.")
        plot_lidar_scan_polar(lidar_data[scan_index], output_path)
    elif plot_type == 'cartesian':
        print(f"Plotting LIDAR scan {scan_index} in Cartesian coordinates.")
        plot_lidar_scan_cartesian(lidar_data[scan_index], output_path)
    else:
        print(f"Invalid plot type: {plot_type}. Valid options are 'polar' or 'cartesian'.")


def plot_lidar_overlay_on_trajectory(trajectory_data, lidar_scan, robot_pose, output_path=None):
    """
    Plot LIDAR data overlaid on the robot trajectory.
    
    Args:
        trajectory_data (dict): Dictionary with trajectory data
        lidar_scan (dict): Dictionary containing LIDAR scan data
        robot_pose (dict): Dictionary with robot position and orientation
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(12, 10))
    
    # Extract data for plotting trajectory
    reference_topic = None
    if 'groundtruth' in trajectory_data:
        reference_topic = 'groundtruth'
        label = 'Ground Truth Trajectory'
    elif 'aruco' in trajectory_data:
        reference_topic = 'aruco'
        label = 'ArUco Trajectory'
    
    if reference_topic:
        northings = trajectory_data[reference_topic]['northings']
        eastings = trajectory_data[reference_topic]['eastings']
        plt.plot(eastings, northings, '-', color='orange', alpha=0.6, label=label)
    # Extract current robot position
    robot_x = robot_pose['position']['x']
    robot_y = robot_pose['position']['y']
    
    # Extract robot orientation (quaternion to euler)
    qw = robot_pose['orientation']['w']
    qx = robot_pose['orientation']['x']
    qy = robot_pose['orientation']['y']
    qz = robot_pose['orientation']['z']
    _, _, robot_yaw = quaternion_to_euler(qw, qx, qy, qz)
    
    # Extract LIDAR data
    angles = np.array(lidar_scan['angles'])
    ranges = np.array(lidar_scan['ranges'])
    
    # Filter out invalid ranges (zeros)
    mask = ranges > 0
    valid_angles = angles[mask]
    valid_ranges = ranges[mask]
    
    # Convert LIDAR scan to Cartesian coordinates in robot frame
    scan_x = valid_ranges * np.cos(valid_angles)
    scan_y = valid_ranges * np.sin(valid_angles)
    
    # Transform LIDAR points to world frame
    world_x = []
    world_y = []
    for sx, sy in zip(scan_x, scan_y):
        # Rotate by robot's yaw and translate to robot's position
        wx = robot_x + sx * np.cos(robot_yaw) - sy * np.sin(robot_yaw)
        wy = robot_y + sx * np.sin(robot_yaw) + sy * np.cos(robot_yaw)
        world_x.append(wx)
        world_y.append(wy)
    
    # Plot LIDAR scan points
    plt.scatter(world_x, world_y, s=15, c='blue', alpha=0.7, label='LIDAR Points')
    
    # Plot robot position
    plt.scatter(robot_x, robot_y, s=150, c='red', marker='x', label='Robot Position')
    
    # Draw robot orientation line
    orientation_length = 0.2  # meters
    plt.arrow(robot_x, robot_y, 
              orientation_length * np.cos(robot_yaw), 
              orientation_length * np.sin(robot_yaw),
              head_width=0.05, head_length=0.1, fc='red', ec='red')
    
    # Set plot properties
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory with LIDAR Data Overlay')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='best')
    
    if output_path:
        plt.savefig(output_path)
        print(f"LIDAR overlay plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()


def read_trajectory_csv(csv_path, x_offset=0.1, y_offset=0.1):
    """
    Read a trajectory CSV file and return the data with optional offsets.
    
    Args:
        csv_path (str): Path to the CSV file
        x_offset (float): Offset to add to x/easting values
        y_offset (float): Offset to add to y/northing values
        
    Returns:
        dict: Dictionary with trajectory data
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Apply offsets to the position data
        eastings = [x + x_offset for x in df['x'].tolist()]
        northings = [y + y_offset for y in df['y'].tolist()]
        
        trajectory_data = {
            'timestamps': df['time'].tolist(),
            'northings': northings,  # Note: northings is y in the CSV
            'eastings': eastings,    # Note: eastings is x in the CSV
            'headings': df['theta'].tolist(),
            'linear_vel': df['linear_vel'].tolist() if 'linear_vel' in df.columns else [],
            'angular_vel': df['angular_vel'].tolist() if 'angular_vel' in df.columns else []
        }
        
        print(f"Loaded {len(trajectory_data['timestamps'])} points from trajectory CSV")
        print(f"Applied offsets: x_offset={x_offset}, y_offset={y_offset}")
        return trajectory_data
    except Exception as e:
        print(f"Error reading trajectory CSV file: {e}")
        return None


def plot_trajectory_with_csv(data, reference_topic, csv_trajectory_data, output_path=None):
    """
    Plot the robot's trajectory along with the generated trajectory from CSV.
    
    Args:
        data (dict): Dictionary with extracted data for each topic
        reference_topic (str): Topic to use as reference ('groundtruth' or 'aruco')
        csv_trajectory_data (dict): Dictionary with trajectory data from CSV
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(12, 10))
    
    # Define plot order for legend control - ArUco first
    plot_order = ['aruco', 'groundtruth', 'est_pose']
    
    # Plot each trajectory
    color_map = {
        'groundtruth': ('orange', 'Ground Truth'),
        'est_pose': ('green', 'Estimated Pose'),
        'aruco': ('blue', 'Aruco Markers')
    }
    
    # Plot all trajectories in the defined order
    for topic in plot_order:
        if topic in data and 'northings' in data[topic] and 'eastings' in data[topic]:
            topic_data = data[topic]
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            # Special case for aruco (blue): only plot markers, no line
            if topic == 'aruco':
                plt.scatter(eastings, northings, color=color, marker='x', label=label, zorder=7)
            
            # Special case for groundtruth (orange): plot line with alpha and markers with no alpha
            elif topic == 'groundtruth':
                # Plot line with transparency
                plt.plot(eastings, northings, '-', color=color, alpha=0.4, label=label, zorder=5)
                # Plot points with full opacity
                plt.scatter(eastings, northings, color=color, marker='.', s=20, zorder=6)
            
            # Default case for other topics
            else:
                if topic == reference_topic:
                    line_style = '-'
                    marker = '.'
                else:
                    line_style = '--' if topic == 'est_pose' else '-.'
                    marker = '.'
                    
                plt.plot(eastings, northings, line_style, label=label, color=color, marker=marker, zorder=5)
    
    # Plot CSV trajectory data in red
    if csv_trajectory_data and 'northings' in csv_trajectory_data and 'eastings' in csv_trajectory_data:
        plt.plot(csv_trajectory_data['eastings'], csv_trajectory_data['northings'], 
                 '-', color='red', linewidth=2, label='Generated Trajectory', zorder=8, alpha=0.4)

    
    # Now add start/end markers for non-aruco topics
    for topic in plot_order:
        if topic == 'aruco':
            continue  # Skip aruco markers
            
        if topic in data and 'northings' in data[topic] and 'eastings' in data[topic]:
            topic_data = data[topic]
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            if len(northings) > 0 and len(eastings) > 0:
                # Use marker='o' and set color separately
                plt.plot(eastings[0], northings[0], 'o', color=color, markersize=12, label=f"{label} Start", zorder=10)
                plt.plot(eastings[-1], northings[-1], 'x', color=color, markersize=16, label=f"{label} End", zorder=10)
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory with Generated Path')
    plt.grid(True)
    plt.axis('equal')
    
    # Create a clean legend (removing duplicates but preserving order)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    
    plt.legend(unique_handles, unique_labels, loc='best')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot with CSV data saved to {output_path}")
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(description='Analyze robot trajectory data by time range.')
    
    # Log file selection options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-file', help='Path to a specific log file to analyze')
    log_group.add_argument('--number', type=int, help='Index number of the log file to analyze (1-indexed)')
    
    parser.add_argument('--log-dir', default='../SHANTAM_LOGS/', help='Directory containing log files')
    parser.add_argument('--aruco-csv', help='Path to ArUco CSV file for ground truth (for real robot)')
    parser.add_argument('--simulation', dest='simulation', action='store_true', help='Use groundtruth as reference (default)')
    parser.add_argument('--real', dest='simulation', action='store_false', help='Use aruco markers as reference')
    parser.add_argument('--output', '-o', help='Path to save plots (without extension)')
    

    # Time-based filtering options
    parser.add_argument('--time', type=str, help='Time range to analyze in seconds (e.g., "5:10" for seconds 5 to 10 from start, or "all" for entire log)')
    
    # Plot options
    parser.add_argument('--plot-wheel-speeds', action='store_true', help='Generate wheel speeds plot')
    parser.add_argument('--plot-headings', action='store_true', help='Generate headings plot')
    parser.add_argument('--debug', action='store_true', help='Print debug information about the data')
    
    # LIDAR plotting options
    parser.add_argument('--plot-lidar', action='store_true', help='Generate LIDAR data plot')
    parser.add_argument('--lidar-scan-index', type=int, default=0, help='Index of the LIDAR scan to plot (default: 0)')
    parser.add_argument('--lidar-plot-type', choices=['polar', 'cartesian', 'overlay'], default='polar', 
                         help='Type of LIDAR plot (polar, cartesian, or overlay with trajectory)')
    
    # Error analysis options
    parser.add_argument('--analyze-errors', action='store_true', help='Calculate and display error metrics between reference and est_pose')


    # In main(), add these arguments to the argument parser:
    parser.add_argument('--trajectory-csv', help='Path to trajectory CSV file to plot')
    parser.add_argument('--x-offset', type=float, default=0.1, 
                        help='Offset to add to x/easting values (default: 0.1)')
    parser.add_argument('--y-offset', type=float, default=0.1, 
                        help='Offset to add to y/northing values (default: 0.1)')


    
    parser.set_defaults(simulation=True)
    
    args = parser.parse_args()
    
    try:
        # Determine which log file to use
        log_file_path = None
        if args.log_file:
            log_file_path = args.log_file
        elif args.number:
            log_file_path = get_log_by_number(args.log_dir, args.number)
        else:
            # Default to the first log file in the directory if neither specified
            log_file_path = get_log_by_number(args.log_dir, 1)
        
        print(f"Current working directory: {os.getcwd()}")
        
        # Load and parse the log file
        data_by_topic = parse_json_file(log_file_path)
        
        # Print available topics
        if args.debug:
            print("\nData organized by topic:")
            for topic, items in data_by_topic.items():
                print(f"{topic}: {len(items)} entries")
                # Print first entry of each topic
                if items:
                    print("First entry sample:")
                    print(f"  {json.dumps(items[0], indent=2)[:300]}...")
                print()
        
        # Extract data from topics
        extracted_data, is_simulation = extract_data_from_topics(data_by_topic, args.aruco_csv)
        
        # Override simulation flag if explicitly specified
        if args.simulation is not None:
            is_simulation = args.simulation
        
        # Determine reference topic based on simulation flag
        reference_topic = 'groundtruth' if is_simulation else 'aruco'
        reference_name = "Ground Truth" if is_simulation else "Aruco Markers"
        
        print(f"Mode: {'Simulation' if is_simulation else 'Real robot'} (using {reference_name} as reference)")
        
        # Check if reference topic exists
        if reference_topic not in extracted_data:
            available_topics = list(extracted_data.keys())
            raise ValueError(f"No {reference_topic} data found. Available topics: {available_topics}")
        
        # Make a copy of data to potentially filter
        filtered_data = extracted_data.copy()
        
        # Filter by time range if specified
        if args.time:
            # Special case for "all" - use all data (no filtering)
            if args.time.lower() == "all":
                print("\nVisualization includes all data from the log (no time filtering)")
            else:
                try:
                    time_parts = args.time.split(':')
                    if len(time_parts) != 2:
                        raise ValueError("Time range must be in format 'start:end' (e.g., '5:10')")
                    
                    # Parse time values
                    relative_start_time = float(time_parts[0])
                    relative_end_time = float(time_parts[1])
                    
                    # Get the reference start time (first timestamp in reference data)
                    if reference_topic in extracted_data and 'timestamps' in extracted_data[reference_topic]:
                        ref_timestamps = extracted_data[reference_topic]['timestamps']
                        
                        if len(ref_timestamps) > 0:
                            ref_start_time = ref_timestamps[0]
                            
                            # Calculate absolute time values based on reference start
                            absolute_start_time = ref_start_time + relative_start_time
                            absolute_end_time = ref_start_time + relative_end_time
                            
                            # Filter all data sources to this time range
                            for topic, topic_data in extracted_data.items():
                                filtered_data[topic] = filter_data_by_time(topic_data, absolute_start_time, absolute_end_time)
                            
                            print(f"\nVisualization filtered to time range {relative_start_time} to {relative_end_time} seconds from start")
                        else:
                            print(f"No timestamps found in reference topic: {reference_topic}")
                    else:
                        print(f"Cannot find timestamps for reference topic: {reference_topic}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing time specification: {e}")
                    print("Using full data set instead")
        
        # Generate output paths
        time_suffix = f"_time_{args.time.replace(':', '-')}" if args.time else ""
        

        # Then, replace the CSV reading code with:
        # Check if trajectory CSV file is provided
        csv_trajectory_data = None
        if args.trajectory_csv:
            csv_trajectory_data = read_trajectory_csv(
                args.trajectory_csv, 
                x_offset=args.x_offset, 
                y_offset=args.y_offset
            )

        # Plot the trajectory
        if csv_trajectory_data:
            # Plot with CSV trajectory data
            trajectory_output = f"{args.output}{time_suffix}_trajectory_with_csv.png" if args.output else None
            plot_trajectory_with_csv(filtered_data, reference_topic, csv_trajectory_data, output_path=trajectory_output)
        else:
            # Use the original trajectory plotting function
            trajectory_output = f"{args.output}{time_suffix}_trajectory.png" if args.output else None
            plot_trajectory(filtered_data, reference_topic, output_path=trajectory_output)
        
        # Rest of the code remains the same...
        # Plot wheel speeds if requested
        if args.plot_wheel_speeds:
            wheel_speeds_output = f"{args.output}{time_suffix}_wheel_speeds.png" if args.output else None
            plot_wheel_speeds(filtered_data, output_path=wheel_speeds_output)
        
        # Plot headings if requested
        if args.plot_headings:
            headings_output = f"{args.output}{time_suffix}_headings.png" if args.output else None
            plot_headings(filtered_data, output_path=headings_output)
        
        # Plot LIDAR data if requested
        if args.plot_lidar:
            print("\nGenerating LIDAR visualization...")
            lidar_output = f"{args.output}{time_suffix}_lidar_{args.lidar_plot_type}.png" if args.output else None
            
            # Extract LIDAR data
            lidar_data = extract_lidar_data(data_by_topic)
            
            if not lidar_data:
                print("No LIDAR data found in the logs.")
            else:
                print(f"Found {len(lidar_data)} LIDAR scans in the logs.")
                
                # Make sure scan index is valid
                scan_index = min(max(0, args.lidar_scan_index), len(lidar_data) - 1)
                if scan_index != args.lidar_scan_index:
                    print(f"Requested scan index {args.lidar_scan_index} out of range. Using scan {scan_index} instead.")
                
                # Get the selected LIDAR scan
                lidar_scan = lidar_data[scan_index]
                
                if args.lidar_plot_type == 'polar':
                    print(f"Plotting LIDAR scan {scan_index} in polar coordinates.")
                    plot_lidar_scan_polar(lidar_scan, lidar_output)
                    
                elif args.lidar_plot_type == 'cartesian':
                    print(f"Plotting LIDAR scan {scan_index} in Cartesian coordinates.")
                    plot_lidar_scan_cartesian(lidar_scan, lidar_output)
                    
                elif args.lidar_plot_type == 'overlay':
                    # Get matching robot pose from groundtruth or aruco
                    lidar_timestamp = lidar_scan['timestamp']
                    
                    if '/groundtruth' in data_by_topic and is_simulation:
                        # Find closest groundtruth timestamp to the lidar scan
                        groundtruth_timestamps = [float(item['timestamp']) for item in data_by_topic['/groundtruth']]
                        closest_idx = np.argmin(np.abs(np.array(groundtruth_timestamps) - lidar_timestamp))
                        
                        # Get robot pose
                        robot_pose = data_by_topic['/groundtruth'][closest_idx]['message']
                        
                        print(f"Plotting LIDAR scan {scan_index} overlaid on trajectory.")
                        plot_lidar_overlay_on_trajectory(filtered_data, lidar_scan, robot_pose, lidar_output)
                    elif '/aruco' in data_by_topic and not is_simulation:
                        # Find closest aruco timestamp to the lidar scan
                        aruco_timestamps = [item['message']['header']['stamp'] for item in data_by_topic['/aruco']]
                        closest_idx = np.argmin(np.abs(np.array(aruco_timestamps) - lidar_timestamp))
                        
                        # Get robot pose from aruco
                        aruco_pose = {
                            'position': data_by_topic['/aruco'][closest_idx]['message']['pose']['position'],
                            'orientation': data_by_topic['/aruco'][closest_idx]['message']['pose']['orientation']
                        }
                        
                        print(f"Plotting LIDAR scan {scan_index} overlaid on trajectory using ArUco data.")
                        plot_lidar_overlay_on_trajectory(filtered_data, lidar_scan, aruco_pose, lidar_output)
                    else:
                        print("No suitable reference data (groundtruth or aruco) available for LIDAR overlay.")

        # Analyze errors if requested
        if args.analyze_errors:
            # Check if both reference and estimated pose data are available
            if reference_topic in filtered_data and 'est_pose' in filtered_data:
                print("\nCalculating error metrics...")
                
                errors = calculate_trajectory_errors(filtered_data[reference_topic], filtered_data['est_pose'])
                
                if errors:
                    # Calculate and display error statistics
                    print_error_statistics(errors)
                    
                    # Generate error plots
                    error_output = f"{args.output}{time_suffix}_errors.png" if args.output else None
                    plot_trajectory_errors(errors, output_path=error_output)
                else:
                    print("Unable to calculate error metrics - no overlapping data found.")
            else:
                if reference_topic not in filtered_data:
                    print(f"Error: Reference data ({reference_topic}) not available for error analysis.")
                if 'est_pose' not in filtered_data:
                    print("Error: Estimated pose data not available for error analysis.")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()









    # Real 

# Very good: 134 144 141 125
# 148 145 142 124 114 136 127
# 131

#Sim
# 110 109 106 130


# python robot-trajectory-analysis-v5.py --real --aruco-csv ../SHANTAM_LOGS/24_Tag_24.csv --number 131 --analyze-error --time 0:60
# Max Run
# - 13:28:21 AM - 13 - Params: Covariance: 1 1 1, Aruco: 0.1 0.1 1, Motion Model: 0.1 0.1 5 0.02 0.01 

# Shantam Run
# - 11:12:02 AM - 113 - Unix: 1741086722
# - 12:30:36 PM - 114 - Unix: 1741091436 -Sim
# - 12:31:59 PM - 115 - Unix: 1741091519
# - 12:32:22 PM - 116 - Unix: 1741091542
# - 12:33:35 PM - 117 - Unix: 1741091615
# - 12:34:23 PM - 118 - Unix: 1741091663
# - 12:35:00 PM - 119 - Unix: 1741091700
# - 12:37:17 PM - 120 - Unix: 1741091837
# - 12:56:47 PM - 121 - Unix: 1741093007
# - 12:59:06 PM - 122 - Unix: 1741093146
# - 1:00:28 PM - 123 - Unix: 1741093228
# - 1:01:59 PM - 124 - Unix: 1741093319 -G
# - 1:05:38 PM - 125 - Unix: 1741093538 -Vg
# - 1:22:04 PM - 126 - Unix: 1741094524 -G
# - 1:25:37 PM - 127 - Unix: 1741094737 -G
# - 1:37:26 PM - 128 - Unix: 1741095446
# - 1:37:45 PM - 129 - Unix: 1741095465
# - 1:40:54 PM - 130 - Unix: 1741095654 -Sim
# - 1:47:08 PM - 131 - Unix: 1741096028 -G NICE 1 lap real 1st half of lab
# - 1:48:17 PM - 132 - Unix: 1741096097
# - 1:48:25 PM - 133 - Unix: 1741096105
# - 1:48:34 PM - 134 - Unix: 1741096114 -Vg - Real many laps
# - 2:21:27 PM - 135 - Unix: 1741098087
# - 2:22:23 PM - 136 - Unix: 1741098143 -G
# - 2:24:40 PM - 137 - Unix: 1741098280
# - 2:25:07 PM - 138 - Unix: 1741098307
# - 2:25:34 PM - 139 - Unix: 1741098334
# - 2:26:26 PM - 140 - Unix: 1741098386
# - 2:31:58 PM - 141 - Unix: 1741098718 -Vg
# - 2:33:40 PM - 142 - Unix: 1741098820 -G - Params: Covariance: 1 1 1, Aruco: 0.001 0.003 0.5, Motion Model: 0.1 0.1 5 0.02 0.05 
# - 2:34:53 PM - 143 - Unix: 1741098893
# - 2:35:16 PM - 144 - Unix: 1741098916 -Vg - Params: Covariance: 1 1 1, Aruco: 0.001 0.003 0.5, Motion Model: 0.2 0.2 5 0.02 0.05 
# - 2:37:47 PM - 145 - Unix: 1741099067 -G
# - 2:39:08 PM - 146 - Unix: 1741099148
# - 2:39:52 PM - 147 - Unix: 1741099192
# - 2:40:40 PM - 148 - Unix: 1741099240 -G
# PF -160




#134!!

# # Loop 0-2
#  - Params: Covariance: 1 1 1, Aruco: 0.1 0.1 1, Motion Model: 0.1 0.1 5 0.02 0.05 
# # Loop 3-5
#  - Params: Covariance: 1 1 1, Aruco: 0.1 0.1 1, Motion Model: 0.01 0.05 3 0.05 0.05 
# # Loop 6-8
#  - Params: Covariance: 1 1 1, Aruco: 0.1 0.1 1, Motion Model: 0.01 0.05 3 0.05 0.05 
# # Loop 9-11
#  - Params: Covariance: 1 1 1, Aruco: 0.1 0.1 1, Motion Model: 0.05 0.05 3 0.05 0.05 
