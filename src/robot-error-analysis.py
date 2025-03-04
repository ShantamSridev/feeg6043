#!/usr/bin/env python3
"""
Robot Localization Analysis Tool

This script analyzes robot positioning data from log files, comparing the robot's estimated position
(from an Extended Kalman Filter or other estimation method) with a reference position 
(either ground truth in simulation or ArUco marker detections in real-world).

Extended Kalman Filter (EKF) Explanation:
- An EKF is a mathematical algorithm used for robot localization (figuring out where the robot is)
- It combines two sources of information:
  1. Motion predictions (where the robot thinks it moved based on wheel encoders)
  2. Sensor measurements (what the robot actually sees, like ArUco markers)
- The EKF balances these sources of information based on their estimated reliability

This script helps evaluate how well the EKF is performing by measuring the difference
between where the robot thinks it is and where it actually is.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

def load_log_file(file_path):
    """
    Load and parse the robot's log file.
    
    The log file contains JSON-formatted lines with timestamped data about the robot's
    estimated position, ground truth position, sensor readings, etc.
    
    Args:
        file_path (str): Path to the log file to analyze
        
    Returns:
        list: A list of parsed JSON entries, each representing a logged data point
    """
    try:
        # Open the file and read all lines, ignoring empty ones
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        parsed_data = []  # Will hold all the successfully parsed JSON objects
        skipped_lines = 0  # Counter for lines that couldn't be parsed
        
        # Try to parse each line as JSON
        for i, line in enumerate(lines):
            try:
                # Convert the text line to a Python dictionary
                data = json.loads(line)
                parsed_data.append(data)
            except json.JSONDecodeError:
                # If line isn't valid JSON, count it and optionally show an error
                skipped_lines += 1
                if skipped_lines <= 5:  # Only show first few errors to avoid console spam
                    print(f"Warning: Skipping invalid JSON at line {i+1}: {line[:50]}...")
                elif skipped_lines == 6:
                    print("Additional invalid JSON lines found (not showing all warnings)...")
        
        # Report how many lines we had to skip
        if skipped_lines > 0:
            print(f"Total of {skipped_lines} invalid JSON lines skipped")
        
        return parsed_data
    
    except Exception as e:
        print(f"Error loading log file: {e}")
        raise  # Re-raise the exception to be handled by the caller

def find_closest_timestamp(target_time, data_list):
    """
    Find the entry in data_list with the timestamp closest to target_time.
    
    This is necessary because the robot records different types of data (position estimates,
    ground truth, sensor readings) at slightly different times. To compare them fairly,
    we need to match up readings that happened at approximately the same time.
    
    Args:
        target_time (float): The timestamp we want to find a match for
        data_list (list): List of data points to search through
        
    Returns:
        tuple: (closest_item, time_difference)
            - closest_item: The entry with the closest timestamp
            - time_difference: How far apart the timestamps are (in seconds)
    """
    closest = None
    min_diff = float('inf')  # Start with infinity as the smallest difference
    
    # Check each item in the list
    for item in data_list:
        timestamp = float(item['timestamp'])
        diff = abs(timestamp - target_time)  # Calculate absolute time difference
        
        # If this is the closest match so far, remember it
        if diff < min_diff:
            min_diff = diff
            closest = item
    
    return closest, min_diff

def extract_position(data_item):
    """
    Extract position coordinates (x, y, z) from different message formats.
    
    Different data sources (groundtruth, EKF, ArUco) might store position information
    in different formats within the log. This function handles all these formats to
    extract the position consistently.
    
    Args:
        data_item (dict): A single data point from the log
        
    Returns:
        dict: Position information with 'x', 'y', 'z' keys
    """
    try:
        # First, check for groundtruth format
        if data_item['topic_name'] == '/groundtruth':
            # Ground truth stores position directly
            return data_item['message']['position']
        
        # Check for common estimation formats (EKF, ArUco, odometry)
        elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
            # These might have different structures - try common patterns
            if 'pose' in data_item['message']:
                return data_item['message']['pose']['position']
            elif 'position' in data_item['message']:
                return data_item['message']['position']
            else:
                # Special case for odometry messages
                if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
                    return data_item['message']['pose']['pose']['position']
        
        # If the above patterns didn't work, try a more general approach
        message = data_item['message']
        if isinstance(message, dict):
            # Search for any key that might contain position information
            for key in message:
                if isinstance(message[key], dict) and 'position' in message[key]:
                    return message[key]['position']
                elif key == 'position' and isinstance(message[key], dict):
                    return message[key]
                
            # Look for nested pose structures
            for key in message:
                if isinstance(message[key], dict) and 'pose' in message[key]:
                    if 'position' in message[key]['pose']:
                        return message[key]['pose']['position']
        
        # If we still can't find position data, give up and report the problem
        raise ValueError(f"Cannot extract position from topic {data_item['topic_name']}. Message structure: {data_item['message']}")
    
    except Exception as e:
        print(f"Error extracting position from {data_item['topic_name']}: {e}")
        print(f"Message structure: {data_item['message']}")
        raise

def extract_orientation(data_item):
    """
    Extract orientation (quaternion) from different message formats.
    
    Similar to extract_position(), but for orientation data. Orientation is typically
    stored as a quaternion (x, y, z, w values) that represents the robot's 3D rotation.
    
    Args:
        data_item (dict): A single data point from the log
        
    Returns:
        dict: Orientation as quaternion with 'x', 'y', 'z', 'w' keys, or None if unavailable
    """
    try:
        # First, check for groundtruth format
        if data_item['topic_name'] == '/groundtruth':
            return data_item['message']['orientation']
        
        # Check for common estimation formats
        elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
            if 'pose' in data_item['message']:
                return data_item['message']['pose']['orientation']
            elif 'orientation' in data_item['message']:
                return data_item['message']['orientation']
            else:
                # Special case for odometry messages
                if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
                    return data_item['message']['pose']['pose']['orientation']
        
        # Try general approach if the above patterns didn't work
        message = data_item['message']
        if isinstance(message, dict):
            # Search through the message for orientation information
            for key in message:
                if isinstance(message[key], dict) and 'orientation' in message[key]:
                    return message[key]['orientation']
                elif key == 'orientation' and isinstance(message[key], dict):
                    return message[key]
                
            # Look for nested pose structures
            for key in message:
                if isinstance(message[key], dict) and 'pose' in message[key]:
                    if 'orientation' in message[key]['pose']:
                        return message[key]['pose']['orientation']
        
        # Unlike with position, we return None if orientation is missing
        # This allows us to still compute position error even without orientation data
        return None
    
    except Exception as e:
        print(f"Error extracting orientation from {data_item['topic_name']}: {e}")
        return None

def calculate_euclidean_distance(pos1, pos2):
    """
    Calculate the straight-line (Euclidean) distance between two 3D positions.
    
    This is the most intuitive measure of "how far apart" two positions are.
    It's calculated using the Pythagorean theorem extended to 3D.
    
    Args:
        pos1 (dict): First position with 'x', 'y', 'z' keys
        pos2 (dict): Second position with 'x', 'y', 'z' keys
        
    Returns:
        float: The Euclidean distance in meters
    """
    # Apply the 3D Pythagorean theorem: sqrt(dx² + dy² + dz²)
    return sqrt(
        (pos1['x'] - pos2['x'])**2 + 
        (pos1['y'] - pos2['y'])**2 + 
        (pos1['z'] - pos2['z'])**2
    )

def calculate_orientation_error(orient1, orient2):
    """
    Calculate the angular difference between two orientations represented as quaternions.
    
    Quaternions are a mathematical way to represent 3D rotations. This function
    calculates the smallest angle (in radians) between the two orientations.
    
    Args:
        orient1 (dict): First orientation as quaternion with 'x', 'y', 'z', 'w' keys
        orient2 (dict): Second orientation as quaternion with 'x', 'y', 'z', 'w' keys
        
    Returns:
        float: The angle between orientations in radians, or None if inputs are invalid
    """
    # Check if we have valid orientation data
    if orient1 is None or orient2 is None:
        return None
    
    # Convert from dictionary format to list format
    q1 = [orient1['x'], orient1['y'], orient1['z'], orient1['w']]
    q2 = [orient2['x'], orient2['y'], orient2['z'], orient2['w']]
    
    # Calculate dot product of quaternions
    # The dot product tells us how "aligned" the quaternions are
    dot_product = sum(a*b for a, b in zip(q1, q2))
    
    # Ensure dot product is within valid range [-1, 1] to avoid math errors
    dot_product = max(-1, min(1, dot_product))
    
    # Calculate angle between quaternions
    # We use absolute value because quaternions q and -q represent the same rotation
    angle = 2 * np.arccos(abs(dot_product))
    
    return angle

def analyze_error(log_data, is_simulation=True, max_time_diff=0.1):
    """
    Analyze position and orientation error over time by comparing estimated and reference positions.
    
    This is the core analysis function that:
    1. Finds corresponding pairs of reference and estimated positions
    2. Calculates the error between them
    3. Compiles the results for plotting and analysis
    
    In simulation, ground truth is used as reference. In real-world, ArUco markers are used.
    
    Args:
        log_data (list): Parsed log data to analyze
        is_simulation (bool): Whether we're analyzing simulation data (True) or real robot data (False)
        max_time_diff (float): Maximum allowed time difference between reference and estimate
        
    Returns:
        list: A list of dictionaries containing error information at each timestep
    """
    # Determine which data source to use as the reference (ground truth)
    # In simulation, we have perfect ground truth
    # In real-world, we use ArUco markers as our best available reference
    reference_topic = '/groundtruth' if is_simulation else '/aruco'
    
    # Determine which data source to use as the estimate
    # We prefer '/est_pose' (the EKF output) but can fall back to '/odom' if needed
    if is_simulation:
        estimation_topics = ['/est_pose', '/odom']
    else:
        estimation_topics = ['/est_pose', '/odom']  # Real robot also prefers est_pose
    
    # Filter log data to get only the reference information
    reference_data = [item for item in log_data if item['topic_name'] == reference_topic]
    
    # Make sure we have reference data
    if not reference_data:
        # If not, list what topics are available to help diagnose the problem
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No {reference_topic} data found in log. Available topics: {available_topics}")
    
    # Try to find the estimation data from our preferred sources
    estimate_topic = None
    estimate_data = []
    
    for topic in estimation_topics:
        temp_data = [item for item in log_data if item['topic_name'] == topic]
        if temp_data:
            estimate_topic = topic
            estimate_data = temp_data
            break
    
    # Make sure we have estimation data
    if not estimate_data:
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No estimation data found. Available topics in log: {available_topics}")
    
    # Report what data sources we're using
    print(f"Using {reference_topic} as reference (found {len(reference_data)} entries)")
    print(f"Using {estimate_topic} as estimation (found {len(estimate_data)} entries)")
    
    # Now analyze the errors by comparing reference and estimation at matching times
    errors = []  # Will hold all our error calculations
    start_time = float(reference_data[0]['timestamp'])  # Use this to calculate relative times
    
    # Counters for skipped and processed data pairs
    skipped_pairs = 0
    processed_pairs = 0
    orientation_errors_computed = 0
    
    # For each reference data point, find the closest matching estimation
    for ref_item in reference_data:
        ref_time = float(ref_item['timestamp'])
        # Find the estimation data point closest in time to this reference point
        est_item, time_diff = find_closest_timestamp(ref_time, estimate_data)
        
        # Only use the pair if the timestamps are close enough
        # This ensures we're comparing measurements that happened at roughly the same time
        if time_diff <= max_time_diff:
            try:
                # Extract position information from both reference and estimation
                ref_pos = extract_position(ref_item)
                est_pos = extract_position(est_item)
                
                # Try to extract orientation information
                ref_orient = extract_orientation(ref_item)
                est_orient = extract_orientation(est_item)
                
                # Calculate position errors in each dimension and overall
                error_x = ref_pos['x'] - est_pos['x']  # Error in x-direction
                error_y = ref_pos['y'] - est_pos['y']  # Error in y-direction
                error_z = ref_pos['z'] - est_pos['z']  # Error in z-direction
                euclidean_error = calculate_euclidean_distance(ref_pos, est_pos)  # Overall distance error
                
                # Calculate orientation error if orientation data is available
                orientation_error = None
                if ref_orient is not None and est_orient is not None:
                    orientation_error = calculate_orientation_error(ref_orient, est_orient)
                    if orientation_error is not None:
                        orientation_errors_computed += 1
                
                # Store all the calculated information for this time point
                errors.append({
                    'time': ref_time - start_time,  # Time relative to start
                    'timestamp': ref_time,          # Absolute timestamp
                    'ref_x': ref_pos['x'],          # Reference x position
                    'ref_y': ref_pos['y'],          # Reference y position
                    'ref_z': ref_pos['z'],          # Reference z position
                    'est_x': est_pos['x'],          # Estimated x position
                    'est_y': est_pos['y'],          # Estimated y position
                    'est_z': est_pos['z'],          # Estimated z position
                    'error_x': error_x,             # Error in x
                    'error_y': error_y,             # Error in y
                    'error_z': error_z,             # Error in z
                    'euclidean_error': euclidean_error,  # Overall distance error
                    'orientation_error': orientation_error,  # Angular error
                    'time_diff': time_diff          # Time difference between readings
                })
                processed_pairs += 1
            except Exception as e:
                print(f"Error processing data point at time {ref_time}: {e}")
                continue  # Skip this point if there was an error
        else:
            # Skip this pair because the timestamps are too far apart
            skipped_pairs += 1
    
    # Report how many data pairs we processed and skipped
    print(f"Processed {processed_pairs} timestamp pairs, skipped {skipped_pairs} pairs due to timestamp difference > {max_time_diff}s")
    
    # Report orientation error statistics
    if orientation_errors_computed > 0:
        print(f"Orientation errors computed for {orientation_errors_computed} pairs")
    else:
        print("No orientation errors computed (missing orientation data)")
    
    # Make sure we have at least some data points
    if not errors:
        raise ValueError("No valid data points found for error calculation. Try increasing max_time_diff.")
    
    return errors

def plot_error(errors, output_path=None, reference_name="Ground Truth"):
    """
    Create plots showing position and orientation error over time.
    
    This visualizes how the robot's estimation error changes throughout the experiment,
    helping identify when and where the localization performed well or poorly.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot titles
        
    Returns:
        tuple: (avg_error, max_error) - The average and maximum position errors
    """
    # Check if we have data to plot
    if not errors:
        print("No error data to plot")
        return
    
    # Extract data for plotting
    times = [error['time'] for error in errors]
    euclidean_errors = [error['euclidean_error'] for error in errors]
    error_x = [error['error_x'] for error in errors]
    error_y = [error['error_y'] for error in errors]
    error_z = [error['error_z'] for error in errors]
    
    # Check if we have orientation error data
    has_orientation = any(error.get('orientation_error') is not None for error in errors)
    if has_orientation:
        orientation_errors = [error.get('orientation_error', 0) for error in errors]
        # Convert radians to degrees for easier interpretation
        orientation_errors = [error * 180 / np.pi if error is not None else 0 for error in orientation_errors]
    
    # Calculate basic statistics about the errors
    avg_error = sum(euclidean_errors) / len(euclidean_errors)
    max_error = max(euclidean_errors)
    
    # Create figures for plotting - either 2 or 3 subplots depending on orientation data
    if has_orientation:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Overall position error (Euclidean distance) over time
    ax1.plot(times, euclidean_errors, 'b-', label=f'Error vs {reference_name}')
    ax1.axhline(y=avg_error, color='r', linestyle='--', label=f'Avg Error: {avg_error:.4f} m')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Euclidean Error (meters)')
    ax1.set_title(f'Position Error vs {reference_name} Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Component-wise position errors (X, Y, Z)
    ax2.plot(times, error_x, 'r-', label='X Error')
    ax2.plot(times, error_y, 'g-', label='Y Error')
    ax2.plot(times, error_z, 'b-', label='Z Error')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Component Error (meters)')
    ax2.set_title('X, Y, Z Component Errors Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Orientation error over time (if available)
    if has_orientation:
        ax3.plot(times, orientation_errors, 'm-', label='Orientation Error')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Orientation Error (degrees)')
        ax3.set_title('Orientation Error Over Time')
        ax3.grid(True)
        ax3.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Either save the plot to a file or display it
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return avg_error, max_error

def plot_trajectory(errors, output_path=None, reference_name="Ground Truth"):
    """
    Plot the robot's trajectory (path) according to both reference and estimated positions.
    
    This shows the actual path the robot took versus the path it thought it was taking,
    giving a visual representation of the localization accuracy.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot labels
    """
    if not errors:
        print("No data to plot trajectory")
        return
    
    # Extract trajectory data - the x,y coordinates of both reference and estimated positions
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Plot reference and estimated trajectories
    plt.plot(ref_x, ref_y, 'b-', label=reference_name)  # Blue solid line for reference
    plt.plot(est_x, est_y, 'r--', label='Estimated Position')  # Red dashed line for estimate
    
    # Add arrows to show the direction of movement
    # We only add a limited number of arrows to avoid cluttering the plot
    arrow_indices = np.linspace(0, len(ref_x)-1, min(20, len(ref_x))).astype(int)
    for i in arrow_indices:
        if i+1 < len(ref_x):
            # Add arrow for reference trajectory
            plt.arrow(ref_x[i], ref_y[i], 
                     (ref_x[i+1]-ref_x[i])*0.5, (ref_y[i+1]-ref_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='b', ec='b')
            
            # Add arrow for estimated trajectory
            plt.arrow(est_x[i], est_y[i], 
                     (est_x[i+1]-est_x[i])*0.5, (est_y[i+1]-est_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='r', ec='r')
    
    # Mark start and end points clearly
    plt.plot(ref_x[0], ref_y[0], 'bo', markersize=10, label=f"{reference_name} Start")
    plt.plot(ref_x[-1], ref_y[-1], 'bx', markersize=10, label=f"{reference_name} End")
    plt.plot(est_x[0], est_y[0], 'ro', markersize=10, label="Estimated Start")
    plt.plot(est_x[-1], est_y[-1], 'rx', markersize=10, label="Estimated End")
    
    # Label the plot
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.axis('equal')  # Ensure x and y axes have equal scale
    plt.legend()
    
    # Either save the plot to a file or display it
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()

def plot_position_over_time(errors, output_path=None, reference_name="Ground Truth"):
    """
    Plot the X and Y positions of the robot over time, comparing reference vs. estimated.
    
    This shows how the robot's position changed throughout the experiment and
    helps identify when the estimation deviated from the reference.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot labels
    """
    if not errors:
        print("No data to plot positions over time")
        return
    
    # Extract time and position data
    times = [error['time'] for error in errors]
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    # Create a figure with two subplots (one for X position, one for Y position)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot X position over time
    ax1.plot(times, ref_x, 'b-', label=f'{reference_name} X')
    ax1.plot(times, est_x, 'r--', label='Estimated X')
    ax1.set_ylabel('X Position (meters)')
    ax1.set_title('X Position Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Y position over time
    ax2.plot(times, ref_y, 'b-', label=f'{reference_name} Y')
    ax2.plot(times, est_y, 'r--', label='Estimated Y')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Y Position (meters)')
    ax2.set_title('Y Position Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Either save the plot to a file or display it
    if output_path:
        plt.savefig(output_path)
        print(f"Position over time plot saved to {output_path}")
    else:
        plt.show()

def main():
    """
    Main function that parses command-line arguments and runs the analysis.
    
    This is the entry point of the script when run from the command line.
    It sets up the command-line interface, processes the arguments, and
    executes the appropriate analysis functions.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Analyze robot positioning error from log files.')
    
    # Define command line arguments
    parser.add_argument('--log-file', default='../logs/20250304_134834_log.json', 
                        help='Path to the log file to analyze')
    parser.add_argument('--simulation', dest='simulation', action='store_true',
                        help='Use groundtruth as reference (default)')
    
    parser.add_argument('--real', dest='simulation', action='store_false',
                        help='Use aruco markers as reference')
    
    parser.add_argument('--max-time-diff', type=float, default=0.05, 
                        help='Maximum time difference between reference and estimate (seconds)')
    
    parser.add_argument('--output', '-o', 
                        help='Path to save the plot (without extension)')
    
    parser.add_argument('--trajectory', '-t', action='store_true', 
                        help='Generate trajectory plot')
    
    parser.add_argument('--debug', action='store_true', 
                        help='Print debug information')
    
    # Set default value for simulation flag
    parser.set_defaults(simulation=True)  # Default to simulation mode
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    try:
        # Step 1: Load and parse the log file
        log_data = load_log_file(args.log_file)
        print(f"Loaded {len(log_data)} log entries")
        
        # Print detailed debug information if requested
        if args.debug:
            print("\nAvailable topics:")
            topics = set(item['topic_name'] for item in log_data)
            for topic in sorted(topics):
                count = sum(1 for item in log_data if item['topic_name'] == topic)
                print(f"  {topic}: {count} entries")
            
            print("\nSample log entries:")
            for i, topic in enumerate(sorted(topics)):
                sample = next((item for item in log_data if item['topic_name'] == topic), None)
                if sample:
                    print(f"\nTopic {i+1}: {topic}")
                    print(json.dumps(sample, indent=2)[0:500] + "...")
        
        # Determine what we're using as the reference source
        reference_name = "Ground Truth" if args.simulation else "Aruco Markers"
        print(f"Mode: {'Simulation' if args.simulation else 'Real robot'} (using {reference_name} as reference)")
        
        # Step 2: Analyze the position error
        errors = analyze_error(log_data, args.simulation, args.max_time_diff)
        
        if not errors:
            print("No matching data points found for error calculation")
            return
        
        # Step 3: Calculate and print error statistics
        euclidean_errors = [error['euclidean_error'] for error in errors]
        avg_error = sum(euclidean_errors) / len(euclidean_errors)
        max_error = max(euclidean_errors)
        min_error = min(euclidean_errors)
        
        print("\nError Statistics:")
        print(f"  Number of data points: {len(errors)}")
        print(f"  Average Error: {avg_error:.4f} meters")
        print(f"  Minimum Error: {min_error:.4f} meters")
        print(f"  Maximum Error: {max_error:.4f} meters")
        print(f"  Initial Error: {errors[0]['euclidean_error']:.4f} meters")
        print(f"  Final Error: {errors[-1]['euclidean_error']:.4f} meters")
        
        # Print orientation error statistics if available
        has_orientation = any(error.get('orientation_error') is not None for error in errors)
        if has_orientation:
            valid_orient_errors = [err['orientation_error'] for err in errors if err.get('orientation_error') is not None]
            if valid_orient_errors:
                avg_orient_error = sum(valid_orient_errors) / len(valid_orient_errors)
                max_orient_error = max(valid_orient_errors)
                # Convert to degrees for easier interpretation
                avg_orient_error_deg = avg_orient_error * 180 / np.pi
                max_orient_error_deg = max_orient_error * 180 / np.pi
                print(f"  Average Orientation Error: {avg_orient_error_deg:.4f} degrees")
                print(f"  Maximum Orientation Error: {max_orient_error_deg:.4f} degrees")
        
        # Step 4: Generate plots
        
        # Plot 1: Error over time
        error_output = f"{args.output}_error.png" if args.output else None
        plot_error(errors, error_output, reference_name)
        
        # Plot 2: Trajectory (optional)
        if args.trajectory:
            traj_output = f"{args.output}_trajectory.png" if args.output else None
            plot_trajectory(errors, traj_output, reference_name)
        
        # Plot 3: Position over time
        pos_output = f"{args.output}_position_time.png" if args.output else None
        plot_position_over_time(errors, pos_output, reference_name)
        
    except FileNotFoundError:
        # Handle case where log file doesn't exist
        print(f"Error: Could not find log file at {args.log_file}")
        print("Current working directory:", os.getcwd())
        print("Please check the file path and try again.")
    except Exception as e:
        # Handle any other exceptions
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    main()















# def log_covariance(self):
#     """
#     Logs the current covariance matrix of the EKF.
    
#     This function creates a JSON-serializable dictionary containing:
#     1. The full covariance matrix
#     2. Current process noise parameters
#     3. Current measurement noise parameters
#     4. Timestamp
    
#     This is essential for evaluating filter consistency, diagnosing issues,
#     and properly analyzing the effects of different parameter values.
#     """
#     # Create a dictionary with timestamp and covariance data
#     covariance_data = {
#         "timestamp": datetime.utcnow().timestamp(),
#         # Main diagonal elements (variances) - most important values
#         "position_uncertainty": {
#             "north_variance": float(self.covariance[self.N, self.N]),
#             "east_variance": float(self.covariance[self.E, self.E]),
#             "heading_variance": float(self.covariance[self.G, self.G]),
#             "velocity_variance": float(self.covariance[self.DOTX, self.DOTX]),
#             "angular_rate_variance": float(self.covariance[self.DOTG, self.DOTG])
#         },
#         # Current process noise parameters
#         "process_noise_params": {
#             "R_N": float(self.R_N),
#             "R_E": float(self.R_E),
#             "R_G": float(self.R_G) if hasattr(self, 'R_G') else 0.0,
#             "dot_x_R_std": float(self.dot_x_R_std[0, 0]) if hasattr(self.dot_x_R_std, 'shape') else float(self.dot_x_R_std),
#             "dot_g_R_std": float(self.dot_g_R_std[0, 0]) if hasattr(self.dot_g_R_std, 'shape') else float(self.dot_g_R_std)
#         },
#         # Current measurement noise parameters
#         "measurement_noise_params": {
#             "NE_Q_std_N": float(self.NE_Q_std[0, 0]),
#             "NE_Q_std_E": float(self.NE_Q_std[0, 1]) if self.NE_Q_std.shape[1] > 1 else float(self.NE_Q_std[0, 0]),
#             "g_Q_std": float(self.g_Q_std[0])
#         }
#     }
    
#     # Log the covariance data
#     self.datalog.log(covariance_data, topic_name="/covariance")