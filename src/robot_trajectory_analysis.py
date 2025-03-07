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

# def extract_data_from_topics(data_by_topic, aruco_path=None):
#     """
#     Extract trajectory data from different topics following the Jupyter notebook approach.
    
#     Args:
#         data_by_topic (dict): Data organized by topic
#         aruco_path (str, optional): Path to the ArUco CSV file for ground truth
        
#     Returns:
#         tuple: (extracted_data, simulation_flag)
#     """
#     extracted_data = {}
    
#     # Check if this is simulation or real robot data
#     simulation_flag = '/groundtruth' in data_by_topic
    
#     # Extract data for /est_pose (same structure as /aruco)
#     if '/est_pose' in data_by_topic:
#         est_pose_timestamps = [item['message']['header']['stamp'] for item in data_by_topic['/est_pose']]
#         est_pose_northings = [item['message']['pose']['position']['x'] for item in data_by_topic['/est_pose']]
#         est_pose_eastings = [item['message']['pose']['position']['y'] for item in data_by_topic['/est_pose']]
#         est_pose_headings = [(quaternion_to_euler(item['message']['pose']['orientation']['w'], 
#                                                 item['message']['pose']['orientation']['x'], 
#                                                 item['message']['pose']['orientation']['y'], 
#                                                 item['message']['pose']['orientation']['z'])[2]) % (2*np.pi)
#                             for item in data_by_topic['/est_pose']]
        
#         extracted_data['est_pose'] = {
#             'timestamps': est_pose_timestamps,
#             'northings': est_pose_northings,
#             'eastings': est_pose_eastings,
#             'headings': est_pose_headings
#         }
    
#     # Extract ArUco data (reference for real robot)
#     if '/aruco' in data_by_topic:
#         aruco_timestamps = [item['message']['header']['stamp'] for item in data_by_topic['/aruco']]
#         aruco_northings = [item['message']['pose']['position']['x'] for item in data_by_topic['/aruco']]
#         aruco_eastings = [item['message']['pose']['position']['y'] for item in data_by_topic['/aruco']]
#         aruco_headings = [(quaternion_to_euler(item['message']['pose']['orientation']['w'], 
#                                               item['message']['pose']['orientation']['x'], 
#                                               item['message']['pose']['orientation']['y'], 
#                                               item['message']['pose']['orientation']['z'])[2]) % (2*np.pi)
#                           for item in data_by_topic['/aruco']]
        
#         extracted_data['aruco'] = {
#             'timestamps': aruco_timestamps,
#             'northings': aruco_northings,
#             'eastings': aruco_eastings,
#             'headings': aruco_headings
#         }
    
#     # Extract wheel speeds
#     if '/true_wheel_speeds' in data_by_topic:
#         true_wheel_timestamps = [float(item['timestamp']) for item in data_by_topic['/true_wheel_speeds']]
#         true_wheel_speeds = [item['message']['vector']['x'] for item in data_by_topic['/true_wheel_speeds']]
        
#         extracted_data['true_wheel_speeds'] = {
#             'timestamps': true_wheel_timestamps,
#             'speeds': true_wheel_speeds
#         }
    
#     if '/wheel_speeds_cmd' in data_by_topic:
#         wheel_speeds_cmd_timestamps = [float(item['timestamp']) for item in data_by_topic['/wheel_speeds_cmd']]
#         wheel_speeds_cmd = [item['message']['vector']['x'] for item in data_by_topic['/wheel_speeds_cmd']]
        
#         extracted_data['wheel_speeds_cmd'] = {
#             'timestamps': wheel_speeds_cmd_timestamps,
#             'speeds': wheel_speeds_cmd
#         }
    
#     # Extract ground truth data
#     if simulation_flag and '/groundtruth' in data_by_topic:
#         # Simulation groundtruth
#         groundtruth_timestamps = [float(item['timestamp']) for item in data_by_topic['/groundtruth']]
#         groundtruth_northings = [item['message']['position']['x'] for item in data_by_topic['/groundtruth']]
#         groundtruth_eastings = [item['message']['position']['y'] for item in data_by_topic['/groundtruth']]
#         groundtruth_headings = [(quaternion_to_euler(item['message']['orientation']['w'], 
#                                                     item['message']['orientation']['x'], 
#                                                     item['message']['orientation']['y'], 
#                                                     item['message']['orientation']['z'])[2]) % (2*np.pi)
#                                 for item in data_by_topic['/groundtruth']]
        
#         extracted_data['groundtruth'] = {
#             'timestamps': groundtruth_timestamps,
#             'northings': groundtruth_northings,
#             'eastings': groundtruth_eastings,
#             'headings': groundtruth_headings
#         }
#     elif not simulation_flag and aruco_path and os.path.exists(aruco_path):
#         # Real robot - use ArUco CSV as ground truth if available
#         df = pd.read_csv(aruco_path)
        
#         # Get time range from estimated pose data if available
#         if 'est_pose' in extracted_data:
#             start_time = min(extracted_data['est_pose']['timestamps'])
#             end_time = max(extracted_data['est_pose']['timestamps'])
#             filtered_data = df.loc[(df['epoch [s]'] >= start_time) & (df['epoch [s]'] <= end_time)]
            
#             groundtruth_timestamps = filtered_data['epoch [s]'].tolist()
#             groundtruth_northings = filtered_data['x [m]'].tolist()
#             groundtruth_eastings = filtered_data['y [m]'].tolist()
#             groundtruth_headings = (np.deg2rad(filtered_data['yaw [deg]']) % (2*np.pi)).tolist()
            
#             extracted_data['groundtruth'] = {
#                 'timestamps': groundtruth_timestamps,
#                 'northings': groundtruth_northings,
#                 'eastings': groundtruth_eastings,
#                 'headings': groundtruth_headings
#             }
    
#     return extracted_data, simulation_flag




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
        # Modified section: For real robot - use ArUco CSV data at est_pose timestamps
        if 'est_pose' in extracted_data:
            # Read the ArUco CSV file
            df = pd.read_csv(aruco_path)
            
            # Get time range from estimated pose data
            est_pose_timestamps = extracted_data['est_pose']['timestamps']
            start_time = np.min(est_pose_timestamps)
            end_time = np.max(est_pose_timestamps)
            
            # Filter data based on the est_pose timestamp range
            filtered_data = df.loc[(df['epoch [s]'] >= start_time) & (df['epoch [s]'] <= end_time)]
            
            # Extract columns
            groundtruth_timestamps = filtered_data['epoch [s]'].tolist()
            groundtruth_northings = filtered_data['x [m]'].tolist()
            groundtruth_eastings = filtered_data['y [m]'].tolist()
            groundtruth_headings = (np.deg2rad(filtered_data['yaw [deg]']) % (2*np.pi)).tolist()
            
            extracted_data['groundtruth'] = {
                'timestamps': groundtruth_timestamps,
                'northings': groundtruth_northings,
                'eastings': groundtruth_eastings,
                'headings': groundtruth_headings
            }
    
    return extracted_data, simulation_flag

def detect_loops(northings, eastings, distance_threshold=0.1, min_loop_size=20):
    """
    Detect loop completions in the robot trajectory.
    
    Args:
        northings (list): List of x positions
        eastings (list): List of y positions
        distance_threshold (float): Threshold to consider a position as "returning to start"
        min_loop_size (int): Minimum number of points needed for loop detection
        
    Returns:
        list: Indices in the errors list where loops start/end
    """
    if not northings or len(northings) < min_loop_size:
        return []
    
    # Get positions as (x, y) pairs
    positions = list(zip(northings, eastings))
    
    # First position is the starting point of the first loop
    start_pos = positions[0]
    
    loop_indices = [0]  # First index is always a loop start
    in_loop = False
    min_distance_from_start = distance_threshold * 4  # Must travel at least this far to consider a loop
    
    # Track maximum distance from start for each potential loop
    max_distance = 0
    
    for i in range(min_loop_size, len(positions)):
        current_pos = positions[i]
        
        # Calculate distance to starting point
        distance_to_start = sqrt((current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2)
        
        # Update maximum distance seen in this potential loop
        max_distance = max(max_distance, distance_to_start)
        
        # If we're close to the starting point and we've moved away first
        if not in_loop and distance_to_start < distance_threshold and max_distance > min_distance_from_start:
            loop_indices.append(i)
            in_loop = True
            start_pos = current_pos  # Update starting position for next loop
            max_distance = 0  # Reset max distance for next loop
        
        # Reset the "in_loop" flag once we've moved away again
        elif in_loop and distance_to_start > distance_threshold * 2:
            in_loop = False
    
    # Add the last index as the final loop end if we haven't already
    if loop_indices[-1] != len(positions) - 1:
        loop_indices.append(len(positions) - 1)
    
    return loop_indices

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

def plot_trajectory(data, reference_topic, loops=None, highlight_loops=None, output_path=None):
    """
    Plot the robot's trajectory, optionally highlighting specific loops.
    
    Args:
        data (dict): Dictionary with extracted data for each topic
        reference_topic (str): Topic to use as reference ('groundtruth' or 'aruco')
        loops (tuple, optional): Tuple of (start_loop_idx, num_loops) to filter the data
        highlight_loops (list, optional): List of loop boundary indices to highlight
        output_path (str, optional): Path to save the plot, or None to display it
    """
    plt.figure(figsize=(12, 10))
    
    # Plot each trajectory
    color_map = {
        'groundtruth': ('orange', 'Ground Truth'),
        'aruco': ('blue', 'Aruco Markers'),
        'est_pose': ('green', 'Estimated Pose')
    }
    
    for topic, topic_data in data.items():
        if topic in color_map and 'northings' in topic_data and 'eastings' in topic_data:
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            if topic == reference_topic:
                line_style = '-'
                marker = 'x' if topic == 'aruco' else '.'
            else:
                line_style = '--' if topic == 'est_pose' else '-.'
                marker = '.'
                
            plt.plot(eastings, northings, line_style, label=label, color=color, marker=marker)
    
    # Highlight loop boundaries if provided
    ref_data = data.get(reference_topic, None)
    if highlight_loops and len(highlight_loops) >= 2 and ref_data:
        ref_northings = ref_data['northings']
        ref_eastings = ref_data['eastings']
        
        for i in range(len(highlight_loops) - 1):
            if i < len(highlight_loops) - 1:  # Make sure we have a next index
                start_idx = highlight_loops[i]
                
                # Check if the index is valid
                if start_idx < len(ref_northings) and start_idx < len(ref_eastings):
                    # Add a marker for loop start/end
                    plt.plot(ref_eastings[start_idx], ref_northings[start_idx], 'o', color='green', markersize=8, 
                            label=f"Loop {i} Start" if i==0 else f"Loop {i} Start/End")
                    
                    # Calculate a midpoint for the label
                    end_idx = highlight_loops[i+1]
                    mid_idx = (start_idx + end_idx) // 2
                    
                    if mid_idx < len(ref_northings) and mid_idx < len(ref_eastings):
                        # Label the loops on the plot
                        plt.text(ref_eastings[mid_idx], ref_northings[mid_idx], f"Loop {i}", 
                                fontsize=12, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add start/end markers for each trajectory
    for topic, topic_data in data.items():
        if topic in color_map and 'northings' in topic_data and 'eastings' in topic_data:
            northings = topic_data['northings']
            eastings = topic_data['eastings']
            color, label = color_map[topic]
            
            if len(northings) > 0 and len(eastings) > 0:
                # Use marker='o' and set color separately
                plt.plot(eastings[0], northings[0], 'o', color=color, markersize=10, label=f"{label} Start")
                plt.plot(eastings[-1], northings[-1], 'x', color=color, markersize=10, label=f"{label} End")
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.axis('equal')
    
    # Create a clean legend (removing duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
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
    
    color_map = {
        'groundtruth': ('orange', 'Ground Truth Heading'),
        'aruco': ('blue', 'Aruco Heading'),
        'est_pose': ('green', 'Estimated Pose Heading')
    }
    
    for topic, topic_data in data.items():
        if topic in color_map and 'timestamps' in topic_data and 'headings' in topic_data:
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

def main():
    parser = argparse.ArgumentParser(description='Analyze robot trajectory data with loop filtering.')
    
    # Log file selection options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-file', help='Path to a specific log file to analyze')
    log_group.add_argument('--number', type=int, help='Index number of the log file to analyze (1-indexed)')
    
    parser.add_argument('--log-dir', default='../SHANTAM_LOGS/', help='Directory containing log files')
    parser.add_argument('--aruco-csv', help='Path to ArUco CSV file for ground truth (for real robot)')
    parser.add_argument('--simulation', dest='simulation', action='store_true', help='Use groundtruth as reference (default)')
    parser.add_argument('--real', dest='simulation', action='store_false', help='Use aruco markers as reference')
    parser.add_argument('--output', '-o', help='Path to save plots (without extension)')
    
    # Loop detection and visualization options
    parser.add_argument('--loop-threshold', type=float, default=0.1, help='Distance threshold for loop detection (meters)')
    parser.add_argument('--loops', type=str, help='Specify which loops to visualize (e.g., "0,1" for first two loops)')
    parser.add_argument('--all-loops', action='store_true', help='Show all detected loops with numbered markers')
    
    # Plot options
    parser.add_argument('--plot-wheel-speeds', action='store_true', help='Generate wheel speeds plot')
    parser.add_argument('--plot-headings', action='store_true', help='Generate headings plot')
    parser.add_argument('--debug', action='store_true', help='Print debug information about the data')
    
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
            
        # Detect loops in reference trajectory
        reference_data = extracted_data[reference_topic]
        loop_indices = detect_loops(
            reference_data['northings'], 
            reference_data['eastings'], 
            distance_threshold=args.loop_threshold
        )
        
        if not loop_indices:
            print(f"No loops detected in the trajectory. Try adjusting --loop-threshold (currently {args.loop_threshold})")
        else:
            print("\nDetected Loops:")
            for i in range(len(loop_indices) - 1):
                loop_start = loop_indices[i]
                loop_end = loop_indices[i+1]
                
                # Calculate loop duration
                if loop_start < len(reference_data['timestamps']) and loop_end < len(reference_data['timestamps']):
                    loop_duration = reference_data['timestamps'][loop_end] - reference_data['timestamps'][loop_start]
                else:
                    loop_duration = 0
                
                # Get start and end positions
                if loop_start < len(reference_data['northings']) and loop_start < len(reference_data['eastings']):
                    start_point = (reference_data['northings'][loop_start], reference_data['eastings'][loop_start])
                else:
                    start_point = (0, 0)
                
                if loop_end < len(reference_data['northings']) and loop_end < len(reference_data['eastings']):
                    end_point = (reference_data['northings'][loop_end], reference_data['eastings'][loop_end])
                else:
                    end_point = (0, 0)
                
                print(f"  Loop {i}: Points {loop_start} to {loop_end} (Duration: {loop_duration:.2f} seconds)")
                print(f"       Start: ({start_point[0]:.2f}, {start_point[1]:.2f}), End: ({end_point[0]:.2f}, {end_point[1]:.2f})")
        
        # Filter data for specified loops
        filtered_data = extracted_data.copy()
        highlight_loops = None
        
        if args.loops and loop_indices:
            try:
                loop_specs = [int(x) for x in args.loops.split(',')]
                if len(loop_specs) == 1:
                    start_loop = loop_specs[0]
                    num_loops = 2  # Default to 2 loops if only one number provided
                else:
                    start_loop = loop_specs[0]
                    num_loops = loop_specs[1] - loop_specs[0] + 1
                
                # Get time range for the selected loops
                ref_timestamps = reference_data['timestamps']
                start_time = ref_timestamps[loop_indices[start_loop]] if loop_indices[start_loop] < len(ref_timestamps) else 0
                end_loop_idx = min(start_loop + num_loops, len(loop_indices) - 1)
                end_time = ref_timestamps[loop_indices[end_loop_idx]] if loop_indices[end_loop_idx] < len(ref_timestamps) else 0
                
                # Filter all data sources to this time range
                for topic, topic_data in extracted_data.items():
                    filtered_data[topic] = filter_data_by_time(topic_data, start_time, end_time)
                
                print(f"\nVisualization filtered to loops {start_loop} to {start_loop + num_loops - 1}")
                
                # Get highlight indices for the selected loops
                if args.all_loops:
                    highlight_loops = loop_indices
                else:
                    highlight_loops = loop_indices[start_loop:start_loop + num_loops + 1]
            except (ValueError, IndexError) as e:
                print(f"Error parsing loop specification: {e}")
                print("Using full data set instead")
                highlight_loops = None
        elif args.all_loops:
            highlight_loops = loop_indices
        
        # Generate output paths
        loop_suffix = f"_loops_{args.loops.replace(',', '-')}" if args.loops else ""
        
        # Plot the trajectory
        trajectory_output = f"{args.output}{loop_suffix}_trajectory.png" if args.output else None
        plot_trajectory(filtered_data, reference_topic, highlight_loops=highlight_loops, output_path=trajectory_output)
        
        # Plot wheel speeds if requested
        if args.plot_wheel_speeds:
            wheel_speeds_output = f"{args.output}{loop_suffix}_wheel_speeds.png" if args.output else None
            plot_wheel_speeds(filtered_data, output_path=wheel_speeds_output)
        
        # Plot headings if requested
        if args.plot_headings:
            headings_output = f"{args.output}{loop_suffix}_headings.png" if args.output else None
            plot_headings(filtered_data, output_path=headings_output)
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


# python robot_trajectory_analysis.py --real --number 134 --plot-wheel-speeds --plot-headings --loop-threshold 0.25 --loops 3,3




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
        # Modified section: For real robot - use ArUco CSV data at est_pose timestamps
        if 'est_pose' in extracted_data:
            # Read the ArUco CSV file
            df = pd.read_csv(aruco_path)
            
            # Get time range from estimated pose data
            est_pose_timestamps = extracted_data['est_pose']['timestamps']
            start_time = np.min(est_pose_timestamps)
            end_time = np.max(est_pose_timestamps)
            
            # Filter data based on the est_pose timestamp range
            filtered_data = df.loc[(df['epoch [s]'] >= start_time) & (df['epoch [s]'] <= end_time)]
            
            # Extract columns
            groundtruth_timestamps = filtered_data['epoch [s]'].tolist()
            groundtruth_northings = filtered_data['x [m]'].tolist()
            groundtruth_eastings = filtered_data['y [m]'].tolist()
            groundtruth_headings = (np.deg2rad(filtered_data['yaw [deg]']) % (2*np.pi)).tolist()
            
            extracted_data['groundtruth'] = {
                'timestamps': groundtruth_timestamps,
                'northings': groundtruth_northings,
                'eastings': groundtruth_eastings,
                'headings': groundtruth_headings
            }
    
    return extracted_data, simulation_flag




    