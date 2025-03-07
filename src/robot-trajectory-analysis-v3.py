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
        
        # Plot the trajectory
        trajectory_output = f"{args.output}{time_suffix}_trajectory.png" if args.output else None
        plot_trajectory(filtered_data, reference_topic, output_path=trajectory_output)
        
        # Plot wheel speeds if requested
        if args.plot_wheel_speeds:
            wheel_speeds_output = f"{args.output}{time_suffix}_wheel_speeds.png" if args.output else None
            plot_wheel_speeds(filtered_data, output_path=wheel_speeds_output)
        
        # Plot headings if requested
        if args.plot_headings:
            headings_output = f"{args.output}{time_suffix}_headings.png" if args.output else None
            plot_headings(filtered_data, output_path=headings_output)
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

# Example usage:
# python robot_trajectory_analysis.py --real --number 134 --plot-wheel-speeds --plot-headings --time 5:10 --aruco-csv ../SHANTAM_LOGS/24_Tag_24.csv
# python robot_trajectory_analysis.py --real --number 134 --plot-wheel-speeds --plot-headings --time all --aruco-csv ../SHANTAM_LOGS/24_Tag_24.csv