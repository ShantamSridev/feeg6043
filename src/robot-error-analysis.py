# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from math import sqrt
# import glob

# def load_log_file(file_path):
#     """
#     Load and parse the robot's log file.
    
#     Args:
#         file_path (str): Path to the log file to analyze
        
#     Returns:
#         list: A list of parsed JSON entries, each representing a logged data point
#     """
#     try:
#         with open(file_path, 'r') as file:
#             lines = [line.strip() for line in file if line.strip()]
        
#         parsed_data = []
#         skipped_lines = 0
        
#         for i, line in enumerate(lines):
#             try:
#                 data = json.loads(line)
#                 parsed_data.append(data)
#             except json.JSONDecodeError:
#                 skipped_lines += 1
#                 if skipped_lines <= 5:
#                     print(f"Warning: Skipping invalid JSON at line {i+1}: {line[:50]}...")
#                 elif skipped_lines == 6:
#                     print("Additional invalid JSON lines found (not showing all warnings)...")
        
#         if skipped_lines > 0:
#             print(f"Total of {skipped_lines} invalid JSON lines skipped")
        
#         return parsed_data
    
#     except Exception as e:
#         print(f"Error loading log file: {e}")
#         raise

# def find_closest_timestamp(target_time, data_list):
#     """
#     Find the entry in data_list with the timestamp closest to target_time.
    
#     Args:
#         target_time (float): The timestamp we want to find a match for
#         data_list (list): List of data points to search through
        
#     Returns:
#         tuple: (closest_item, time_difference)
#     """
#     closest = None
#     min_diff = float('inf')
    
#     for item in data_list:
#         timestamp = float(item['timestamp'])
#         diff = abs(timestamp - target_time)
        
#         if diff < min_diff:
#             min_diff = diff
#             closest = item
    
#     return closest, min_diff

# def extract_position(data_item):
#     """
#     Extract position coordinates (x, y, z) from different message formats.
    
#     Args:
#         data_item (dict): A single data point from the log
        
#     Returns:
#         dict: Position information with 'x', 'y', 'z' keys
#     """
#     try:
#         if data_item['topic_name'] == '/groundtruth':
#             return data_item['message']['position']
        
#         elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
#             if 'pose' in data_item['message']:
#                 return data_item['message']['pose']['position']
#             elif 'position' in data_item['message']:
#                 return data_item['message']['position']
#             else:
#                 if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
#                     return data_item['message']['pose']['pose']['position']
        
#         message = data_item['message']
#         if isinstance(message, dict):
#             for key in message:
#                 if isinstance(message[key], dict) and 'position' in message[key]:
#                     return message[key]['position']
#                 elif key == 'position' and isinstance(message[key], dict):
#                     return message[key]
                
#             for key in message:
#                 if isinstance(message[key], dict) and 'pose' in message[key]:
#                     if 'position' in message[key]['pose']:
#                         return message[key]['pose']['position']
        
#         raise ValueError(f"Cannot extract position from topic {data_item['topic_name']}. Message structure: {data_item['message']}")
    
#     except Exception as e:
#         print(f"Error extracting position from {data_item['topic_name']}: {e}")
#         print(f"Message structure: {data_item['message']}")
#         raise

# def extract_orientation(data_item):
#     """
#     Extract orientation (quaternion) from different message formats.
    
#     Args:
#         data_item (dict): A single data point from the log
        
#     Returns:
#         dict: Orientation as quaternion with 'x', 'y', 'z', 'w' keys, or None if unavailable
#     """
#     try:
#         if data_item['topic_name'] == '/groundtruth':
#             return data_item['message']['orientation']
        
#         elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
#             if 'pose' in data_item['message']:
#                 return data_item['message']['pose']['orientation']
#             elif 'orientation' in data_item['message']:
#                 return data_item['message']['orientation']
#             else:
#                 if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
#                     return data_item['message']['pose']['pose']['orientation']
        
#         message = data_item['message']
#         if isinstance(message, dict):
#             for key in message:
#                 if isinstance(message[key], dict) and 'orientation' in message[key]:
#                     return message[key]['orientation']
#                 elif key == 'orientation' and isinstance(message[key], dict):
#                     return message[key]
                
#             for key in message:
#                 if isinstance(message[key], dict) and 'pose' in message[key]:
#                     if 'orientation' in message[key]['pose']:
#                         return message[key]['pose']['orientation']
        
#         return None
    
#     except Exception as e:
#         print(f"Error extracting orientation from {data_item['topic_name']}: {e}")
#         return None

# def calculate_euclidean_distance(pos1, pos2):
#     """
#     Calculate the straight-line (Euclidean) distance between two 3D positions.
    
#     Args:
#         pos1 (dict): First position with 'x', 'y', 'z' keys
#         pos2 (dict): Second position with 'x', 'y', 'z' keys
        
#     Returns:
#         float: The Euclidean distance in meters
#     """
#     return sqrt(
#         (pos1['x'] - pos2['x'])**2 + 
#         (pos1['y'] - pos2['y'])**2 + 
#         (pos1['z'] - pos2['z'])**2
#     )

# def calculate_orientation_error(orient1, orient2):
#     """
#     Calculate the angular difference between two orientations represented as quaternions.
    
#     Args:
#         orient1 (dict): First orientation as quaternion with 'x', 'y', 'z', 'w' keys
#         orient2 (dict): Second orientation as quaternion with 'x', 'y', 'z', 'w' keys
        
#     Returns:
#         float: The angle between orientations in radians, or None if inputs are invalid
#     """
#     if orient1 is None or orient2 is None:
#         return None
    
#     q1 = [orient1['x'], orient1['y'], orient1['z'], orient1['w']]
#     q2 = [orient2['x'], orient2['y'], orient2['z'], orient2['w']]
    
#     dot_product = sum(a*b for a, b in zip(q1, q2))
#     dot_product = max(-1, min(1, dot_product))
    
#     angle = 2 * np.arccos(abs(dot_product))
    
#     return angle

# def analyze_error(log_data, is_simulation=True, max_time_diff=0.1):
#     """
#     Analyze position and orientation error over time by comparing estimated and reference positions.
    
#     Args:
#         log_data (list): Parsed log data to analyze
#         is_simulation (bool): Whether we're analyzing simulation data (True) or real robot data (False)
#         max_time_diff (float): Maximum allowed time difference between reference and estimate
        
#     Returns:
#         list: A list of dictionaries containing error information at each timestep
#     """
#     reference_topic = '/groundtruth' if is_simulation else '/aruco'
    
#     if is_simulation:
#         estimation_topics = ['/est_pose', '/odom']
#     else:
#         estimation_topics = ['/est_pose', '/odom']
    
#     reference_data = [item for item in log_data if item['topic_name'] == reference_topic]
    
#     if not reference_data:
#         available_topics = set(item['topic_name'] for item in log_data)
#         raise ValueError(f"No {reference_topic} data found in log. Available topics: {available_topics}")
    
#     estimate_topic = None
#     estimate_data = []
    
#     for topic in estimation_topics:
#         temp_data = [item for item in log_data if item['topic_name'] == topic]
#         if temp_data:
#             estimate_topic = topic
#             estimate_data = temp_data
#             break
    
#     if not estimate_data:
#         available_topics = set(item['topic_name'] for item in log_data)
#         raise ValueError(f"No estimation data found. Available topics in log: {available_topics}")
    
#     print(f"Using {reference_topic} as reference (found {len(reference_data)} entries)")
#     print(f"Using {estimate_topic} as estimation (found {len(estimate_data)} entries)")
    
#     errors = []
#     start_time = float(reference_data[0]['timestamp'])
    
#     skipped_pairs = 0
#     processed_pairs = 0
#     orientation_errors_computed = 0
    
#     for ref_item in reference_data:
#         ref_time = float(ref_item['timestamp'])
#         est_item, time_diff = find_closest_timestamp(ref_time, estimate_data)
        
#         if time_diff <= max_time_diff:
#             try:
#                 ref_pos = extract_position(ref_item)
#                 est_pos = extract_position(est_item)
                
#                 ref_orient = extract_orientation(ref_item)
#                 est_orient = extract_orientation(est_item)
                
#                 error_x = ref_pos['x'] - est_pos['x']
#                 error_y = ref_pos['y'] - est_pos['y']
#                 error_z = ref_pos['z'] - est_pos['z']
#                 euclidean_error = calculate_euclidean_distance(ref_pos, est_pos)
                
#                 orientation_error = None
#                 if ref_orient is not None and est_orient is not None:
#                     orientation_error = calculate_orientation_error(ref_orient, est_orient)
#                     if orientation_error is not None:
#                         orientation_errors_computed += 1
                
#                 errors.append({
#                     'time': ref_time - start_time,
#                     'timestamp': ref_time,
#                     'ref_x': ref_pos['x'],
#                     'ref_y': ref_pos['y'],
#                     'ref_z': ref_pos['z'],
#                     'est_x': est_pos['x'],
#                     'est_y': est_pos['y'],
#                     'est_z': est_pos['z'],
#                     'error_x': error_x,
#                     'error_y': error_y,
#                     'error_z': error_z,
#                     'euclidean_error': euclidean_error,
#                     'orientation_error': orientation_error,
#                     'time_diff': time_diff
#                 })
#                 processed_pairs += 1
#             except Exception as e:
#                 print(f"Error processing data point at time {ref_time}: {e}")
#                 continue
#         else:
#             skipped_pairs += 1
    
#     print(f"Processed {processed_pairs} timestamp pairs, skipped {skipped_pairs} pairs due to timestamp difference > {max_time_diff}s")
    
#     if orientation_errors_computed > 0:
#         print(f"Orientation errors computed for {orientation_errors_computed} pairs")
#     else:
#         print("No orientation errors computed (missing orientation data)")
    
#     if not errors:
#         raise ValueError("No valid data points found for error calculation. Try increasing max_time_diff.")
    
#     return errors

# def plot_error(errors, output_path=None, reference_name="Ground Truth"):
#     """
#     Create plots showing position and orientation error over time.
    
#     Args:
#         errors (list): List of error data points from analyze_error()
#         output_path (str, optional): Path to save the plot, or None to display it
#         reference_name (str): Name of the reference source for the plot titles
        
#     Returns:
#         tuple: (avg_error, max_error) - The average and maximum position errors
#     """
#     if not errors:
#         print("No error data to plot")
#         return
    
#     times = [error['time'] for error in errors]
#     euclidean_errors = [error['euclidean_error'] for error in errors]
#     error_x = [error['error_x'] for error in errors]
#     error_y = [error['error_y'] for error in errors]
#     error_z = [error['error_z'] for error in errors]
    
#     has_orientation = any(error.get('orientation_error') is not None for error in errors)
#     if has_orientation:
#         orientation_errors = [error.get('orientation_error', 0) for error in errors]
#         orientation_errors = [error * 180 / np.pi if error is not None else 0 for error in orientation_errors]
    
#     avg_error = sum(euclidean_errors) / len(euclidean_errors)
#     max_error = max(euclidean_errors)
    
#     if has_orientation:
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
#     else:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
#     ax1.plot(times, euclidean_errors, 'b-', label=f'Error vs {reference_name}')
#     ax1.axhline(y=avg_error, color='r', linestyle='--', label=f'Avg Error: {avg_error:.4f} m')
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Euclidean Error (meters)')
#     ax1.set_title(f'Position Error vs {reference_name} Over Time')
#     ax1.grid(True)
#     ax1.legend()
    
#     ax2.plot(times, error_x, 'r-', label='X Error')
#     ax2.plot(times, error_y, 'g-', label='Y Error')
#     ax2.plot(times, error_z, 'b-', label='Z Error')
#     ax2.set_xlabel('Time (seconds)')
#     ax2.set_ylabel('Component Error (meters)')
#     ax2.set_title('X, Y, Z Component Errors Over Time')
#     ax2.grid(True)
#     ax2.legend()
    
#     if has_orientation:
#         ax3.plot(times, orientation_errors, 'm-', label='Orientation Error')
#         ax3.set_xlabel('Time (seconds)')
#         ax3.set_ylabel('Orientation Error (degrees)')
#         ax3.set_title('Orientation Error Over Time')
#         ax3.grid(True)
#         ax3.legend()
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"Plot saved to {output_path}")
#     else:
#         plt.show()
    
#     return avg_error, max_error

# def plot_trajectory(errors, output_path=None, reference_name="Ground Truth"):
#     """
#     Plot the robot's trajectory (path) according to both reference and estimated positions.
    
#     Args:
#         errors (list): List of error data points from analyze_error()
#         output_path (str, optional): Path to save the plot, or None to display it
#         reference_name (str): Name of the reference source for the plot labels
#     """
#     if not errors:
#         print("No data to plot trajectory")
#         return
    
#     ref_x = [error['ref_x'] for error in errors]
#     ref_y = [error['ref_y'] for error in errors]
#     est_x = [error['est_x'] for error in errors]
#     est_y = [error['est_y'] for error in errors]
    
#     plt.figure(figsize=(10, 8))
    
#     plt.plot(ref_x, ref_y, 'b-', label=reference_name)
#     plt.plot(est_x, est_y, 'r--', label='Estimated Position')
    
#     arrow_indices = np.linspace(0, len(ref_x)-1, min(20, len(ref_x))).astype(int)
#     for i in arrow_indices:
#         if i+1 < len(ref_x):
#             plt.arrow(ref_x[i], ref_y[i], 
#                      (ref_x[i+1]-ref_x[i])*0.5, (ref_y[i+1]-ref_y[i])*0.5, 
#                      head_width=0.01, head_length=0.02, fc='b', ec='b')
            
#             plt.arrow(est_x[i], est_y[i], 
#                      (est_x[i+1]-est_x[i])*0.5, (est_y[i+1]-est_y[i])*0.5, 
#                      head_width=0.01, head_length=0.02, fc='r', ec='r')
    
#     plt.plot(ref_x[0], ref_y[0], 'bo', markersize=10, label=f"{reference_name} Start")
#     plt.plot(ref_x[-1], ref_y[-1], 'bx', markersize=10, label=f"{reference_name} End")
#     plt.plot(est_x[0], est_y[0], 'ro', markersize=10, label="Estimated Start")
#     plt.plot(est_x[-1], est_y[-1], 'rx', markersize=10, label="Estimated End")
    
#     plt.xlabel('X Position (meters)')
#     plt.ylabel('Y Position (meters)')
#     plt.title('Robot Trajectory')
#     plt.grid(True)
#     plt.axis('equal')
#     plt.legend()
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"Trajectory plot saved to {output_path}")
#     else:
#         plt.show()

# def plot_position_over_time(errors, output_path=None, reference_name="Ground Truth"):
#     """
#     Plot the X and Y positions of the robot over time, comparing reference vs. estimated.
    
#     Args:
#         errors (list): List of error data points from analyze_error()
#         output_path (str, optional): Path to save the plot, or None to display it
#         reference_name (str): Name of the reference source for the plot labels
#     """
#     if not errors:
#         print("No data to plot positions over time")
#         return
    
#     times = [error['time'] for error in errors]
#     ref_x = [error['ref_x'] for error in errors]
#     ref_y = [error['ref_y'] for error in errors]
#     est_x = [error['est_x'] for error in errors]
#     est_y = [error['est_y'] for error in errors]
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
#     ax1.plot(times, ref_x, 'b-', label=f'{reference_name} X')
#     ax1.plot(times, est_x, 'r--', label='Estimated X')
#     ax1.set_ylabel('X Position (meters)')
#     ax1.set_title('X Position Over Time')
#     ax1.grid(True)
#     ax1.legend()
    
#     ax2.plot(times, ref_y, 'b-', label=f'{reference_name} Y')
#     ax2.plot(times, est_y, 'r--', label='Estimated Y')
#     ax2.set_xlabel('Time (seconds)')
#     ax2.set_ylabel('Y Position (meters)')
#     ax2.set_title('Y Position Over Time')
#     ax2.grid(True)
#     ax2.legend()
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"Position over time plot saved to {output_path}")
#     else:
#         plt.show()

# def get_log_by_number(log_dir, log_number):
#     """
#     Get a log file by its index number in the directory (sorted by name).
    
#     Args:
#         log_dir (str): Directory containing log files
#         log_number (int): Index of the log file to select (1-indexed)
        
#     Returns:
#         str: Full path to the selected log file
#     """
#     # Get all log files in the directory with .json extension
#     log_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    
#     if not log_files:
#         raise FileNotFoundError(f"No log files found in directory: {log_dir}")
    
#     # Adjust for 1-indexed input
#     if log_number < 1 or log_number > len(log_files):
#         raise ValueError(f"Log number must be between 1 and {len(log_files)}")
    
#     selected_log = log_files[log_number - 1]
#     print(f"Selected log file {log_number} of {len(log_files)}: {os.path.basename(selected_log)}")
    
#     return selected_log

# def main():
#     """
#     Main function that parses command-line arguments and runs the analysis.
#     """
#     parser = argparse.ArgumentParser(description='Analyze robot positioning error from log files.')
    
#     # Use a mutually exclusive group for log selection options
#     log_group = parser.add_mutually_exclusive_group()
    
#     log_group.add_argument('--log-file', 
#                         help='Path to a specific log file to analyze')
    
#     log_group.add_argument('--number', type=int,
#                         help='Index number of the log file to analyze (1-indexed)')
    
#     parser.add_argument('--log-dir', default='../SHANTAM_LOGS/',
#                         help='Directory containing log files')
    
#     parser.add_argument('--simulation', dest='simulation', action='store_true',
#                         help='Use groundtruth as reference (default)')
    
#     parser.add_argument('--real', dest='simulation', action='store_false',
#                         help='Use aruco markers as reference')
    
#     parser.add_argument('--max-time-diff', type=float, default=0.05, 
#                         help='Maximum time difference between reference and estimate (seconds)')
    
#     parser.add_argument('--output', '-o', 
#                         help='Path to save the plot (without extension)')
    
#     parser.add_argument('--trajectory', '-t', action='store_true', 
#                         help='Generate trajectory plot')
    
#     parser.add_argument('--debug', action='store_true', 
#                         help='Print debug information')
    
#     parser.set_defaults(simulation=True)
    
#     args = parser.parse_args()
    
#     try:
#         # Determine which log file to use
#         log_file_path = None
#         if args.log_file:
#             log_file_path = args.log_file
#         elif args.number:
#             log_file_path = get_log_by_number(args.log_dir, args.number)
#         else:
#             # Default to the first log file in the directory if neither specified
#             log_file_path = get_log_by_number(args.log_dir, 1)
        
#         # Load and parse the log file
#         log_data = load_log_file(log_file_path)
#         print(f"Loaded {len(log_data)} log entries from {log_file_path}")
        
#         # Print debug information if requested
#         if args.debug:
#             print("\nAvailable topics:")
#             topics = set(item['topic_name'] for item in log_data)
#             for topic in sorted(topics):
#                 count = sum(1 for item in log_data if item['topic_name'] == topic)
#                 print(f"  {topic}: {count} entries")
            
#             print("\nSample log entries:")
#             for i, topic in enumerate(sorted(topics)):
#                 sample = next((item for item in log_data if item['topic_name'] == topic), None)
#                 if sample:
#                     print(f"\nTopic {i+1}: {topic}")
#                     print(json.dumps(sample, indent=2)[0:500] + "...")
        
#         # Determine what we're using as the reference source
#         reference_name = "Ground Truth" if args.simulation else "Aruco Markers"
#         print(f"Mode: {'Simulation' if args.simulation else 'Real robot'} (using {reference_name} as reference)")
        
#         # Analyze the position error
#         errors = analyze_error(log_data, args.simulation, args.max_time_diff)
        
#         if not errors:
#             print("No matching data points found for error calculation")
#             return
        
#         # Calculate and print error statistics
#         euclidean_errors = [error['euclidean_error'] for error in errors]
#         avg_error = sum(euclidean_errors) / len(euclidean_errors)
#         max_error = max(euclidean_errors)
#         min_error = min(euclidean_errors)
        
#         print("\nError Statistics:")
#         print(f"  Number of data points: {len(errors)}")
#         print(f"  Average Error: {avg_error:.4f} meters")
#         print(f"  Minimum Error: {min_error:.4f} meters")
#         print(f"  Maximum Error: {max_error:.4f} meters")
#         print(f"  Initial Error: {errors[0]['euclidean_error']:.4f} meters")
#         print(f"  Final Error: {errors[-1]['euclidean_error']:.4f} meters")
        
#         # Print orientation error statistics if available
#         has_orientation = any(error.get('orientation_error') is not None for error in errors)
#         if has_orientation:
#             valid_orient_errors = [err['orientation_error'] for err in errors if err.get('orientation_error') is not None]
#             if valid_orient_errors:
#                 avg_orient_error = sum(valid_orient_errors) / len(valid_orient_errors)
#                 max_orient_error = max(valid_orient_errors)
#                 avg_orient_error_deg = avg_orient_error * 180 / np.pi
#                 max_orient_error_deg = max_orient_error * 180 / np.pi
#                 print(f"  Average Orientation Error: {avg_orient_error_deg:.4f} degrees")
#                 print(f"  Maximum Orientation Error: {max_orient_error_deg:.4f} degrees")
        
#         # Generate plots
#         error_output = f"{args.output}_error.png" if args.output else None
#         plot_error(errors, error_output, reference_name)
        
#         if args.trajectory:
#             traj_output = f"{args.output}_trajectory.png" if args.output else None
#             plot_trajectory(errors, traj_output, reference_name)
        
#         pos_output = f"{args.output}_position_time.png" if args.output else None
#         plot_position_over_time(errors, pos_output, reference_name)
        
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Current working directory:", os.getcwd())
#     except Exception as e:
#         print(f"Error: {e}")
#         if args.debug:
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     main()


# python robot-error-analysis.py --trajectory --real --number 134 --loops 0,4 --loop-threshold 0.2

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt
import glob

def load_log_file(file_path):
    """
    Load and parse the robot's log file.
    
    Args:
        file_path (str): Path to the log file to analyze
        
    Returns:
        list: A list of parsed JSON entries, each representing a logged data point
    """
    try:
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        parsed_data = []
        skipped_lines = 0
        
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                parsed_data.append(data)
            except json.JSONDecodeError:
                skipped_lines += 1
                if skipped_lines <= 5:
                    print(f"Warning: Skipping invalid JSON at line {i+1}: {line[:50]}...")
                elif skipped_lines == 6:
                    print("Additional invalid JSON lines found (not showing all warnings)...")
        
        if skipped_lines > 0:
            print(f"Total of {skipped_lines} invalid JSON lines skipped")
        
        return parsed_data
    
    except Exception as e:
        print(f"Error loading log file: {e}")
        raise

def find_closest_timestamp(target_time, data_list):
    """
    Find the entry in data_list with the timestamp closest to target_time.
    
    Args:
        target_time (float): The timestamp we want to find a match for
        data_list (list): List of data points to search through
        
    Returns:
        tuple: (closest_item, time_difference)
    """
    closest = None
    min_diff = float('inf')
    
    for item in data_list:
        timestamp = float(item['timestamp'])
        diff = abs(timestamp - target_time)
        
        if diff < min_diff:
            min_diff = diff
            closest = item
    
    return closest, min_diff

def extract_position(data_item):
    """
    Extract position coordinates (x, y, z) from different message formats.
    
    Args:
        data_item (dict): A single data point from the log
        
    Returns:
        dict: Position information with 'x', 'y', 'z' keys
    """
    try:
        if data_item['topic_name'] == '/groundtruth':
            return data_item['message']['position']
        
        elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
            if 'pose' in data_item['message']:
                return data_item['message']['pose']['position']
            elif 'position' in data_item['message']:
                return data_item['message']['position']
            else:
                if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
                    return data_item['message']['pose']['pose']['position']
        
        message = data_item['message']
        if isinstance(message, dict):
            for key in message:
                if isinstance(message[key], dict) and 'position' in message[key]:
                    return message[key]['position']
                elif key == 'position' and isinstance(message[key], dict):
                    return message[key]
                
            for key in message:
                if isinstance(message[key], dict) and 'pose' in message[key]:
                    if 'position' in message[key]['pose']:
                        return message[key]['pose']['position']
        
        raise ValueError(f"Cannot extract position from topic {data_item['topic_name']}. Message structure: {data_item['message']}")
    
    except Exception as e:
        print(f"Error extracting position from {data_item['topic_name']}: {e}")
        print(f"Message structure: {data_item['message']}")
        raise

def extract_orientation(data_item):
    """
    Extract orientation (quaternion) from different message formats.
    
    Args:
        data_item (dict): A single data point from the log
        
    Returns:
        dict: Orientation as quaternion with 'x', 'y', 'z', 'w' keys, or None if unavailable
    """
    try:
        if data_item['topic_name'] == '/groundtruth':
            return data_item['message']['orientation']
        
        elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
            if 'pose' in data_item['message']:
                return data_item['message']['pose']['orientation']
            elif 'orientation' in data_item['message']:
                return data_item['message']['orientation']
            else:
                if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
                    return data_item['message']['pose']['pose']['orientation']
        
        message = data_item['message']
        if isinstance(message, dict):
            for key in message:
                if isinstance(message[key], dict) and 'orientation' in message[key]:
                    return message[key]['orientation']
                elif key == 'orientation' and isinstance(message[key], dict):
                    return message[key]
                
            for key in message:
                if isinstance(message[key], dict) and 'pose' in message[key]:
                    if 'orientation' in message[key]['pose']:
                        return message[key]['pose']['orientation']
        
        return None
    
    except Exception as e:
        print(f"Error extracting orientation from {data_item['topic_name']}: {e}")
        return None

def calculate_euclidean_distance(pos1, pos2):
    """
    Calculate the straight-line (Euclidean) distance between two 3D positions.
    
    Args:
        pos1 (dict): First position with 'x', 'y', 'z' keys
        pos2 (dict): Second position with 'x', 'y', 'z' keys
        
    Returns:
        float: The Euclidean distance in meters
    """
    return sqrt(
        (pos1['x'] - pos2['x'])**2 + 
        (pos1['y'] - pos2['y'])**2 + 
        (pos1['z'] - pos2['z'])**2
    )

def calculate_orientation_error(orient1, orient2):
    """
    Calculate the angular difference between two orientations represented as quaternions.
    
    Args:
        orient1 (dict): First orientation as quaternion with 'x', 'y', 'z', 'w' keys
        orient2 (dict): Second orientation as quaternion with 'x', 'y', 'z', 'w' keys
        
    Returns:
        float: The angle between orientations in radians, or None if inputs are invalid
    """
    if orient1 is None or orient2 is None:
        return None
    
    q1 = [orient1['x'], orient1['y'], orient1['z'], orient1['w']]
    q2 = [orient2['x'], orient2['y'], orient2['z'], orient2['w']]
    
    dot_product = sum(a*b for a, b in zip(q1, q2))
    dot_product = max(-1, min(1, dot_product))
    
    angle = 2 * np.arccos(abs(dot_product))
    
    return angle

def analyze_error(log_data, is_simulation=True, max_time_diff=0.1):
    """
    Analyze position and orientation error over time by comparing estimated and reference positions.
    
    Args:
        log_data (list): Parsed log data to analyze
        is_simulation (bool): Whether we're analyzing simulation data (True) or real robot data (False)
        max_time_diff (float): Maximum allowed time difference between reference and estimate
        
    Returns:
        list: A list of dictionaries containing error information at each timestep
    """
    reference_topic = '/groundtruth' if is_simulation else '/aruco'
    
    if is_simulation:
        estimation_topics = ['/est_pose', '/odom']
    else:
        estimation_topics = ['/est_pose', '/odom']
    
    reference_data = [item for item in log_data if item['topic_name'] == reference_topic]
    
    if not reference_data:
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No {reference_topic} data found in log. Available topics: {available_topics}")
    
    estimate_topic = None
    estimate_data = []
    
    for topic in estimation_topics:
        temp_data = [item for item in log_data if item['topic_name'] == topic]
        if temp_data:
            estimate_topic = topic
            estimate_data = temp_data
            break
    
    if not estimate_data:
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No estimation data found. Available topics in log: {available_topics}")
    
    print(f"Using {reference_topic} as reference (found {len(reference_data)} entries)")
    print(f"Using {estimate_topic} as estimation (found {len(estimate_data)} entries)")
    
    errors = []
    start_time = float(reference_data[0]['timestamp'])
    
    skipped_pairs = 0
    processed_pairs = 0
    orientation_errors_computed = 0
    
    for ref_item in reference_data:
        ref_time = float(ref_item['timestamp'])
        est_item, time_diff = find_closest_timestamp(ref_time, estimate_data)
        
        if time_diff <= max_time_diff:
            try:
                ref_pos = extract_position(ref_item)
                est_pos = extract_position(est_item)
                
                ref_orient = extract_orientation(ref_item)
                est_orient = extract_orientation(est_item)
                
                error_x = ref_pos['x'] - est_pos['x']
                error_y = ref_pos['y'] - est_pos['y']
                error_z = ref_pos['z'] - est_pos['z']
                euclidean_error = calculate_euclidean_distance(ref_pos, est_pos)
                
                orientation_error = None
                if ref_orient is not None and est_orient is not None:
                    orientation_error = calculate_orientation_error(ref_orient, est_orient)
                    if orientation_error is not None:
                        orientation_errors_computed += 1
                
                errors.append({
                    'time': ref_time - start_time,
                    'timestamp': ref_time,
                    'ref_x': ref_pos['x'],
                    'ref_y': ref_pos['y'],
                    'ref_z': ref_pos['z'],
                    'est_x': est_pos['x'],
                    'est_y': est_pos['y'],
                    'est_z': est_pos['z'],
                    'error_x': error_x,
                    'error_y': error_y,
                    'error_z': error_z,
                    'euclidean_error': euclidean_error,
                    'orientation_error': orientation_error,
                    'time_diff': time_diff
                })
                processed_pairs += 1
            except Exception as e:
                print(f"Error processing data point at time {ref_time}: {e}")
                continue
        else:
            skipped_pairs += 1
    
    print(f"Processed {processed_pairs} timestamp pairs, skipped {skipped_pairs} pairs due to timestamp difference > {max_time_diff}s")
    
    if orientation_errors_computed > 0:
        print(f"Orientation errors computed for {orientation_errors_computed} pairs")
    else:
        print("No orientation errors computed (missing orientation data)")
    
    if not errors:
        raise ValueError("No valid data points found for error calculation. Try increasing max_time_diff.")
    
    return errors

def detect_loops(errors, distance_threshold=0.1, min_loop_size=20):
    """
    Detect loop completions in the robot trajectory.
    
    Args:
        errors (list): List of error data points from analyze_error()
        distance_threshold (float): Threshold to consider a position as "returning to start"
        min_loop_size (int): Minimum number of points needed for loop detection
        
    Returns:
        list: Indices in the errors list where loops start/end
    """
    if not errors or len(errors) < min_loop_size:
        return []
    
    # Get reference positions
    ref_positions = [(error['ref_x'], error['ref_y']) for error in errors]
    
    # First position is the starting point of the first loop
    start_pos = ref_positions[0]
    
    loop_indices = [0]  # First index is always a loop start
    in_loop = False
    min_distance_from_start = distance_threshold * 4  # Must travel at least this far to consider a loop
    
    # Track maximum distance from start for each potential loop
    max_distance = 0
    
    for i in range(min_loop_size, len(ref_positions)):
        current_pos = ref_positions[i]
        
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
    if loop_indices[-1] != len(errors) - 1:
        loop_indices.append(len(errors) - 1)
    
    return loop_indices

def get_loops_data(errors, loop_indices, start_loop_idx=0, num_loops=2):
    """
    Extract data for a specific range of loops.
    
    Args:
        errors (list): List of error data points from analyze_error()
        loop_indices (list): Indices in the errors list where loops start/end
        start_loop_idx (int): Starting loop index to visualize (0-indexed)
        num_loops (int): Number of consecutive loops to visualize
        
    Returns:
        list: Subset of errors list containing only the specified loops
    """
    if not loop_indices or len(loop_indices) < 2:
        return errors
    
    # Validate loop indices
    if start_loop_idx < 0 or start_loop_idx >= len(loop_indices) - 1:
        print(f"Warning: Start loop index {start_loop_idx} is out of range (0-{len(loop_indices)-2})")
        start_loop_idx = 0
    
    end_loop_idx = min(start_loop_idx + num_loops, len(loop_indices) - 1)
    
    # Get start and end indices in the errors list
    start_idx = loop_indices[start_loop_idx]
    end_idx = loop_indices[end_loop_idx]
    
    return errors[start_idx:end_idx + 1]

def plot_error(errors, output_path=None, reference_name="Ground Truth"):
    """
    Create plots showing position and orientation error over time.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot titles
        
    Returns:
        tuple: (avg_error, max_error) - The average and maximum position errors
    """
    if not errors:
        print("No error data to plot")
        return
    
    times = [error['time'] for error in errors]
    euclidean_errors = [error['euclidean_error'] for error in errors]
    error_x = [error['error_x'] for error in errors]
    error_y = [error['error_y'] for error in errors]
    error_z = [error['error_z'] for error in errors]
    
    has_orientation = any(error.get('orientation_error') is not None for error in errors)
    if has_orientation:
        orientation_errors = [error.get('orientation_error', 0) for error in errors]
        orientation_errors = [error * 180 / np.pi if error is not None else 0 for error in orientation_errors]
    
    avg_error = sum(euclidean_errors) / len(euclidean_errors)
    max_error = max(euclidean_errors)
    
    if has_orientation:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(times, euclidean_errors, 'b-', label=f'Error vs {reference_name}')
    ax1.axhline(y=avg_error, color='r', linestyle='--', label=f'Avg Error: {avg_error:.4f} m')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Euclidean Error (meters)')
    ax1.set_title(f'Position Error vs {reference_name} Over Time')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(times, error_x, 'r-', label='X Error')
    ax2.plot(times, error_y, 'g-', label='Y Error')
    ax2.plot(times, error_z, 'b-', label='Z Error')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Component Error (meters)')
    ax2.set_title('X, Y, Z Component Errors Over Time')
    ax2.grid(True)
    ax2.legend()
    
    if has_orientation:
        ax3.plot(times, orientation_errors, 'm-', label='Orientation Error')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Orientation Error (degrees)')
        ax3.set_title('Orientation Error Over Time')
        ax3.grid(True)
        ax3.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return avg_error, max_error

def plot_trajectory(errors, output_path=None, reference_name="Ground Truth", highlight_loops=None):
    """
    Plot the robot's trajectory (path) according to both reference and estimated positions.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot labels
        highlight_loops (list, optional): List of loop boundary indices to highlight
    """
    if not errors:
        print("No data to plot trajectory")
        return
    
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory lines
    plt.plot(ref_x, ref_y, 'b-', label=reference_name, linewidth=1.5)
    plt.plot(est_x, est_y, 'r--', label='Estimated Position', linewidth=1.5)
    
    # Add directional arrows to show path flow
    arrow_indices = np.linspace(0, len(ref_x)-1, min(20, len(ref_x))).astype(int)
    for i in arrow_indices:
        if i+1 < len(ref_x):
            plt.arrow(ref_x[i], ref_y[i], 
                     (ref_x[i+1]-ref_x[i])*0.5, (ref_y[i+1]-ref_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='b', ec='b')
            
            plt.arrow(est_x[i], est_y[i], 
                     (est_x[i+1]-est_x[i])*0.5, (est_y[i+1]-est_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='r', ec='r')
    
    # Highlight loop boundaries if provided
    if highlight_loops and len(highlight_loops) >= 2:
        for i in range(len(highlight_loops) - 1):
            start_idx = highlight_loops[i]
            end_idx = highlight_loops[i+1]
            
            # Add a marker for loop start/end
            plt.plot(ref_x[start_idx], ref_y[start_idx], 'go', markersize=8, 
                    label=f"Loop {i} Start" if i==0 else f"Loop {i} Start/End")
            
            # Label the loops on the plot
            mid_idx = (start_idx + end_idx) // 2
            plt.text(ref_x[mid_idx], ref_y[mid_idx], f"Loop {i}", 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Mark start and end points
    plt.plot(ref_x[0], ref_y[0], 'bo', markersize=10, label=f"{reference_name} Start")
    plt.plot(ref_x[-1], ref_y[-1], 'bx', markersize=10, label=f"{reference_name} End")
    plt.plot(est_x[0], est_y[0], 'ro', markersize=10, label="Estimated Start")
    plt.plot(est_x[-1], est_y[-1], 'rx', markersize=10, label="Estimated End")
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()

def plot_position_over_time(errors, output_path=None, reference_name="Ground Truth"):
    """
    Plot the X and Y positions of the robot over time, comparing reference vs. estimated.
    
    Args:
        errors (list): List of error data points from analyze_error()
        output_path (str, optional): Path to save the plot, or None to display it
        reference_name (str): Name of the reference source for the plot labels
    """
    if not errors:
        print("No data to plot positions over time")
        return
    
    times = [error['time'] for error in errors]
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(times, ref_x, 'b-', label=f'{reference_name} X')
    ax1.plot(times, est_x, 'r--', label='Estimated X')
    ax1.set_ylabel('X Position (meters)')
    ax1.set_title('X Position Over Time')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(times, ref_y, 'b-', label=f'{reference_name} Y')
    ax2.plot(times, est_y, 'r--', label='Estimated Y')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Y Position (meters)')
    ax2.set_title('Y Position Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Position over time plot saved to {output_path}")
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
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in directory: {log_dir}")
    
    # Adjust for 1-indexed input
    if log_number < 1 or log_number > len(log_files):
        raise ValueError(f"Log number must be between 1 and {len(log_files)}")
    
    selected_log = log_files[log_number - 1]
    print(f"Selected log file {log_number} of {len(log_files)}: {os.path.basename(selected_log)}")
    
    return selected_log

def main():
    """
    Main function that parses command-line arguments and runs the analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze robot positioning error from log files.')
    
    # Use a mutually exclusive group for log selection options
    log_group = parser.add_mutually_exclusive_group()
    
    log_group.add_argument('--log-file', 
                        help='Path to a specific log file to analyze')
    
    log_group.add_argument('--number', type=int,
                        help='Index number of the log file to analyze (1-indexed)')
    
    parser.add_argument('--log-dir', default='../SHANTAM_LOGS/',
                        help='Directory containing log files')
    
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
    
    # New loop-related arguments
    parser.add_argument('--detect-loops', action='store_true',
                        help='Detect and display loop information')
    
    parser.add_argument('--loop-threshold', type=float, default=0.1,
                        help='Distance threshold for loop detection (meters)')
    
    parser.add_argument('--loops', type=str,
                        help='Specify which loops to visualize (e.g., "0,1" for first two loops)')
    
    parser.add_argument('--all-loops', action='store_true',
                        help='Show all detected loops with numbered markers')
    
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
        
        # Load and parse the log file
        log_data = load_log_file(log_file_path)
        print(f"Loaded {len(log_data)} log entries from {log_file_path}")
        
        # Print debug information if requested
        if args.debug:
            print("\nAvailable topics:")
            topics = set(item['topic_name'] for item in log_data)
            for topic in sorted(topics):
                count = sum(1 for item in log_data if item['topic_name'] == topic)
                print(f"  {topic}: {count} entries")
        
        # Determine what we're using as the reference source
        reference_name = "Ground Truth" if args.simulation else "Aruco Markers"
        print(f"Mode: {'Simulation' if args.simulation else 'Real robot'} (using {reference_name} as reference)")
        
        # Analyze the position error
        errors = analyze_error(log_data, args.simulation, args.max_time_diff)
        
        if not errors:
            print("No matching data points found for error calculation")
            return
        
        # Detect loops if requested
        loop_indices = []
        if args.detect_loops or args.loops or args.all_loops:
            loop_indices = detect_loops(errors, args.loop_threshold)
            
            if not loop_indices:
                print("No loops detected in the trajectory. Try adjusting --loop-threshold")
            else:
                print("\nDetected Loops:")
                for i in range(len(loop_indices) - 1):
                    loop_start = loop_indices[i]
                    loop_end = loop_indices[i+1]
                    loop_duration = errors[loop_end]['time'] - errors[loop_start]['time']
                    start_point = (errors[loop_start]['ref_x'], errors[loop_start]['ref_y'])
                    end_point = (errors[loop_end]['ref_x'], errors[loop_end]['ref_y'])
                    print(f"  Loop {i}: Points {loop_start} to {loop_end} (Duration: {loop_duration:.2f} seconds)")
                    print(f"       Start: ({start_point[0]:.2f}, {start_point[1]:.2f}), End: ({end_point[0]:.2f}, {end_point[1]:.2f})")
        
        # Filter data for specified loops
        filtered_errors = errors
        loop_range_str = "all data"
        
        if args.loops and loop_indices:
            try:
                loop_specs = [int(x) for x in args.loops.split(',')]
                if len(loop_specs) == 1:
                    start_loop = loop_specs[0]
                    num_loops = 2  # Default to 2 loops if only one number provided
                else:
                    start_loop = loop_specs[0]
                    num_loops = loop_specs[1] - loop_specs[0] + 1
                
                filtered_errors = get_loops_data(errors, loop_indices, start_loop, num_loops)
                loop_range_str = f"loops {start_loop} to {start_loop + num_loops - 1}"
                print(f"\nVisualization filtered to {loop_range_str} ({len(filtered_errors)} data points)")
            except (ValueError, IndexError) as e:
                print(f"Error parsing loop specification: {e}")
                print("Using full data set instead")
        
        # Calculate and print error statistics
        euclidean_errors = [error['euclidean_error'] for error in filtered_errors]
        avg_error = sum(euclidean_errors) / len(euclidean_errors)
        max_error = max(euclidean_errors)
        min_error = min(euclidean_errors)
        
        print(f"\nError Statistics for {loop_range_str}:")
        print(f"  Number of data points: {len(filtered_errors)}")
        print(f"  Average Error: {avg_error:.4f} meters")
        print(f"  Minimum Error: {min_error:.4f} meters")
        print(f"  Maximum Error: {max_error:.4f} meters")
        print(f"  Initial Error: {filtered_errors[0]['euclidean_error']:.4f} meters")
        print(f"  Final Error: {filtered_errors[-1]['euclidean_error']:.4f} meters")
        
        # Print orientation error statistics if available
        has_orientation = any(error.get('orientation_error') is not None for error in filtered_errors)
        if has_orientation:
            valid_orient_errors = [err['orientation_error'] for err in filtered_errors if err.get('orientation_error') is not None]
            if valid_orient_errors:
                avg_orient_error = sum(valid_orient_errors) / len(valid_orient_errors)
                max_orient_error = max(valid_orient_errors)
                avg_orient_error_deg = avg_orient_error * 180 / np.pi
                max_orient_error_deg = max_orient_error * 180 / np.pi
                print(f"  Average Orientation Error: {avg_orient_error_deg:.4f} degrees")
                print(f"  Maximum Orientation Error: {max_orient_error_deg:.4f} degrees")
        
        # Generate plots
        loop_suffix = f"_loops_{args.loops.replace(',', '-')}" if args.loops else ""
        
        error_output = f"{args.output}{loop_suffix}_error.png" if args.output else None
        plot_error(filtered_errors, error_output, reference_name)
        
        # Create trajectory subplot with loop markers if requested
        # Note: For trajectory plot, I always show all loops for context
        if args.trajectory:
            traj_output = f"{args.output}{loop_suffix}_trajectory.png" if args.output else None
            
            # Decide which loop indices to highlight
            highlight_indices = None
            if args.all_loops and loop_indices:
                highlight_indices = loop_indices
            elif args.loops and loop_indices:
                try:
                    # Just highlight the loops we're visualizing
                    loop_specs = [int(x) for x in args.loops.split(',')]
                    start_loop = loop_specs[0]
                    num_loops = 2 if len(loop_specs) == 1 else (loop_specs[1] - loop_specs[0] + 1)
                    end_loop = start_loop + num_loops
                    highlight_indices = loop_indices[start_loop:end_loop+1]
                except Exception:
                    highlight_indices = None
            
            # plot_trajectory(errors, traj_output, reference_name, highlight_indices)
            plot_trajectory(filtered_errors, traj_output, reference_name, highlight_indices)
        pos_output = f"{args.output}{loop_suffix}_position_time.png" if args.output else None
        plot_position_over_time(filtered_errors, pos_output, reference_name)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Current working directory:", os.getcwd())
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



# 11:12:02 AM - 113 - Unix: 1741086722
# 12:30:36 PM - 114 - Unix: 1741091436 -G
# 12:31:59 PM - 115 - Unix: 1741091519
# 12:32:22 PM - 116 - Unix: 1741091542
# 12:33:35 PM - 117 - Unix: 1741091615
# 12:34:23 PM - 118 - Unix: 1741091663
# 12:35:00 PM - 119 - Unix: 1741091700
# 12:37:17 PM - 120 - Unix: 1741091837
# 12:56:47 PM - 121 - Unix: 1741093007
# 12:59:06 PM - 122 - Unix: 1741093146
# 1:00:28 PM - 123 - Unix: 1741093228
# 1:01:59 PM - 124 - Unix: 1741093319 -G
# 1:05:38 PM - 125 - Unix: 1741093538 -Vg
# 1:22:04 PM - 126 - Unix: 1741094524 -G
# 1:25:37 PM - 127 - Unix: 1741094737 -G - Real
# 1:37:26 PM - 128 - Unix: 1741095446
# 1:37:45 PM - 129 - Unix: 1741095465
# 1:40:54 PM - 130 - Unix: 1741095654 -Sim
# 1:47:08 PM - 131 - Unix: 1741096028 -G
# 1:48:17 PM - 132 - Unix: 1741096097
# 1:48:25 PM - 133 - Unix: 1741096105
# 1:48:34 PM - 134 - Unix: 1741096114 -Vg - Real many laps
# 2:21:27 PM - 135 - Unix: 1741098087
# 2:22:23 PM - 136 - Unix: 1741098143 -G
# 2:24:40 PM - 137 - Unix: 1741098280
# 2:25:07 PM - 138 - Unix: 1741098307
# 2:25:34 PM - 139 - Unix: 1741098334
# 2:26:26 PM - 140 - Unix: 1741098386
# 2:31:58 PM - 141 - Unix: 1741098718 -Vg
# 2:33:40 PM - 142 - Unix: 1741098820 -G
# 2:34:53 PM - 143 - Unix: 1741098893
# 2:35:16 PM - 144 - Unix: 1741098916 -Vg
# 2:37:47 PM - 145 - Unix: 1741099067 -G
# 2:39:08 PM - 146 - Unix: 1741099148
# 2:39:52 PM - 147 - Unix: 1741099192
# 2:40:40 PM - 148 - Unix: 1741099240 -G



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