#!/usr/bin/env python3
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

def load_log_file(file_path):
    """Load and parse the log file."""
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
                if skipped_lines <= 5:  # Only print the first few errors to avoid flooding the console
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
    """Find the entry with the closest timestamp to the target time."""
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
    """Extract position coordinates from different message formats."""
    try:
        if data_item['topic_name'] == '/groundtruth':
            # Direct position in groundtruth
            return data_item['message']['position']
        elif data_item['topic_name'] in ['/est_pose', '/aruco', '/odom']:
            # Check different potential structures
            if 'pose' in data_item['message']:
                return data_item['message']['pose']['position']
            elif 'position' in data_item['message']:
                return data_item['message']['position']
            else:
                # For odom messages
                if 'pose' in data_item['message'] and 'pose' in data_item['message']['pose']:
                    return data_item['message']['pose']['pose']['position']
        
        # If we can't find a position with the above patterns, print the message structure
        # and try a more general approach
        message = data_item['message']
        if isinstance(message, dict):
            # Try to find any key that might contain position information
            for key in message:
                if isinstance(message[key], dict) and 'position' in message[key]:
                    return message[key]['position']
                elif key == 'position' and isinstance(message[key], dict):
                    return message[key]
                
            # Try to find nested pose structure
            for key in message:
                if isinstance(message[key], dict) and 'pose' in message[key]:
                    if 'position' in message[key]['pose']:
                        return message[key]['pose']['position']
        
        # If we still can't find a position, raise an error
        raise ValueError(f"Cannot extract position from topic {data_item['topic_name']}. Message structure: {data_item['message']}")
    
    except Exception as e:
        print(f"Error extracting position from {data_item['topic_name']}: {e}")
        print(f"Message structure: {data_item['message']}")
        raise

def extract_orientation(data_item):
    """Extract orientation coordinates from different message formats."""
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
        
        # Try more general approach if the above doesn't work
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
        
        # If we can't find orientation, return None instead of failing
        # This allows us to still compute position error even if orientation is missing
        return None
    
    except Exception as e:
        print(f"Error extracting orientation from {data_item['topic_name']}: {e}")
        return None

def calculate_euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return sqrt(
        (pos1['x'] - pos2['x'])**2 + 
        (pos1['y'] - pos2['y'])**2 + 
        (pos1['z'] - pos2['z'])**2
    )

def calculate_orientation_error(orient1, orient2):
    """Calculate orientation error between two quaternions.
    Returns the angle in radians between the two quaternions.
    """
    if orient1 is None or orient2 is None:
        return None
    
    # Convert quaternions to proper format
    q1 = [orient1['x'], orient1['y'], orient1['z'], orient1['w']]
    q2 = [orient2['x'], orient2['y'], orient2['z'], orient2['w']]
    
    # Calculate dot product
    dot_product = sum(a*b for a, b in zip(q1, q2))
    
    # Make sure dot product is in range [-1, 1]
    dot_product = max(-1, min(1, dot_product))
    
    # Calculate angle
    angle = 2 * np.arccos(abs(dot_product))
    
    return angle

def analyze_error(log_data, is_simulation=True, max_time_diff=0.1):
    """Analyze position error over time."""
    # Determine reference source based on simulation flag
    reference_topic = '/groundtruth' if is_simulation else '/aruco'
    
    # Try different estimation topics in order of preference
    # For this specific dataset, we prioritize /aruco over /est_pose if in real robot mode
    if is_simulation:
        estimation_topics = ['/est_pose', '/odom']
    else:
        estimation_topics = ['/est_pose', '/odom']  # still try est_pose first in non-simulation mode
    
    # Filter by topic type
    reference_data = [item for item in log_data if item['topic_name'] == reference_topic]
    
    # Check if reference data exists
    if not reference_data:
        # List all available topics for debugging
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No {reference_topic} data found in log. Available topics: {available_topics}")
    
    # Try to find estimation data from different possible topics
    estimate_topic = None
    estimate_data = []
    
    for topic in estimation_topics:
        temp_data = [item for item in log_data if item['topic_name'] == topic]
        if temp_data:
            estimate_topic = topic
            estimate_data = temp_data
            break
    
    if not estimate_data:
        # List all available topics in the log for debugging
        available_topics = set(item['topic_name'] for item in log_data)
        raise ValueError(f"No estimation data found. Available topics in log: {available_topics}")
    
    print(f"Using {reference_topic} as reference (found {len(reference_data)} entries)")
    print(f"Using {estimate_topic} as estimation (found {len(estimate_data)} entries)")
    
    # Calculate error at each reference timestamp
    errors = []
    start_time = float(reference_data[0]['timestamp'])
    
    # We'll count how many pairs we're skipping due to timestamp difference
    skipped_pairs = 0
    processed_pairs = 0
    orientation_errors_computed = 0
    
    for ref_item in reference_data:
        ref_time = float(ref_item['timestamp'])
        est_item, time_diff = find_closest_timestamp(ref_time, estimate_data)
        
        # Only use the pair if timestamps are close enough
        if time_diff <= max_time_diff:
            try:
                ref_pos = extract_position(ref_item)
                est_pos = extract_position(est_item)
                
                # Try to extract orientation
                ref_orient = extract_orientation(ref_item)
                est_orient = extract_orientation(est_item)
                
                # Calculate position errors
                error_x = ref_pos['x'] - est_pos['x']
                error_y = ref_pos['y'] - est_pos['y']
                error_z = ref_pos['z'] - est_pos['z']
                euclidean_error = calculate_euclidean_distance(ref_pos, est_pos)
                
                # Calculate orientation error if available
                orientation_error = None
                if ref_orient is not None and est_orient is not None:
                    orientation_error = calculate_orientation_error(ref_orient, est_orient)
                    if orientation_error is not None:
                        orientation_errors_computed += 1
                
                errors.append({
                    'time': ref_time - start_time,  # Relative time
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

def plot_error(errors, output_path=None, reference_name="Ground Truth"):
    """Plot the error data."""
    if not errors:
        print("No error data to plot")
        return
    
    # Extract data for plotting
    times = [error['time'] for error in errors]
    euclidean_errors = [error['euclidean_error'] for error in errors]
    error_x = [error['error_x'] for error in errors]
    error_y = [error['error_y'] for error in errors]
    error_z = [error['error_z'] for error in errors]
    
    # Extract orientation errors if available
    has_orientation = any(error.get('orientation_error') is not None for error in errors)
    if has_orientation:
        orientation_errors = [error.get('orientation_error', 0) for error in errors]
        # Convert to degrees for better readability
        orientation_errors = [error * 180 / np.pi if error is not None else 0 for error in orientation_errors]
    
    # Calculate statistics
    avg_error = sum(euclidean_errors) / len(euclidean_errors)
    max_error = max(euclidean_errors)
    
    # Create a figure with multiple subplots - add one more if we have orientation data
    if has_orientation:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot Euclidean distance error
    ax1.plot(times, euclidean_errors, 'b-', label=f'Error vs {reference_name}')
    ax1.axhline(y=avg_error, color='r', linestyle='--', label=f'Avg Error: {avg_error:.4f} m')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Euclidean Error (meters)')
    ax1.set_title(f'Position Error vs {reference_name} Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot component errors
    ax2.plot(times, error_x, 'r-', label='X Error')
    ax2.plot(times, error_y, 'g-', label='Y Error')
    ax2.plot(times, error_z, 'b-', label='Z Error')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Component Error (meters)')
    ax2.set_title('X, Y, Z Component Errors Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Plot orientation error if available
    if has_orientation:
        ax3.plot(times, orientation_errors, 'm-', label='Orientation Error')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Orientation Error (degrees)')
        ax3.set_title('Orientation Error Over Time')
        ax3.grid(True)
        ax3.legend()
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return avg_error, max_error

def plot_trajectory(errors, output_path=None, reference_name="Ground Truth"):
    """Plot the trajectory of both reference and estimated positions."""
    if not errors:
        print("No data to plot trajectory")
        return
    
    # Extract trajectory data
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    plt.figure(figsize=(10, 8))
    plt.plot(ref_x, ref_y, 'b-', label=reference_name)
    plt.plot(est_x, est_y, 'r--', label='Estimated Position')
    
    # Add arrows to show direction
    arrow_indices = np.linspace(0, len(ref_x)-1, min(20, len(ref_x))).astype(int)
    for i in arrow_indices:
        if i+1 < len(ref_x):
            plt.arrow(ref_x[i], ref_y[i], 
                     (ref_x[i+1]-ref_x[i])*0.5, (ref_y[i+1]-ref_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='b', ec='b')
            
            plt.arrow(est_x[i], est_y[i], 
                     (est_x[i+1]-est_x[i])*0.5, (est_y[i+1]-est_y[i])*0.5, 
                     head_width=0.01, head_length=0.02, fc='r', ec='r')
    
    # Add markers for start and end points
    plt.plot(ref_x[0], ref_y[0], 'bo', markersize=10, label=f"{reference_name} Start")
    plt.plot(ref_x[-1], ref_y[-1], 'bx', markersize=10, label=f"{reference_name} End")
    plt.plot(est_x[0], est_y[0], 'ro', markersize=10, label="Estimated Start")
    plt.plot(est_x[-1], est_y[-1], 'rx', markersize=10, label="Estimated End")
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title('Robot Trajectory')
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()

def plot_position_over_time(errors, output_path=None, reference_name="Ground Truth"):
    """Plot X and Y positions over time for both reference and estimated position."""
    if not errors:
        print("No data to plot positions over time")
        return
    
    # Extract time and position data
    times = [error['time'] for error in errors]
    ref_x = [error['ref_x'] for error in errors]
    ref_y = [error['ref_y'] for error in errors]
    est_x = [error['est_x'] for error in errors]
    est_y = [error['est_y'] for error in errors]
    
    # Create a figure with two subplots (one for X, one for Y)
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
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Position over time plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze robot positioning error from log files.')
    parser.add_argument('--log-file', default='./logs/20250217_194555_log.json', 
                        help='Path to the log file (default: paste.txt)')
    parser.add_argument('--simulation', dest='simulation', action='store_true',
                        help='Use groundtruth as reference (default)')
    parser.add_argument('--real', dest='simulation', action='store_false',
                        help='Use aruco markers as reference')
    parser.add_argument('--max-time-diff', type=float, default=0.1, 
                        help='Maximum time difference between reference and estimate (seconds)')
    parser.add_argument('--output', '-o', help='Path to save the plot')
    parser.add_argument('--trajectory', '-t', action='store_true', help='Generate trajectory plot')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    parser.set_defaults(simulation=True)  # Default to simulation mode
    args = parser.parse_args()
    
    try:
        # Load and parse log file
        log_data = load_log_file(args.log_file)
        print(f"Loaded {len(log_data)} log entries")
        
        # Print sample entries for debugging if requested
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
        
        # Determine reference source based on the simulation flag
        reference_name = "Ground Truth" if args.simulation else "Aruco Markers"
        print(f"Mode: {'Simulation' if args.simulation else 'Real robot'} (using {reference_name} as reference)")
        
        # Analyze error
        errors = analyze_error(log_data, args.simulation, args.max_time_diff)
        
        if not errors:
            print("No matching data points found for error calculation")
            return
        
        # Print error statistics
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
        
        # Check if we have orientation errors
        has_orientation = any(error.get('orientation_error') is not None for error in errors)
        if has_orientation:
            valid_orient_errors = [err['orientation_error'] for err in errors if err.get('orientation_error') is not None]
            if valid_orient_errors:
                avg_orient_error = sum(valid_orient_errors) / len(valid_orient_errors)
                max_orient_error = max(valid_orient_errors)
                # Convert to degrees
                avg_orient_error_deg = avg_orient_error * 180 / np.pi
                max_orient_error_deg = max_orient_error * 180 / np.pi
                print(f"  Average Orientation Error: {avg_orient_error_deg:.4f} degrees")
                print(f"  Maximum Orientation Error: {max_orient_error_deg:.4f} degrees")
        
        # Plot error over time
        error_output = f"{args.output}_error.png" if args.output else None
        plot_error(errors, error_output, reference_name)
        
        # Generate trajectory plot if requested
        if args.trajectory:
            traj_output = f"{args.output}_trajectory.png" if args.output else None
            plot_trajectory(errors, traj_output, reference_name)
        
        # Always generate the position over time plot
        pos_output = f"{args.output}_position_time.png" if args.output else None
        plot_position_over_time(errors, pos_output, reference_name)
        
    except FileNotFoundError:
        print(f"Error: Could not find log file at {args.log_file}")
        print("Current working directory:", os.getcwd())
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()