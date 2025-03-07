#!/usr/bin/env python3
"""
Standalone trajectory generator.
This script generates a CSV file with the trajectory data for a robot path.
"""
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
import argparse
import time

# These imports are needed from the original code - you might need to adjust paths
try:
    from math_feeg6043 import Vector, l2m
    from model_feeg6043 import TrajectoryGenerate
except ImportError:
    print("WARNING: Cannot import math_feeg6043 and model_feeg6043 modules.")
    print("This script requires these modules to generate the trajectory.")
    print("Please make sure they are available in your Python path.")
    print("Exiting...")
    import sys
    sys.exit(1)

def generate_trajectory_csv(output_dir=".", sample_rate=10.0, start_x=0.0, start_y=0.0):
    """
    Generate a trajectory CSV file based on the LaptopPilot path.
    
    Args:
        output_dir (str): Directory to save the CSV file
        sample_rate (float): Rate at which to sample the trajectory (Hz)
        start_x (float): Starting X position (eastings)
        start_y (float): Starting Y position (northings)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create timestamp for the filename
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"trajectory_{timestamp}.csv"
    csv_path = Path(output_dir) / csv_filename
    
    # Define the square path waypoints (same as in LaptopPilot)
    northings_path = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    eastings_path = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    
    # Offset waypoints by starting position
    for i in range(len(northings_path)):
        northings_path[i] += start_y
        eastings_path[i] += start_x
    
    # Convert path to matrix for TrajectoryGenerate
    C = l2m([northings_path, eastings_path])
    
    # Create trajectory generator instance
    path = TrajectoryGenerate(C[:,0], C[:,1])
    
    # Set trajectory parameters (same as in LaptopPilot)
    velocity = 0.1  # m/s
    acceleration = velocity/3  # m/s²
    turning_radius = 0.35  # meters - minimum turning radius
    
    # Generate the trajectory
    path.path_to_trajectory(velocity, acceleration)
    path.turning_arcs(turning_radius)
    path.wp_id = 0  # Initialize waypoint ID
    
    print(f"Generating trajectory with:")
    print(f"  - {len(northings_path)} waypoints")
    print(f"  - Starting position: ({start_x}, {start_y})")
    print(f"  - Velocity: {velocity} m/s")
    print(f"  - Acceleration: {acceleration} m/s²")
    print(f"  - Turning radius: {turning_radius} m")
    print(f"  - Sample rate: {sample_rate} Hz")
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['time', 'x', 'y', 'theta', 'linear_vel', 'angular_vel'])
        
        # Initialize time
        t = 0.0
        dt = 1.0 / sample_rate
        
        # Dummy state vector for wp_progress (we don't need it for plotting)
        dummy_state = Vector(3)
        dummy_state[0] = start_y
        dummy_state[1] = start_x
        dummy_state[2] = 0.0  # Initial yaw
        
        # Track points for convergence detection
        last_positions = []
        converged_count = 0
        
        # Sample trajectory until it's completed or max time reached
        max_time = 300.0  # 5 minutes max to prevent infinite loops
        points_count = 0
        
        while t < max_time:
            # Update waypoint progress
            path.wp_progress(t, dummy_state, turning_radius)
            
            # Sample the path at current time
            p_ref, u_ref = path.p_u_sample(t)
            
            # Check if we've reached the end (waypoint ID is at the end and position isn't changing)
            current_pos = (p_ref[0, 0], p_ref[1, 0])
            last_positions.append(current_pos)
            if len(last_positions) > 10:
                last_positions.pop(0)
                
                # Check if position has converged (not moving)
                if path.wp_id >= len(northings_path) - 1:
                    positions_converged = all(
                        abs(pos[0] - last_positions[-1][0]) < 0.001 and 
                        abs(pos[1] - last_positions[-1][1]) < 0.001 
                        for pos in last_positions
                    )
                    
                    if positions_converged:
                        converged_count += 1
                        if converged_count > 10:  # Wait for several converged points to confirm
                            print(f"Trajectory completed at t={t:.2f}s (position converged)")
                            break
                    else:
                        converged_count = 0
            
            # Write the trajectory point to CSV
            csv_writer.writerow([
                t,               # time
                p_ref[1, 0],     # x (eastings)
                p_ref[0, 0],     # y (northings)
                p_ref[2, 0],     # theta (yaw)
                u_ref[0, 0],     # linear velocity
                u_ref[1, 0]      # angular velocity
            ])
            points_count += 1
            
            # Increment time
            t += dt
            
            # Status update every second
            if points_count % int(sample_rate) == 0:
                print(f"Generated {points_count} points, t={t:.1f}s, waypoint {path.wp_id}/{len(northings_path)-1}")
    
    print(f"Trajectory saved to: {csv_path}")
    print(f"Generated {points_count} points over {t:.2f} seconds")
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Generate robot trajectory CSV file')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Directory to save the CSV file (default: current directory)')
    parser.add_argument('--sample-rate', '-r', type=float, default=10.0,
                        help='Sample rate in Hz (default: 10.0)')
    parser.add_argument('--start-x', '-x', type=float, default=0.0,
                        help='Starting X position in meters (default: 0.0)')
    parser.add_argument('--start-y', '-y', type=float, default=0.0,
                        help='Starting Y position in meters (default: 0.0)')
    args = parser.parse_args()
    
    csv_path = generate_trajectory_csv(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        start_x=args.start_x,
        start_y=args.start_y
    )
    
    print(f"CSV file generated: {csv_path}")
    
if __name__ == "__main__":
    main()