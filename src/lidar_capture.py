import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def load_laserscans(file_path):
    laser_scans = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("class") == "LaserScan":
                timestamp = float(data.get("timestamp", 0))
                msg = data.get("message", {})
                angles = msg.get("angles", [])
                ranges = msg.get("ranges", [])
                if angles and ranges and len(angles) == len(ranges):
                    laser_scans.append({
                        "timestamp": timestamp,
                        "angles": angles,
                        "ranges": ranges
                    })
    return laser_scans

def plot_scan(scan, index):
    angles = np.array(scan["angles"])
    ranges = np.array(scan["ranges"])
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5)
    plt.title(f"LIDAR Scan {index + 1} (timestamp: {scan['timestamp']})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_laserscans.py <filename.json> [start_time end_time]")
        sys.exit(1)

    filename = sys.argv[1]
    file_path = os.path.join("..", "logs", filename)

    scans = load_laserscans(file_path)
    print(f"Total LaserScan messages found: {len(scans)}")

    if len(sys.argv) == 4:
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3])
        scans = [s for s in scans if start_time <= s["timestamp"] <= end_time]
        print(f"Plotting {len(scans)} scans between {start_time} and {end_time}...")
    else:
        print("No time range provided. Plotting all scans...")

    for i, scan in enumerate(scans):
        plot_scan(scan, i)

if __name__ == "__main__":
    main()
