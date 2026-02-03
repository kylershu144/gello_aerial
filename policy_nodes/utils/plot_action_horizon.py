#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def load_action_csv(csv_path):
    """Load action data from CSV file."""
    steps = []
    del_x = []
    del_y = []
    del_z = []
    del_yaw = []
    gripper = []

    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            steps.append(int(row['step']))
            del_x.append(float(row['del_x']))
            del_y.append(float(row['del_y']))
            del_z.append(float(row['del_z']))
            del_yaw.append(float(row['del_yaw']))
            gripper.append(float(row['gripper']))

    return np.array(steps), np.array(del_x), np.array(del_y), np.array(del_z), np.array(del_yaw), np.array(gripper)

def plot_actions(csv_path):
    """Plot the 50 action horizon from CSV file."""
    steps, del_x, del_y, del_z, del_yaw, gripper = load_action_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Action Horizon - {os.path.basename(csv_path)}', fontsize=16)

    # Plot del_x
    axes[0, 0].plot(steps, del_x, 'b-', linewidth=2)
    axes[0, 0].set_title('Delta X', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('del_x')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot del_y
    axes[0, 1].plot(steps, del_y, 'g-', linewidth=2)
    axes[0, 1].set_title('Delta Y', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('del_y')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot del_z
    axes[1, 0].plot(steps, del_z, 'r-', linewidth=2)
    axes[1, 0].set_title('Delta Z', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('del_z')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot del_yaw
    axes[1, 1].plot(steps, del_yaw, 'm-', linewidth=2)
    axes[1, 1].set_title('Delta Yaw', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('del_yaw')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot gripper
    axes[2, 0].plot(steps, gripper, 'c-', linewidth=2)
    axes[2, 0].set_title('Gripper', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('gripper')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 3D trajectory (XYZ)
    ax_3d = fig.add_subplot(3, 2, 6, projection='3d')
    # Compute cumulative position for visualization
    cum_x = np.cumsum(del_x)
    cum_y = np.cumsum(del_y)
    cum_z = np.cumsum(del_z)
    ax_3d.plot(cum_x, cum_y, cum_z, 'k-', linewidth=2)
    ax_3d.scatter(cum_x[0], cum_y[0], cum_z[0], c='g', s=100, marker='o', label='Start')
    ax_3d.scatter(cum_x[-1], cum_y[-1], cum_z[-1], c='r', s=100, marker='*', label='End')
    ax_3d.set_title('Cumulative 3D Trajectory', fontsize=12, fontweight='bold')
    ax_3d.set_xlabel('Cumulative X')
    ax_3d.set_ylabel('Cumulative Y')
    ax_3d.set_zlabel('Cumulative Z')
    ax_3d.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = csv_path.replace('.csv', '_plot.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")

    # Show plot
    plt.show()

def find_latest_csv():
    """Find the most recent action_horizon_*.csv file in the current directory."""
    csv_files = glob.glob('action_horizon_*.csv')
    if not csv_files:
        return None
    # Sort by modification time and return the latest
    latest_csv = max(csv_files, key=os.path.getmtime)
    return latest_csv

def main():
    parser = argparse.ArgumentParser(description='Plot action horizon from CSV file')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (default: latest action_horizon_*.csv)')
    args = parser.parse_args()

    csv_path = args.csv

    # If no CSV file specified, find the latest one
    if csv_path is None:
        csv_path = find_latest_csv()
        if csv_path is None:
            print("Error: No action_horizon_*.csv files found in current directory.")
            print("Please specify a CSV file with --csv option.")
            return
        print(f"Using latest CSV file: {csv_path}")

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    # Plot the actions
    plot_actions(csv_path)

if __name__ == '__main__':
    main()
