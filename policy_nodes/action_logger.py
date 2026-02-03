#!/usr/bin/env python3
"""
Action logging utility for debugging policy execution.
Records all actions, poses, and timing for offline analysis.
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class ActionLogger:
    """Logs all actions and poses for debugging."""

    def __init__(self, log_dir="logs", prefix=""):
        """Initialize action logger.

        Args:
            log_dir: Directory to save logs
            prefix: Prefix for log filename (e.g., "rtc" or "openloop")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{prefix}_{timestamp}.jsonl"

        self.start_time = time.time()
        self.action_count = 0

        print(f"ActionLogger: Logging to {self.log_file}")

    def log_action(self, action, current_pose, target_pose, gripper_cmd,
                   chunk_idx=None, inference_count=None, **kwargs):
        """Log a single action execution.

        Args:
            action: Raw action from policy [x, y, z, yaw, ...]
            current_pose: Current robot pose [x, y, z, yaw]
            target_pose: Target pose being sent to robot [x, y, z, yaw]
            gripper_cmd: Gripper command (0 or 1)
            chunk_idx: Index within current chunk (for RTC)
            inference_count: Total number of inferences so far
            **kwargs: Additional data to log
        """
        elapsed = time.time() - self.start_time

        log_entry = {
            'timestamp': elapsed,
            'action_count': self.action_count,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'current_pose': current_pose.tolist() if isinstance(current_pose, np.ndarray) else current_pose,
            'target_pose': target_pose.tolist() if isinstance(target_pose, np.ndarray) else target_pose,
            'gripper_cmd': float(gripper_cmd),
        }

        if chunk_idx is not None:
            log_entry['chunk_idx'] = int(chunk_idx)
        if inference_count is not None:
            log_entry['inference_count'] = int(inference_count)

        # Add any extra data
        log_entry.update(kwargs)

        # Write to file (JSONL format - one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        self.action_count += 1

    def log_inference(self, obs_summary, actions, inference_time_ms, **kwargs):
        """Log an inference event.

        Args:
            obs_summary: Summary of observation (e.g., pose, gripper state)
            actions: Full action chunk from inference
            inference_time_ms: Inference time in milliseconds
            **kwargs: Additional data (e.g., RTC parameters)
        """
        elapsed = time.time() - self.start_time

        log_entry = {
            'timestamp': elapsed,
            'event': 'inference',
            'obs_summary': obs_summary,
            'actions_shape': list(actions.shape) if isinstance(actions, np.ndarray) else None,
            'actions_norm': float(np.linalg.norm(actions)) if isinstance(actions, np.ndarray) else None,
            'first_action': actions[0].tolist() if isinstance(actions, np.ndarray) and len(actions) > 0 else None,
            'last_action': actions[-1].tolist() if isinstance(actions, np.ndarray) and len(actions) > 0 else None,
            'inference_time_ms': float(inference_time_ms),
        }

        # Add any extra data
        log_entry.update(kwargs)

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def close(self):
        """Finalize logging."""
        print(f"ActionLogger: Logged {self.action_count} actions to {self.log_file}")


def compare_logs(log_file1, log_file2, output_file=None):
    """Compare two action logs and generate a report.

    Args:
        log_file1: Path to first log (e.g., open-loop)
        log_file2: Path to second log (e.g., RTC)
        output_file: Optional output file for comparison report
    """
    import matplotlib.pyplot as plt

    # Load logs
    def load_log(path):
        actions = []
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if 'action' in entry:  # Only action entries
                    actions.append(entry)
        return actions

    actions1 = load_log(log_file1)
    actions2 = load_log(log_file2)

    print(f"Log 1: {len(actions1)} actions from {log_file1}")
    print(f"Log 2: {len(actions2)} actions from {log_file2}")

    if len(actions1) == 0 or len(actions2) == 0:
        print("⚠ One or both logs are empty!")
        return

    # Extract trajectories
    def extract_trajectory(actions):
        times = [a['timestamp'] for a in actions]
        targets = np.array([a['target_pose'][:3] for a in actions])  # xyz only
        currents = np.array([a['current_pose'][:3] for a in actions])
        return times, targets, currents

    times1, targets1, currents1 = extract_trajectory(actions1)
    times2, targets2, currents2 = extract_trajectory(actions2)

    # Plot comparison
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Action Log Comparison', fontsize=16)

    axes = axes.flatten()

    # Plot targets
    for i, (label, color) in enumerate([('x', 'r'), ('y', 'g'), ('z', 'b')]):
        ax = axes[i]
        ax.plot(times1, targets1[:, i], f'{color}-', label='Log 1 (target)', alpha=0.7)
        ax.plot(times1, currents1[:, i], f'{color}--', label='Log 1 (current)', alpha=0.5)
        ax.plot(times2, targets2[:, i], f'{color}-', label='Log 2 (target)', alpha=0.7, linewidth=2)
        ax.plot(times2, currents2[:, i], f'{color}--', label='Log 2 (current)', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} position (m)')
        ax.set_title(f'{label.upper()} Position')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3D trajectory
    ax = axes[3]
    ax = fig.add_subplot(3, 2, 4, projection='3d')
    ax.plot(targets1[:, 0], targets1[:, 1], targets1[:, 2], 'b-', label='Log 1', alpha=0.7)
    ax.plot(targets2[:, 0], targets2[:, 1], targets2[:, 2], 'r-', label='Log 2', alpha=0.7, linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()

    # Plot tracking error
    ax = axes[4]
    error1 = np.linalg.norm(targets1 - currents1, axis=1)
    error2 = np.linalg.norm(targets2 - currents2, axis=1)
    ax.plot(times1, error1, 'b-', label='Log 1', alpha=0.7)
    ax.plot(times2, error2, 'r-', label='Log 2', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking error (m)')
    ax.set_title('Target vs Current Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[5]
    ax.axis('off')
    summary = f"""
    COMPARISON SUMMARY

    Log 1: {Path(log_file1).name}
    - Actions: {len(actions1)}
    - Duration: {times1[-1]:.1f}s
    - Mean error: {np.mean(error1):.4f}m
    - Max error: {np.max(error1):.4f}m

    Log 2: {Path(log_file2).name}
    - Actions: {len(actions2)}
    - Duration: {times2[-1]:.1f}s
    - Mean error: {np.mean(error2):.4f}m
    - Max error: {np.max(error2):.4f}m
    """
    ax.text(0.1, 0.5, summary, fontfamily='monospace', fontsize=10,
            verticalalignment='center')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {output_file}")
    else:
        plt.savefig('action_log_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to action_log_comparison.png")

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare two action logs')
    parser.add_argument('log1', type=str, help='First log file (e.g., open-loop)')
    parser.add_argument('log2', type=str, help='Second log file (e.g., RTC)')
    parser.add_argument('--output', type=str, default=None, help='Output plot file')
    args = parser.parse_args()

    compare_logs(args.log1, args.log2, args.output)
