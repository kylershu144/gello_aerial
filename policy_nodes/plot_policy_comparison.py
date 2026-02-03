#!/usr/bin/env python3
"""
Script to compare expected (training data) vs predicted (policy) actions.
Loads parquet files, feeds states to the policy server, and plots the comparison.
"""

import sys
from pathlib import Path

# Add the repository root to Python path so we can import openpi_client
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend by default for reliable plot saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse
import cv2

# Import policy client
from openpi_client import image_tools

# Import policy infrastructure from common
try:
    from policy_nodes.common import (
        PolicyType,
        create_policy_client,
        resolve_policy_host,
        DEFAULT_PORTS,
        DEFAULT_HOSTS,
    )
except Exception:
    # Fallback if import fails (e.g., running from different directory)
    from common import (
        PolicyType,
        create_policy_client,
        resolve_policy_host,
        DEFAULT_PORTS,
        DEFAULT_HOSTS,
    )

# Task definition (from common.py)
TASKS = {
    "gate_nav": "go through the gate and hover over the stuffed animal",
    "penguin_grasp_blue": "grab the stuffed animal and put it in the blue bucket",
    "chip_grasp": "grab the chip bag and put it in the blue bucket",
}

def load_parquet_files(data_dir: str):
    """Load all parquet files from the specified directory."""
    parquet_files = glob.glob(f"{data_dir}/**/*.parquet", recursive=True)
    print(f"Found {len(parquet_files)} parquet files")
    
    episodes = []
    for pq_file in sorted(parquet_files):
        try:
            df = pd.read_parquet(pq_file)
            episodes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {pq_file}: {e}")
    
    return episodes

def decode_image_from_parquet(image_dict):
    """Decode image from parquet dictionary format.
    
    Args:
        image_dict: Dictionary with 'bytes' and 'path' keys
        
    Returns:
        numpy array of decoded image in BGR format
    """
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        img_bytes = image_dict['bytes']
        if img_bytes is not None:
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    # If already a numpy array, return as-is
    elif isinstance(image_dict, np.ndarray):
        return image_dict
    
    raise ValueError(f"Could not decode image from type {type(image_dict)}")

def prepare_observation(row, task_prompt: str, policy_type: PolicyType = PolicyType.OPENPI):
    """Prepare observation dictionary from a parquet row.

    Args:
        row: DataFrame row with state, image, wrist_image, 3pov_1
        task_prompt: Task description string
        policy_type: Type of policy (affects observation format)

    Returns:
        Observation dictionary formatted for the policy type
    """
    # State is already in the correct format
    state = row['state']

    # Decode images from parquet format (they're stored as compressed bytes)
    front_image = decode_image_from_parquet(row['image'])
    wrist_image = decode_image_from_parquet(row['wrist_image'])
    third_pov = decode_image_from_parquet(row['3pov_1'])

    # Resize and convert images using image_tools
    front_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(front_image, 256, 256)
    )
    wrist_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, 256, 256)
    )

    # Third person view - resize to 256x256
    if third_pov.shape[:2] != (256, 256):
        third_pov_resized = cv2.resize(third_pov, (256, 256))
    else:
        third_pov_resized = third_pov

    # Build observation dict based on policy type
    if policy_type == PolicyType.ACT or policy_type == PolicyType.DIFFUSION:
        # gRPC format (ACT and Diffusion use same server)
        obs_dict = {
            "observation.state": state.astype(np.float32),
            "observation.images.image": front_resized,
            "observation.images.wrist_image": wrist_resized,
            "observation.images.3pov_1": third_pov_resized,
        }
    else:
        # OpenPI format (WebSocket)
        obs_dict = {
            "observation/state": state.astype(np.float32),
            "observation/image": front_resized,
            "observation/wrist_image": wrist_resized,
            "observation/3pov_1": third_pov_resized,
            "prompt": task_prompt
        }

    return obs_dict

def collect_predictions(episodes, client, task_prompt: str, max_samples: int = None, sample_strategy: str = 'sequential', policy_type: PolicyType = PolicyType.OPENPI):
    """
    Collect predictions from the policy for all timesteps in episodes.

    Args:
        episodes: List of DataFrames with parquet data
        client: Policy client instance
        task_prompt: Task description string
        max_samples: Maximum number of samples to process (None = all)
        sample_strategy: 'sequential' (default), 'random', or 'stratified'
        policy_type: Type of policy (affects observation format)

    Returns:
        expected_actions: Array of expected actions from training data
        predicted_actions: Array of predicted actions from policy
        states: Array of states used for predictions
        inference_times: Array of inference times in milliseconds
    """
    expected_actions = []
    predicted_actions = []
    states = []
    inference_times = []

    # Calculate total samples and create index list
    total_samples = sum(len(ep) for ep in episodes)

    # Build list of (episode_idx, row_idx) tuples
    sample_indices = []
    for episode_idx, episode_df in enumerate(episodes):
        for row_idx in range(len(episode_df)):
            sample_indices.append((episode_idx, row_idx))

    # Apply sampling strategy
    if max_samples is not None and max_samples < len(sample_indices):
        if sample_strategy == 'random':
            # Randomly select starting points, then take sequential samples
            import random
            random.seed(42)  # For reproducibility

            # Calculate how many sequences we can fit
            # If we have episodes of varying lengths, we need to be careful
            sample_indices = []
            remaining_samples = max_samples

            # Shuffle episodes to randomize which ones we sample from
            episode_order = list(range(len(episodes)))
            random.shuffle(episode_order)

            for episode_idx in episode_order:
                if remaining_samples <= 0:
                    break

                episode_df = episodes[episode_idx]
                episode_len = len(episode_df)

                # Decide how many samples to take from this episode
                samples_from_episode = min(remaining_samples, episode_len)

                # Random starting point in episode (ensure we can take consecutive samples)
                max_start = max(0, episode_len - samples_from_episode)
                start_idx = random.randint(0, max_start) if max_start > 0 else 0

                # Take consecutive samples starting from random point
                for i in range(samples_from_episode):
                    sample_indices.append((episode_idx, start_idx + i))

                remaining_samples -= samples_from_episode

            print(f"Random sequential sampling: {len(sample_indices)} consecutive samples from random episode segments")
        elif sample_strategy == 'stratified':
            # Sample evenly from each episode, maintaining sequential ordering
            samples_per_episode = max_samples // len(episodes)
            remainder = max_samples % len(episodes)
            sample_indices = []
            for episode_idx, episode_df in enumerate(episodes):
                ep_samples = samples_per_episode + (1 if episode_idx < remainder else 0)
                ep_samples = min(ep_samples, len(episode_df))
                # Take consecutive samples from evenly spaced starting points
                step = len(episode_df) / ep_samples if ep_samples > 0 else 0
                for i in range(ep_samples):
                    start_idx = int(i * step)
                    sample_indices.append((episode_idx, start_idx))
            print(f"Stratified sampling: {len(sample_indices)} samples evenly spaced across {len(episodes)} episodes")
        else:  # sequential
            sample_indices = sample_indices[:max_samples]
            print(f"Sequential sampling: {max_samples} samples from {total_samples} total")
    else:
        print(f"Processing all {total_samples} samples...")

    # Process samples
    with tqdm(total=len(sample_indices)) as pbar:
        for episode_idx, row_idx in sample_indices:
            try:
                episode_df = episodes[episode_idx]
                row = episode_df.iloc[row_idx]

                # Prepare observation
                obs_dict = prepare_observation(row, task_prompt, policy_type)

                # Get policy prediction and measure inference time
                start_time = time.perf_counter()
                result = client.infer(obs_dict)
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

                predicted_action = result["actions"][0]  # First action in horizon

                # Get expected action from training data
                expected_action = row['actions']

                # Get state
                state = row['state']

                # Store results
                expected_actions.append(expected_action)
                predicted_actions.append(predicted_action)
                states.append(state)
                inference_times.append(inference_time_ms)

                pbar.update(1)

            except Exception as e:
                print(f"\nWarning: Error processing episode {episode_idx}, row {row_idx}: {e}")
                continue
    
    expected_actions = np.array(expected_actions)
    predicted_actions = np.array(predicted_actions)
    states = np.array(states)
    inference_times = np.array(inference_times)

    print(f"\nCollected {len(expected_actions)} samples")
    print(f"Expected actions shape: {expected_actions.shape}")
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"States shape: {states.shape}")
    print(f"Inference times: mean={inference_times.mean():.2f}ms, std={inference_times.std():.2f}ms, min={inference_times.min():.2f}ms, max={inference_times.max():.2f}ms")

    return expected_actions, predicted_actions, states, inference_times

def plot_comparison(expected_actions, predicted_actions, save_path: str = None, show_interactive: bool = False):
    """
    Create subplots comparing expected vs predicted actions for each dimension.
    
    Args:
        expected_actions: Array of shape (N, 7) with expected actions
        predicted_actions: Array of shape (N, 7) with predicted actions
        save_path: Optional path to save the figure
        show_interactive: Whether to display plot interactively (default: False)
    """
    action_dims = 7
    action_labels = ['X', 'Y', 'Z', 'Yaw', 'Dim 4', 'Dim 5', 'Gripper']
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for dim in range(action_dims):
        ax = axes[dim]
        
        # Get data for this dimension
        expected = expected_actions[:, dim]
        predicted = predicted_actions[:, dim]
        
        # Scatter plot
        ax.scatter(expected, predicted, alpha=0.5, s=10)
        
        # Add diagonal line (perfect prediction)
        min_val = min(expected.min(), predicted.min())
        max_val = max(expected.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        # Calculate metrics
        mse = np.mean((expected - predicted) ** 2)
        mae = np.mean(np.abs(expected - predicted))
        correlation = np.corrcoef(expected, predicted)[0, 1]
        
        # Set labels and title
        ax.set_xlabel(f'Expected {action_labels[dim]}', fontsize=12)
        ax.set_ylabel(f'Predicted {action_labels[dim]}', fontsize=12)
        ax.set_title(f'{action_labels[dim]} - MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {correlation:.3f}', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot (we have 7 dimensions, 8 subplots)
    axes[-1].remove()
    
    plt.suptitle('Expected vs Predicted Actions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Scatter plot saved to: {save_path}")
    
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)

def plot_time_series(expected_actions, predicted_actions, max_timesteps: int = 500, save_path: str = None, show_interactive: bool = False):
    """
    Create time-series plots showing expected vs predicted over time.
    
    Args:
        expected_actions: Array of shape (N, 7) with expected actions
        predicted_actions: Array of shape (N, 7) with predicted actions
        max_timesteps: Maximum number of timesteps to plot
        save_path: Optional path to save the figure
        show_interactive: Whether to display plot interactively (default: False)
    """
    action_dims = 7
    action_labels = ['X', 'Y', 'Z', 'Yaw', 'Dim 4', 'Dim 5', 'Gripper']
    
    # Limit number of samples for clarity
    n_samples = min(len(expected_actions), max_timesteps)
    timesteps = np.arange(n_samples)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for dim in range(action_dims):
        ax = axes[dim]
        
        # Get data for this dimension
        expected = expected_actions[:n_samples, dim]
        predicted = predicted_actions[:n_samples, dim]
        
        # Plot time series
        ax.plot(timesteps, expected, label='Expected', alpha=0.7, linewidth=1.5)
        ax.plot(timesteps, predicted, label='Predicted', alpha=0.7, linewidth=1.5)
        
        # Calculate metrics
        mse = np.mean((expected - predicted) ** 2)
        mae = np.mean(np.abs(expected - predicted))
        
        # Set labels and title
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel(f'{action_labels[dim]}', fontsize=12)
        ax.set_title(f'{action_labels[dim]} - MSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot
    axes[-1].remove()
    
    plt.suptitle('Expected vs Predicted Actions Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Time series plot saved to: {save_path}")
    
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)

def plot_state_vs_action(states, predicted_actions, save_path: str = None, show_interactive: bool = False):
    """
    Create subplots comparing state values to predicted actions for each dimension.
    
    This shows how the policy's predictions relate to the current state.
    For example: how does current X position relate to predicted X action?
    
    Args:
        states: Array of shape (N, 7) with state observations
        predicted_actions: Array of shape (N, 7) with predicted actions
        save_path: Optional path to save the figure
        show_interactive: Whether to display plot interactively (default: False)
    """
    action_dims = 7
    dimension_labels = ['X', 'Y', 'Z', 'Yaw', 'Dim 4', 'Dim 5', 'Gripper']
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for dim in range(action_dims):
        ax = axes[dim]
        
        # Get data for this dimension
        state_dim = states[:, dim]
        action_dim = predicted_actions[:, dim]
        
        # Scatter plot: state value vs predicted action
        ax.scatter(state_dim, action_dim, alpha=0.5, s=10, c='purple')
        
        # Calculate metrics (with error handling for zero variance)
        try:
            # Check for zero variance
            if np.std(state_dim) < 1e-10 or np.std(action_dim) < 1e-10:
                correlation = 0.0
                correlation_str = "N/A (zero variance)"
            else:
                correlation = np.corrcoef(state_dim, action_dim)[0, 1]
                if np.isnan(correlation):
                    correlation_str = "N/A"
                else:
                    correlation_str = f"{correlation:.3f}"
        except Exception:
            correlation_str = "N/A"
        
        # Linear fit to show trend (with error handling)
        fit_label = None
        try:
            if len(state_dim) > 1 and np.std(state_dim) > 1e-10:
                # Remove any NaN or Inf values
                valid_mask = np.isfinite(state_dim) & np.isfinite(action_dim)
                if np.sum(valid_mask) > 1:
                    state_valid = state_dim[valid_mask]
                    action_valid = action_dim[valid_mask]
                    
                    z = np.polyfit(state_valid, action_valid, 1)
                    p = np.poly1d(z)
                    state_sorted = np.sort(state_valid)
                    ax.plot(state_sorted, p(state_sorted), "r--", linewidth=2, 
                           label=f'Linear fit (slope={z[0]:.3f})')
                    fit_label = True
        except Exception as e:
            # If polyfit fails, just skip the linear fit
            pass
        
        # Set labels and title
        ax.set_xlabel(f'State {dimension_labels[dim]}', fontsize=12)
        ax.set_ylabel(f'Predicted Action {dimension_labels[dim]}', fontsize=12)
        ax.set_title(f'{dimension_labels[dim]} - Correlation: {correlation_str}', fontsize=10)
        if fit_label:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot (we have 7 dimensions, 8 subplots)
    axes[-1].remove()
    
    plt.suptitle('State vs Predicted Action (Policy Behavior)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ State vs Action plot saved to: {save_path}")
    
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)

def plot_inference_times(inference_times, save_path: str = None, show_interactive: bool = False):
    """
    Create plots showing inference time statistics.

    Args:
        inference_times: Array of inference times in milliseconds
        save_path: Optional path to save the figure
        show_interactive: Whether to display plot interactively (default: False)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time series of inference times
    ax1 = axes[0, 0]
    ax1.plot(inference_times, alpha=0.7, linewidth=0.8)
    ax1.axhline(y=inference_times.mean(), color='r', linestyle='--', label=f'Mean: {inference_times.mean():.2f}ms')
    ax1.fill_between(range(len(inference_times)),
                     inference_times.mean() - inference_times.std(),
                     inference_times.mean() + inference_times.std(),
                     alpha=0.2, color='r', label=f'Std: {inference_times.std():.2f}ms')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Inference Time Over Samples', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of inference times
    ax2 = axes[0, 1]
    ax2.hist(inference_times, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=inference_times.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {inference_times.mean():.2f}ms')
    ax2.axvline(x=np.median(inference_times), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(inference_times):.2f}ms')
    ax2.set_xlabel('Inference Time (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Inference Times', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Rolling average (to see trends)
    ax3 = axes[1, 0]
    window_size = min(50, len(inference_times) // 5) if len(inference_times) > 10 else 1
    if window_size > 1:
        rolling_mean = np.convolve(inference_times, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size-1, len(inference_times)), rolling_mean, linewidth=1.5, label=f'Rolling mean (window={window_size})')
    ax3.plot(inference_times, alpha=0.3, linewidth=0.5, label='Raw')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Inference Time (ms)', fontsize=12)
    ax3.set_title('Inference Time Trend', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics text box
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Inference Time Statistics

    Samples:     {len(inference_times)}
    Mean:        {inference_times.mean():.2f} ms
    Std:         {inference_times.std():.2f} ms
    Median:      {np.median(inference_times):.2f} ms
    Min:         {inference_times.min():.2f} ms
    Max:         {inference_times.max():.2f} ms

    Percentiles:
      5th:       {np.percentile(inference_times, 5):.2f} ms
      25th:      {np.percentile(inference_times, 25):.2f} ms
      75th:      {np.percentile(inference_times, 75):.2f} ms
      95th:      {np.percentile(inference_times, 95):.2f} ms
      99th:      {np.percentile(inference_times, 99):.2f} ms

    Throughput:  {1000/inference_times.mean():.1f} Hz (based on mean)
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Policy Inference Time Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inference time plot saved to: {save_path}")

    if show_interactive:
        plt.show()
    else:
        plt.close(fig)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare expected vs predicted actions from policy server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/javier/Downloads/poop/gate_nav",
        help="Directory containing parquet training data files"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="openpi",
        choices=["openpi", "act", "diffusion"],
        help="Type of policy to use"
    )
    parser.add_argument(
        "--policy-host",
        type=str,
        default=None,
        help='Policy server hostname (default depends on policy type)'
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=None,
        help="Policy server port (default depends on policy type)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gate_nav",
        choices=list(TASKS.keys()),
        help="Task name for prompt"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to process (0 or negative for all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--max-timesteps-plot",
        type=int,
        default=500,
        help="Maximum timesteps to show in time series plot"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display plots interactively (default: just save to files)"
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=["sequential", "random", "stratified"],
        default="stratified",
        help="Sample selection strategy: sequential (first N), random (random N), or stratified (evenly across episodes)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Only use a specific episode (0-indexed). If not specified, uses all episodes."
    )
    # ACT/Diffusion specific options
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to pretrained model (ACT/Diffusion only, relative to server cwd)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: cuda, cpu, mps (ACT/Diffusion only)"
    )

    args = parser.parse_args()

    # Resolve policy type
    policy_type = PolicyType(args.policy_type.lower())

    # Use defaults if not specified
    policy_host = args.policy_host
    if policy_host is None:
        policy_host = DEFAULT_HOSTS[policy_type]

    policy_port = args.policy_port
    if policy_port is None:
        policy_port = DEFAULT_PORTS[policy_type]

    # Resolve pretrained path for ACT/Diffusion
    pretrained_path = args.pretrained_path
    if pretrained_path is None:
        if policy_type == PolicyType.ACT:
            pretrained_path = "outputs/penguin_grasp/act"
        elif policy_type == PolicyType.DIFFUSION:
            pretrained_path = "outputs/penguin_grasp/diffusion"

    # Configuration
    DATA_DIR = args.data_dir
    POLICY_HOST = resolve_policy_host(policy_host)
    POLICY_PORT = policy_port
    TASK_PROMPT = TASKS[args.task]
    MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None
    OUTPUT_DIR = args.output_dir

    # Ensure output directory exists
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Policy Prediction Comparison Script")
    print("=" * 60)
    print(f"Policy type: {policy_type.value}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Policy server: {POLICY_HOST}:{POLICY_PORT}")
    if policy_type in (PolicyType.ACT, PolicyType.DIFFUSION):
        print(f"Pretrained path: {pretrained_path}")
        print(f"Device: {args.device}")
    print(f"Task prompt: {TASK_PROMPT}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"Output directory: {output_path.resolve()}")
    print("=" * 60)
    
    # Load episodes
    print("\nLoading parquet files...")
    episodes = load_parquet_files(DATA_DIR)

    if len(episodes) == 0:
        print("ERROR: No episodes loaded!")
        sys.exit(1)

    print(f"Loaded {len(episodes)} episodes")

    # Filter to specific episode if requested
    if args.episode is not None:
        if args.episode < 0 or args.episode >= len(episodes):
            print(f"ERROR: Episode {args.episode} out of range (0-{len(episodes)-1})")
            sys.exit(1)
        episodes = [episodes[args.episode]]
        print(f"Filtering to episode {args.episode} ({len(episodes[0])} timesteps)")
    
    # Connect to policy server
    print(f"\nConnecting to policy server at {POLICY_HOST}:{POLICY_PORT}...")
    try:
        client = create_policy_client(
            policy_type=policy_type,
            host=POLICY_HOST,
            port=POLICY_PORT,
            pretrained_path=pretrained_path,
            device=args.device,
        )
        print("Successfully connected to policy server!")
    except Exception as e:
        print(f"ERROR: Could not connect to policy server: {e}")
        sys.exit(1)
    
    # Collect predictions
    print("\nCollecting predictions from policy...")
    expected_actions, predicted_actions, states, inference_times = collect_predictions(
        episodes, client, TASK_PROMPT, max_samples=MAX_SAMPLES, sample_strategy=args.sample_strategy, policy_type=policy_type
    )
    
    if len(expected_actions) == 0:
        print("ERROR: No predictions collected!")
        sys.exit(1)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    scatter_path = f"{OUTPUT_DIR}/policy_comparison_scatter.png"
    plot_comparison(
        expected_actions, predicted_actions,
        save_path=scatter_path,
        show_interactive=args.show_plots
    )
    
    timeseries_path = f"{OUTPUT_DIR}/policy_comparison_timeseries.png"
    plot_time_series(
        expected_actions, predicted_actions,
        max_timesteps=args.max_timesteps_plot,
        save_path=timeseries_path,
        show_interactive=args.show_plots
    )
    
    state_action_path = f"{OUTPUT_DIR}/policy_state_vs_action.png"
    plot_state_vs_action(
        states, predicted_actions,
        save_path=state_action_path,
        show_interactive=args.show_plots
    )

    inference_time_path = f"{OUTPUT_DIR}/policy_inference_times.png"
    plot_inference_times(
        inference_times,
        save_path=inference_time_path,
        show_interactive=args.show_plots
    )

    # Cleanup
    if hasattr(client, 'close'):
        client.close()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

