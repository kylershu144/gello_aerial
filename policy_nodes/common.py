#!/usr/bin/env python3

"""Common utilities and constants for policy nodes."""

from typing import Dict, Optional
from abc import ABC, abstractmethod
from enum import Enum
from termcolor import colored
import time
import numpy as np
import cv2
import subprocess
import pickle

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray, Bool
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from openpi_client import websocket_client_policy


# =============================================================================
# Policy Client Types and Factory
# =============================================================================

class PolicyType(Enum):
	"""Supported policy types."""
	OPENPI = "openpi"
	ACT = "act"
	DIFFUSION = "diffusion"


class BasePolicyClient(ABC):
	"""Abstract base class for policy clients."""

	@abstractmethod
	def infer(self, obs: Dict) -> Dict:
		"""Run inference on observation dict, return dict with 'actions' key."""
		pass

	@abstractmethod
	def reset(self) -> None:
		"""Reset the policy state (if applicable)."""
		pass

	def close(self) -> None:
		"""Close any connections (optional override)."""
		pass


class OpenPIClient(BasePolicyClient):
	"""Wrapper around the OpenPI websocket client."""

	def __init__(self, host: str, port: int, timeout: float = 30.0):
		self.host = host
		self.port = port
		print(f"Connecting to OpenPI server at {host}:{port}...")
		self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
		print(f"Connected to OpenPI server at {host}:{port}")

	def infer(self, obs: Dict) -> Dict:
		return self._client.infer(obs)

	def reset(self) -> None:
		self._client.reset()


class ACTClient(BasePolicyClient):
	"""gRPC client for ACT policy server."""

	def __init__(
		self,
		host: str,
		port: int,
		timeout: float = 30.0,
		policy_type: str = "act",
		pretrained_path: str = "outputs/penguin_grasp/act",
		device: str = "cuda",
	):
		self.host = host
		self.port = port
		self.timeout = timeout
		self.policy_type = policy_type
		self.pretrained_path = pretrained_path
		self.device = device
		self._stub = None
		self._channel = None
		self._timestep = 0
		self._connect()

	def _connect(self):
		"""Establish gRPC connection and send policy configuration."""
		import grpc
		import os
		from lerobot.transport import services_pb2, services_pb2_grpc
		from lerobot.async_inference.helpers import RemotePolicyConfig

		print(f"[gRPC] Connecting to {self.host}:{self.port}...")
		self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
		self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)

		# Handshake with server
		print(f"[gRPC] Handshaking with server...")
		self._stub.Ready(services_pb2.Empty())

		# Use relative path (relative to SERVER's working directory, not client's)
		# Do NOT use absolute paths or ./ prefix - just the bare relative path
		pretrained_path = self.pretrained_path
		print(f"[gRPC] Using pretrained path (relative to server cwd): {pretrained_path}")

		# Send policy configuration
		print(f"[gRPC] Sending policy configuration...")
		lerobot_features = {
			"observation.state": {
				"dtype": "float32",
				"shape": (7,),
				"names": ["motor_0", "motor_1", "motor_2", "motor_3", "motor_4", "motor_5", "motor_6"],
			},
			"observation.images.image": {
				"dtype": "image",
				"shape": (256, 256, 3),
				"names": ["height", "width", "channels"],
			},
			"observation.images.wrist_image": {
				"dtype": "image",
				"shape": (256, 256, 3),
				"names": ["height", "width", "channels"],
			},
			"observation.images.3pov_1": {
				"dtype": "image",
				"shape": (256, 256, 3),
				"names": ["height", "width", "channels"],
			},
		}

		policy_config = RemotePolicyConfig(
			policy_type=self.policy_type,
			pretrained_name_or_path=pretrained_path,
			lerobot_features=lerobot_features,
			actions_per_chunk=50,
			device=self.device,
		)

		policy_config_bytes = pickle.dumps(policy_config)
		policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
		self._stub.SendPolicyInstructions(policy_setup)

		print(f"[gRPC] Connected to ACT server at {self.host}:{self.port}")

	def _validate_observation(self, obs: Dict, prefix: str = "[ACT]") -> bool:
		"""Validate observation dict for None values and correct types."""
		issues = []

		# Check required keys
		required_keys = [
			'observation.state',
			'observation.images.image',
			'observation.images.wrist_image',
			'observation.images.3pov_1',
		]

		for key in required_keys:
			value = obs.get(key)
			if value is None:
				issues.append(f"{key} is None")
			elif hasattr(value, 'shape'):
				# It's an array - check for valid shape
				if any(d == 0 for d in value.shape):
					issues.append(f"{key} has zero dimension: {value.shape}")

		if issues:
			print(f"{prefix} WARNING - Observation issues: {', '.join(issues)}")
			return False
		else:
			# Debug: print observation shapes
			state = obs.get('observation.state')
			img = obs.get('observation.images.image')
			wrist = obs.get('observation.images.wrist_image')
			third = obs.get('observation.images.3pov_1')
			print(f"{prefix} Observation OK - state: {state.shape if state is not None else None}, "
			      f"image: {img.shape if img is not None else None}, "
			      f"wrist: {wrist.shape if wrist is not None else None}, "
			      f"3pov: {third.shape if third is not None else None}")
			return True

	def infer(self, obs: Dict) -> Dict:
		"""Send observation via gRPC and get actions back."""
		import time as time_module
		from lerobot.transport import services_pb2
		from lerobot.async_inference.helpers import TimedObservation, RawObservation
		from lerobot.transport.utils import send_bytes_in_chunks

		if self._stub is None:
			self._connect()

		# Validate observation before sending
		prefix = f"[{self.policy_type.upper()}]"
		self._validate_observation(obs, prefix=prefix)

		# Build RawObservation with individual motor values
		state_vector = obs.get('observation.state', np.zeros(7, dtype=np.float32))

		# Get images (check for None)
		image = obs.get('observation.images.image')
		wrist_image = obs.get('observation.images.wrist_image')
		third_image = obs.get('observation.images.3pov_1')

		# Debug: print image info
		print(f"{prefix} Building RawObservation:")
		print(f"  - image: {type(image).__name__}, shape={getattr(image, 'shape', 'N/A') if image is not None else 'None'}")
		print(f"  - wrist_image: {type(wrist_image).__name__}, shape={getattr(wrist_image, 'shape', 'N/A') if wrist_image is not None else 'None'}")
		print(f"  - 3pov_1: {type(third_image).__name__}, shape={getattr(third_image, 'shape', 'N/A') if third_image is not None else 'None'}")
		print(f"  - state: {state_vector}")

		raw_obs = RawObservation({
			"image": image,
			"wrist_image": wrist_image,
			"3pov_1": third_image,
			"motor_0": float(state_vector[0]),
			"motor_1": float(state_vector[1]),
			"motor_2": float(state_vector[2]),
			"motor_3": float(state_vector[3]),
			"motor_4": float(state_vector[4]),
			"motor_5": float(state_vector[5]),
			"motor_6": float(state_vector[6]),
		})

		# Wrap in TimedObservation
		timed_obs = TimedObservation(
			timestamp=time_module.time(),
			observation=raw_obs,
			timestep=self._timestep,
			must_go=True,
		)

		print(f"{prefix} Sending observation timestep={self._timestep}, must_go=True")

		# Send observation
		observation_bytes = pickle.dumps(timed_obs)
		print(f"{prefix} Serialized observation size: {len(observation_bytes)} bytes")

		observation_iterator = send_bytes_in_chunks(
			observation_bytes,
			services_pb2.Observation,
			log_prefix=f"{prefix} Observation",
			silent=True,
		)
		self._stub.SendObservations(observation_iterator)
		print(f"{prefix} Observation sent, requesting actions...")

		# Get actions
		actions_response = self._stub.GetActions(services_pb2.Empty())
		print(f"{prefix} Got response, data length: {len(actions_response.data)} bytes")

		if len(actions_response.data) == 0:
			raise RuntimeError("Server returned empty action response")

		# Deserialize actions
		timed_actions = pickle.loads(actions_response.data)

		# Extract actions as numpy array
		actions_list = []
		for ta in timed_actions:
			action = ta.get_action()
			if hasattr(action, 'cpu'):
				action = action.cpu().numpy()
			else:
				action = np.array(action)
			actions_list.append(action)

		actions = np.array(actions_list)
		self._timestep += 1

		return {'actions': actions}

	def reset(self) -> None:
		self._timestep = 0

	def close(self) -> None:
		if self._channel is not None:
			self._channel.close()
			self._channel = None
			self._stub = None


class DiffusionClient(ACTClient):
	"""gRPC client for Diffusion policy server.

	Same protocol as ACT, just different policy_type.
	Server handles observation history internally.
	"""

	def __init__(
		self,
		host: str,
		port: int,
		timeout: float = 30.0,
		pretrained_path: str = "outputs/penguin_grasp/diffusion",
		device: str = "cuda",
	):
		super().__init__(
			host=host,
			port=port,
			timeout=timeout,
			policy_type="diffusion",
			pretrained_path=pretrained_path,
			device=device,
		)


def create_policy_client(
	policy_type: PolicyType | str,
	host: str,
	port: int,
	timeout: float = 10.0,
	# gRPC policy options (ACT and Diffusion)
	pretrained_path: str = "outputs/penguin_grasp/act",
	device: str = "cuda",
) -> BasePolicyClient:
	"""Factory function to create the appropriate policy client.

	Args:
		policy_type: Type of policy (openpi, act, diffusion)
		host: Server hostname
		port: Server port
		timeout: Connection timeout in seconds
		pretrained_path: Path to pretrained model (ACT/Diffusion - relative to server cwd)
		device: Device to run on (ACT/Diffusion - server's device)

	Returns:
		BasePolicyClient instance
	"""
	if isinstance(policy_type, str):
		policy_type = PolicyType(policy_type.lower())

	if policy_type == PolicyType.OPENPI:
		return OpenPIClient(host=host, port=port, timeout=timeout)
	elif policy_type == PolicyType.ACT:
		return ACTClient(
			host=host,
			port=port,
			timeout=timeout,
			pretrained_path=pretrained_path,
			device=device,
		)
	elif policy_type == PolicyType.DIFFUSION:
		return DiffusionClient(
			host=host,
			port=port,
			timeout=timeout,
			pretrained_path=pretrained_path,
			device=device,
		)
	else:
		raise ValueError(f"Unknown policy type: {policy_type}")


# Default ports for each policy type
DEFAULT_PORTS = {
	PolicyType.OPENPI: 8000,
	PolicyType.ACT: 8080,
	PolicyType.DIFFUSION: 8080,  # Same gRPC server as ACT
}

# Default hosts for each policy type
DEFAULT_HOSTS = {
	PolicyType.OPENPI: "moraband",
	PolicyType.ACT: "coruscant",
	PolicyType.DIFFUSION: "coruscant",  # Same gRPC server as ACT
}


# =============================================================================
# Policy Host Mapping
# =============================================================================

# Policy host mapping
POLICY_HOSTS = {
	"moraband": "moraband.stanford.edu",
	"manaan": "SOE-50TJK74.stanford.edu",
	"coruscant": "coruscant.stanford.edu",
}

def resolve_policy_host(host: str) -> str:
	"""Resolve policy host shorthand to full hostname.

	Args:
		host: Either a shorthand name (e.g., "moraband", "manaan") or a full hostname

	Returns:
		Full hostname (e.g., "moraband.stanford.edu")
	"""
	# If it's a shorthand name, resolve it
	if host in POLICY_HOSTS:
		return POLICY_HOSTS[host]
	# Otherwise, assume it's already a full hostname
	return host

def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return

# Task definitions
tasks = {
	"gate_nav": "go through the gate and hover over the stuffed animal",
	"gate_nav_left": "go through the gate on the left and hover over the stuffed animal",
	"gate_nav_right": "go through the gate on the right and hover over the stuffed animal",
	"gate_nav_grasp": "go through the gate and grab the stuffed animal and put it in the blue bucket",
	"hover" : "do not move",
	"penguin_grasp": "grab the stuffed animal and put it in the blue bucket",
	"penguin_grasp_blue": "grab the stuffed animal and put it in the blue bucket",
	"penguin_grasp_purple": "grab the stuffed animal and put it in the purple bucket",
	"chip_grasp": "grab the chip bag and put it in the blue bucket",
	"box_grasp": "grab the box and put it in the blue bucket",
	"sandwich_grasp": "grab the red and white sandwich and put it in the blue bucket",
	"helicopter_grasp": "grab the stuffed animal and put it on the helicopter landing pad",
	"goomba_grasp": "grab the brown stuffed animal and put it in the blue bucket",
	"not_goomba_grasp": "grab the penguin stuffed animal and put it in the blue bucket",
}

# Log IDs that should be enabled for debugging
enabled_log_ids = [
	# "TRIM",
	"GRIP",
	"WAIT",
	#"INF",
	#"POL",
	#"RATE",
	#"PUB",
	"STATE",
	"SKIP_SUM",
	"ACT_SUM",
	# "DISC",
]


class DelayLogMixin:
	"""Mixin class that provides delayed/throttled logging functionality.

	Classes using this mixin will have logging-related instance variables initialized.
	The class using this mixin should define:
	- self.debug: bool flag for debug mode
	"""

	def __init__(self, *args, **kwargs):
		"""Initialize logging-related instance variables."""
		super().__init__(*args, **kwargs)
		self._log_counts = {}
		self._log_colors = {}
		self._dbg_colors = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

	def delay_log(self, msg: str, id: str, log_every: int = 1, debug_only: bool = True) -> bool:
		"""Log a message with throttling and color-coding.

		Args:
			msg: Message to log
			id: Unique identifier for this log type
			log_every: Log every N calls (1 = always log)
			debug_only: Only log if debug mode is enabled

		Returns:
			True if the message was printed, False otherwise
		"""
		printed = False
		if (self.debug or not debug_only) and id in enabled_log_ids:
			self._log_counts[id] = (self._log_counts.get(id, -1) + 1) % log_every
			if id not in self._log_colors:
				self._log_colors[id] = self._dbg_colors[len(self._log_counts) - 1]
			if self._log_counts[id] == 0:
				bold_id = colored(f"[{id}]", self._log_colors[id], attrs=["bold"])
				print(f"{bold_id} {msg}")
				printed = True
		return printed


class GripperControlMixin:
	"""Mixin class that provides gripper control and debugging functionality.

	Features:
	- Consensus-based opening guard to prevent rapid toggling
	- Optional debug logging and graph generation

	Classes using this mixin should define:
	- self.grip_threshold: float threshold for legacy gripper toggle
	- self.legacy_actions: bool flag for legacy action format
	- self._last_published_gripper: float current gripper state
	- self._latest_gripper: Optional[float] latest gripper sensor reading
	"""

	def __init__(self, *args, debug_gripper: bool = False, gripper_open_consensus: int = 5, **kwargs):
		"""Initialize gripper debugging.

		Args:
			debug_gripper: Enable gripper debugging (logs and graphs)
			gripper_open_consensus: Number of consecutive "open" commands needed to open a closed gripper
		"""
		super().__init__(*args, **kwargs)
		self.debug_gripper = bool(debug_gripper)
		self.gripper_open_consensus = int(gripper_open_consensus)
		self._gripper_command_history = []  # Track recent gripper commands
		self._gripper_debug_data = {
			'timestamps': [],           # Time since start
			'action_6_values': [],      # Raw action[6] from policy
			'gripper_commands': [],     # Published gripper commands (0 or 1)
			'gripper_states': [],       # Actual gripper state from sensor
		}
		self._debug_start_time = time.time()

	def compute_gripper_command(self, action: np.ndarray) -> float:
		"""Compute gripper command with consensus logic to prevent rapid toggling.

		Args:
			action: Action array from policy

		Returns:
			Gripper command (0.0 = open, 1.0 = closed)
		"""
		# Get raw action[6] value (if it exists)
		raw_action_6 = float(action[6]) if len(action) > 6 else 0.0

		# Compute desired gripper command from action
		legacy_actions = getattr(self, 'legacy_actions', False)
		if legacy_actions:
			# Legacy: gripper at action[4], continuous value with toggle on threshold
			should_toggle = (abs(action[4]) >= self.grip_threshold)
			prev_grip = float(self._last_published_gripper)
			desired_cmd = float(1.0 - prev_grip) if should_toggle else prev_grip
		else:
			# New format: gripper at action[6], binary 0/1
			desired_cmd = 1.0 if raw_action_6 > 0.5 else 0.0

		# Add to command history
		self._gripper_command_history.append(desired_cmd)
		if len(self._gripper_command_history) > self.gripper_open_consensus:
			self._gripper_command_history.pop(0)

		# Apply consensus logic: only open if gripper is closed AND we have consensus
		current_state = float(self._last_published_gripper)
		if current_state > 0.5:  # Gripper is currently closed
			# Check if we have N consecutive "open" (0.0) commands
			if len(self._gripper_command_history) >= self.gripper_open_consensus:
				if all(cmd < 0.5 for cmd in self._gripper_command_history):
					# Consensus reached: open the gripper
					return 0.0
				else:
					# No consensus yet: keep closed
					return 1.0
			else:
				# Not enough history yet: keep closed
				return 1.0
		else:
			# Gripper is currently open: allow immediate closing
			return desired_cmd

	def log_gripper_debug(self, action: np.ndarray, gripper_cmd: float):
		"""Log gripper debugging data for a single action.

		Args:
			action: Action array from policy
			gripper_cmd: The gripper command that will be published
		"""
		if not self.debug_gripper:
			return

		# Get raw action[6] value (if it exists)
		raw_action_6 = float(action[6]) if len(action) > 6 else 0.0

		# Store debug data with action number instead of timestamp
		action_number = len(self._gripper_debug_data['action_6_values'])
		self._gripper_debug_data['timestamps'].append(action_number)
		self._gripper_debug_data['action_6_values'].append(raw_action_6)
		self._gripper_debug_data['gripper_commands'].append(gripper_cmd)
		self._gripper_debug_data['gripper_states'].append(
			self._latest_gripper if self._latest_gripper is not None else 0.0
		)

	def save_gripper_debug_graph(self):
		"""Save gripper debug data to a graph file."""
		if not self.debug_gripper or len(self._gripper_debug_data['timestamps']) == 0:
			return

		try:
			import matplotlib
			matplotlib.use('Agg')  # Non-interactive backend
			import matplotlib.pyplot as plt

			timestamps = self._gripper_debug_data['timestamps']
			action_6 = self._gripper_debug_data['action_6_values']
			gripper_cmds = self._gripper_debug_data['gripper_commands']
			gripper_states = self._gripper_debug_data['gripper_states']

			# Create figure with subplots
			_, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

			# Plot action[6] values
			ax1.plot(timestamps, action_6, 'b-', linewidth=1.5, marker='o', markersize=3, label='action[6] (policy output)')
			ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='threshold (0.5)')
			ax1.set_ylabel('Action[6] Value', fontsize=11)
			ax1.set_title('Gripper Debugging Over Actions', fontsize=13, fontweight='bold')
			ax1.grid(True, alpha=0.3)
			ax1.legend(loc='upper right')
			ax1.set_ylim(-0.1, 1.1)

			# Plot gripper commands
			ax2.plot(timestamps, gripper_cmds, 'r-', linewidth=1.5, marker='s', markersize=3, label='gripper command (published)')
			ax2.set_ylabel('Gripper Command', fontsize=11)
			ax2.grid(True, alpha=0.3)
			ax2.legend(loc='upper right')
			ax2.set_ylim(-0.1, 1.1)

			# Plot gripper state
			ax3.plot(timestamps, gripper_states, 'g-', linewidth=1.5, marker='^', markersize=3, label='gripper state (sensor)')
			ax3.set_ylabel('Gripper State', fontsize=11)
			ax3.set_xlabel('Action Number', fontsize=11)
			ax3.grid(True, alpha=0.3)
			ax3.legend(loc='upper right')
			ax3.set_ylim(-0.1, 1.1)

			# Set x-axis ticks: grid every 5 actions, labels every 20 actions
			max_actions = len(timestamps)
			if max_actions > 0:
				import matplotlib.ticker as ticker
				ax3.xaxis.set_major_locator(ticker.MultipleLocator(20))  # Labels every 20
				ax3.xaxis.set_minor_locator(ticker.MultipleLocator(5))   # Grid lines every 5
				ax3.grid(which='minor', alpha=0.2, linestyle=':')         # Minor grid (every 5)
				ax3.grid(which='major', alpha=0.4, linestyle='-')         # Major grid (every 20)

			plt.tight_layout()

			# Save to file with timestamp
			filename = f"gripper_debug_{int(time.time())}.png"
			plt.savefig(filename, dpi=150, bbox_inches='tight')
			plt.close()

			print(f"Saved gripper debug graph to: {filename}")

		except Exception as e:
			print(f"Failed to save gripper debug graph: {e}")


class PolicyNodeBase(Node):
	"""Base class for policy nodes with common ROS infrastructure.
	
	Provides:
	- QoS profile setup
	- ROS publishers and subscribers for standard topics
	- Callback methods for observations
	- Policy client initialization
	- Prompt/task resolution
	- Observation storage (latest values + timestamps)
	- Staleness tracking and alerts
	
	Subclasses should implement their specific policy execution logic.
	"""
	
	def __init__(
		self,
		node_name: str,
		resize_hw: int = 255,
		task: Optional[str] = None,
		prompt: Optional[str] = None,
		grip_threshold: float = 0.15,
		policy_host: str = 'moraband',
		policy_port: int = 8000,
		staleness_threshold_s: float = 0.5,
		enable_staleness_alerts: bool = True,
		absolute_actions: bool = False,
		disable_yaw: bool = False,
		skip_client: bool = False,
		mask_third_person_camera: bool = False,
		**kwargs,
	):
		"""Initialize common infrastructure for policy nodes.

		Args:
			node_name: Name for the ROS node
			resize_hw: Image resize dimension (square)
			task: Task name for prompt resolution
			prompt: Override prompt text (takes precedence over task)
			grip_threshold: Threshold for gripper toggle
			policy_host: WebSocket policy server host (can be shorthand "moraband"/"manaan" or full hostname)
			policy_port: WebSocket policy server port
			staleness_threshold_s: Time threshold for staleness warnings
			enable_staleness_alerts: Whether to enable loud alerts for very stale streams
			absolute_actions: If True, actions are absolute positions; if False, actions are deltas
			skip_client: If True, skip creating the websocket policy client (for custom clients)
			mask_third_person_camera: If True, pass zeros instead of actual third person camera images
		"""
		super().__init__(node_name, **kwargs)

		# Parameters
		self.resize_hw = int(resize_hw)
		self.grip_threshold = float(grip_threshold)
		self._staleness_threshold_s = float(staleness_threshold_s)
		self._enable_staleness_alerts = bool(enable_staleness_alerts)
		self.absolute_actions = bool(absolute_actions)
		self.disable_yaw = bool(disable_yaw)
		self.mask_third_person_camera = bool(mask_third_person_camera)

		# QoS configuration
		self._qos = QoSProfile(
			depth=10,
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.VOLATILE,
			history=HistoryPolicy.KEEP_LAST,
		)
		
		# CV Bridge
		self.bridge = CvBridge()
		
		# Prompt/task resolution
		self.task = task
		self.prompt_override = prompt
		if self.prompt_override is not None:
			self.prompt_text = str(self.prompt_override)
		else:
			if self.task is not None and self.task in tasks:
				self.prompt_text = tasks[self.task]
			else:
				self.prompt_text = tasks.get('penguin_grasp_blue', 'grab the stuffed animal and put it in the blue bucket')
		
		# Latest observations (always available)
		self._latest_pose: Optional[np.ndarray] = None
		self._latest_gripper: Optional[float] = None
		self._latest_front_image: Optional[np.ndarray] = None
		self._latest_down_image: Optional[np.ndarray] = None
		self._latest_third_pov: Optional[np.ndarray] = None
		
		# Observation timestamps (monotonic time for staleness checking)
		self._t_pose = 0.0
		self._t_gripper = 0.0
		self._t_front = 0.0
		self._t_down = 0.0
		self._t_third = 0.0
		
		# Staleness alert tracking (for 5+ second stale streams)
		self._staleness_start_time = {}
		self._last_loud_alert_time = 0.0
		self._last_warning_time = 0.0
		
		# Freshness flags - track if observations are new since last inference
		self._new_pose = False
		self._new_gripper = False
		self._new_front = False
		self._new_down = False
		self._new_third = False
		
		# Action execution state (for gripper tracking)
		self._last_published_gripper = 0.0
		
		# ROS Publishers
		self.cmd_xyz_pub = self.create_publisher(Float32MultiArray, '/control', 10)
		self.cmd_gripper_pub = self.create_publisher(Float32, '/control_gripper', 10)
		self.camera_restart_pub = self.create_publisher(Bool, '/camera_restart_request', 10)
		
		# ROS Subscribers
		self.create_subscription(PoseStamped, '/vrpn_mocap/fmu/pose', self._pose_cb, self._qos)
		self.create_subscription(Float32, '/gripper_state', self._gripper_cb, 10)
		self.create_subscription(Image, '/hires_front_small_color', self._front_cb, 10)
		self.create_subscription(Image, '/hires_down_small_color', self._down_cb, 10)
		self.create_subscription(CompressedImage, '/camera1/image_compressed', self._third_cb, 10)

		# Policy client - resolve hostname shorthand (skip if custom client will be used)
		if not skip_client:
			resolved_host = resolve_policy_host(policy_host)
			self.client = websocket_client_policy.WebsocketClientPolicy(host=resolved_host, port=policy_port)
			self.get_logger().info(f"Policy server: {resolved_host}:{policy_port}")

		self.get_logger().info(f"{node_name} base initialized with task: {self.prompt_text}")
	
	# ----- ROS Callbacks -----
	
	def _pose_cb(self, msg: PoseStamped):
		"""Store latest pose with yaw extracted from quaternion."""
		pos = msg.pose.position
		quat = msg.pose.orientation
		try:
			from scipy.spatial.transform import Rotation as R
			yaw = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')[2]
		except Exception as e:
			self.get_logger().error(f"Quaternion conversion failed: {e}")
			return
		
		self._latest_pose = np.array([pos.x, pos.y, pos.z, yaw], dtype=np.float32)
		self._t_pose = time.monotonic()
		self._new_pose = True
	
	def _gripper_cb(self, msg: Float32):
		"""Store latest gripper state."""
		self._latest_gripper = float(msg.data)
		self._t_gripper = time.monotonic()
		self._new_gripper = True
	
	def _front_cb(self, msg: Image):
		"""Store latest front camera image."""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
			self._latest_front_image = cv_image
			self._t_front = time.monotonic()
			self._new_front = True
		except Exception as e:
			self.get_logger().error(f"Front cam callback error: {e}")
	
	def _down_cb(self, msg: Image):
		"""Store latest down camera image."""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
			self._latest_down_image = cv_image
			self._t_down = time.monotonic()
			self._new_down = True
		except Exception as e:
			self.get_logger().error(f"Down cam callback error: {e}")
	
	def _third_cb(self, msg: CompressedImage):
		"""Store latest third person camera image (compressed)."""
		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
			# Resize third person view
			cv_image = cv2.resize(cv_image, (self.resize_hw, self.resize_hw))
			self._latest_third_pov = cv_image
			self._t_third = time.monotonic()
			self._new_third = True
		except Exception as e:
			self.get_logger().error(f"Third cam callback error: {e}")
	
	# ----- Observation Management -----
	
	def _check_ready(self) -> bool:
		"""Check if all observations are available."""
		return (
			self._latest_pose is not None
			and self._latest_gripper is not None
			and self._latest_front_image is not None
			and self._latest_down_image is not None
			and self._latest_third_pov is not None
		)

	def _get_third_pov_image(self) -> np.ndarray:
		"""Get third person camera image, or zeros if masking is enabled.

		Returns:
			The actual third person image, or zeros of the same shape if
			mask_third_person_camera is True.
		"""
		if self.mask_third_person_camera:
			# Return zeros with the same shape as the resized third POV image
			return np.zeros((self.resize_hw, self.resize_hw, 3), dtype=np.uint8)
		return self._latest_third_pov

	def _check_all_fresh(self) -> bool:
		"""Check if all observations are fresh (new since last inference).
		
		Returns:
			True if all observation streams have new data, False otherwise.
		"""
		return (
			self._new_pose 
			and self._new_gripper 
			and self._new_front 
			and self._new_down 
			and self._new_third
		)
	
	def reset_freshness(self):
		"""Reset all freshness flags.
		
		Should be called after inference/action execution completes to ensure
		the next cycle waits for new observations.
		"""
		self._new_pose = False
		self._new_gripper = False
		self._new_front = False
		self._new_down = False
		self._new_third = False
	
	def _check_staleness(self) -> bool:
		"""Check for stale observations and make loud noise if stale > 5 seconds.

		Returns:
			True if any observations are stale, False otherwise.
		"""
		if not self._enable_staleness_alerts:
			return False

		now = time.monotonic()
		stale_streams = []
		very_stale_streams = []  # Streams stale for 5+ seconds

		# Check each stream
		streams_to_check = [
			('pose', self._latest_pose, self._t_pose),
			('gripper', self._latest_gripper, self._t_gripper),
			('front_cam', self._latest_front_image, self._t_front),
			('down_cam', self._latest_down_image, self._t_down),
			('third_pov', self._latest_third_pov, self._t_third),
		]

		for stream_name, latest_data, last_update_time in streams_to_check:
			if latest_data is not None:
				age = now - last_update_time
				if age > self._staleness_threshold_s:
					stale_streams.append(f"{stream_name} ({age:.2f}s old)")

					# Track when this stream first became stale
					if stream_name not in self._staleness_start_time:
						self._staleness_start_time[stream_name] = now

					# Check if continuously stale for 5+ seconds
					stale_duration = now - self._staleness_start_time[stream_name]
					if stale_duration >= 5.0:
						very_stale_streams.append(f"{stream_name} ({age:.2f}s old, stale for {stale_duration:.1f}s)")
				else:
					# Stream is fresh, reset staleness tracking
					if stream_name in self._staleness_start_time:
						del self._staleness_start_time[stream_name]

		# Regular warning once every 2 seconds
		if stale_streams and (now - self._last_warning_time) > 2.0:
			self.get_logger().warn(f"Stale observation streams: {', '.join(stale_streams)}")
			self._last_warning_time = now

		# LOUD ALERT for very stale streams (5+ seconds), with 10s cooldown
		if very_stale_streams and (now - self._last_loud_alert_time) > 10.0:
			# Terminal bell + bright red blinking warning
			alert_msg = f"\a\a\a\033[1;31;5m!!! CRITICAL: Observation streams stale for 5+ seconds: {', '.join(very_stale_streams)} !!!\033[0m"
			print(alert_msg)
			self.get_logger().error(f"CRITICAL STALENESS: {', '.join(very_stale_streams)}")

			# Request camera server restart
			restart_msg = Bool()
			restart_msg.data = True
			self.camera_restart_pub.publish(restart_msg)
			self.get_logger().warn("Sent camera restart request to control node")

			# Try to make actual system beep sound
			try:
				subprocess.Popen(['paplay', '/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga'],
				                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
			except Exception:
				try:
					# Fallback to simple beep command
					subprocess.Popen(['beep', '-f', '1000', '-l', '500'],
					                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
				except Exception:
					pass  # If no sound system available, just use terminal bell
			self._last_loud_alert_time = now

		# Return True if any streams are stale
		return len(stale_streams) > 0
	
	def _build_obs_dict(self) -> dict:
		"""Build observation dictionary from latest observations.
		
		Returns standard observation dict with resized images.
		Subclasses can override for custom observation formats.
		"""
		# Build state vector [x, y, z, yaw, gripper, 0, 0]
		state_np = np.zeros((7,), dtype=np.float32)
		if self._latest_pose is not None:
			state_np[:4] = self._latest_pose
		if self._latest_gripper is not None:
			state_np[4] = self._latest_gripper
		
		# Resize images
		front_resized = cv2.resize(self._latest_front_image, (self.resize_hw, self.resize_hw)) if self._latest_front_image is not None else None
		down_resized = cv2.resize(self._latest_down_image, (self.resize_hw, self.resize_hw)) if self._latest_down_image is not None else None
		
		obs = {
			"observation/state": state_np,
			"observation/front_1": front_resized,
			"observation/down_1": down_resized,
			"observation/3pov_1": self._latest_third_pov,  # Already resized in callback
			"prompt": self.prompt_text,
		}
		return obs
	
	# ----- Action Execution -----
	
	def publish_movement(
		self,
		state_np: np.ndarray,
		total_del_x: float,
		total_del_y: float,
		total_del_z: float,
		total_del_yaw: float,
		action: np.ndarray,
		legacy_actions: bool = False,
		action_delay: float = 0.1,
	):
		"""Unified movement publishing for both legacy and new action formats.

		Args:
			state_np: Current state [x, y, z, yaw, ...]
			total_del_x: Accumulated x delta (or absolute x position if self.absolute_actions=True)
			total_del_y: Accumulated y delta (or absolute y position if self.absolute_actions=True)
			total_del_z: Accumulated z delta (or absolute z position if self.absolute_actions=True)
			total_del_yaw: Accumulated yaw delta (or absolute yaw if self.absolute_actions=True)
			action: Single action array
			legacy_actions: If True, use legacy format (inverted deltas, gripper at [4] with threshold)
			                If False, use new format (normal deltas, gripper at [7] as binary)
			action_delay: Sleep duration after publishing
		"""
		# Check for dry run mode
		if hasattr(self, 'dry_run') and self.dry_run:
			# Dry run: don't publish, just sleep and return
			time.sleep(action_delay)
			return

		# Publish position setpoint
		cmd_msg = Float32MultiArray()
		if self.absolute_actions:
			# Absolute actions: use the values directly as absolute positions
			yaw_val = state_np[3] if self.disable_yaw else total_del_yaw
			cmd_msg.data = [
				total_del_x,  # Actually absolute x
				total_del_y,  # Actually absolute y
				total_del_z,  # Actually absolute z
				yaw_val,  # Absolute yaw (or current yaw if disabled)
			]
		elif legacy_actions:
			# Legacy: actions are inverted, so subtract deltas
			yaw_delta = 0.0 if self.disable_yaw else total_del_yaw
			cmd_msg.data = [
				state_np[0] - total_del_x,
				state_np[1] - total_del_y,
				state_np[2] - total_del_z,
				state_np[3] - yaw_delta,
			]
		else:
			# New format: actions are normal, so add deltas
			yaw_delta = 0.0 if self.disable_yaw else total_del_yaw
			cmd_msg.data = [
				state_np[0] + total_del_x,
				state_np[1] + total_del_y,
				state_np[2] + total_del_z,
				state_np[3] + yaw_delta,
			]
		
		self.cmd_xyz_pub.publish(cmd_msg)
		
		# Gripper handling
		cmd_gripper_msg = Float32()

		# Use consensus-based gripper command if mixin is available
		if hasattr(self, 'compute_gripper_command'):
			gripper_val = self.compute_gripper_command(action)
			self._last_published_gripper = gripper_val
			cmd_gripper_msg.data = gripper_val
		elif legacy_actions:
			# Legacy: gripper at action[4], continuous value with toggle on threshold
			should_toggle = (abs(action[4]) >= self.grip_threshold)
			prev_grip = float(self._last_published_gripper)
			new_grip = float(1.0 - prev_grip) if should_toggle else prev_grip
			self._last_published_gripper = new_grip
			cmd_gripper_msg.data = new_grip
		else:
			# New format: gripper at action[7], binary 0/1
			if action.shape[0] > 6:
				gripper_val = 1.0 if float(action[6]) > .5 else 0.0
				self._last_published_gripper = gripper_val
				cmd_gripper_msg.data = gripper_val
			else:
				# Fallback: no gripper action, keep previous state
				cmd_gripper_msg.data = self._last_published_gripper

		self.cmd_gripper_pub.publish(cmd_gripper_msg)
		
		# Delay between actions
		time.sleep(action_delay)

