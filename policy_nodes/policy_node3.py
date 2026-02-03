#!/usr/bin/env python3

import rclpy
import time
import argparse
import numpy as np
import math
import threading
import sys
import signal
import os

from std_msgs.msg import Float32, Float32MultiArray
from openpi_client import image_tools

from policy_nodes.common import PolicyNodeBase, GripperControlMixin, tasks


class DronePolicyNode2(GripperControlMixin, PolicyNodeBase):
	"""Improved drone policy node with latest-only observations and freshness tracking."""
	
	def __init__(
		self,
		rate_hz: float = 10.0,
		task: str = None,
		prompt: str = None,
		grip_threshold: float = 0.1,
		resize_hw: int = 255,
		staleness_threshold_s: float = 0.5,
		action_delay: float = 0.1,
		post_action_sleep: float = 1.0,
		# Dynamic action scaling (noop-based early stop)
		use_dynamic_actions: bool = False,
		max_actions: int = 50,
		# Action magnitude scaling
		action_scale: float = 1.0,
		action_scale_minus_z: float = None,  # Scale for negative z only (overrides action_scale for z if set)
		# Action format
		legacy_actions: bool = True,
		absolute_actions: bool = False,
		disable_yaw: bool = False,
		# Policy server
		policy_host: str = 'moraband',
		policy_port: int = 8000,
		# Debugging
		debug_gripper: bool = False,
		gripper_open_consensus: int = 5,
		# Camera masking
		mask_third_person_camera: bool = False,
	):
		# Initialize base class with common infrastructure
		super().__init__(
			node_name='drone_policy_node2',
			resize_hw=resize_hw,
			task=task,
			prompt=prompt,
			grip_threshold=grip_threshold,
			policy_host=policy_host,
			policy_port=policy_port,
			staleness_threshold_s=staleness_threshold_s,
			enable_staleness_alerts=True,
			absolute_actions=absolute_actions,
			disable_yaw=disable_yaw,
			debug_gripper=debug_gripper,
			gripper_open_consensus=gripper_open_consensus,
			mask_third_person_camera=mask_third_person_camera,
		)
		
		# Node-specific parameters
		self.rate_hz = float(rate_hz)
		self.action_delay = float(action_delay)
		self.post_action_sleep = float(post_action_sleep)
		self.legacy_actions = bool(legacy_actions)
		
		# Dynamic action scaling parameters
		self.use_dynamic_actions = bool(use_dynamic_actions)
		self.max_actions = int(max_actions)

		# Action magnitude scaling
		self.action_scale = float(action_scale)
		self.action_scale_minus_z = float(action_scale_minus_z) if action_scale_minus_z is not None else None

		# Noop detection parameters for early reinference
		self.noop_threshold = 0.01  # 5mm movement = noop
		self.noop_window_size = 10  # Size of rolling window
		self.noop_density_threshold = 0.8  # If 80% of window are noops, reinfer

		# Action execution threading
		self._action_lock = threading.Lock()
		self._is_executing_actions = False
		self._action_thread = None
		self._shutdown_requested = False  # For clean Ctrl+C shutdown

		# Timer for policy execution
		self.timer = self.create_timer(1.0 / self.rate_hz, self._run_policy)
		
		if self.use_dynamic_actions:
			self.get_logger().info(
				f"DronePolicyNode2 initialized at {self.rate_hz} Hz "
				f"(DYNAMIC mode: noop early stop enabled, window={self.noop_window_size})"
			)
		else:
			self.get_logger().info(f"DronePolicyNode2 initialized at {self.rate_hz} Hz")

		if self.action_scale != 1.0:
			self.get_logger().info(f"Action scaling enabled: {self.action_scale}x")
		if self.action_scale_minus_z is not None:
			self.get_logger().info(f"Action scaling for -Z enabled: {self.action_scale_minus_z}x (overrides action_scale for negative z)")

		if self.debug_gripper:
			self.get_logger().info("Gripper debugging ENABLED - will log and save graphs")
	
	# ----- Observation Building (Override) -----
	def _build_obs(self) -> dict:
		"""Build observation dictionary from latest observations."""
		state_np = np.zeros((7,), dtype=np.float32)
		state_np[:4] = self._latest_pose
		state_np[6] = self._latest_gripper
		
		obs = {
			"observation/state": state_np,
			"observation/image": image_tools.convert_to_uint8(
				image_tools.resize_with_pad(self._latest_front_image, self.resize_hw, self.resize_hw)
			),
			"observation/wrist_image": image_tools.convert_to_uint8(
				image_tools.resize_with_pad(self._latest_down_image, self.resize_hw, self.resize_hw)
			),
			"observation/3pov_1": self._get_third_pov_image(),
			"prompt": self.prompt_text,
		}
		return obs, state_np

	# ----- Action Execution (runs in separate thread) -----
	def _execute_actions_thread(self, actions: np.ndarray, state_np: np.ndarray, num_actions_to_execute: int):
		"""Execute actions in a separate thread with rolling window noop detection."""
		try:
			# Rolling window of recent movements
			movement_window = []
			total_del_x, total_del_y, total_del_z, total_del_yaw = 0.0, 0.0, 0.0, 0.0
			actions_executed = 0
			early_reinfer = False

			for i in range(0, num_actions_to_execute):
				# Check for shutdown request
				if self._shutdown_requested:
					print("Shutdown requested, stopping action execution")
					break

				# Get pose before action
				pose_before = self._latest_pose.copy() if self._latest_pose is not None else None

				# Execute action
				action = actions[i]

				try:
					if self.absolute_actions:
						# For absolute actions: use values directly as positions
						abs_x, abs_y, abs_z, abs_yaw = float(action[0]), float(action[1]), float(action[2]), float(action[3])
						self.publish_movement(state_np, abs_x, abs_y, abs_z, abs_yaw, action,
						                      legacy_actions=False, action_delay=self.action_delay)
					else:
						# For delta actions: accumulate deltas (scaled by action_scale)
						del_x = float(action[0]) * self.action_scale
						del_y = float(action[1]) * self.action_scale
						raw_del_z = float(action[2])
						# Use action_scale_minus_z for negative z if set, otherwise use action_scale
						if raw_del_z < 0 and self.action_scale_minus_z is not None:
							del_z = raw_del_z * self.action_scale_minus_z
						else:
							del_z = raw_del_z * self.action_scale
						del_yaw = float(action[3]) * self.action_scale
						total_del_x += del_x
						total_del_y += del_y
						total_del_z += del_z
						total_del_yaw += del_yaw
						self.publish_movement(state_np, total_del_x, total_del_y, total_del_z, total_del_yaw, action,
						                      legacy_actions=self.legacy_actions, action_delay=self.action_delay)

					# Log gripper debug data after publishing (so we log the actual published command)
					self.log_gripper_debug(action, self._last_published_gripper)

					actions_executed = i + 1
				except Exception as e:
					# Handle case where node is being destroyed during publish
					if "InvalidHandle" in str(type(e)) or "Destroyable" in str(e):
						print("Node shutting down during action execution")
						break
					else:
						raise

				# Get pose after action (after sleep)
				pose_after = self._latest_pose.copy() if self._latest_pose is not None else None

				# Compute actual movement from pose delta
				if pose_before is not None and pose_after is not None:
					actual_delta = pose_after[:3] - pose_before[:3]  # x, y, z only
					actual_movement = np.linalg.norm(actual_delta)

					# Add to rolling window
					movement_window.append(actual_movement)
					if len(movement_window) > self.noop_window_size:
						movement_window.pop(0)  # Remove oldest

					# Check for noop window (only if window is full and dynamic actions enabled)
					if self.use_dynamic_actions and len(movement_window) >= self.noop_window_size:
						noop_count = sum(1 for m in movement_window if m < self.noop_threshold)
						noop_density = noop_count / len(movement_window)

						if noop_density >= self.noop_density_threshold:
							early_reinfer = True
							self.get_logger().info(
								f"Early reinfer: {noop_count}/{len(movement_window)} noops in window "
								f"at action {actions_executed}/{num_actions_to_execute} "
								f"(window: {[f'{m*1000:.1f}' for m in movement_window]}mm)"
							)
							break

			# Logging (skip if shutting down)
			if not self._shutdown_requested:
				if self.use_dynamic_actions:
					avg_movement = np.mean(movement_window) if movement_window else 0.0
					status = "EARLY_REINFER" if early_reinfer else "COMPLETED"
					self.get_logger().info(
						f"[{status}] Executed: {actions_executed}/{num_actions_to_execute} | "
						f"Avg movement: {avg_movement*1000:.2f}mm | "
						f"Gripper: {self._last_published_gripper}"
					)
				else:
					if self.absolute_actions:
						self.get_logger().info(
							f"Final absolute position: x={abs_x:.3f}, y={abs_y:.3f}, z={abs_z:.3f}, "
							f"yaw={abs_yaw:.3f}, gripper={self._last_published_gripper}"
						)
					else:
						self.get_logger().info(
							f"Applied deltas: x={-total_del_x:.3f}, y={-total_del_y:.3f}, z={-total_del_z:.3f}, "
							f"yaw={-total_del_yaw:.3f}, gripper={self._last_published_gripper}"
						)

			# Sleep after all actions are done (skip if shutting down)
			if not self._shutdown_requested:
				time.sleep(self.post_action_sleep)

		finally:
			# Mark action execution as complete
			with self._action_lock:
				self._is_executing_actions = False
			# Don't log during shutdown
			if not self._shutdown_requested:
				self.get_logger().debug("Action execution thread completed")

	# ----- Policy Execution -----
	def _run_policy(self):
		"""Run policy inference and execute actions (non-blocking)."""
		# Check for shutdown request
		if self._shutdown_requested:
			return

		# Check if actions are currently being executed
		with self._action_lock:
			if self._is_executing_actions:
				self.get_logger().debug("Skipping policy cycle - previous actions still executing")
				return

		# Check if all observations are available
		if not self._check_ready():
			missing = []
			if self._latest_pose is None:
				missing.append('pose')
			if self._latest_gripper is None:
				missing.append('gripper')
			if self._latest_front_image is None:
				missing.append('front_cam')
			if self._latest_down_image is None:
				missing.append('down_cam')
			if self._latest_third_pov is None:
				missing.append('third_pov')
			self.get_logger().info(f"Waiting for observations: {', '.join(missing)}")
			return

		# Check for stale observations - skip if stale
		is_stale = self._check_staleness()
		if is_stale:
			self.get_logger().info("Skipping inference due to stale observations")
			return

		# Build observation
		obs, state_np = self._build_obs()

		# Run inference
		t0 = time.time()
		try:
			result = self.client.infer(obs)
			actions = np.array(result.get('actions', []), dtype=np.float32)
		except Exception as e:
			self.get_logger().error(f"Policy inference error: {e}")
			return
		dt = time.time() - t0

		# Always execute max actions, relying on early stop for adaptive behavior
		num_actions_to_execute = self.max_actions
		self.get_logger().info(
			f"Inference: {dt*1000:.1f}ms | Actions shape: {actions.shape} | "
			f"Attempting: {num_actions_to_execute}"
		)

		# Mark that we're starting action execution
		with self._action_lock:
			self._is_executing_actions = True

		# Start action execution in a separate thread (non-blocking)
		self._action_thread = threading.Thread(
			target=self._execute_actions_thread,
			args=(actions, state_np, num_actions_to_execute),
			daemon=True
		)
		self._action_thread.start()
		self.get_logger().debug("Started action execution thread")


def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--rate_hz', type=float, default=10.0, help='Policy execution rate in Hz')
	parser.add_argument('--task', type=str, default=None, choices=list(tasks.keys()), help='Task name')
	parser.add_argument('--prompt', type=str, default=None, help='Custom prompt (overrides task)')
	parser.add_argument('--grip_threshold', type=float, default=0.05, help='Gripper toggle threshold')
	parser.add_argument('--resize', type=int, default=256, help='Image resize dimension')
	parser.add_argument('--staleness_threshold', type=float, default=0.5, help='Warn if observation older than this (seconds)')
	parser.add_argument('--action_delay', type=float, default=0.09, help='Delay between individual actions (seconds)')
	parser.add_argument('--post_action_sleep', type=float, default=1.0, help='Sleep after all actions are done (seconds)')
	parser.add_argument('--use_dynamic_actions', action='store_true', help='Enable noop-based early stop')
	parser.add_argument('--max_actions', type=int, default=50, help='Maximum actions to execute per inference')
	parser.add_argument('--action_scale', type=float, default=1.0, help='Scale factor for action magnitudes (e.g., 1.5 = 50% larger movements)')
	parser.add_argument('--action_scale_minus_z', type=float, default=None, help='Scale factor for negative z actions only (overrides action_scale for z when z < 0)')
	parser.add_argument('--legacy_actions', action='store_true', default=False, help='Use legacy action format (inverted deltas, gripper at [4])')
	parser.add_argument('--absolute_actions', action='store_true', default=False, help='Use absolute actions instead of delta actions')
	parser.add_argument('--policy_host', type=str, default='moraband', help='Policy server host (shorthand: "moraband" or "manaan", or full hostname)')
	parser.add_argument('--policy_port', type=int, default=8000, help='Policy server port')
	parser.add_argument('--debug_gripper', action='store_true', default=False, help='Enable gripper debugging (logs and graphs)')
	parser.add_argument('--gripper_open_consensus', type=int, default=5, help='Number of consecutive "open" commands needed to open a closed gripper')
	parser.add_argument('--disable_yaw', action='store_true', default=False, help='Disable yaw control (keep yaw constant)')
	parser.add_argument('--mask_third_person_camera', action='store_true', default=False, help='Pass zeros instead of third person camera images')
	cli_args, unknown_args = parser.parse_known_args()
	if unknown_args:
		print(f"WARNING: Unknown arguments (ignored): {unknown_args}")

	rclpy.init(args=args)
	node = DronePolicyNode2(
		rate_hz=cli_args.rate_hz,
		task=cli_args.task,
		prompt=cli_args.prompt,
		grip_threshold=cli_args.grip_threshold,
		resize_hw=cli_args.resize,
		staleness_threshold_s=cli_args.staleness_threshold,
		action_delay=cli_args.action_delay,
		post_action_sleep=cli_args.post_action_sleep,
		use_dynamic_actions=cli_args.use_dynamic_actions,
		max_actions=cli_args.max_actions,
		action_scale=cli_args.action_scale,
		action_scale_minus_z=cli_args.action_scale_minus_z,
		legacy_actions=cli_args.legacy_actions,
		absolute_actions=cli_args.absolute_actions,
		disable_yaw=cli_args.disable_yaw,
		policy_host=cli_args.policy_host,
		policy_port=cli_args.policy_port,
		debug_gripper=cli_args.debug_gripper,
		gripper_open_consensus=cli_args.gripper_open_consensus,
		mask_third_person_camera=cli_args.mask_third_person_camera,
	)

	def signal_handler(_sig, _frame):
		print("\nShutdown requested (Ctrl+C)...")
		node._shutdown_requested = True
		if node.debug_gripper:
			node.save_gripper_debug_graph()
		try:
			node.destroy_node()
			if rclpy.ok():
				rclpy.shutdown()
		except Exception:
			pass
		os._exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	rclpy.spin(node)


if __name__ == '__main__':
	main()