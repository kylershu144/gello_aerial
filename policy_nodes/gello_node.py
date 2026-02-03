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

from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from policy_nodes.common import PolicyNodeBase, GripperControlMixin, tasks

class DroneGelloNode(Node):
    def __init__(self, rate_hz: float = 10.0,):
        super().__init__(node_name="drone_gello_node")
        self.rate_hz = rate_hz

        # ROS Publishers: Need to send xyz, yaw, and gripper commands to drone
        self.cmd_xyz_pub = self.create_publisher(Float32MultiArray, '/control', 10)
        self.cmd_gripper_pub = self.create_publisher(Float32, '/control_gripper', 10)

        # Subscribe to GELLO position
        self.gello_xyz_sub = self.create_subscription(
            Float32MultiArray,
            '/gello_xyz',
            self.gello_xyz_cb,
            10)
        self.gello_gripper_sub = self.create_subscription(
            Float32,
            '/gello_gripper',
            self.gello_gripper_cb,
            10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_action)

    def gello_xyz_cb(self, msg: Float32MultiArray):
        self.gello_position = np.array(msg.data)

    def gello_gripper_cb(self, msg: Float32):
        self.gripper_value = msg.data

    def publish_action(self, action: np.ndarray):
        # Publish XYZ command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = action.tolist()
        self.cmd_xyz_pub.publish(cmd_msg)

        # Publish gripper command
        gripper_msg = Float32()
        gripper_msg.data = action[3]  # Assuming 4th element is gripper control
        self.cmd_gripper_pub.publish(gripper_msg)


def main(args=None):
	parser = argparse.ArgumentParser()
	# parser.add_argument('--rate_hz', type=float, default=10.0, help='Policy execution rate in Hz')

	cli_args, unknown_args = parser.parse_known_args()
	if unknown_args:
		print(f"WARNING: Unknown arguments (ignored): {unknown_args}")

	rclpy.init(args=args)
	node = DroneGelloNode() #args here

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