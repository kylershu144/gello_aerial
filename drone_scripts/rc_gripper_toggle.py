#!/usr/bin/env python3

import time
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class RCGripperToggleNode(Node):
    """
    Node that monitors RC controller button and toggles gripper state.
    Only sends commands on button press to avoid overriding the policy continuously.
    """
    def __init__(self):
        super().__init__('rc_gripper_toggle_node')
        
        # Publisher to control_gripper topic (used by gripper_control.py)
        self.publisher_ = self.create_publisher(Float32, '/control_gripper', 10)
        
        timer_period = 0.05  # 20Hz update rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Connect to MAVLink
        self.get_logger().info("Connecting to MAVLink for RC input...")
        self.master = mavutil.mavlink_connection("udpin:127.0.0.1:14551")
        self.master.wait_heartbeat()
        self.get_logger().info("MAVLink heartbeat received!")

        # State tracking for button press detection
        self.prev_switch_state = None  # Track previous switch position
        self.current_gripper_state = 0.0  # Current gripper state (0.0 = open, 1.0 = closed, -1.0 = stow)
        
        self.get_logger().info("RC Gripper Toggle Node initialized. Monitoring channel 8...")
        self.get_logger().info("Button press will toggle gripper state without overriding policy.")

    def timer_callback(self):
        # Read RC channels from MAVLink
        msg = self.master.recv_match(type=['RC_CHANNELS'], blocking=False)
        if msg and msg.get_type() == 'RC_CHANNELS':
            switch = msg.chan8_raw

            # Determine switch position state
            # < 1200: position 1 (closed)
            # 1200-1800: position 2 (open)
            # > 1800: position 3 (stow)
            if switch < 1200:
                switch_state = 'closed'
            elif switch > 1800:
                switch_state = 'stow'
            else:
                switch_state = 'open'

            # Detect state change (button toggle)
            if self.prev_switch_state is not None and switch_state != self.prev_switch_state:
                # Button state changed - send gripper command
                if switch_state == 'closed':
                    self.current_gripper_state = 1.0
                elif switch_state == 'stow':
                    self.current_gripper_state = -1.0
                else:  # open
                    self.current_gripper_state = 0.0

                # Publish the new gripper state
                out_msg = Float32()
                out_msg.data = self.current_gripper_state
                self.publisher_.publish(out_msg)
                
                self.get_logger().info(f"RC button toggled: {switch_state} -> gripper state: {self.current_gripper_state}")

            # Update previous state
            self.prev_switch_state = switch_state

def main(args=None):
    rclpy.init(args=args)
    node = RCGripperToggleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

