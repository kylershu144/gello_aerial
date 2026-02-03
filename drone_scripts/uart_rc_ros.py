#!/usr/bin/env python3

import time
import serial
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class GripperStateNode(Node):
    def __init__(self):
        super().__init__('gripper_state_node')
        self.publisher_ = self.create_publisher(Float32, 'gripper_state', 10)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Connect to MAVLink
        self.get_logger().info("Connecting to MAVLink...")
        self.master = mavutil.mavlink_connection("udpin:127.0.0.1:14551")
        self.master.wait_heartbeat()
        self.get_logger().info("MAVLink heartbeat received!")

        # Setup serial connection
        try:
            self.ser = serial.Serial(
                port='/dev/ttyHS1',
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.get_logger().info("Connected to serial port")
        except serial.SerialException as e:
            self.get_logger().error(f"Error opening serial port: {e}")
            rclpy.shutdown()

    def timer_callback(self):
        msg = self.master.recv_match(type=['RC_CHANNELS'], blocking=False)
        if msg and msg.get_type() == 'RC_CHANNELS':
            switch = msg.chan8_raw

            # Map the RC channel value to gripper state:
            # < 1200: gripper closed (1.0)
            # 1200-1800: gripper open (0.0)
            # > 1800: gripper stow (-1.0)
            if switch < 1200:
                switch = 1150
                gripper_state = 1.0
            elif switch > 1800:
                gripper_state = -1.0
            else:
                gripper_state = 0.0

            # Publish the gripper state
            out_msg = Float32()
            out_msg.data = gripper_state
            self.publisher_.publish(out_msg)

            # Optionally, write the raw switch value to the serial port
            self.ser.write(f"{switch}\n".encode())
            self.ser.flush()

def main(args=None):
    rclpy.init(args=args)
    node = GripperStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
