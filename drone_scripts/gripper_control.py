#!/usr/bin/env python3

import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class GripperCommandNode(Node):
    def __init__(self):
        super().__init__('gripper_command_node')
        # Subscribe to the "gripper" topic
        # Timer to publish the current state regularly
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.subscription = self.create_subscription(
            Float32,
            '/control_gripper',
            self.gripper_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Publisher for the gripper state on "gripper_state" topic
        self.publisher = self.create_publisher(Float32, '/gripper_state', 10)

        
        self.gripper_state = 0.0
        self.prev_gripper_state = -1.0

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

        # Initialize the current state to 0.0 (before any command is received)
        self.current_state = 0.0
        self.print_freq = 100
        self.print_count = 0


    def gripper_callback(self, msg: Float32):
        # The incoming message data is the gripper command
        self.gripper_state = msg.data
        self.set_gripper()

    def set_gripper(self):
        # Map the float value to a command string.
        # if self.prev_gripper_state == self.gripper_state:
        #     return
        
        if abs(self.gripper_state - 1.0) < 0.1:
            command = 1030 # close
        elif abs(self.gripper_state - 0.0) < 0.1:
            command = 1430 # open
        elif abs(self.gripper_state + 1.0) < 0.1:
            command = 1980 # stow
        else:
            command = f"UNKNOWN:{self.gripper_state}"

        # self.get_logger().info(f"Received gripper command: {self.gripper_state} -> {command}")
        self.prev_gripper_state = self.gripper_state
        # Send the command string over the serial port.
        self.ser.write(f"{command}\n".encode())
        self.ser.flush()

    def timer_callback(self):
        # Publish the current gripper state (default is 0.0 until updated)

        # self.set_gripper()
        msg = Float32()
        msg.data = self.gripper_state
        self.publisher.publish(msg)
        self.print_count = (self.print_count + 1) % self.print_freq
        if self.print_count % self.print_freq == 0:
            self.get_logger().info(f"Published gripper state: {self.gripper_state}")

def main(args=None):
    rclpy.init(args=args)
    node = GripperCommandNode()
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
