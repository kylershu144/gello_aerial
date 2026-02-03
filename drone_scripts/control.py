import rclpy
import math
import time
import numpy
import subprocess
import argparse
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus


# Predefined starting positions (NED frame: x forward, y right, z down)
# Positions from ENU-like frame are converted by negating y and z
# Yaw extracted from quaternion [x, y, z, w] using: atan2(2*(w*z + x*y), 1 - 2*(y² + z²)), then negated for NED
STARTING_POSITIONS = {
    'default_start': {
        'pos': [0.0, 0.0, -1.5],
        'yaw': 0.0
    },
    'penguin_hover': {
        'pos': [1.4, 0.35, -1.5],
        'yaw': 0.0
    },
    'left_gate': {
        # Original: pos=[0.29, 0.32, 1.5], quat=[-0.04, -0.0116, 0.355, 0.9337]
        # NED: negate y,z -> [0.29, -0.32, -1.5], yaw from quat ~0.727 rad, negated -> -0.727
        'pos': [0.19, -0.32, -1.5],
        'yaw': -0.727
    },
    'right_gate': {
        # Original: pos=[0.0, -0.505, 1.5], quat=[-0.027, -0.015, -0.480, 0.87647]
        # NED: negate y,z -> [0.0, 0.505, -1.5], yaw from quat ~-1.0 rad, negated -> 1.0
        'pos': [0.0, 0.505, -1.5],
        'yaw': 1.0
    },
}


class OffboardNode(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self, starting_position: str = 'default_start'):
        super().__init__('offboard_node')

        self.get_logger().info("Offboard Node Alive!")

        # Get starting position configuration
        if starting_position not in STARTING_POSITIONS:
            self.get_logger().warn(
                f"Unknown starting position '{starting_position}', using 'default_start'. "
                f"Available: {list(STARTING_POSITIONS.keys())}"
            )
            starting_position = 'default_start'

        start_config = STARTING_POSITIONS[starting_position]
        self.get_logger().info(
            f"Starting position: {starting_position} -> "
            f"pos={start_config['pos']}, yaw={start_config['yaw']:.3f} rad ({math.degrees(start_config['yaw']):.1f}°)"
        )

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.control_subscriber = self.create_subscription(Float32MultiArray, '/control', self.control_callback, 1)

        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback, qos_profile)

        # Subscriber for camera restart signal
        self.camera_restart_subscriber = self.create_subscription(
            Bool, '/camera_restart_request',
            self.camera_restart_callback, 10)

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()


        self.timer = self.create_timer(0.01, self.timer_callback)

        # Initialize desired position and yaw from starting position config
        self.des_pos = list(start_config['pos'])  # Make a copy
        self.des_vel = [0.0, 0.0]
        self.des_acc = [0.0, 0.0]
        self.des_yaw = start_config['yaw']

        self.estop = False
        self.print_freq = 100
        self.print_count = 0


    def vehicle_status_callback(self, msg):
        """Callback for vehicle status."""
        self.vehicle_status = msg

    def camera_restart_callback(self, msg):
        """Callback for camera restart request."""
        if msg.data:
            self.get_logger().warn("Received camera restart request - restarting voxl-camera-server...")
            try:
                subprocess.run(['systemctl', 'restart', 'voxl-camera-server'],
                             check=True,
                             timeout=10.0)
                self.get_logger().info("Successfully restarted voxl-camera-server")
            except subprocess.TimeoutExpired:
                self.get_logger().error("Camera restart timed out after 10 seconds")
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f"Failed to restart camera server: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error restarting camera: {e}")
        
    def timer_callback(self) -> None:
        """Callback function for the timer."""
        # Logging
        self.print_count = (self.print_count + 1) % self.print_freq
        if self.print_count % self.print_freq == 0:
            nav_state = "UNKNOWN"
            if self.vehicle_status.nav_state == 14:  # OFFBOARD
                nav_state = "OFFBOARD"
            elif self.vehicle_status.nav_state == 2:  # ALTCTL
                nav_state = "HOVER"
            else:
                nav_state = f"MANUAL"
            self.get_logger().info(f'pos to be sent {self.des_pos} | Nav: {nav_state}')
        
        # Always stream offboard control mode and setpoints.
        self.publish_offboard_control_mode()
        if self.vehicle_status.nav_state == 14:
            self.offboard_move_callback()
    
    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def control_callback(self, msg):
        self.des_pos[0] = msg.data[0]
        self.des_pos[1] = -msg.data[1]   # /vrpn y and z are negated to px4 messages
        self.des_pos[2] = -msg.data[2]
        self.des_pos[2] = self.des_pos[2] if self.des_pos[2] < -.79 else -.79
        self.des_pos[2] = self.des_pos[2] if self.des_pos[2] > -1.75 else -1.75
        self.des_yaw = -msg.data[3]
        # self.des_yaw = 0.0

    def offboard_move_callback(self):
        msg = TrajectorySetpoint()
        
        # In OFFBOARD: publish policy/commanded setpoint
        msg.position = self.des_pos
        msg.yaw = self.des_yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

def main(args=None) -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Offboard control node for PX4')
    parser.add_argument(
        '--start-pos', '--start_pos',
        type=str,
        default='default_start',
        choices=list(STARTING_POSITIONS.keys()),
        help=f"Starting position preset. Available: {list(STARTING_POSITIONS.keys())}"
    )
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    offboard_node = OffboardNode(starting_position=cli_args.start_pos)
    rclpy.spin(offboard_node)
    offboard_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
