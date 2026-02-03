#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped, TwistStamped
from pymavlink import mavutil
import threading
import time
import serial
import math

# ---------------- Config (defaults) ----------------
DEFAULTS = {
    'xy_max': 20.0,
    'z_up': 10.0,
    'z_dn': 4.0,
    'yaw_max': 0.785398,  # 45 deg/s
    'expo_xy': 0.45,
    'expo_z': 0.10,
    'expo_yaw': 0.30,
}


def expo_blend(n, e):
    # PX4-like expo: blend linear and cubic
    return (1.0 - e) * n + e * (n ** 3)


def clamp_unit(x):
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)


class CombinedRCGripperPublisher(Node):
    def __init__(self):
        super().__init__('rc_gripper_publisher')
        
        # Declare parameters with defaults
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)
        self.xy_max  = float(self.get_parameter('xy_max').value)
        self.z_up    = float(self.get_parameter('z_up').value)
        self.z_dn    = float(self.get_parameter('z_dn').value)
        self.yaw_max = float(self.get_parameter('yaw_max').value)
        self.ex_xy   = float(self.get_parameter('expo_xy').value)
        self.ex_z    = float(self.get_parameter('expo_z').value)
        self.ex_yaw  = float(self.get_parameter('expo_yaw').value)
        
        # Create publishers
        self.pub_cmd_vel = self.create_publisher(TwistStamped, '/pilot/cmd_vel_setpoint', 10)
        self.thrust_pub = self.create_publisher(Vector3Stamped, 'vehicle_thrust', 10)
        self.gripper_pub = self.create_publisher(Float32, 'gripper_state', 10)
        self.ccw_neu = True  # publish angular.z as CCW-positive (NEU convention)
        
        # Connect to MAVLink
        self.get_logger().info('Connecting to MAVLink...')
        self.mav = mavutil.mavlink_connection('udp:127.0.0.1:14551')
        self.mav.wait_heartbeat()
        self.get_logger().info('MAVLink connected!')
        
        # Setup serial connection for gripper
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
            self.ser = None
        
        # Shared state variables
        self.latest_rc = None
        self.latest_attitude = None
        self.lock = threading.Lock()
        self.running = True
        
        # Start receiver thread
        threading.Thread(target=self.receive_mavlink, daemon=True).start()
        
        # Publish at 50Hz
        self.create_timer(0.02, self.publish_all)
        
        self.get_logger().info(
            f'RC Gripper with CmdVel → xy_max={self.xy_max:.3f}, z_up={self.z_up:.3f}, z_dn={self.z_dn:.3f}, '
            f'yaw_max={self.yaw_max:.3f}, expo(xy,z,yaw)=({self.ex_xy:.2f},{self.ex_z:.2f},{self.ex_yaw:.2f})'
        )
    
    def _norm_pwm(self, raw, center=1500, deadband=50):
        """Normalize RC channel from PWM to -1.0 to 1.0"""
        if raw is None:
            return 0.0
        if abs(raw - center) < deadband:
            return 0.0
        return (raw - center) / 500.0  # 1000..2000 → -1..+1 approx (clip later)
    
    def receive_mavlink(self):
        """Receive and store MAVLink messages"""
        while self.running:
            # Non-blocking receive to check multiple message types
            msg = self.mav.recv_match(blocking=True, timeout=0.01)
            
            if msg:
                msg_type = msg.get_type()
                
                with self.lock:
                    if msg_type == 'RC_CHANNELS':
                        self.latest_rc = msg
                    
                    elif msg_type == 'ATTITUDE':
                        # Attitude for thrust calculation
                        self.latest_attitude = {
                            'roll': msg.roll,
                            'pitch': msg.pitch,
                            'yaw': msg.yaw,
                            'rollspeed': msg.rollspeed,
                            'pitchspeed': msg.pitchspeed,
                            'yawspeed': msg.yawspeed
                        }
    
    def publish_all(self):
        """Publish all state information"""
        with self.lock:
            # Publish pilot cmd_vel setpoint and gripper state
            if self.latest_rc:
                self.publish_pilot_cmd_vel(self.latest_rc)
                self.publish_gripper_state(self.latest_rc)
            
            # Publish thrust estimate
            if self.latest_rc and self.latest_attitude:
                self.publish_thrust(self.latest_rc, self.latest_attitude)
    
    def publish_pilot_cmd_vel(self, rc):
        """Publish pilot commanded velocity setpoint in NEU frame"""
        # Normalize → [-1,1]
        roll     = clamp_unit(self._norm_pwm(getattr(rc, 'chan1_raw', 1500)))  # right stick L/R
        pitch    = clamp_unit(self._norm_pwm(getattr(rc, 'chan2_raw', 1500)))  # right stick F/B
        throttle = clamp_unit(self._norm_pwm(getattr(rc, 'chan3_raw', 1500)))  # left  stick U/D
        yaw      = clamp_unit(self._norm_pwm(getattr(rc, 'chan4_raw', 1500)))  # left  stick yaw

        # Apply expo
        roll_e  = expo_blend(roll,     self.ex_xy)
        pitch_e = expo_blend(pitch,    self.ex_xy)
        thr_e   = expo_blend(throttle, self.ex_z)
        yaw_e   = expo_blend(yaw,      self.ex_yaw)

        # Map to **NEU** setpoints
        sp_vx_neu = pitch_e * self.xy_max             # forward stick → +N
        sp_vy_neu = roll_e  * self.xy_max             # right   stick → +E
        sp_vz_neu = thr_e * (self.z_up if thr_e >= 0 else self.z_dn)  # up positive
        sp_yaw_ned = yaw_e * self.yaw_max
        sp_yaw_neu = -sp_yaw_ned if self.ccw_neu else sp_yaw_ned      # flip to CCW+ in NEU

        # Publish commanded setpoint (NEU)
        sp = TwistStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = 'neu'
        sp.twist.linear.x = float(sp_vx_neu)
        sp.twist.linear.y = float(sp_vy_neu)
        sp.twist.linear.z = float(sp_vz_neu)
        sp.twist.angular.z = float(sp_yaw_neu)
        self.pub_cmd_vel.publish(sp)
    
    def publish_gripper_state(self, rc):
        """Publish gripper state based on RC channel 8"""
        switch = rc.chan8_raw
        
        # Map the RC channel value to gripper state:
        # < 1200: gripper closed (1.0)
        # 1200-1800: gripper open (0.0)
        # > 1800: gripper stow (-1.0)
        if switch < 1200:
            switch_val = 1150
            gripper_state = 1.0
        elif switch > 1800:
            gripper_state = -1.0
            switch_val = switch
        else:
            gripper_state = 0.0
            switch_val = switch
        
        # Publish the gripper state
        msg = Float32()
        msg.data = gripper_state
        self.gripper_pub.publish(msg)
        
        # Write to serial port if available
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{switch_val}\n".encode())
                self.ser.flush()
            except serial.SerialException as e:
                self.get_logger().error(f"Serial write error: {e}")
    
    def publish_thrust(self, rc, attitude):
        """Estimate and publish thrust from RC throttle and attitude"""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Normalize throttle (assuming 1000-2000 range)
        throttle_normalized = (rc.chan3_raw - 1000.0) / 1000.0
        throttle_normalized = max(0.0, min(1.0, throttle_normalized))
        
        # Estimate total thrust magnitude
        # Calibrate max_thrust for your specific drone (in Newtons)
        max_thrust = 20.0  # Adjust based on your drone's specs
        thrust_magnitude = throttle_normalized * max_thrust
        
        # Get attitude angles
        roll = attitude['roll']   # rotation around x-axis
        pitch = attitude['pitch'] # rotation around y-axis
        
        # Convert thrust from body frame to inertial frame
        # In body frame, thrust points upward (negative z in NED convention)
        # When drone tilts, thrust vector rotates with it
        
        # Thrust components in inertial frame (NED convention):
        # Positive pitch (nose up) -> negative x thrust (backward)
        # Positive roll (right wing down) -> positive y thrust (right)
        # Z component is thrust times cos(roll)*cos(pitch)
        
        msg.vector.x = -thrust_magnitude * math.sin(pitch)
        msg.vector.y = thrust_magnitude * math.sin(roll) * math.cos(pitch)
        msg.vector.z = -thrust_magnitude * math.cos(roll) * math.cos(pitch)
        
        self.thrust_pub.publish(msg)
    
    def destroy_node(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = CombinedRCGripperPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
