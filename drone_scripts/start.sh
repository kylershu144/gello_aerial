#!/bin/bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2_node &

# Wait for systemctl restart before running uart_rc.py
systemctl restart voxl-px4 && python3 uart_rc_ros_combined.py 

#python3 uart_rc.py &

#sspython3 gripper_control.py