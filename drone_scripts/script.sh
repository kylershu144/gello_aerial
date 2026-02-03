#!/bin/bash
ros2 run voxl_mpa_to_ros2 voxl_mpa_to_ros2_node &

# Wait for systemctl restart before running uart_rc.py
systemctl restart voxl-px4
until ros2 topic list | grep -q "/tracking_down"; do
  echo "Waiting for PX4 to finish booting..."
	sleep 1
done
