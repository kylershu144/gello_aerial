#!/bin/bash

# Usage: ./control.sh [start_position]
# Available positions: default_start, penguin_hover, left_gate, right_gate
# Default: default_start

START_POS="${1:-default_start}"

echo "Starting control with position: $START_POS"

# Start all three processes in background
python3 control.py --start-pos "$START_POS" &
PID1=$!
python3 gripper_control.py &
PID2=$!
python3 rc_gripper_toggle.py &
PID3=$!

# Kill all on Ctrl+C
trap "echo 'Caught Ctrl+C, killing processes...'; kill $PID1 $PID2 $PID3; wait; exit" SIGINT

# Wait for all
wait
