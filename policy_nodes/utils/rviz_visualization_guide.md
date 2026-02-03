# RViz Visualization Guide for Action Horizon

The policy node now publishes the 50 action horizon to three RViz-compatible topics:

## Topics Published

1. **`/drone_current_position`** (visualization_msgs/Marker)
   - Shows the current drone position as a **blue arrow**
   - Arrow points in the direction the drone is facing (based on yaw)
   - Frame ID: "world"

2. **`/action_horizon_path`** (nav_msgs/Path)
   - Shows the predicted trajectory path as a line
   - Frame ID: "world"

3. **`/action_horizon_markers`** (visualization_msgs/MarkerArray)
   - Shows individual action points as colored spheres
   - **Green spheres**: Gripper open (action[4] >= 0)
   - **Red spheres**: Gripper closed (action[4] < 0)
   - Frame ID: "world"

## How to Visualize in RViz

1. Launch RViz:
   ```bash
   rviz2
   ```

2. **IMPORTANT - Set the Fixed Frame:**
   - In RViz, set the **Fixed Frame** to `world`

3. Add the Current Drone Position:
   - Click "Add" button
   - Select "By topic" tab
   - Find `/drone_current_position` → Marker
   - Click OK
   - You'll see a **blue arrow** at the drone's location

4. Add the Path visualization:
   - Click "Add" button
   - Select "By topic" tab
   - Find `/action_horizon_path` → Path
   - Click OK
   - Optionally adjust the color and line width in the Display properties

5. Add the Action Markers:
   - Click "Add" button
   - Select "By topic" tab
   - Find `/action_horizon_markers` → MarkerArray
   - Click OK

## What You'll See

- A **blue arrow** showing the current drone position and orientation
- A **line path** showing the predicted 50-step trajectory the drone will follow
- **Colored spheres** at each action point:
  - Green = gripper will be open
  - Red = gripper will be closed
- The trajectory accounts for the inverted actions (subtracting deltas as in `publish_movement`)

## Troubleshooting

If markers don't show up in RViz:

1. **Check the Fixed Frame**:
   - In RViz, make sure "Fixed Frame" is set to `world`
   - Look for the log message: `Published 50 action horizon to RViz topics (frame: world)`

2. **Check TF tree** (if needed):
   ```bash
   ros2 run tf2_tools view_frames
   ```
   - Make sure the "world" frame exists in your TF tree
   - If your system uses a different frame name, you may need to modify the code

## Disabling Visualization

To disable RViz visualization (for performance or when not needed), run the node with the `--no-rviz` flag:

```bash
python3 policy_nodes/policy_node.py --no-rviz
```

This will:
- Skip creating the RViz publishers (saves resources)
- Skip publishing visualization messages
- Log "RViz visualization disabled" at startup

## Notes

- The visualization updates each time the policy runs
- The path shows cumulative positions from the current drone state
- All visualizations use the "world" frame
- RViz visualization is enabled by default
