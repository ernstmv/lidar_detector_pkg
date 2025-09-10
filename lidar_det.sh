#!/bin/bash

cd $HOME/ros2_ws

echo "[1/3] Building package"
colcon build --packages-select lidar_detector_pkg 
if [  $? -eq 1 ]; then
	echo "ERROR: Error building package. Please check"
	exit
else
	echo "Build done"
fi

echo
echo "[2/3] Sourcing ROS2 overlay"
source install/setup.bash

if [  $? -eq 1 ]; then
	echo "ERROR: Error sourcing environment. Please check"
	exit
else
	echo "Environment loaded"
fi

echo
echo "[3/3] Launching package"
ros2 launch lidar_detector_pkg lidar_detect.launch.py
