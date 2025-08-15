# 3D LiDAR Detector

A ROS2 package for real-time 3D object detection in LiDAR point clouds using the PointPillars deep learning model from mmdet3d.

## Overview

This package implements a CUDA-accelerated 3D object detection system capable of identifying pedestrians, cars, and cyclists from LiDAR point cloud data. It consists of two main components:

- **Detection Node**: Processes point clouds and generates 3D bounding boxes
- **Projection Node**: Visualizes detections by projecting 3D boxes onto camera images

## Features

- ⚡ **High Performance**: GPU-accelerated inference using NVIDIA CUDA
- 🎯 **Multi-Class Detection**: Detects pedestrians, cars, and cyclists
- 🔄 **Real-time Processing**: Optimized for real-time robotics applications
- 📷 **Visual Feedback**: Optional image projection for visualization
- ⚙️ **Configurable**: Easy parameter tuning through YAML configuration

## System Requirements

### Hardware
- NVIDIA GPU with CUDA capability (compute capability ≥ 6.0 recommended)
- Minimum 8GB GPU memory for optimal performance

### Software Dependencies
- **ROS2 Humble Hawksbill**
- **CUDA Toolkit** (compatible with your PyTorch version)
- **PyTorch** with CUDA support
- **mmdet3d** detection framework
- **Custom message package**: [marker_array_stamped](https://github.com/ernstmv/marker_array_stamped.git)

## Installation

### 1. Install mmdet3d

The mmdet3d installation varies depending on your system configuration. You have two options:

**Option A: Automated Installation (No Warranty)**
```bash
# Use the provided script (at your own risk)
./install.sh
```

**Option B: Manual Installation (Recommended)**
Follow the official [mmdet3d installation guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) for your specific setup.

### 2. Install Dependencies

Clone and build the required message package:
```bash
cd ~/your_ros2_workspace/src
git clone git@github.com:ernstmv/marker_array_stamped.git
cd ..
colcon build --packages-select markerarraystamped
```

### 3. Install This Package

```bash
# Clone this repository
cd ~/your_ros2_workspace/src
git clone git@github.com:ernstmv/3d_lidar_detector.git

# Build the workspace
cd ~/your_ros2_workspace
colcon build --packages-select marker_array_stamped lidar_detector_pkg

# Source the workspace
source install/setup.bash
```

## Configuration

Configure the package through `config/lidar_detector_pkg.config.yaml`:

### Detection Node Parameters
```yaml
lidar_detector_node:
  ros__parameters:
    # Model Configuration
    model_path: 'path/to/your/pointpillars_model.py'
    weights_path: 'path/to/your/weights.pth'
    device: 'cuda:0'  # GPU device
    model_threshold: 0.5  # Detection confidence threshold
    
    # Topics
    pointcloud_topic: '/lidar/points'
    bounding_box_topic: '/lidar_detector/detected_bounding_boxes'
```

### Projection Node Parameters
```yaml
projection_node:
  ros__parameters:
    # Input Topics
    calib_topic: '/camera/calibration'
    image_input_topic: '/camera/image_raw'
    detections_topic: '/lidar_detector/detected_bounding_boxes'
    
    # Output Topic
    image_publisher_topic: '/lidar_detector/image'
```

### Important Notes
- **Update file paths** in the configuration to match your system
- **Ensure GPU device** matches your hardware (cuda:0, cuda:1, etc.)
- **Adjust threshold** based on your detection requirements

## Usage

### Quick Start
```bash
# Launch both detection and projection nodes
ros2 launch lidar_detector_pkg lidar_detect.launch.py
```

### Run Individual Nodes
```bash
# Detection only (lower computational load)
ros2 run lidar_detector_pkg lidar_detector_node --ros-args --params-file install/lidar_detector_pkg/share/lidar_detector_pkg/config/lidar_detector_pkg.config.yaml

# Projection only (requires detection node running)
ros2 run lidar_detector_pkg projection_node --ros-args --params-file install/lidar_detector_pkg/share/lidar_detector_pkg/config/lidar_detector_pkg.config.yaml
```

### Optional: System-wide Script Installation
```bash
# Make script executable and install globally
chmod +x launch_lidar_detector
sudo mv launch_lidar_detector /usr/local/bin/

# Now you can launch from anywhere
launch_lidar_detector
```

## Topics

### Subscribed Topics
| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/lidar/points` | `sensor_msgs/PointCloud2` | Input LiDAR point cloud |
| `/camera/calibration` | `sensor_msgs/CameraInfo` | Camera calibration parameters |
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera image |

### Published Topics
| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/lidar_detector/detected_bounding_boxes` | `marker_array_stamped/MarkerArrayStamped` | 3D bounding boxes |
| `/lidar_detector/image` | `sensor_msgs/Image` | Image with projected boxes |

## Architecture

```
┌─────────────────┐    PointCloud2     ┌──────────────────┐    MarkerArrayStamped    ┌─────────────────┐
│   LiDAR Sensor  │ ────────────────► │ Detection Node   │ ──────────────────────► │ Projection Node │
└─────────────────┘                    └──────────────────┘                          └─────────────────┘
                                              │                                              │
                                              ▼                                              ▼
                                       3D Bounding Boxes                              Annotated Image
```

## Model Information

This package uses the **PointPillars** architecture:
- **Classes**: Pedestrian, Car, Cyclist
- **Framework**: mmdet3d

## Performance Tips

- **GPU Memory**: Ensure sufficient GPU memory (8GB+ recommended)
- **Detection Only**: Disable projection node if visualization isn't needed
- **Threshold Tuning**: Adjust `model_threshold` based on precision/recall needs

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce point cloud density
- Use a GPU with more memory
- Lower the model input resolution

**mmdet3d Import Error**
- Verify mmdet3d installation
- Check PyTorch and CUDA compatibility
- Ensure all dependencies are installed

**No Detections**
- Check `model_threshold` (try lowering it)
- Verify point cloud format and coordinate system
- Ensure model weights are compatible

**Topics Not Publishing**
- Verify topic names in configuration
- Check ROS2 node connectivity with `ros2 topic list`
- Ensure all dependencies are sourced

## Development

To extend this package:
1. Fork the repository
2. Create feature branches
3. Test thoroughly with different point cloud data
4. Submit pull requests with detailed descriptions

## License

MIT

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.
