# ROS 2 LiDAR-Camera 3D Object Tracking & Fusion

This repository contains my personal learning work and exploration in 3D object detection, tracking, and sensor fusion using LiDAR and Camera data from KITTI dataset.

It documents my experiments with the KITTI dataset, object detection algorithms (YOLOv11), tracking, and fusion techniques, bridging raw point clouds and camera feeds into semantically labeled 3D bounding boxes.

It’s meant as a research and experimentation playground to understand how LiDAR and camera data can be combined for robust 3D perception. The project includes implementations in both Python and C++.



https://github.com/user-attachments/assets/8413acd3-cea0-4bd6-a0e6-e7ae9cbb8b2d



## Pipeline Overview

The system is broken down into modular ROS 2 nodes (available in both Python and C++):

1. **LiDAR Preprocessing (`lidar_preprocessing_node`)**
  * Subscribes to the raw LiDAR point cloud.
  * Performs voxel grid downsampling for performance.
  * Applies ground plane segmentation (RANSAC) to separate the road from obstacles.

2. **Lidar Cluster Detector (`lidar_cluster_detector_node`)**
  * Subscribes to non-ground LiDAR point clouds.
  * Performs spatial clustering DBSCAN (in Python) / Euclidean Cluster Extraction (in C++).
  * Applies geometric filtering to extract vehicle obstacles.
  * Outputs 3D Axis-Aligned Bounding Boxes (AABB).

3. **3D Object Tracker (`lidar_tracker_node`)**
  * Tracks moving obstacles across frames using a 3D Constant Velocity Kalman Filter.
  * Calculates dynamic dt using ROS message timestamps for robust prediction.
  * Uses Hungarian (in Python) / Greedy Nearest-Neighbor (in C++) algorithms for data association.
 
4. **Camera Object Detection (`camera_object_detection_node`)**
  * Uses deep learning (YOLO11 Nano) to detect objects in the 2D camera feed.
  * Filters for traffic-relevant COCO classes (cars, person, etc.).
  * Publishes standard ROS 2 `vision_msgs`.

5. **Image Overlay (`image_overlay_node`)**
  * Uses KITTI homogeneous calibration matrices to project 3D LiDAR tracks onto the 2D image plane.
  * Calculates Intersection over Union (IoU) to match 2D YOLO boxes with 3D projected boxes.
  * Generates a visual debugging feed (`/fusion/image_overlay`) with ID, Distance, and Class overlays.

6. **LiDAR-Camera Fusion (`lidar_camera_fusion_node`)**
  * The core fusion engine. Matches semantic YOLO labels to persistent 3D Kalman Filter tracks.
  * Implements Label Memory: Tracks remember their semantic class even if the camera temporarily loses the detection.
  * Routes semantically colored 3D markers to Foxglove for real-time visualization.

## ROS 2 Topics Summary

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/kitti/point_cloud` | `PointCloud2` | rosbag | Raw LiDAR data |
| `/kitti/image/color/left` | `Image` | rosbag | Input raw camera feed |
| `/lidar/non_ground_pcd` | `PointCloud2` | lidar_preprocessing_node | Filtered non-ground LiDAR data |
| `/lidar/ground_pcd` | `PointCloud2` | lidar_preprocessing_node | Filtered ground LiDAR data |
| `/lidar/bounding_boxes` | `MarkerArray` | lidar_cluster_detector_node | Raw 3D bounding boxes from clustering |
| `/lidar/tracked_objects` | `MarkerArray` | lidar_tracker_node | Persistent 3D bounding boxes with IDs |
| `/camera/object_detections` | `Detection2DArray` |  camera_object_detection_node | 2D bounding boxes and class scores |
| `/fusion/image_overlay` | `Image` | image_overlay_node | 2D debug image with projected 3D data |
| `/fusion/identified_objects` | `MarkerArray` | lidar_camera_fusion_node | Colored 3D boxes with known semantic classes |
| `/fusion/unknown_objects` | `MarkerArray` | lidar_camera_fusion_node | Unclassified tracked 3D objects |

##  Environment

* Ubuntu 22.04 LTS
* ROS2 Humble

##  Setup

### 1. Download & Prepare the KITTI Dataset
To run this pipeline, you need the KITTI Raw Data. This project using the `2011_09_26_drive_0009` sequence.
* Download the synced+rectified data and calibration files from the [KITTI Raw Data website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
* Use the [umtclskn/ros2_kitti_publishers](https://github.com/umtclskn/ros2_kitti_publishers) tool to play the raw dataset files as ROS 2 topics.
* (Optional but Recommended): Record the output of the publisher as a ROS 2 bag file (e.g., `kitti_dataset_0009`) so you don't have to run the publisher script every time.

### 2. Clone the Repository
Clone this package into the `src` directory of your ROS 2 workspace and build it:
```bash
cd ~/ros2_ws/src
git clone [https://github.com/YOUR_USERNAME/lidar_camera_perception.git](https://github.com/YOUR_USERNAME/lidar_camera_perception.git)
cd ~/ros2_ws
colcon build --packages-select lidar_camera_perception
source install/setup.bash
```

##  Running the system

### 1. Start the ROS Bag
Open a terminal and start playing the recorded KITTI dataset. `

```bash
cd bags/kitti_rosbag/
ros2 bag play kitti_dataset_0009/
```

### 2. Run the Perception Stack
You can either launch the entire system at once or run the pipelines individually for debugging.

IMPORTANT: The camera nodes rely on the ultralytics package (YOLO). If you installed YOLO inside a Python virtual environment (venv), you must activate that environment before launching the camera or full stack launch files!

```bash
source ~/ros2_venv_ws/venv/bin/activate
```

Option A: Run the Full Stack
Launches the LiDAR, Camera, and Fusion nodes together:
```bash
ros2 launch lidar_camera_perception perception_stack.launch.py
```

Option B: Run Individual Pipelines

Terminal 1 (LiDAR Pipeline):
```bash
ros2 launch lidar_camera_perception lidar_pipeline.launch.py
```

Terminal 2 (Camera Pipeline - Requires VENV):
```bash
ros2 launch lidar_camera_perception camera_pipeline.launch.py
```

Terminal 3 (Fusion Pipeline):
```bash
ros2 launch lidar_camera_perception fusion_pipeline.launch.py
```

### 3. Visualization
You can visualize the 3D bounding boxes, tracked IDs, and fused image overlays using either RViz2 or Foxglove Studio.

For Foxglove Studio, open a new terminal and launch the Foxglove bridge to connect your ROS 2 topics to the application:
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
Once the bridge is running, open the Foxglove Studio app, establish a WebSocket connection to ws://localhost:8765, and configure your panels for visualization.

## Learning Objectives
This project explores key concepts in robotic perception:
- LiDAR point cloud processing
- multi-object tracking
- camera object detection
- multi-sensor fusion









