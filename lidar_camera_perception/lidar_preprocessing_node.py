#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import open3d as o3d
import numpy as np

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('lidar_preprocessing_node')

        self.subscription = self.create_subscription(PointCloud2, '/kitti/point_cloud', self.lidar_callback, 10)
        
        self.publisher_ = self.create_publisher(PointCloud2, '/lidar/non_ground_pcd', 10)
        self.ground_publisher_ = self.create_publisher(PointCloud2, '/lidar/ground_pcd', 10)
        
        self.get_logger().info("Lidar Preprocessing Node Started...")

    def lidar_callback(self, msg):
        
        point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_array = np.array(list(point_generator))
        
        if points_array.size == 0:
            return
            
        # THE BULLETPROOF SHAPE FIX
        if points_array.ndim == 1:
            if points_array.dtype.names is not None:
                points_array = np.column_stack((points_array['x'], points_array['y'], points_array['z']))
            else:
                points_array = points_array.reshape(-1, 3)
        else:
            points_array = points_array[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_array)

        # Voxel Downsampling
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.1)

        # ROI Filtering
        points = np.asarray(pcd_downsampled.points)
        x_min, x_max = -20.0, 50.0     # Forward/Back
        y_min, y_max = -10.0, 10.0     # Left/Right

        roi_mask = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max)
        )
        # LiDAR sensor is at Z = 0, and the ground is roughly Z = -1.73 meters
        roi_points = points[roi_mask]
        bottom_mask = roi_points[:, 2] < -0.5
        top_mask = roi_points[:, 2] >= -0.5
        bottom_points = roi_points[bottom_mask]
        top_points = roi_points[top_mask]

        bottom_pcd = o3d.geometry.PointCloud()
        bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)

        # RANSAC Plane Fitting
        plane_model, inliers = bottom_pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=1000)

        # Extract Non-Ground AND Ground
        # Non-Ground
        bottom_non_ground_pcd = bottom_pcd.select_by_index(inliers, invert=True)
        all_non_ground_points = np.vstack((top_points, np.asarray(bottom_non_ground_pcd.points)))
        
        # Ground
        ground_pcd = bottom_pcd.select_by_index(inliers)
        ground_points = np.asarray(ground_pcd.points)

        # Convert back to ROS PointCloud2 and Publish
        header = Header()
        header.stamp = msg.header.stamp       
        header.frame_id = msg.header.frame_id 

        # Publish Non-Ground
        non_ground_msg = pc2.create_cloud_xyz32(header, all_non_ground_points.tolist())
        self.publisher_.publish(non_ground_msg)
        
        # Publish Ground
        ground_msg = pc2.create_cloud_xyz32(header, ground_points.tolist())
        self.ground_publisher_.publish(ground_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()