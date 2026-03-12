#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import open3d as o3d
import numpy as np

class ObstacleClusteringNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_detector_node')
        
        self.pcd_sub = self.create_subscription(PointCloud2, '/lidar/non_ground_pcd', self.obstacle_callback, 10)

        self.obstacles_pub = self.create_publisher(PointCloud2, '/lidar/clustered_obstacles_pcd', 10)
        self.bbox_pub = self.create_publisher(MarkerArray, '/lidar/bounding_boxes', 10)
        
        self.get_logger().info("Lidar Cluster Detector Node Started...")

    def obstacle_callback(self, msg):
        point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_array = np.array(list(point_generator))
        
        if points_array.size == 0:
            return
            
        if points_array.ndim == 1:
            if points_array.dtype.names is not None:
                points_array = np.column_stack((points_array['x'], points_array['y'], points_array['z']))
            else:
                points_array = points_array.reshape(-1, 3)
        else:
            points_array = points_array[:, :3]
        
        # Load into Open3D
        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(points_array)

        # DBSCAN Clustering 
        labels = np.array(non_ground_pcd.cluster_dbscan(eps=0.6, min_points=10, print_progress=False))
        if len(labels) == 0:
            return
            
        max_label = labels.max()
        
        # Data containers for this frame
        obstacle_points_list = []
        marker_array = MarkerArray()
        delete_all_marker = Marker()
        delete_all_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_all_marker)

        obstacle_count = 0

        # Generate and Filter 3D Bounding Boxes
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]

            if len(cluster_indices) < 30:
                continue
            
            cluster_pcd = non_ground_pcd.select_by_index(cluster_indices)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            
            extent = aabb.get_extent()
            size_x, size_y, size_z = extent[0], extent[1], extent[2]
            
            max_horizontal = max(size_x, size_y)
            min_horizontal = min(size_x, size_y)
            
            # Geometric Size Filter
            if max_horizontal > 6.0 or min_horizontal < 1.1:
                continue 
            if size_z > 3.5 or size_z < 0.8:
                continue 

            obstacle_count += 1
            obstacle_points_list.append(np.asarray(cluster_pcd.points))

            # 3D Bounding Box Marker
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = msg.header.stamp
            marker.ns = "detected_obstacle"
            marker.id = obstacle_count
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            center = aabb.get_center()
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = size_x
            marker.scale.y = size_y
            marker.scale.z = size_z
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 0.3 

            marker_array.markers.append(marker)

        self.bbox_pub.publish(marker_array)

        # Publish Obstacle Points
        if len(obstacle_points_list) > 0:
            all_obstacle_points = np.vstack(obstacle_points_list)
            header = Header()
            header.stamp = msg.header.stamp       
            header.frame_id = msg.header.frame_id 
            valid_msg = pc2.create_cloud_xyz32(header, all_obstacle_points.tolist())
            self.obstacles_pub.publish(valid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleClusteringNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()