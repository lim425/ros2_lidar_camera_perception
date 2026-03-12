#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class ImageOverlayNode(Node):
    def __init__(self):
        super().__init__('image_overlay_node')
        self.bridge = CvBridge()
        
        self.declare_parameter('show_unmatched_tracks', False)
        self.show_unmatched_tracks = self.get_parameter('show_unmatched_tracks').get_parameter_value().bool_value

        self.declare_parameter('overlap_threshold', 0.15)
        self.overlap_threshold = self.get_parameter('overlap_threshold').get_parameter_value().double_value

        # KITTI Calibration (lidar and camera)
        self.Tr_velo_to_cam = np.eye(4)
        self.Tr_velo_to_cam[:3, :3] = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02]).reshape(3, 3)
        self.Tr_velo_to_cam[:3, 3] = [-4.069766e-03, -7.631618e-02, -2.717806e-01]
        self.R_rect_00 = np.eye(4)
        self.R_rect_00[:3, :3] = np.array([9.999239e-01, 9.837760e-03, -7.445048e-03, -9.869795e-03, 9.999421e-01, -4.278459e-03, 7.402527e-03, 4.351614e-03, 9.999631e-01]).reshape(3, 3)
        self.P_rect_02 = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]).reshape(3, 4)

        self.latest_yolo_detections = []
        self.latest_tracks = []

        self.create_subscription(Detection2DArray, '/camera/object_detections', self.yolo_callback, 10)
        self.create_subscription(MarkerArray, '/lidar/tracked_objects', self.mot_callback, 10)
        self.create_subscription(Image, '/kitti/image/color/left', self.image_callback, 10)

        self.pub = self.create_publisher(Image, '/fusion/image_overlay', 10)

        self.get_logger().info("Image Overlay Node Started...")

    def yolo_callback(self, msg):
        self.latest_yolo_detections = msg.detections

    def mot_callback(self, msg):
        self.latest_tracks = [m for m in msg.markers if m.type == Marker.CUBE and m.action == Marker.ADD]

    def project_3d(self, x, y, z):
        p = self.P_rect_02 @ self.R_rect_00 @ self.Tr_velo_to_cam @ np.array([x, y, z, 1.0])
        if p[2] <= 0: return None
        return [int(p[0]/p[2]), int(p[1]/p[2])]

    def calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter_area / float(areaA + areaB - inter_area + 1e-6)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Parse YOLO Detections
        yolo_boxes = []
        yolo_labels = []
        for det in self.latest_yolo_detections:
            y_box = [
                det.bbox.center.position.x - det.bbox.size_x/2,
                det.bbox.center.position.y - det.bbox.size_y/2,
                det.bbox.center.position.x + det.bbox.size_x/2,
                det.bbox.center.position.y + det.bbox.size_y/2
            ]
            yolo_boxes.append(y_box)
            yolo_labels.append(det.results[0].hypothesis.class_id)

        # Parse and Project Valid MOT Tracks
        # Sort tracks from furthest to closest for correct visual occlusion drawing
        self.latest_tracks.sort(key=lambda m: m.pose.position.x, reverse=True)
        
        valid_tracks = []
        valid_mot_boxes = []
        distances = []

        for track in self.latest_tracks:
            pos = track.pose.position
            dim = track.scale
            dist = float(np.sqrt(pos.x**2 + pos.y**2 + pos.z**2))
            
            corners = []
            for dx in [-dim.x/2, dim.x/2]:
                for dy in [-dim.y/2, dim.y/2]:
                    for dz in [-dim.z/2, dim.z/2]:
                        p = self.project_3d(pos.x + dx, pos.y + dy, pos.z + dz)
                        if p: corners.append(p)
            
            if len(corners) < 4: continue
            corners = np.array(corners)
            mot_2d_box = [np.min(corners[:,0]), np.min(corners[:,1]), np.max(corners[:,0]), np.max(corners[:,1])]
            
            valid_tracks.append(track)
            valid_mot_boxes.append(mot_2d_box)
            distances.append(dist)

        # Hungarian Algorithm (Global Optimal Matching)
        matched_mot_indices = {}
        
        if len(valid_mot_boxes) > 0 and len(yolo_boxes) > 0:
            # Create Cost Matrix (1 - IoU)
            iou_matrix = np.zeros((len(valid_mot_boxes), len(yolo_boxes)))
            for i, m_box in enumerate(valid_mot_boxes):
                for j, y_box in enumerate(yolo_boxes):
                    iou_matrix[i, j] = self.calculate_iou(m_box, y_box)
            
            # Linear Sum Assignment minimizes cost
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter matches by overlap threshold
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > self.overlap_threshold:
                    matched_mot_indices[r] = c

        # Visualization
        for i, track in enumerate(valid_tracks):
            mot_2d_box = valid_mot_boxes[i]
            dist = distances[i]
            
            if i in matched_mot_indices:
                # Match found
                yolo_idx = matched_mot_indices[i]
                draw_box = yolo_boxes[yolo_idx]
                best_label = yolo_labels[yolo_idx]
                color = (255, 255, 0)
                thickness = 2
            else:
                # No match
                if not self.show_unmatched_tracks:
                    continue

                draw_box = mot_2d_box
                best_label = "Unknown"
                color = (150, 150, 150)
                thickness = 1
            
            cv2.rectangle(img, (int(draw_box[0]), int(draw_box[1])), (int(draw_box[2]), int(draw_box[3])), color, thickness)
            
            label_text = f"{best_label.upper()} | ID:{track.id//2} | {dist:.1f}m"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(draw_box[0]), int(draw_box[1]) - h - 5), (int(draw_box[0]) + w, int(draw_box[1])), color, -1)
            cv2.putText(img, label_text, (int(draw_box[0]), int(draw_box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        self.pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

def main():
    rclpy.init()
    node = ImageOverlayNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()