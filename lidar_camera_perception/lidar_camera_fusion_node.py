#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import ColorRGBA
import numpy as np
from scipy.optimize import linear_sum_assignment

class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')
        
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
        self.label_memory = {}   
        self.active_ids = set()  

        self.create_subscription(Detection2DArray, '/camera/object_detections', self.yolo_callback, 10)
        self.create_subscription(MarkerArray, '/lidar/tracked_objects', self.mot_callback, 10)
        
        self.pub_identified = self.create_publisher(MarkerArray, '/fusion/identified_objects', 10)
        self.pub_unknown = self.create_publisher(MarkerArray, '/fusion/unknown_objects', 10)
        
        self.get_logger().info("Lidar Camera Fusion Node Started...")

    def yolo_callback(self, msg):
        self.latest_yolo_detections = msg.detections

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

    def get_semantic_color(self, label):
        label = label.lower()
        color = ColorRGBA(a=0.7)
        if label in ['car', 'truck', 'bus', "bicycle", "motorcycle"]:
            color.r, color.g, color.b = 1.0, 0.5, 0.0
        elif label in ['person']:
            color.r, color.g, color.b = 1.0, 0.0, 0.0
        elif label in ['traffic light', 'stop sign']:
            color.r, color.g, color.b = 1.0, 1.0, 0.0
        elif label == 'unknown':
            color.r, color.g, color.b = 0.0, 1.0, 0.0
        else:
            color.r, color.g, color.b = 0.0, 1.0, 1.0
        return color
    
    def mot_callback(self, msg):
        identified_array = MarkerArray()
        unknown_array = MarkerArray()
        
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

        current_live_ids = set()
        live_tracks = []
        valid_mot_boxes = []
        valid_mot_indices = [] 
        
        ref_header = msg.markers[0].header if msg.markers else None

        for m in msg.markers:
            if m.action == Marker.ADD and m.type == Marker.CUBE:
                real_id = m.id // 2
                current_live_ids.add(real_id)
                live_tracks.append(m)
                
                pos = m.pose.position
                dim = m.scale
                corners = []
                for dx in [-dim.x/2, dim.x/2]:
                    for dy in [-dim.y/2, dim.y/2]:
                        for dz in [-dim.z/2, dim.z/2]:
                            p = self.project_3d(pos.x + dx, pos.y + dy, pos.z + dz)
                            if p: corners.append(p)
                
                if len(corners) >= 4:
                    corners = np.array(corners)
                    mot_2d_box = [np.min(corners[:,0]), np.min(corners[:,1]), np.max(corners[:,0]), np.max(corners[:,1])]
                    valid_mot_boxes.append(mot_2d_box)
                    valid_mot_indices.append(len(live_tracks) - 1)

        # DELETION LOGIC
        dead_ids = self.active_ids - current_live_ids
        for dead_id in dead_ids:
            cube_delete = Marker()
            if ref_header: cube_delete.header = ref_header
            cube_delete.ns = 'semantic_cubes'
            cube_delete.id = dead_id
            cube_delete.action = Marker.DELETE
            
            text_delete = Marker()
            if ref_header: text_delete.header = ref_header
            text_delete.ns = 'semantic_labels'
            text_delete.id = dead_id + 10000
            text_delete.action = Marker.DELETE
            
            identified_array.markers.extend([cube_delete, text_delete])
            unknown_array.markers.extend([cube_delete, text_delete])
            
            if dead_id in self.label_memory:
                del self.label_memory[dead_id]

        self.active_ids = current_live_ids

        # Hungarian Algorithm
        matched_live_track_indices = {}
        if len(valid_mot_boxes) > 0 and len(yolo_boxes) > 0:
            iou_matrix = np.zeros((len(valid_mot_boxes), len(yolo_boxes)))
            for i, m_box in enumerate(valid_mot_boxes):
                for j, y_box in enumerate(yolo_boxes):
                    iou_matrix[i, j] = self.calculate_iou(m_box, y_box)
            
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] > self.overlap_threshold:
                    actual_track_idx = valid_mot_indices[r]
                    matched_live_track_indices[actual_track_idx] = c

        # Generate Markers
        for i, track in enumerate(live_tracks):
            real_id = track.id // 2
            
            if i in matched_live_track_indices:
                new_label = yolo_labels[matched_live_track_indices[i]]
                self.label_memory[real_id] = new_label
            
            final_label = self.label_memory.get(real_id, "Unknown")
            semantic_color = self.get_semantic_color(final_label)
            
            cube = Marker()
            cube.header = track.header
            cube.ns = 'semantic_cubes'
            cube.id = real_id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = track.pose
            cube.scale = track.scale
            cube.color = semantic_color

            text = Marker()
            text.header = track.header
            text.ns = 'semantic_labels'
            text.id = real_id + 10000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            px, py, pz = track.pose.position.x, track.pose.position.y, track.pose.position.z
            distance = np.sqrt(px**2 + py**2 + pz**2)
            
            text.pose.position.x = px
            text.pose.position.y = py
            text.pose.position.z = pz + (track.scale.z / 2.0) + 1.0 
            text.scale.z = 0.8
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f"ID: {real_id}\n{final_label.upper()}\n{distance:.1f}m"

            # Cross-DELETE Markers (To prevent Ghosts)
            cube_del = Marker()
            cube_del.header = track.header
            cube_del.ns = 'semantic_cubes'
            cube_del.id = real_id
            cube_del.action = Marker.DELETE

            text_del = Marker()
            text_del.header = track.header
            text_del.ns = 'semantic_labels'
            text_del.id = real_id + 10000
            text_del.action = Marker.DELETE

            # Route logic
            if final_label.lower() == 'unknown':
                # Route to unknown, delete from identified
                unknown_array.markers.extend([cube, text])
                identified_array.markers.extend([cube_del, text_del])
            else:
                # Route to identified, delete from unknown
                identified_array.markers.extend([cube, text])
                unknown_array.markers.extend([cube_del, text_del])

        if len(identified_array.markers) > 0:
            self.pub_identified.publish(identified_array)
            
        if len(unknown_array.markers) > 0:
            self.pub_unknown.publish(unknown_array)

def main():
    rclpy.init()
    node = LidarCameraFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()