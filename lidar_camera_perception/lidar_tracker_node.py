#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, detection, dimensions, track_id):
        self.track_id = track_id
        self.age = 1              
        self.hits = 1            
        self.time_since_update = 0 
        self.size = dimensions
        
        # Initialize a Constant Velocity Kalman Filter (6 states: x, y, z, vx, vy, vz)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # Initial State: Strict column vector with double brackets
        self.kf.x = np.array([[detection[0], detection[1], detection[2], 0.0, 0.0, 0.0]]).T
        
        # Initialize F with a dummy dt (will be updated in predict)
        self.kf.F = np.eye(6)
        
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])
        # P (initial covariance)
        # Large P:Trust measurements more. Small P: Trust prediction more
        self.kf.P *= 10.0 
        # R (measurement noise)
        # If LiDAR is noisy: Increase R. If LiDAR is very accurate: Lower R.
        self.kf.R *= 0.5  
        # Q (process noise)
        # Large Q: Allows rapid velocity change More responsive, Small Q: Smoother motion But slower to react
        self.kf.Q *= 0.1  

    def predict(self, dt):
        # Update State Transition Matrix: Position = Position + (Velocity * dt)
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:3].flatten()

    def update(self, detection, dimensions):
        self.kf.update(detection)
        self.size = dimensions
        self.time_since_update = 0 
        self.hits += 1

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('lidar_tracker_node')
        
        self.subscription = self.create_subscription(
            MarkerArray, '/lidar/bounding_boxes', self.detection_callback, 10)
        
        self.tracked_pub = self.create_publisher(MarkerArray, '/lidar/tracked_objects', 10)
        
        self.tracks = []
        self.next_track_id = 0
        
        # TUNED PARAMETERS
        # How long the Kalman Filter will keep predicting a car's movement if the LiDAR loses sight of it
        # Tune it DOWN if: You see "ghost" boxes continuing to drive straight through walls after a car has turned a corner
        # Tune it UP if: A car's ID number changes every time a pedestrian walks in front of it blocking the LiDAR
        self.declare_parameter('max_age', 3)
        self.max_age = self.get_parameter('max_age').get_parameter_value().integer_value

        # The maximum distance a car is allowed to move between two frames
        # Tune it DOWN if: ID numbers are randomly swapping between two cars driving close to each other
        # Tune it UP if: Fast-moving cars are constantly getting new ID numbers (the tracker thinks the fast car is a completely new object because it moved past the threshold)
        self.declare_parameter('distance_threshold', 4.0)
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        
        # DYNAMIC DT VARIABLE
        self.last_timestamp = None 
        
        self.get_logger().info("Lidar Tracker Node Started...")

    def detection_callback(self, msg):
        # 1. Safely extract the original sensor timestamp
        current_stamp = None
        current_frame = "map"
        
        for marker in msg.markers:
            if marker.action == Marker.ADD:
                current_stamp = marker.header.stamp
                current_frame = marker.header.frame_id
                break 

        if current_stamp is None:
            return

        # CALCULATE DYNAMIC DT
        current_time_float = current_stamp.sec + (current_stamp.nanosec * 1e-9)
        
        if self.last_timestamp is None:
            dt = 0.1 # Default for the very first frame
        else:
            dt = current_time_float - self.last_timestamp
            
        if dt <= 0 or dt > 2.0:
            dt = 0.1
            
        self.last_timestamp = current_time_float

        # Extract raw detections
        detections = []
        dimensions = []
        for marker in msg.markers:
            if marker.action == Marker.ADD:
                detections.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                dimensions.append([marker.scale.x, marker.scale.y, marker.scale.z])
                
        detections = np.array(detections) if len(detections) > 0 else np.array([])

        # Predict with Dynamic dt
        predictions = np.array([track.predict(dt) for track in self.tracks])

        # Data Association (Hungarian Algorithm)
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Only run matching if we have BOTH tracks and new detections
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, trk_pred in enumerate(predictions):
                for d, det in enumerate(detections):
                    cost_matrix[t, d] = np.linalg.norm(trk_pred - det)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.distance_threshold:
                    matched_indices.append((r, c))
                    if r in unmatched_tracks: unmatched_tracks.remove(r)
                    if c in unmatched_detections: unmatched_detections.remove(c)

        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx], dimensions[det_idx])

        # Create new tracks
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx], dimensions[det_idx], self.next_track_id)
            self.tracks.append(new_track)
            self.next_track_id += 1

        # Delete dead tracks (ALWAYS do this)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Publish to Foxglove/RViz (ALWAYS do this)
        self.publish_markers(current_frame, current_stamp)

    def publish_markers(self, frame_id, stamp):
        marker_array = MarkerArray()
        
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # Only publish if it has been successfully matched
        valid_tracks = [t for t in self.tracks if t.hits >= 3]

        for track in valid_tracks:
            # FLATTEN the (6,1) column vector into a standard 1D array of 6 numbers
            state = track.kf.x.flatten()
            x, y, z, vx, vy, vz = state
            
            # Bounding Box
            box_marker = Marker()
            box_marker.header.frame_id = frame_id
            box_marker.header.stamp = stamp
            box_marker.ns = "tracked_boxes"
            box_marker.id = track.track_id * 2
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = float(x)
            box_marker.pose.position.y = float(y)
            box_marker.pose.position.z = float(z)
            box_marker.scale.x, box_marker.scale.y, box_marker.scale.z = track.size
            box_marker.color.r, box_marker.color.g, box_marker.color.a = 0.0, 1.0, 0.5
            marker_array.markers.append(box_marker)

            # Text
            text_marker = Marker()
            text_marker.header = box_marker.header
            text_marker.ns = "tracked_text"
            text_marker.id = (track.track_id * 2) + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.pose.position.x = float(x)
            text_marker.pose.position.y = float(y)
            text_marker.pose.position.z = float(z) + (track.size[2]/2) + 0.5
            text_marker.scale.z = 0.6
            text_marker.color.r = text_marker.color.g = text_marker.color.b = text_marker.color.a = 1.0
            
            text_marker.text = f"ID: {track.track_id}"
            marker_array.markers.append(text_marker)

        self.tracked_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()