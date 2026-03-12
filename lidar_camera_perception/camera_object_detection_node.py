#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('camera_object_detection_node')
        
        self.model = YOLO('/home/junyi/ros2_ws/yolo11n.pt') 
        self.confidence_threshold = 0.5
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(Image, '/kitti/image/color/left',self.image_callback, 10)

        self.detection_pub = self.create_publisher(Detection2DArray, '/camera/object_detections', 10)
        self.debug_pub = self.create_publisher(Image, '/camera/yolo_detections', 10)

        self.get_logger().info("Camera Object Detection Node Started...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck ,9:trafficlight, 11:stopsign
        results = self.model(cv_image, conf=self.confidence_threshold, classes=[0,1,2,3,5,7,9,11], verbose=False)
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        result = results[0]
        
        for box in result.boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self.model.names[class_id]
                 
            detection = Detection2D()
            detection.header = msg.header
            detection.bbox.center.position.x = x_center
            detection.bbox.center.position.y = y_center
            detection.bbox.size_x = width
            detection.bbox.size_y = height
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)

        annotated_frame = result.plot()
        debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()