#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using std::placeholders::_1;

class ImageOverlayNode : public rclcpp::Node
{
public:
    ImageOverlayNode() : Node("image_overlay_node")
    {
        this->declare_parameter("show_unmatched_tracks", false);
        show_unmatched_tracks_ = this->get_parameter("show_unmatched_tracks").as_bool();

        this->declare_parameter("overlap_threshold", 0.15);
        overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();

        setup_calibration_matrices();

        yolo_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "/camera/object_detections", 10, std::bind(&ImageOverlayNode::yolo_callback, this, _1));
        
        mot_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lidar/tracked_objects", 10, std::bind(&ImageOverlayNode::mot_callback, this, _1));
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/kitti/image/color/left", 10, std::bind(&ImageOverlayNode::image_callback, this, _1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/fusion/image_overlay", 10);

        RCLCPP_INFO(this->get_logger(), "Image Overlay Node Started...");
    }

private:
    bool show_unmatched_tracks_;
    double overlap_threshold_;
    cv::Mat projection_matrix_;

    vision_msgs::msg::Detection2DArray latest_yolo_detections_;
    std::vector<visualization_msgs::msg::Marker> latest_tracks_;

    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr mot_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    void setup_calibration_matrices()
    {
        // Tr_velo_to_cam (4x4)
        cv::Mat Tr = cv::Mat::eye(4, 4, CV_64F);
        Tr.at<double>(0,0) = 7.533745e-03; Tr.at<double>(0,1) = -9.999714e-01; Tr.at<double>(0,2) = -6.166020e-04; Tr.at<double>(0,3) = -4.069766e-03;
        Tr.at<double>(1,0) = 1.480249e-02; Tr.at<double>(1,1) = 7.280733e-04;  Tr.at<double>(1,2) = -9.998902e-01; Tr.at<double>(1,3) = -7.631618e-02;
        Tr.at<double>(2,0) = 9.998621e-01; Tr.at<double>(2,1) = 7.523790e-03;  Tr.at<double>(2,2) = 1.480755e-02;  Tr.at<double>(2,3) = -2.717806e-01;

        // R_rect_00 (4x4)
        cv::Mat R_rect = cv::Mat::eye(4, 4, CV_64F);
        R_rect.at<double>(0,0) = 9.999239e-01; R_rect.at<double>(0,1) = 9.837760e-03;  R_rect.at<double>(0,2) = -7.445048e-03;
        R_rect.at<double>(1,0) = -9.869795e-03; R_rect.at<double>(1,1) = 9.999421e-01; R_rect.at<double>(1,2) = -4.278459e-03;
        R_rect.at<double>(2,0) = 7.402527e-03; R_rect.at<double>(2,1) = 4.351614e-03;  R_rect.at<double>(2,2) = 9.999631e-01;

        // P_rect_02 (3x4)
        cv::Mat P_rect = cv::Mat::zeros(3, 4, CV_64F);
        P_rect.at<double>(0,0) = 7.215377e+02; P_rect.at<double>(0,2) = 6.095593e+02; P_rect.at<double>(0,3) = 4.485728e+01;
        P_rect.at<double>(1,1) = 7.215377e+02; P_rect.at<double>(1,2) = 1.728540e+02; P_rect.at<double>(1,3) = 2.163791e-01;
        P_rect.at<double>(2,2) = 1.000000e+00; P_rect.at<double>(2,3) = 2.745884e-03;

        // Pre-compute the full projection matrix for massive speedup
        projection_matrix_ = P_rect * R_rect * Tr;
    }

    void yolo_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
        latest_yolo_detections_ = *msg;
    }

    void mot_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        latest_tracks_.clear();
        for (const auto& marker : msg->markers) {
            if (marker.type == visualization_msgs::msg::Marker::CUBE && marker.action == visualization_msgs::msg::Marker::ADD) {
                latest_tracks_.push_back(marker);
            }
        }
    }

    cv::Point2f project_3d(double x, double y, double z) {
        cv::Mat pt_3d = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
        cv::Mat pt_2d = projection_matrix_ * pt_3d;
        
        double w = pt_2d.at<double>(2, 0);
        if (w <= 0) return cv::Point2f(-1, -1);
        
        return cv::Point2f(pt_2d.at<double>(0, 0) / w, pt_2d.at<double>(1, 0) / w);
    }

    double calculate_iou(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
        cv::Rect2f intersection = boxA & boxB;
        double inter_area = intersection.area();
        double union_area = boxA.area() + boxB.area() - inter_area;
        if (union_area <= 0) return 0.0;
        return inter_area / union_area;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat img = cv_ptr->image;

        // Parse YOLO Detections
        std::vector<cv::Rect2f> yolo_boxes;
        std::vector<std::string> yolo_labels;
        for (const auto& det : latest_yolo_detections_.detections) {
            float x1 = det.bbox.center.position.x - det.bbox.size_x / 2.0;
            float y1 = det.bbox.center.position.y - det.bbox.size_y / 2.0;
            yolo_boxes.push_back(cv::Rect2f(x1, y1, det.bbox.size_x, det.bbox.size_y));
            yolo_labels.push_back(det.results[0].hypothesis.class_id);
        }

        // Sort MOT tracks by distance (furthest to closest for proper drawing occlusion)
        std::sort(latest_tracks_.begin(), latest_tracks_.end(),
            [](const visualization_msgs::msg::Marker& a, const visualization_msgs::msg::Marker& b) {
                return a.pose.position.x > b.pose.position.x;
            });

        std::vector<visualization_msgs::msg::Marker> valid_tracks;
        std::vector<cv::Rect2f> valid_mot_boxes;
        std::vector<double> distances;

        // Parse and Project Valid MOT Tracks
        for (const auto& track : latest_tracks_) {
            auto pos = track.pose.position;
            auto dim = track.scale;
            double dist = std::sqrt(pos.x*pos.x + pos.y*pos.y);

            std::vector<cv::Point2f> corners;
            double dx_vals[] = {-dim.x/2, dim.x/2};
            double dy_vals[] = {-dim.y/2, dim.y/2};
            double dz_vals[] = {-dim.z/2, dim.z/2};

            for (double dx : dx_vals) {
                for (double dy : dy_vals) {
                    for (double dz : dz_vals) {
                        cv::Point2f p = project_3d(pos.x + dx, pos.y + dy, pos.z + dz);
                        if (p.x != -1) corners.push_back(p);
                    }
                }
            }

            if (corners.size() < 4) continue;

            // Get 2D bounding box from 3D projected corners
            cv::Rect2f mot_2d_box = cv::boundingRect(corners);
            
            valid_tracks.push_back(track);
            valid_mot_boxes.push_back(mot_2d_box);
            distances.push_back(dist);
        }

        // Greedy Matching Algorithm (Fast alternative to Hungarian)
        std::map<int, int> matched_mot_indices; // mot_index -> yolo_index
        
        struct Match { int mot_idx; int yolo_idx; double iou; };
        std::vector<Match> potential_matches;

        for (size_t i = 0; i < valid_mot_boxes.size(); ++i) {
            for (size_t j = 0; j < yolo_boxes.size(); ++j) {
                double iou = calculate_iou(valid_mot_boxes[i], yolo_boxes[j]);
                if (iou > overlap_threshold_) {
                    potential_matches.push_back({(int)i, (int)j, iou});
                }
            }
        }

        // Sort matches by highest IoU first
        std::sort(potential_matches.begin(), potential_matches.end(),
            [](const Match& a, const Match& b) { return a.iou > b.iou; });

        std::vector<bool> yolo_used(yolo_boxes.size(), false);
        for (const auto& match : potential_matches) {
            if (matched_mot_indices.find(match.mot_idx) == matched_mot_indices.end() && !yolo_used[match.yolo_idx]) {
                matched_mot_indices[match.mot_idx] = match.yolo_idx;
                yolo_used[match.yolo_idx] = true;
            }
        }

        // Visualization
        for (size_t i = 0; i < valid_tracks.size(); ++i) {
            cv::Rect2f draw_box;
            std::string best_label;
            cv::Scalar color;
            int thickness;

            if (matched_mot_indices.count(i)) {
                // Match Found
                int yolo_idx = matched_mot_indices[i];
                draw_box = yolo_boxes[yolo_idx];
                best_label = yolo_labels[yolo_idx];
                color = cv::Scalar(255, 255, 0);
                thickness = 2;
            } else {
                // No Match
                if (!show_unmatched_tracks_) continue;
                draw_box = valid_mot_boxes[i];
                best_label = "Object";
                color = cv::Scalar(150, 150, 150);
                thickness = 1;
            }

            
            cv::rectangle(img, draw_box, color, thickness);
            char label_text[100];
            std::transform(best_label.begin(), best_label.end(), best_label.begin(), ::toupper);
            snprintf(label_text, sizeof(label_text), "%s | ID:%d | %.1fm", best_label.c_str(), valid_tracks[i].id / 2, distances[i]);
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            
            cv::rectangle(img, 
                cv::Point(draw_box.x, draw_box.y - text_size.height - 5),
                cv::Point(draw_box.x + text_size.width, draw_box.y),
                color, cv::FILLED);
                
            cv::putText(img, label_text, cv::Point(draw_box.x, draw_box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
        }

        image_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", img).toImageMsg());
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageOverlayNode>());
    rclcpp::shutdown();
    return 0;
}