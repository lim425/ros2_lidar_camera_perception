#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

using std::placeholders::_1;

class LidarCameraFusionNode : public rclcpp::Node
{
public:
    LidarCameraFusionNode() : Node("lidar_camera_fusion_node")
    {
        this->declare_parameter("overlap_threshold", 0.15);
        overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();

        setup_calibration_matrices();

        yolo_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "/camera/object_detections", 10, std::bind(&LidarCameraFusionNode::yolo_callback, this, _1));
        
        mot_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lidar/tracked_objects", 10, std::bind(&LidarCameraFusionNode::mot_callback, this, _1));
        
        pub_identified_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fusion/identified_objects", 10);
        
        pub_unknown_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fusion/unknown_objects", 10);

        RCLCPP_INFO(this->get_logger(), "Lidar Camera Fusion Node Started...");
    }

private:
    double overlap_threshold_;
    cv::Mat projection_matrix_;
    vision_msgs::msg::Detection2DArray latest_yolo_detections_;
    
    std::map<int, std::string> label_memory_;
    std::set<int> active_ids_;

    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr mot_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_identified_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_unknown_;

    void setup_calibration_matrices()
    {
        cv::Mat Tr = cv::Mat::eye(4, 4, CV_64F);
        Tr.at<double>(0,0) = 7.533745e-03; Tr.at<double>(0,1) = -9.999714e-01; Tr.at<double>(0,2) = -6.166020e-04; Tr.at<double>(0,3) = -4.069766e-03;
        Tr.at<double>(1,0) = 1.480249e-02; Tr.at<double>(1,1) = 7.280733e-04;  Tr.at<double>(1,2) = -9.998902e-01; Tr.at<double>(1,3) = -7.631618e-02;
        Tr.at<double>(2,0) = 9.998621e-01; Tr.at<double>(2,1) = 7.523790e-03;  Tr.at<double>(2,2) = 1.480755e-02;  Tr.at<double>(2,3) = -2.717806e-01;

        cv::Mat R_rect = cv::Mat::eye(4, 4, CV_64F);
        R_rect.at<double>(0,0) = 9.999239e-01; R_rect.at<double>(0,1) = 9.837760e-03;  R_rect.at<double>(0,2) = -7.445048e-03;
        R_rect.at<double>(1,0) = -9.869795e-03; R_rect.at<double>(1,1) = 9.999421e-01; R_rect.at<double>(1,2) = -4.278459e-03;
        R_rect.at<double>(2,0) = 7.402527e-03; R_rect.at<double>(2,1) = 4.351614e-03;  R_rect.at<double>(2,2) = 9.999631e-01;

        cv::Mat P_rect = cv::Mat::zeros(3, 4, CV_64F);
        P_rect.at<double>(0,0) = 7.215377e+02; P_rect.at<double>(0,2) = 6.095593e+02; P_rect.at<double>(0,3) = 4.485728e+01;
        P_rect.at<double>(1,1) = 7.215377e+02; P_rect.at<double>(1,2) = 1.728540e+02; P_rect.at<double>(1,3) = 2.163791e-01;
        P_rect.at<double>(2,2) = 1.000000e+00; P_rect.at<double>(2,3) = 2.745884e-03;

        projection_matrix_ = P_rect * R_rect * Tr;
    }

    void yolo_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
        latest_yolo_detections_ = *msg;
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
        return (union_area <= 0) ? 0.0 : (inter_area / union_area);
    }

    std_msgs::msg::ColorRGBA get_semantic_color(std::string label) {
        std::transform(label.begin(), label.end(), label.begin(), ::tolower);
        std_msgs::msg::ColorRGBA color;
        color.a = 0.7;
        
        if (label == "car" || label == "truck" || label == "bus" || label == "bicycle" || label == "motorcycle") {
            color.r = 1.0; color.g = 0.5; color.b = 0.0;
        } else if (label == "person") {
            color.r = 1.0; color.g = 0.0; color.b = 0.0; 
        } else if (label == "traffic light" || label == "stop sign") {
            color.r = 1.0; color.g = 1.0; color.b = 0.0; 
        } else if (label == "unknown") {
            color.r = 0.0; color.g = 1.0; color.b = 0.0; 
        } else {
            color.r = 0.0; color.g = 1.0; color.b = 1.0;
        }
        return color;
    }

    void mot_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        visualization_msgs::msg::MarkerArray identified_array;
        visualization_msgs::msg::MarkerArray unknown_array;

        // Extract YOLO
        std::vector<cv::Rect2f> yolo_boxes;
        std::vector<std::string> yolo_labels;
        for (const auto& det : latest_yolo_detections_.detections) {
            float x = det.bbox.center.position.x - det.bbox.size_x / 2.0;
            float y = det.bbox.center.position.y - det.bbox.size_y / 2.0;
            yolo_boxes.push_back(cv::Rect2f(x, y, det.bbox.size_x, det.bbox.size_y));
            yolo_labels.push_back(det.results[0].hypothesis.class_id);
        }

        std::set<int> current_live_ids;
        std::vector<visualization_msgs::msg::Marker> live_tracks;
        std::vector<cv::Rect2f> valid_mot_boxes;
        std::vector<int> valid_mot_indices;

        std_msgs::msg::Header ref_header;
        if (!msg->markers.empty()) ref_header = msg->markers[0].header;

        // Parse LIVE Tracks
        for (const auto& m : msg->markers) {
            if (m.action == visualization_msgs::msg::Marker::ADD && m.type == visualization_msgs::msg::Marker::CUBE) {
                int real_id = m.id / 2;
                current_live_ids.insert(real_id);
                live_tracks.push_back(m);

                auto pos = m.pose.position;
                auto dim = m.scale;
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

                if (corners.size() >= 4) {
                    valid_mot_boxes.push_back(cv::boundingRect(corners));
                    valid_mot_indices.push_back(live_tracks.size() - 1);
                }
            }
        }

        // DELETION LOGIC (Cleanup stale memory)
        std::vector<int> dead_ids;
        std::set_difference(active_ids_.begin(), active_ids_.end(),
                            current_live_ids.begin(), current_live_ids.end(),
                            std::inserter(dead_ids, dead_ids.begin()));

        for (int dead_id : dead_ids) {
            visualization_msgs::msg::Marker cube_delete;
            cube_delete.header = ref_header;
            cube_delete.ns = "semantic_cubes";
            cube_delete.id = dead_id;
            cube_delete.action = visualization_msgs::msg::Marker::DELETE;

            visualization_msgs::msg::Marker text_delete;
            text_delete.header = ref_header;
            text_delete.ns = "semantic_labels";
            text_delete.id = dead_id + 10000;
            text_delete.action = visualization_msgs::msg::Marker::DELETE;

            identified_array.markers.push_back(cube_delete);
            identified_array.markers.push_back(text_delete);
            
            unknown_array.markers.push_back(cube_delete);
            unknown_array.markers.push_back(text_delete);

            label_memory_.erase(dead_id);
        }
        active_ids_ = current_live_ids;

        // GREEDY MATCHING
        std::map<int, int> matched_live_track_indices; 
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

        std::sort(potential_matches.begin(), potential_matches.end(),
            [](const Match& a, const Match& b) { return a.iou > b.iou; });

        std::vector<bool> yolo_used(yolo_boxes.size(), false);
        for (const auto& match : potential_matches) {
            int actual_track_idx = valid_mot_indices[match.mot_idx];
            if (matched_live_track_indices.find(actual_track_idx) == matched_live_track_indices.end() && !yolo_used[match.yolo_idx]) {
                matched_live_track_indices[actual_track_idx] = match.yolo_idx;
                yolo_used[match.yolo_idx] = true;
            }
        }

        // GENERATE FUSED MARKERS
        for (size_t i = 0; i < live_tracks.size(); ++i) {
            auto& track = live_tracks[i];
            int real_id = track.id / 2;

            if (matched_live_track_indices.count(i)) {
                label_memory_[real_id] = yolo_labels[matched_live_track_indices[i]];
            }

            std::string final_label = label_memory_.count(real_id) ? label_memory_[real_id] : "Unknown";
            
            // Generate ADD Markers
            visualization_msgs::msg::Marker cube = track;
            cube.ns = "semantic_cubes";
            cube.id = real_id;
            cube.color = get_semantic_color(final_label);

            visualization_msgs::msg::Marker text;
            text.header = track.header;
            text.ns = "semantic_labels";
            text.id = real_id + 10000;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;

            double px = track.pose.position.x;
            double py = track.pose.position.y;
            double pz = track.pose.position.z;
            double distance = std::sqrt(px*px + py*py + pz*pz);

            text.pose.position.x = px;
            text.pose.position.y = py;
            text.pose.position.z = pz + (track.scale.z / 2.0) + 1.0;
            text.scale.z = 0.8;
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0;
            
            std::string upper_label = final_label;
            std::transform(upper_label.begin(), upper_label.end(), upper_label.begin(), ::toupper);
            char buffer[100];
            snprintf(buffer, sizeof(buffer), "ID: %d\n%s\n%.1fm", real_id, upper_label.c_str(), distance);
            text.text = buffer;

            // Generate Cross-DELETE Markers (To prevent Ghosts)
            visualization_msgs::msg::Marker cube_del;
            cube_del.header = track.header;
            cube_del.ns = "semantic_cubes";
            cube_del.id = real_id;
            cube_del.action = visualization_msgs::msg::Marker::DELETE;

            visualization_msgs::msg::Marker text_del;
            text_del.header = track.header;
            text_del.ns = "semantic_labels";
            text_del.id = real_id + 10000;
            text_del.action = visualization_msgs::msg::Marker::DELETE;

            // Route Logic
            std::string lower_label = final_label;
            std::transform(lower_label.begin(), lower_label.end(), lower_label.begin(), ::tolower);
            
            if (lower_label == "unknown") {
                unknown_array.markers.push_back(cube);
                unknown_array.markers.push_back(text);
                identified_array.markers.push_back(cube_del);
                identified_array.markers.push_back(text_del);
            } else {
                identified_array.markers.push_back(cube);
                identified_array.markers.push_back(text);
                unknown_array.markers.push_back(cube_del);
                unknown_array.markers.push_back(text_del);
            }
        }

        if (!identified_array.markers.empty()) {
            pub_identified_->publish(identified_array);
        }
        if (!unknown_array.markers.empty()) {
            pub_unknown_->publish(unknown_array);
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarCameraFusionNode>());
    rclcpp::shutdown();
    return 0;
}