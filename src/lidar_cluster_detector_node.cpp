#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>

class ObstacleClusteringNode : public rclcpp::Node
{
public:
    ObstacleClusteringNode() : Node("lidar_cluster_detector_node")
    {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/non_ground_pcd", 10,
            std::bind(&ObstacleClusteringNode::callback, this, std::placeholders::_1));

        obstacles_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/lidar/clustered_obstacles_pcd", 10);

        bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/lidar/bounding_boxes", 10);

        RCLCPP_INFO(this->get_logger(), "Lidar Cluster Detector Node Started...");
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
            return;

        // Create a KdTree for fast spatial searching
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        // Euclidean Clustering
        std::vector<pcl::PointIndices> cluster_indices;

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.6);   // eps = 0.6m
        ec.setMinClusterSize(30);      
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        if (cluster_indices.empty())
            return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr valid_obstacles(
            new pcl::PointCloud<pcl::PointXYZ>());

        visualization_msgs::msg::MarkerArray marker_array;

        // DELETE ALL previous markers
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        int obstacle_id = 0;

        // Process Each Cluster
        for (const auto& indices : cluster_indices)
        {
            if (indices.indices.size() < 30)
                continue;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(
                new pcl::PointCloud<pcl::PointXYZ>());

            for (int idx : indices.indices)
                cluster->points.push_back(cloud->points[idx]);

            // Compute AABB
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            float size_x = max_pt.x - min_pt.x;
            float size_y = max_pt.y - min_pt.y;
            float size_z = max_pt.z - min_pt.z;

            float max_horizontal = std::max(size_x, size_y);
            float min_horizontal = std::min(size_x, size_y);

            // Geometric Filtering
            if (max_horizontal > 6.0 || min_horizontal < 1.1)
                continue;

            if (size_z > 3.5 || size_z < 0.8)
                continue;

            obstacle_id++;

            // Add to obstacle cloud
            *valid_obstacles += *cluster;

            // Create Bounding Box Marker
            visualization_msgs::msg::Marker marker;
            marker.header = msg->header;
            marker.ns = "detected_obstacle";
            marker.id = obstacle_id;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
            marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
            marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = size_x;
            marker.scale.y = size_y;
            marker.scale.z = size_z;

            marker.color.r = 0.0;
            marker.color.g = 0.5;
            marker.color.b = 1.0;
            marker.color.a = 0.3;

            marker_array.markers.push_back(marker);

        }

        bbox_pub_->publish(marker_array);
        if (!valid_obstacles->empty())
        {
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*valid_obstacles, output_msg);
            output_msg.header = msg->header;
            obstacles_pub_->publish(output_msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacles_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleClusteringNode>());
    rclcpp::shutdown();
    return 0;
}
