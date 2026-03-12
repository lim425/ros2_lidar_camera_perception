#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

class SegmentationNode : public rclcpp::Node
{
public:
    SegmentationNode() : Node("lidar_preprocessing_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/kitti/point_cloud",
            10,
            std::bind(&SegmentationNode::lidarCallback, this, std::placeholders::_1)
        );

        non_ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/non_ground_pcd", 10);
        ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/ground_pcd", 10);

        RCLCPP_INFO(this->get_logger(), "Lidar Preprocessing Node Started...");
    }

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS2 PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
            return;

        // Voxel Downsampling
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel.filter(*cloud_filtered);

        // ROI Filtering
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud_filtered);
        crop.setMin(Eigen::Vector4f(-20.0, -10.0, -3.0, 1.0)); // (backward, right)
        crop.setMax(Eigen::Vector4f(50.0, 10.0, 5.0, 1.0)); // (forward, left)

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>());
        crop.filter(*cloud_roi);

        pcl::PointCloud<pcl::PointXYZ>::Ptr bottom_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &point : cloud_roi->points)
        {
            if (point.z < -0.5)
                bottom_cloud->points.push_back(point);
            else
                top_cloud->points.push_back(point);
        }

        if (bottom_cloud->empty())
            return;

        // RANSAC Plane Segmentation
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.25);
        seg.setMaxIterations(1000);
        seg.setInputCloud(bottom_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
            return;

        // Extract Ground & Non-Ground
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr bottom_non_ground(new pcl::PointCloud<pcl::PointXYZ>());

        extract.setInputCloud(bottom_cloud);
        extract.setIndices(inliers);

        // Ground
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        // Non-ground (bottom part)
        extract.setNegative(true);
        extract.filter(*bottom_non_ground);

        // Combine top + bottom_non_ground
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_non_ground(new pcl::PointCloud<pcl::PointXYZ>());
        *final_non_ground = *top_cloud + *bottom_non_ground;

        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*ground_cloud, ground_msg);
        ground_msg.header = msg->header;
        ground_pub_->publish(ground_msg);

        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*final_non_ground, non_ground_msg);
        non_ground_msg.header = msg->header;
        non_ground_pub_->publish(non_ground_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr non_ground_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentationNode>());
    rclcpp::shutdown();
    return 0;
}
