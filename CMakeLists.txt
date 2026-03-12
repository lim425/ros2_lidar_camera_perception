cmake_minimum_required(VERSION 3.8)
project(lidar_camera_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# 1. lidar_preprocessing_node
add_executable(lidar_preprocessing_node src/lidar_preprocessing_node.cpp)
ament_target_dependencies(lidar_preprocessing_node
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(lidar_preprocessing_node ${PCL_LIBRARIES})

# 2. lidar_cluster_detector_node
add_executable(lidar_cluster_detector_node src/lidar_cluster_detector_node.cpp)
ament_target_dependencies(lidar_cluster_detector_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)
target_link_libraries(lidar_cluster_detector_node ${PCL_LIBRARIES})

# 3. lidar_tracker_node
add_executable(lidar_tracker_node src/lidar_tracker_node.cpp)
ament_target_dependencies(lidar_tracker_node
  rclcpp
  visualization_msgs
)
target_link_libraries(lidar_tracker_node Eigen3::Eigen)

# 4. image_overlay_node
add_executable(image_overlay_node src/image_overlay_node.cpp)
ament_target_dependencies(image_overlay_node
rclcpp
sensor_msgs
vision_msgs
visualization_msgs
cv_bridge
)
target_link_libraries(image_overlay_node ${OpenCV_LIBS})

# 5. lidar_camera_fusion_node
add_executable(lidar_camera_fusion_node src/lidar_camera_fusion_node.cpp)
target_include_directories(lidar_camera_fusion_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(lidar_camera_fusion_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  std_msgs
  cv_bridge
)
target_link_libraries(lidar_camera_fusion_node ${OpenCV_LIBRARIES})

install(TARGETS
  lidar_preprocessing_node
  lidar_cluster_detector_node
  lidar_tracker_node
  image_overlay_node
  lidar_camera_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# Python
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ${PROJECT_NAME}/lidar_preprocessing_node.py
  ${PROJECT_NAME}/lidar_cluster_detector_node.py
  ${PROJECT_NAME}/lidar_tracker_node.py
  ${PROJECT_NAME}/camera_object_detection_node.py
  ${PROJECT_NAME}/image_overlay_node.py
  ${PROJECT_NAME}/lidar_camera_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
