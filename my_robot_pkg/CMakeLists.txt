cmake_minimum_required(VERSION 3.8)
project(my_robot_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Build C++ node
add_executable(motor_driver 
  src/motor_driver.cpp
  src/SparkBase.cpp
  src/SparkMax.cpp
)
target_include_directories(motor_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(motor_driver rclcpp std_msgs sensor_msgs)

# Install the C++ executable
install(TARGETS motor_driver
  RUNTIME DESTINATION lib/${PROJECT_NAME})

# Install headers so other packages can consume them
install(DIRECTORY include/
  DESTINATION include)

# Install Python scripts as executables
install(PROGRAMS
  scripts/ros2_controller.py
  scripts/costmap.py
  scripts/lidar_processing.py
  scripts/mapping.py
  scripts/odometry.py
  scripts/pathfinding.py
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export include directories and dependencies
ament_export_include_directories(include)
ament_export_dependencies(rclcpp std_msgs sensor_msgs)

ament_package()
