cmake_minimum_required(VERSION 3.8)

project(dynosam_utils)

# set(CMAKE_C_FLAGS "-std=gnu11 -Wall -Wextra -O3 -march=sandybridge -flto")
set(CMAKE_C_FLAGS "-Wall -Wextra")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dynosam REQUIRED)

# setup targets
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
include_directories(include)



# ament_target_dependencies(${PROJECT_NAME}
#   rclcpp
#   std_msgs
#   dynosam
# )

# target_link_libraries(${PROJECT_NAME}
#   ${rclcpp_LIBRARIES}
#   glog
#   gflags
# )
# ament_export_libraries(${PROJECT_NAME}
#   glog
#   gflags
# )


# target_include_directories(${PROJECT_NAME}
#   PUBLIC
#   ${rclcpp_INCLUDE_DIRS}
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>)

# ament_export_include_directories(include)
# ament_export_libraries(${PROJECT_NAME})

# install(
#   DIRECTORY include/
#   DESTINATION include
# )
# install(
#   TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}
# )

# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}
# )

#####exec test#########
# add_executable(kitti_semantic_to_motion src/KittiSemaniticToMotion.cc)

# ament_target_dependencies(kitti_semantic_to_motion
#   rclcpp
#   std_msgs
#   dynosam
# )

# ament_target_dependencies(kitti_semantic_to_motion "rclcpp" "std_msgs"
# "dynosam"
# )

# install(TARGETS
# kitti_semantic_to_motion
#   DESTINATION lib/${PROJECT_NAME})





add_executable(InstanceVizReader src/InstanceVizReader.cc)

target_link_libraries(InstanceVizReader
png
)

ament_target_dependencies(InstanceVizReader
rclcpp
std_msgs
dynosam
)

ament_target_dependencies(InstanceVizReader "rclcpp" "std_msgs"
"dynosam"
)

install(TARGETS
InstanceVizReader
DESTINATION lib/${PROJECT_NAME})



# if(BUILD_TESTING)
#   # find_package(ament_cmake_gtest REQUIRED)
#   find_package(ament_cmake_gmock REQUIRED)
#   find_package(ament_lint_auto REQUIRED)

#   ament_lint_auto_find_test_dependencies()

#   # Run all lint tests in package.xml except those listed above
#   ament_lint_auto_find_test_dependencies()
#   ament_add_gmock(${PROJECT_NAME}_test
#     test/test_main.cc
# )

#   target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${rclcpp_LIBRARIES})

#   install(TARGETS
#   ${PROJECT_NAME}_test
#       DESTINATION test/)

# endif()


ament_package()
