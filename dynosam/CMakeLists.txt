cmake_minimum_required(VERSION 3.8)

project(dynosam)

# set(CMAKE_C_FLAGS "-std=gnu11 -Wall -Wextra -O3 -march=sandybridge -flto")
set(CMAKE_C_FLAGS "-Wall -Wextra")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)


list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)


find_package(Boost REQUIRED)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
  INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()


include(VerifyGtsamConfig)
option(DYNO_VERIFY_GTSAM_CONFIG "Check that GTSAM was compiled with the right options" ON)
if (DYNO_VERIFY_GTSAM_CONFIG)
    verify_gtsam_config()
endif()


# setup targets
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# build library
add_library(${PROJECT_NAME} SHARED

  src/common/Types.cc

  src/dataprovider/DataProviderModule.cc
  src/dataprovider/DataProvider.cc
  src/dataprovider/DatasetLoader.cc
  src/dataprovider/DataProviderUtils.cc

  src/frontend/imu/ThreadSafeImuBuffer.cc
  src/frontend/FrontendPipeline.cc
  src/frontend/FrontendModule.cc
  src/frontend/RGBDInstanceFrontendModule.cc

  src/utils/OpenCVUtils.cc
  src/utils/GtsamUtils.cc
  src/utils/Spinner.cc
)

# target_compile_definitions(${PROJECT_NAME}
#   PRIVATE "MINIMAL_COMPOSITION_DLL")

# eigen3_cmake_module
# Eigen3

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  sensor_msgs
  diagnostic_updater)

target_link_libraries(${PROJECT_NAME}
# PRIVATE
  opengv
# PUBLIC
  ${rclcpp_LIBRARIES}
  glog
  gflags
  gtsam
  gtsam_unstable
  Boost::boost
  # Eigen3::Eigen
  ${OpenCV_LIBRARIES}
)


ament_export_libraries(${PROJECT_NAME}
  glog
  gflags
  opengv
  Boost::boost
)

# ament_export_dependencies(rclcpp std_msgs sensor_msgs eigen3_cmake_module Eigen3)
ament_export_dependencies(rclcpp std_msgs sensor_msgs)


target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)


#####exec test#########
add_executable(dyno_sam example/dyno_sam.cc)
target_link_libraries(dyno_sam
  ${PROJECT_NAME}
)
ament_target_dependencies(dyno_sam "rclcpp" "std_msgs")



install(TARGETS
  dyno_sam
  DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  # find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
  ament_add_gmock(${PROJECT_NAME}_test
    test/test_main.cc
    test/thread_safe_queue_tests.cc
    test/thread_safe_temporal_buffer_test.cc
    test/test_threadsafe_imu_buffer.cc
    test/test_dataset_provider.cc
    test/test_types.cc
    test/internal/tmp_file.cc
)

  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${rclcpp_LIBRARIES})

  install(TARGETS
  ${PROJECT_NAME}_test
      DESTINATION test/)

endif()


ament_package()
