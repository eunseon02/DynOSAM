cmake_minimum_required(VERSION 2.8.3)
project(dynosam_ros)

find_package(catkin_simple REQUIRED)

### System Dependencies
find_package(Boost REQUIRED COMPONENTS
  serialization system filesystem thread date_time regex timer chrono)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
  INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()

catkin_simple()

cs_add_library(${PROJECT_NAME}
  # Include files are just here for IDEs
  include/dynosam_ros/tmp_ros.h
  src/tmp_ros.cpp
)
target_link_libraries(${PROJECT_NAME} dynosam_ros)

cs_add_executable(${PROJECT_NAME}_node
  src/app.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

cs_install()

cs_export()
