cmake_minimum_required (VERSION 3.5)
project(dynosam_odom VERSION 1.0 LANGUAGES CXX C)

# Make sure that custom modules like FindXXXX are found
# Prefer to use system installation of gflags/glog
set(GFLAGS_PREFER_EXPORTED_GFLAGS_CMAKE_CONFIGURATION TRUE)
set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION TRUE)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

### System Dependencies
find_package(Boost REQUIRED COMPONENTS
  filesystem serialization regex timer date_time thread)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
  INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()

find_package(Gflags REQUIRED)
find_package(Glog 0.3.5 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(opengv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(DBoW2 REQUIRED)
if(NOT TARGET DBoW2::DBoW2)
  add_library(DBoW2::DBoW2 INTERFACE IMPORTED)
  set_target_properties(DBoW2::DBoW2 PROPERTIES
  INTERFACE_LINK_LIBRARIES "${DBoW2_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${DBoW2_INCLUDE_DIRS}")
endif()
# Pangolin is optional
find_package(Pangolin QUIET)

include(VerifyGtsamConfig)
option(DYNOSAM_VERIFY_GTSAM_CONFIG "Check that GTSAM was compiled with the right options" ON)
if (DYNOSAM_VERIFY_GTSAM_CONFIG)
    verify_gtsam_config()
endif()

### External Dependencies
include(ExternalProject)

### Download and unzip the vocabularly file
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.yml)
  message(STATUS "Downloading vocabulary file from dropbox.")
  file(DOWNLOAD
       http://www.dropbox.com/s/lyo0qgbdxn6eg6o/ORBvoc.zip?dl=1
       ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.zip
       SHOW_PROGRESS
       STATUS orbvoc_download_success
       TIMEOUT 60)
  if(orbvoc_download_success)
    message(STATUS "Unzipping vocabulary file.")

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc.zip
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/)
    message(STATUS "Moving vocabulary file.")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc/
                    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/)
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary/ORBvoc/)
  else(orbvoc_download_success)
    message(STATUS "Failed to download vocabulary file. Please download manually.")
  endif(orbvoc_download_success)
else()
  message(STATUS "Vocabulary file exists, will not download.")
endif()

### Compile the code
add_library(${PROJECT_NAME} SHARED "")

add_subdirectory(include/dynosam_odom/tmp)
add_subdirectory(src/tmp)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    opengv
    DBoW2::DBoW2
  PUBLIC
    Boost::boost
    ${OpenCV_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    gtsam
    gtsam_unstable
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Pangolin is optional
if(Pangolin_FOUND)
  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      ${Pangolin_LIBRARIES}
  )
  target_include_directories(${PROJECT_NAME}
    PUBLIC
      ${Pangolin_INCLUDE_DIRS}
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE Pangolin_FOUND=1)
else(Pangolin_FOUND)
  message(STATUS "Pangolin not found.")
endif(Pangolin_FOUND)

target_compile_options(${PROJECT_NAME}
  PRIVATE -Wall -pipe
  PRIVATE -march=native
)

# We would just need to say cxx_std_11 if we were using cmake 3.8
target_compile_features(${PROJECT_NAME} PUBLIC
        cxx_auto_type cxx_constexpr cxx_range_for cxx_nullptr cxx_override)

# Add an alias so that library can be used inside the build tree,
# e.g. when testing
add_library(dynosam_odom::dynosam_odom ALIAS dynosam_odom)

# add_executable(stereoVIOEuroc ./examples/KimeraVIO.cpp)
# target_link_libraries(stereoVIOEuroc PUBLIC kimera_vio::kimera_vio)

############################### INSTALL/EXPORT #################################
## We install the export that we defined above
## Export the targets to a script
## This will install the import script dynosam_odomTargets.cmake
## When findpackaged by other scripts, it will load the targets defined
## in the export dynosam_odom-export.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/dynosam_odomConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Create dynosam_odomConfig.cmake with extra info from dynosam_odomConfig.cmake.in
# This file is necessary to find_package the library dynosam_odom.
set(INSTALL_CONFIGDIR lib/cmake/dynosam_odom)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/dynosam_odomConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dynosam_odomConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

option(EXPORT_DYNOSAM "Export dynosam_odom instead of installing it." OFF)
if(EXPORT_DYNOSAM)
  export(TARGETS ${PROJECT_NAME}
         FILE dynosam_odomTargets.cmake)
       export(PACKAGE ${PROJECT_NAME})
else(EXPORT_DYNOSAM)
  include(GNUInstallDirs)
  ## First of all, we need to install the library itself.
  install(TARGETS ${PROJECT_NAME}
      EXPORT dynosam_odom-export
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      # Perhaps use this, instead of installing include dir
      #PUBLIC_HEADER DESTINATION include
      INCLUDES DESTINATION include # We need this right?
      RUNTIME DESTINATION bin
  )
  install(EXPORT dynosam_odom-export
    FILE
      dynosam_odomTargets.cmake
    DESTINATION
      ${INSTALL_CONFIGDIR}
  )
  # Install header files
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")

  ## Install the config and configversion
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/dynosam_odomConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/dynosam_odomConfigVersion.cmake
      ${CMAKE_CURRENT_LIST_DIR}/cmake/FindGflags.cmake
      ${CMAKE_CURRENT_LIST_DIR}/cmake/FindGlog.cmake
      DESTINATION ${INSTALL_CONFIGDIR}
  )
endif(EXPORT_DYNOSAM)

################################################################################
# Print configuration variables
message(STATUS "===============================================================")
message(STATUS "================  Configuration Options  ======================")
message(STATUS "CMAKE_CXX_COMPILER_ID type                : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION                : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build flags                                               ")
if(NOT MSVC AND NOT XCODE_VERSION)
    message(STATUS "  Build type                              : ${CMAKE_BUILD_TYPE}")
    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release")
      message(WARNING "  Build type is not in RELEASE, this will slow down the code.")
    endif()
    message(STATUS "  C compilation flags (Release)           : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "  C++ compilation flags (Release)         : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
