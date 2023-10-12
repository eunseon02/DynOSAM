## Notes on Install (for now)

- need to build GTSAM with -DGTSAM_USE_SYSTEM_EIGEN=ON to avoid issues with ROS and OpenCV-Eigen compatability. Confirmed from https://discourse.ros.org/t/announcing-gtsam-as-a-ros-1-2-package/32739 which explains that the GTSAM ROS package is build with this flag set to ON (and describes it as "problematic"). We still want to build GTSAM from source so we can control the version and other compiler flags.
Kimera-VIO's install instructions indicate that OpenGV must use the same version of Eigen as GTSAM, which can be set using compiler flags. Since we are using the ROS install Eigen, I have removed these flags and hope that the package manager with CMake can find the right (and only) version
