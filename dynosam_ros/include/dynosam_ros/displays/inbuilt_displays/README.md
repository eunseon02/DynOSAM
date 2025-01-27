# Inbuilt ROS displays

Version of the front/back-end displays that use `visualization_msgs` to display the object states per frame:

- object pose
- object paths
- object id
- bounding box

Inbuilt refers to the fact that third-party deps are needed for the visualisation. These classes are conditionally compiled:
```
colcon build --cmake-args -DENABLE_DYNAMIC_SLAM_INTERFACES=OFF
```
and used as the `FrontendDisplayRos` and `BackendDisplayRos`, as defined in [DisplaysImpl.hpp](../DisplaysImpl.hpp).