# Inbuilt ROS displays

Version of the front/back-end displays that use `visualization_msgs` to display the object states per frame:

- object pose
- object paths
- object id
- bounding box

Inbuilt refers to the fact that no third-party deps are needed for the visualisation, instead we use standard ROS messages (mainly `visualization_msgs`) to display the object states.
This setup is therefore more complex, and results in many more advertised topics to achieve a similar (and less flexible) display than using the custom plugin/interface combination.

These classes are conditionally compiled:
```
colcon build --cmake-args -DENABLE_DYNAMIC_SLAM_INTERFACES=OFF
```
and used as the `FrontendDisplayRos` and `BackendDisplayRos`, as defined in [DisplaysImpl.hpp](../DisplaysImpl.hpp).
