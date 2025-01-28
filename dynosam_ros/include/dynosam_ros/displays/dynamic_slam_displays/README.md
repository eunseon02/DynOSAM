# Dynamic SLAM Displays

Version of the front/back-end displays that depends on `dynamic_slam_interfaces` to publish the object states per frame:

- object pose
- object paths
- object motions (as velocity)
- object id

Each object is represented using the `dynamic_slam_interfaces::msg::ObjectOdometry` message which can be displayed in RVIZ using the [rviz_dynamic_slam_plugins](https://github.com/ACFR-RPG/rviz_dynamic_slam_plugins) plugin.
These classes are compiled by __default__ but this configuration can be specified with:
```
colcon build --cmake-args -DENABLE_DYNAMIC_SLAM_INTERFACES=ON
```
and used as the `FrontendDisplayRos` and `BackendDisplayRos`, as defined in [DisplaysImpl.hpp](../DisplaysImpl.hpp).
