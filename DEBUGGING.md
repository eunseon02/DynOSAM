
### Build
```
cd /home/user/dev_ws && colcon build --cmake-args -DENABLE_DYNAMIC_SLAM_INTERFACES=ON
```


### Running using RBG-D camera

To run from data provided by an RGB-D camera use
```
ros2 launch dynosam_ros dyno_sam_online_rgbd_launch.py
```