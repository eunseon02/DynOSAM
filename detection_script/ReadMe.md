# Preparing Your Data

## Requirements
Tested with Python 3.7. Ensure these packages are installed: `numpy`, `opencv`, `json`, `glob`, `yaml`, `scipy`, `math`, `yolov8` (ultralytics).

## File Descriptions
- **find_xxx_corresponds.py**: Generates a basic JSON file (e.g., `diamond_vr_slow.json`), which includes paths to RGB and depth images, and ground truth poses. It's a utility script for the following file generation.
- **generate_rgbd_association.py**: Creates a `.txt` file to facilitate RGBD SLAM by matching timestamps with RGB and depth images. It's utilized by ORB-SLAM2.
- **generate_detection_files.py**: The primary script for creating JSON files for our VOOM system. It records detection data as follows:
  ```python
  det = dict()
  det["category_id"] = category_id
  det["detection_score"] = np.float64(conf)
  det["bbox"] = list(box)
  det["ellipse"] = list(ellipse_data)
- **process_diamond_gt.py**: Adjusts the ground truth format of ICL data to match that of TUM, used solely for evaluation purposes.
- **camera_pose_object.py**: Handles pose processing.
  
## Dataset
### TUM
- Path to the dataset directory should contain rgb/, depth/, and groundtruth.txt
```shell
python yolo_detection.py --data_path=/path/to/dataset
```
### ICL-data
```shell
```
### Recorded data
Assume we have a ros file:

