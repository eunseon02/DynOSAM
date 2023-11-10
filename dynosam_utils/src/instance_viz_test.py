import colorsys
import cv2
# import matplotlib.pyplot as plt
import numpy as np

import skimage.io as io

HSV_mapping = {
#  ID:  [ H_min, H_max,    S,   v ]
    0:  [  0.00,  1.00, 0.00, 0.0 ], # Background
    1:  [  0.00,  1.00, 1.00, 0.9 ], # Vehicle linked to box
    2:  [  0.00,  1.00, 1.00, 0.9 ], # Pedestrian linked to box
    3:  [  0.00,  1.00, 0.40, 0.6 ], # Vehicle or Pedestrian without box
}


def map_instance_to_color(instances, HSV_mapping=HSV_mapping):
    """Map instances to the corresponding colors.

    Args:
        instances (np.array): Instance array. [ H x W ]
        HSV_mapping (dict, optional): Dictionary containing for every instance class a list with [hue_min, hue_max, saturation, value].

    Returns:
        np.array: Array containing the color coded instances [ H x W x 3]
    """
    assert isinstance(instances, np.ndarray), f"'instances' must be of type np.array, not {type(instances)}"
    # Get unique instances
    uniques, instance_map = np.unique(instances, return_inverse=True)
    print(uniques)
    # for i in instance_map:
    #     print(i)
    #     print(" ")
    # Make color_map
    color_map = []
    for instance_id in uniques:
        # Determine color range
        segm_class = instance_id//1000
        box_id = instance_id%1000
        hue_min, hue_max, saturation, value = HSV_mapping.get(segm_class, [0, 1, 1, 1])
        # Get color for each instance
        color_map.append(get_rgb_from_id(int(box_id), hue_min, hue_max, saturation=saturation, value=value))
    color_map = np.array(color_map)
    # Create colored instance image
    instance_map = color_map[instance_map.reshape(instances.shape)]
    return instance_map

def get_rgb_from_id(instance_id, hue_min=0, hue_max=1, saturation=0.9, value=0.9):
    """Map an instance/bbox id to a unique color.

    Args:
        instance_id (int): Value representing a unique id.
        hue_min (float, optional): Minimum value for the hue range in which this id should be mapped. Should be greater than or equal to 0. Default: 0
        hue_max (float, optional): Maximum value for the hue range in which this id should be mapped. If larger than 1, hue value will loop back to 0. Default: 1
        saturation (float, optional): Value between 0 and 1 for saturation. Default: 0.9
        value (float, optional): Value between 0 and 1 for brightness. Default: 0.9

    Returns:
        np.array[(3,)]: Array containing the RGB values for this instance id, with values between 0 and 1.
    """
    assert isinstance(instance_id, (int, np.int32)), f"instance_id should be of type 'int', not '{type(instance_id)}'"
    assert isinstance(hue_min, (int, float)) and isinstance(hue_max, (int, float)), f"hue_min and hue_max should be of type 'float', not '{type(hue_min)}' and '{type(hue_max)}'"
    assert isinstance(saturation, (int, float)) and saturation >= 0 and saturation <= 1, f"saturation should be a float between 0 and 1, not '{saturation}'"
    assert isinstance(value, (int, float)) and value >= 0 and value <= 1, f"value should be a float between 0 and 1, not '{value}'"

    # Golden angle: equally distributed colors but as far appart as possible; Hue will be between 0 and 1
    golden_angle = 137/360
    h = (instance_id*golden_angle) % 1
    # Scale hue in range (hue_min, hue_max)
    h = h*(hue_max-hue_min) + hue_min
    # Get rgb values
    rgb = np.array(colorsys.hsv_to_rgb(h, saturation, value))
    return rgb


img_path = "/root/data/virtual_kitti/vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"
inst_path = "/root/data/virtual_kitti/vkitti_2.0.3_instanceSegmentation/Scene01/clone/frames/instanceSegmentation/Camera_0/instancegt_00000.png"

img_inst = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

img_inst = cv2.cvtColor(img_inst, cv2.COLOR_BGR2RGB)
inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)

# img_inst = io.imread(img_path)
# inst = io.imread(inst_path)

alpha = 0.6

inst_color = map_instance_to_color(inst)
print(type(inst_color))
print(inst_color.shape)
# mask = inst_color.sum(axis=2)
# img_inst[:] = img_inst[:]*0.5
# img_inst[mask!=0,:] = img_inst[mask!=0,:]*(1-alpha) + inst_color[mask!=0,:]*256*alpha

cv2.imshow("Thing", inst_color)
cv2.waitKey(0)
