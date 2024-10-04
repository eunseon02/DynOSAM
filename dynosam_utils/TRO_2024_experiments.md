# TRO 2024 paper experiments readme

## Notes taken during experiments that may be helpful to future Jesse or other researchers.

> **Note**
These comments were made DURING experiments and may not reflect the final version of the code or parameters upon release. I have tried to keep things dated (gitblame will also help) but I do not guaratee it.

### KITTI dataset
- (12.9.24) Requires the propogate mask flag to be turned on otehrwise tracking of the cyclist starts really late. I think this is becuase we're using an old segmentation network for this sequence and so there is a lot of flickering of the masks.
- (25.9.24) for some datasets we still need the "track dynamic" on - which essenially only tries to determine an object is dynamic or not via scene flow - this is relevant only for this sequence becuase some of the dynamic points (from the depth) is really crap and 'stetches' out the object, making it look static. We want to classify it as static to stop these points being tracked so the object motion is not effected. This is akind of a hack :)
- (1.10.24) Turn off propogate mask with KITTI 0005/06 - still minor bug where it track's the end of an object (when it disappears) and propogates those points to the static points behind it. This results in the motion being horrible for the last few frames!!
