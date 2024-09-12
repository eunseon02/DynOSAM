# TRO 2024 paper experiments readme

## Notes taken during experiments that may be helpful to future Jesse or other researchers.

> **Note**
These comments were made DURING experiments and may not reflect the final version of the code or parameters upon release. I have tried to keep things dated (gitblame will also help) but I do not guaratee it.

### KITTI dataset
- (12.9.24) Requires the propogate mask flag to be turned on otehrwise tracking of the cyclist starts really late. I think this is becuase we're using an old segmentation network for this sequence and so there is a lot of flickering of the masks.
