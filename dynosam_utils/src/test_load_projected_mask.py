import cv2
import open3d as o3d
import numpy as np

path_to_projection_mask = "/root/results/DynoSAM/test_omd/project_mask/3.yml"


# Load the matrix using FileStorage
fs = cv2.FileStorage(path_to_projection_mask, cv2.FILE_STORAGE_READ)
projected_mask = fs.getNode("matrix").mat()
fs.release()
# projected_mask = cv2.imread(path_to_projection_mask, cv2.IMREAD_UNCHANGED).astype(np.float64)
# print(projected_mask.shape)

point_cloud = np.array(projected_mask[:,:,0:3])
print(point_cloud)
point_cloud = point_cloud.reshape(-1, 3)
print(point_cloud.shape)

mask = projected_mask[:,:,3].astype(np.uint8) * 255
mask = mask.reshape((*mask.shape, 1))
print(point_cloud.dtype)

cv2.imshow("Mask", mask)
cv2.waitKey(0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

o3d.visualization.draw_geometries([pcd])
