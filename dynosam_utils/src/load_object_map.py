import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("/root/results/DynoSAM/incremental_kitti_00_test/object_map_k11_j1.pcd")
# pcd = o3d.io.read_point_cloud("/root/results/DynoSAM/incremental_omd_test/object_map_k8_j2.pcd")

# pcd = o3d.io.read_point_cloud("/root/results/Dynosam_ecmr2024/cluster_l1/object_map_k82_j24.pcd")
pcd.paint_uniform_color([0.1, 0.1, 0.7])

xyz_load = np.asarray(pcd.points)
print(np.asarray(pcd.colors))

o3d.visualization.draw_geometries([pcd])
