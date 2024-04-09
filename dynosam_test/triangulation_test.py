
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

import csv

def gen_random_pose(xyz_mean, rpy_mean, xyz_range, rpy_range):
  # rpy mean and range are all in degree
  # xyz mean and range are all in meter

  xyz = xyz_mean + np.array([random.uniform(-xyz_range[0], xyz_range[0]), random.uniform(-xyz_range[1], xyz_range[1]), random.uniform(-xyz_range[2], xyz_range[2])])
  rpy = rpy_mean + np.array([random.uniform(-rpy_range[0], rpy_range[0]), random.uniform(-rpy_range[1], rpy_range[1]), random.uniform(-rpy_range[2], rpy_range[2])])


  rot = R.from_euler('zyx', rpy, degrees=True)
  rot_matrix = rot.as_matrix()

  pose = np.identity(4)
  pose[0:3, 0:3] = rot_matrix
  pose[0:3, 3:4] = xyz[np.newaxis].T

  return pose

def gen_random_points(number, centre, point_range):
  points = np.tile(centre[np.newaxis].T, (1, number))

  for i in range(number):
    noise = np.array([random.uniform(-point_range[0], point_range[0]), random.uniform(-point_range[1], point_range[1]), random.uniform(-point_range[2], point_range[2])])
    points[0:3, i:i+1] = points[0:3, i:i+1] + noise[np.newaxis].T

  return points

def gen_cubic_points(number, centre, point_range):
  points = np.tile(centre[np.newaxis].T, (1, number))

  points[0][0] -= point_range[0]/2
  points[1][0] += point_range[1]/2
  points[2][0] -= point_range[2]/2
  points[0][1] += point_range[0]/2
  points[1][1] += point_range[1]/2
  points[2][1] -= point_range[2]/2
  points[0][2] -= point_range[0]/2
  points[1][2] += point_range[1]/2
  points[2][2] += point_range[2]/2
  points[0][3] += point_range[0]/2
  points[1][3] += point_range[1]/2
  points[2][3] += point_range[2]/2
  points[0][4] -= point_range[0]/2
  points[1][4] -= point_range[1]/2
  points[2][4] -= point_range[2]/2
  points[0][5] += point_range[0]/2
  points[1][5] -= point_range[1]/2
  points[2][5] -= point_range[2]/2
  points[0][6] -= point_range[0]/2
  points[1][6] -= point_range[1]/2
  points[2][6] += point_range[2]/2
  points[0][7] += point_range[0]/2
  points[1][7] -= point_range[1]/2
  points[2][7] += point_range[2]/2

  return points

def homogeneous_points(points):
  points_shape = np.shape(points)
  if points_shape[0] == 3:
    length = points_shape[1]
    homo_pad = np.ones(length)
    points_homo = np.vstack((points, homo_pad))

    return points_homo
  else:
    print("We expect 3D points")
    print(points_shape)

def transform_points(points, motion):
  # Assume 3D points
  points_shape = np.shape(points)
  if points_shape[0] == 3:
    points_origin_homo = homogeneous_points(points)
    points_target_homo = np.matmul(motion, points_origin_homo)
    points_target = points_target_homo[0:3, :]

    return points_target

  else:
    print("We expect 3D points")
    print(points_shape)

def point_frame_robotics_to_cv(points):
  points_tmp = np.copy(points)
  points_camera = np.copy(points)
  points_camera[0, :] = -points_tmp[1, :]
  points_camera[1, :] = -points_tmp[2, :]
  points_camera[2, :] = points_tmp[0, :]
  return points_camera

def project_points(points, intrinsic):
  # Assume 3D points instead of homogeneous
  points_projected = np.matmul(intrinsic, points)
  depths = points_projected[2, :]
  points_image = points_projected[0:2, :]
  points_image[0, :] = np.divide(points_image[0, :], depths)
  points_image[1, :] = np.divide(points_image[1, :], depths)
  return depths, points_image

def compute_projection_matrix(camera_pose, intrinsic):
  pose_inv = np.linalg.inv(camera_pose)
  projection_matrix = np.matmul(intrinsic, pose_inv[0:3, :])
  return projection_matrix

def plot_whole_test(points_gt, points_est, points_exp, cam_poses):
  fig = plt.figure(figsize = (10,10))
  ax = plt.axes(projection="3d")

  x_axis = np.array([1, 0, 0, 1])
  y_axis = np.array([0, 1, 0, 1])
  z_axis = np.array([0, 0, 1, 1])

  for step in range(len(cam_poses)):
    ax.scatter3D(points_gt[step][0, :], points_gt[step][1, :], points_gt[step][2, :], label="Point (GT) in world at "+str(step))
    if step > 0:
      ax.scatter3D(points_est[step-1][0, :], points_est[step-1][1, :], points_est[step-1][2, :], label="Point (Est) in world from "+str(step-1)+" and "+str(step)+" at "+str(step-1))
      ax.scatter3D(points_exp[step-1][0, :], points_exp[step-1][1, :], points_exp[step-1][2, :], label="Point (Exp) in world from "+str(step-1)+" and "+str(step)+" at "+str(step-1))

    x_axis_cam = np.matmul(cam_poses[step], x_axis[np.newaxis].T)[0:3]
    y_axis_cam = np.matmul(cam_poses[step], y_axis[np.newaxis].T)[0:3]
    z_axis_cam = np.matmul(cam_poses[step], z_axis[np.newaxis].T)[0:3]
    o_cam = cam_poses[step][0:3, 3][np.newaxis].T
    ax.plot3D(np.array([o_cam[0], x_axis_cam[0]]), np.array([o_cam[1], x_axis_cam[1]]), np.array([o_cam[2], x_axis_cam[2]]), 'red')
    ax.plot3D(np.array([o_cam[0], y_axis_cam[0]]), np.array([o_cam[1], y_axis_cam[1]]), np.array([o_cam[2], y_axis_cam[2]]), 'green')
    ax.plot3D(np.array([o_cam[0], z_axis_cam[0]]), np.array([o_cam[1], z_axis_cam[1]]), np.array([o_cam[2], z_axis_cam[2]]), 'blue')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()

    ax.set_aspect('equal', adjustable='datalim')
    # ax.set_box_aspect([1, 1, 1])

  plt.show()

def write_whole_test(points_gt, points_est, cam_poses):
  file = open('csv/triangulation_1.csv', 'w')
  writer = csv.writer(file)

  for step in range(len(cam_poses)):
    writer.writerows(cam_poses[step])
    writer.writerow([])

  for step in range(len(cam_poses)):
    this_points_gt = points_gt[step]

    writer.writerows(this_points_gt.T)
    writer.writerow([])

    if step > 0:
      this_points_est = points_est[step-1]
      writer.writerows(this_points_est.T)
      writer.writerow([])

  file.close()


def triangulate_with_rotation_expanded(cam_pose_origin, cam_pose_target, intrinsic, obv_origin, obv_target, obj_rot):
  cam_rot_origin = cam_pose_origin[0:3, 0:3]
  cam_tran_origin = cam_pose_origin[0:3, 3]
  cam_rot_target = cam_pose_target[0:3, 0:3]
  cam_tran_target = cam_pose_target[0:3, 3]
  K_inv = np.linalg.inv(intrinsic)
  project_inv_origin = np.matmul(cam_rot_origin, K_inv)
  project_inv_target = np.matmul(cam_rot_target, K_inv)

  centroid_2d_origin = np.append(np.mean(obv_origin, axis=1), 1)[np.newaxis].T
  centroid_3d_origin = np.matmul(K_inv, centroid_2d_origin)
  coeffs_scale = np.matmul((obj_rot - np.identity(3)), centroid_3d_origin)
  print("Coeff scales")
  print(coeffs_scale)

  # For every observation, the result of the 3 equations constructed are identical to each other (?)
  results_single = np.matmul(obj_rot, cam_tran_origin) - cam_tran_target
  print("Results single")
  print(results_single)
  results = results_single

  n_obv = len(obv_origin[0, :])
  # Every observation contributes a pair of depth variables, and all of them share a scale variable
  # Hence the coefficient matrix is has 1+2*n_obv columns
  # Each observation helps construct 3 equations
  # Hence the coefficient matrix has 3*n_obv rows
  coeffs = np.zeros((3*n_obv, 1+2*n_obv), dtype=float)
  points_world = np.zeros((3, n_obv), dtype=float)
  for i_obv in range(n_obv):
    this_obv_origin = np.append(obv_origin[:, i_obv], 1)[np.newaxis].T
    this_obv_target = np.append(obv_target[:, i_obv], 1)[np.newaxis].T
    coeffs_origin = np.matmul(obj_rot, np.matmul(project_inv_origin, this_obv_origin))
    coeffs_target = np.matmul(project_inv_target, this_obv_target)

    # Static case:
    # project_inv matrices are 3*3: rotation*intrinsic^{-1}
    # coeffs is a 3*1 column vector: project_inv*obv
    # the function constructed is:
    # coeff_target*depth_target - coeff_origin*depth_origin = cam_tran_origin - cam_tran_target
    # 2 unknowns 3 functions: we will use the first 2 to solve for a solution first

    # Dynamic case:
    # target project_inv matrix is 3*3: rotation*intrinsic^{-1}
    # origin project_inv matrix: obj_rot*rotation*intrinsic^{-1}
    # coeffs is a 3*1 column vector: project_inv*obv
    # the function constructed is:
    # coeff_target*depth_target - coeff_origin*depth_origin = obj_rot*cam_tran_origin + obj_tran(unknown) - cam_tran_target
    # 2 unknowns 3 functions: we will use the first 2 to solve for a solution first
    coeffs[3*i_obv, 0] = coeffs_scale[0]
    coeffs[3*i_obv+1, 0] = coeffs_scale[1]
    coeffs[3*i_obv+2, 0] = coeffs_scale[2]
    coeffs[3*i_obv:(3*i_obv+3), (1+2*i_obv):(3+2*i_obv)] = np.concatenate((coeffs_target, -coeffs_origin), axis=1)
    if i_obv > 0:
      results = np.concatenate((results, results_single), axis=0)

  # print(coeffs)
  depths_lstsq, residuals, rank, singulars = np.linalg.lstsq(coeffs, results)
  print(depths_lstsq)

  # lhs = coeffs[0:2, 0:2]
  # rhs = results[0:2][np.newaxis].T
  # depths = np.matmul(np.linalg.inv(lhs), rhs)
  # print(depths)

  for i_obv in range(n_obv):
    this_obv_origin = np.append(obv_origin[:, i_obv], 1)[np.newaxis].T
    this_obv_target = np.append(obv_target[:, i_obv], 1)[np.newaxis].T
    depths = depths_lstsq[(1+2*i_obv):(3+2*i_obv)][np.newaxis].T

    # # Check with function 3
    # print(coeffs[2, 0] * depths[0] + coeffs[2, 1] * depths[1])
    # print(results[2])

    this_point_target = np.matmul(K_inv, depths[0]*this_obv_target)
    this_point_origin = np.matmul(K_inv, depths[1]*this_obv_origin)
    this_point_world = transform_points(this_point_origin, cam_pose_origin)
    this_point_world_test = transform_points(this_point_target, cam_pose_target)

    print("Point in world from origin observation", i_obv, ": \n", this_point_world)
    print("Point in world from target observation", i_obv, ": \n", this_point_world_test)

    points_world[0, i_obv] = this_point_world[0]
    points_world[1, i_obv] = this_point_world[1]
    points_world[2, i_obv] = this_point_world[2]

  # print(points_world)

  return points_world

def triangulate_with_rotation(cam_pose_origin, cam_pose_target, intrinsic, obv_origin, obv_target, obj_rot):
  cam_rot_origin = cam_pose_origin[0:3, 0:3]
  cam_tran_origin = cam_pose_origin[0:3, 3]
  cam_rot_target = cam_pose_target[0:3, 0:3]
  cam_tran_target = cam_pose_target[0:3, 3]
  K_inv = np.linalg.inv(intrinsic)
  project_inv_origin = np.matmul(cam_rot_origin, K_inv)
  project_inv_target = np.matmul(cam_rot_target, K_inv)

  n_obv = len(obv_origin[0, :])
  points_world = np.zeros((3, n_obv), dtype=float)
  for i_obv in range(n_obv):
    this_obv_origin = np.append(obv_origin[:, i_obv], 1)[np.newaxis].T
    this_obv_target = np.append(obv_target[:, i_obv], 1)[np.newaxis].T
    coeffs_origin = np.matmul(obj_rot, np.matmul(project_inv_origin, this_obv_origin))
    coeffs_target = np.matmul(project_inv_target, this_obv_target)

    # Static case:
    # project_inv matrices are 3*3: rotation*intrinsic^{-1}
    # coeffs is a 3*1 column vector: project_inv*obv
    # the function constructed is:
    # coeff_target*depth_target - coeff_origin*depth_origin = cam_tran_origin - cam_tran_target
    # 2 unknowns 3 functions: we will use the first 2 to solve for a solution first

    # Dynamic case:
    # target project_inv matrix is 3*3: rotation*intrinsic^{-1}
    # origin project_inv matrix: obj_rot*rotation*intrinsic^{-1}
    # coeffs is a 3*1 column vector: project_inv*obv
    # the function constructed is:
    # coeff_target*depth_target - coeff_origin*depth_origin = obj_rot*cam_tran_origin + obj_tran(unknown) - cam_tran_target
    # 2 unknowns 3 functions: we will use the first 2 to solve for a solution first
    coeffs = np.concatenate((coeffs_target, -coeffs_origin), axis=1)
    results = np.matmul(obj_rot, cam_tran_origin) - cam_tran_target

    depths_lstsq, residuals, rank, singulars = np.linalg.lstsq(coeffs, results)
    print(depths_lstsq)

    lhs = coeffs[0:2, :]
    rhs = results[0:2][np.newaxis].T
    depths = np.matmul(np.linalg.inv(lhs), rhs)
    print(depths)

    depths = depths_lstsq[np.newaxis].T

    # # Check with function 3
    # print(coeffs[2, 0] * depths[0] + coeffs[2, 1] * depths[1])
    # print(results[2])

    this_point_target = np.matmul(K_inv, depths[0]*this_obv_target)
    this_point_origin = np.matmul(K_inv, depths[1]*this_obv_origin)
    this_point_world = transform_points(this_point_origin, cam_pose_origin)
    this_point_world_test = transform_points(this_point_target, cam_pose_target)

    # print("Point in world from origin observation", i_obv, ": \n", this_point_world)
    # print("Point in world from target observation", i_obv, ": \n", this_point_world_test)

    points_world[0, i_obv] = this_point_world[0]
    points_world[1, i_obv] = this_point_world[1]
    points_world[2, i_obv] = this_point_world[2]

  print(points_world)

  return points_world

def main():
  example_length = 2

  # Set GT camera poses and points

  cam_ori_xyz_mean = np.array([0., 0., 0.]) # in meter
  cam_ori_rpy_mean = np.array([0., 0., 0.]) # in degree
  cam_ori_xyz_range = np.array([0., 0., 0.]) # in meter
  cam_ori_rpy_range = np.array([0., 0., 0.]) # in degree
  cam_pose_origin = gen_random_pose(cam_ori_xyz_mean, cam_ori_rpy_mean, cam_ori_xyz_range, cam_ori_rpy_range)

  cam_poses = []
  cam_poses.append(cam_pose_origin)

  cam_mot_xyz_mean = np.array([0.1, 0.1, 2.0]) # in meter
  cam_mot_rpy_mean = np.array([0., 15., 0.]) # in degree
  cam_mot_xyz_range = np.array([0.0, 0.0, 0.0]) # in meter
  cam_mot_rpy_range = np.array([0., 0., 0.]) # in degree

  for step in range(example_length-1):
    cam_motion = gen_random_pose(cam_mot_xyz_mean, cam_mot_rpy_mean, cam_mot_xyz_range, cam_mot_rpy_range)
    cam_pose_target = np.matmul(cam_motion, cam_poses[len(cam_poses)-1])
    cam_poses.append(cam_pose_target)

  for step in range(len(cam_poses)):
    print("Camera pose", step, ": \n", cam_poses[step])

  print()

  number_points = 8
  points_origin_centre = np.array([6., 0., 6.])
  points_origin_range = np.array([2., 2., 2.])
  # points_origin = gen_random_points(number_points, points_origin_centre, points_origin_range)
  points_origin = gen_cubic_points(number_points, points_origin_centre, points_origin_range)

  points = []
  points.append(points_origin)
  for step in range(example_length-1):
    points_mot_xyz_mean = np.array([2.0, 0.0, 0.0]) # in meter
    points_mot_rpy_mean = np.array([5., 10., 15.]) # in degree
    points_mot_xyz_range = np.array([0., 0., 0.]) # in meter
    points_mot_rpy_range = np.array([0., 0., 0.]) # in degree
    points_motion = gen_random_pose(points_mot_xyz_mean, points_mot_rpy_mean, points_mot_xyz_range, points_mot_rpy_range)
    print(points_motion)
    # points_motion = np.identity(4, dtype=float)
    points_target = transform_points(points[step], points_motion)
    points.append(points_target)

  for step in range(len(points)):
    print("Points", step, ": \n", points[step])

  print()

  # Compute measurements

  intrinsic = np.identity(3, dtype=float);
  intrinsic[0, 0] = 320.
  intrinsic[1, 1] = 320.
  intrinsic[0, 2] = 320.
  intrinsic[1, 2] = 240.

  pixels_observation = []

  for step in range(example_length):
    points_obv = transform_points(points[step], np.linalg.inv(cam_poses[step]))
    depth, points_obv_image = project_points(points_obv, intrinsic)
    pixels_observation.append(points_obv_image)

  for step in range(len(pixels_observation)):
    print("Points observes", step, ": \n", pixels_observation[step])

  print()

  points_estimated = []
  points_expanded = []

  for step in range(example_length):
    if step > 0:
      projection_matrix_prev = compute_projection_matrix(cam_poses[step-1], intrinsic)
      projection_matrix_curr = compute_projection_matrix(cam_poses[step], intrinsic)
      # print("Projection matrix", step, ":\n", projection_matrix)

      points_triangulated = cv2.triangulatePoints(projection_matrix_prev, projection_matrix_curr, pixels_observation[step-1], pixels_observation[step])
      points_triangulated_test = triangulate_with_rotation(cam_poses[step-1], cam_poses[step], intrinsic, pixels_observation[step-1], pixels_observation[step], points_motion[0:3, 0:3])
      points_triangulated_expanded = triangulate_with_rotation_expanded(cam_poses[step-1], cam_poses[step], intrinsic, pixels_observation[step-1], pixels_observation[step], points_motion[0:3, 0:3])

      points_normalised = np.zeros((3, number_points))
      for i in range(number_points):
        points_normalised[0:3, i] = np.array([points_triangulated[0, i]/points_triangulated[3, i], points_triangulated[1, i]/points_triangulated[3, i], points_triangulated[2, i]/points_triangulated[3, i]])

      print("Non-normalised points", step-1, "to", step, ": \n", points_triangulated)
      print("Triangulated points", step-1, "to", step, ": \n", points_normalised)
      print("Rotation compensated points", step-1, "to", step, ": \n", points_triangulated_test)
      print("Expanded points", step-1, "to", step, ": \n", points_triangulated_expanded)

      # points_estimated.append(points_normalised)
      points_estimated.append(points_triangulated_test)
      points_expanded.append(points_triangulated_expanded)

  plot_whole_test(points, points_estimated, points_expanded, cam_poses)

  # write_whole_test(points, points_estimated, cam_poses)

if __name__ == "__main__":
  main()
