
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

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

def plot_whole_test(points_gt, points_est, cam_poses):
  fig = plt.figure(figsize = (10,10))
  ax = plt.axes(projection="3d")

  x_axis = np.array([1, 0, 0, 1])
  y_axis = np.array([0, 1, 0, 1])
  z_axis = np.array([0, 0, 1, 1])

  for step in range(len(cam_poses)):
    ax.scatter3D(points_gt[step][0, :], points_gt[step][1, :], points_gt[step][2, :])
    if step > 0:
      ax.scatter3D(points_est[step-1][0, :], points_est[step-1][1, :], points_est[step-1][2, :])
  
    x_axis_cam = np.matmul(cam_poses[step], x_axis[np.newaxis].T)[0:3]
    y_axis_cam = np.matmul(cam_poses[step], y_axis[np.newaxis].T)[0:3]
    z_axis_cam = np.matmul(cam_poses[step], z_axis[np.newaxis].T)[0:3]
    o_cam = cam_poses[step][0:3, 3][np.newaxis].T
    ax.plot3D(np.array([o_cam[0], x_axis_cam[0]]), np.array([o_cam[1], x_axis_cam[1]]), np.array([o_cam[2], x_axis_cam[2]]), 'red')
    ax.plot3D(np.array([o_cam[0], y_axis_cam[0]]), np.array([o_cam[1], y_axis_cam[1]]), np.array([o_cam[2], y_axis_cam[2]]), 'green')
    ax.plot3D(np.array([o_cam[0], z_axis_cam[0]]), np.array([o_cam[1], z_axis_cam[1]]), np.array([o_cam[2], z_axis_cam[2]]), 'blue')

  ax.set_box_aspect([1,1,1])

  plt.show()

def main():
  example_length = 3

  # Set GT camera poses and points

  cam_ori_xyz_mean = np.array([0., 0., 0.]) # in meter
  cam_ori_rpy_mean = np.array([0., 0., 0.]) # in degree 
  cam_ori_xyz_range = np.array([0., 0., 0.]) # in meter
  cam_ori_rpy_range = np.array([0., 0., 0.]) # in degree 
  cam_pose_origin = gen_random_pose(cam_ori_xyz_mean, cam_ori_rpy_mean, cam_ori_xyz_range, cam_ori_rpy_range)

  cam_poses = []
  cam_poses.append(cam_pose_origin)

  cam_mot_xyz_mean = np.array([0.5, 0.5, 3.]) # in meter
  cam_mot_rpy_mean = np.array([5., 15., 5.]) # in degree 
  cam_mot_xyz_range = np.array([0.1, 0.1, 1.]) # in meter
  cam_mot_rpy_range = np.array([3., 3., 3.]) # in degree 

  for step in range(example_length-1):
    cam_motion = gen_random_pose(cam_mot_xyz_mean, cam_mot_rpy_mean, cam_mot_xyz_range, cam_mot_rpy_range)
    cam_pose_target = np.matmul(cam_motion, cam_poses[len(cam_poses)-1])
    cam_poses.append(cam_pose_target)

  for step in range(len(cam_poses)):
    print("Camera pose", step, ": \n", cam_poses[step])

  print()

  number_points = 8
  points_origin_centre = np.array([5., 0., 10.])
  points_origin_range = np.array([2., 2., 2.])
  points_origin = gen_random_points(number_points, points_origin_centre, points_origin_range)

  points = []
  points.append(points_origin)
  for step in range(example_length-1):
    points_mot_xyz_mean = np.array([-3., 0.5, 0.5]) # in meter
    points_mot_rpy_mean = np.array([0., 0., 0.]) # in degree 
    points_mot_xyz_range = np.array([1., 0.1, 0.1]) # in meter
    points_mot_rpy_range = np.array([0., 0., 0.]) # in degree 
    points_motion = gen_random_pose(points_mot_xyz_mean, points_mot_rpy_mean, points_mot_xyz_range, points_mot_rpy_range)
    points_target = transform_points(points_origin, points_motion)
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

  for step in range(example_length):
    if step > 0:
      projection_matrix_prev = compute_projection_matrix(cam_poses[step-1], intrinsic)
      projection_matrix_curr = compute_projection_matrix(cam_poses[step], intrinsic)
      # print("Projection matrix", step, ":\n", projection_matrix)

      points_triangulated = cv2.triangulatePoints(projection_matrix_prev, projection_matrix_curr, pixels_observation[step-1], pixels_observation[step])

      points_normalised = np.zeros((3, number_points))
      for i in range(number_points):
        points_normalised[0:3, i] = np.array([points_triangulated[0, i]/points_triangulated[3, i], points_triangulated[1, i]/points_triangulated[3, i], points_triangulated[2, i]/points_triangulated[3, i]])

      print("Triangulated points", step-1, "to", step, ": \n", points_normalised)

      points_estimated.append(points_normalised)

  plot_whole_test(points, points_estimated, cam_poses)

if __name__ == "__main__":
  main()

