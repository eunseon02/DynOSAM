
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import random


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
    print(noise)
    points[0:3, i:i+1] = points[0:3, i:i+1] + noise[np.newaxis].T

  return points


def main():
  cam_pose_origin = np.identity(4)

  cam_mot_xyz_mean = np.array([10., 11., 12.]) # in meter
  cam_mot_rpy_mean = np.array([30., 45., 60.]) # in degree 
  cam_mot_xyz_range = np.array([2., 2., 2.]) # in meter
  cam_mot_rpy_range = np.array([5., 5., 5.]) # in degree 
  cam_pose_target = gen_random_pose(cam_mot_xyz_mean, cam_mot_rpy_mean, cam_mot_xyz_range, cam_mot_rpy_range)


  print(cam_pose_origin)
  print(cam_pose_target)

  # # check target pose rotation
  # rot = cam_pose_target[0:3, 0:3]
  # print(rot)
  # det = np.linalg.det(rot)
  # orth = np.matmul(rot.T, rot)
  # print(det)
  # print(orth)

  points_origin_centre = np.array([5., 6., 7.])
  points_origin_range = np.array([1., 1., 1.])
  points_origin = gen_random_points(5, points_origin_centre, points_origin_range)

  print(points_origin)


if __name__ == "__main__":
  main()

