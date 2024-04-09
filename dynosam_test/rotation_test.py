
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

def decompose_essential(essential):
  U, S, Vh = np.linalg.svd(essential, full_matrices=True)
  t = U[:, 2]
  t = t/np.linalg.norm(t)

  W = np.zeros((3, 3))
  W[0, 1] = -1.
  W[1, 0] = 1.
  W[2, 2] = 1.

  R1 = U @ (W @ Vh)
  if np.linalg.det(R1) < 0:
    R1 = -R1

  R2 = U @ (W.T @ Vh)
  if np.linalg.det(R2) < 0:
    R2 = -R2

  return R1, R2, t

def compute_essential(cam_pose_target, points_motion, cam_pose_origin):
  cam_rot = cam_pose_target[0:3, 0:3]
  cam_rot_inv = cam_rot.T
  cam_t = cam_pose_target[0:3, 3][np.newaxis].T

  ori_rot = cam_pose_origin[0:3, 0:3]
  ori_rot_inv = ori_rot.T
  ori_t = cam_pose_origin[0:3, 3][np.newaxis].T

  mot_rot = points_motion[0:3, 0:3]
  mot_t = points_motion[0:3, 3][np.newaxis].T

  translation = np.matmul(cam_rot_inv, (np.matmul(mot_rot, ori_t) + mot_t - cam_t))
  translation_cross = np.zeros((3, 3))
  translation_cross[0, 1] = -translation[2]
  translation_cross[1, 0] = translation[2]
  translation_cross[0, 2] = translation[1]
  translation_cross[2, 0] = -translation[1]
  translation_cross[1, 2] = -translation[0]
  translation_cross[2, 1] = translation[0]

  rotation = np.matmul(cam_rot_inv, np.matmul(mot_rot, ori_rot))
  essential = np.matmul(translation_cross, rotation)

  return essential, translation

def main():
  cam_ori_xyz_mean = np.array([0., 0., 0.]) # in meter
  cam_ori_rpy_mean = np.array([0., 0., 0.]) # in degree
  cam_ori_xyz_range = np.array([2., 2., 2.]) # in meter
  cam_ori_rpy_range = np.array([5., 5., 5.]) # in degree
  cam_pose_origin = gen_random_pose(cam_ori_xyz_mean, cam_ori_rpy_mean, cam_ori_xyz_range, cam_ori_rpy_range)
  # cam_pose_origin = np.identity(4)

  cam_mot_xyz_mean = np.array([10., 11., 12.]) # in meter
  cam_mot_rpy_mean = np.array([30., 45., 60.]) # in degree
  cam_mot_xyz_range = np.array([2., 2., 2.]) # in meter
  cam_mot_rpy_range = np.array([5., 5., 5.]) # in degree
  # cam_mot_xyz_mean = np.array([11., 0., 0.]) # in meter
  # cam_mot_rpy_mean = np.array([0., 0., 0.]) # in degree
  # cam_mot_xyz_range = np.array([0., 0., 0.]) # in meter
  # cam_mot_rpy_range = np.array([0., 0., 0.]) # in degree
  cam_motion = gen_random_pose(cam_mot_xyz_mean, cam_mot_rpy_mean, cam_mot_xyz_range, cam_mot_rpy_range)
  cam_pose_target = np.matmul(cam_motion, cam_pose_origin)

  print("Camera pose origin:")
  print(cam_pose_origin)
  print("Camera pose target:")
  print(cam_pose_target)
  print()

  # # check target pose rotation
  # rot = cam_pose_target[0:3, 0:3]
  # print(rot)
  # det = np.linalg.det(rot)
  # orth = np.matmul(rot.T, rot)
  # print(det)
  # print(orth)

  number_points = 8
  points_origin_centre = np.array([5., 6., 7.])
  points_origin_range = np.array([3., 3., 3.])
  points_origin = gen_random_points(number_points, points_origin_centre, points_origin_range)

  points_mot_xyz_mean = np.array([10., 11., 12.]) # in meter
  points_mot_rpy_mean = np.array([30., 45., 60.]) # in degree
  points_mot_xyz_range = np.array([2., 2., 2.]) # in meter
  points_mot_rpy_range = np.array([5., 5., 5.]) # in degree
  points_motion = gen_random_pose(cam_mot_xyz_mean, cam_mot_rpy_mean, cam_mot_xyz_range, cam_mot_rpy_range)
  # points_motion = np.identity(4)

  points_target = transform_points(points_origin, points_motion)

  print("Points origin:")
  print(points_origin)
  print("Points target:")
  print(points_target)

  intrinsic = np.identity(3);
  intrinsic[0, 0] = 320
  intrinsic[1, 1] = 320
  intrinsic[0, 2] = 320
  intrinsic[1, 2] = 240

  points_origin_local = transform_points(points_origin, np.linalg.inv(cam_pose_origin))
  points_target_local = transform_points(points_target, np.linalg.inv(cam_pose_target))

  print("Points origin local:")
  print(points_origin_local)
  print("Points target local:")
  print(points_target_local)

  # points_origin_cv = point_frame_robotics_to_cv(points_origin_local)
  # points_target_cv = point_frame_robotics_to_cv(points_target_local)

  # depth_origin, points_origin_image = project_points(points_origin_cv, intrinsic)
  # depth_target, points_target_image = project_points(points_target_cv, intrinsic)
  depth_origin, points_origin_image = project_points(points_origin_local, intrinsic)
  depth_target, points_target_image = project_points(points_target_local, intrinsic)

  print("Points origin image:")
  print(points_origin_image)
  print("Points target image:")
  print(points_target_image)
  print()

  essential, mask = cv2.findEssentialMat(points_origin_image.T, points_target_image.T, intrinsic)

  print("Essential matrix:")
  print(essential)
  print()

  R1, R2, t = cv2.decomposeEssentialMat(essential)

  print("Rotation 1:")
  print(R1)
  print("Rotation 2:")
  print(R2)
  print("Translation:")
  print(t)
  print()

  test_rotation = np.linalg.inv(cam_pose_target[0:3, 0:3]) @ ( points_motion[0:3, 0:3] @ cam_pose_origin[0:3, 0:3])
  # test_rotation = np.matmul(np.linalg.inv(cam_pose_target[0:3, 0:3]), np.matmul(points_motion[0:3, 0:3], cam_pose_origin[0:3, 0:3]))

  print("Test expected rotational component:")
  print(test_rotation)

  test_essential, test_translation = compute_essential(cam_pose_target, points_motion, cam_pose_origin)

  print("Test expected translation component:")
  print(test_translation)
  print("Test expected essential matrix")
  print(test_essential)

  R1_test, R2_test, t_test = cv2.decomposeEssentialMat(test_essential)

  print("Rotation 1 test:")
  print(R1_test)
  print("Rotation 2 test:")
  print(R2_test)
  print("Translation test:")
  print(t_test)

if __name__ == "__main__":
  main()
