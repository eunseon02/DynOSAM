import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct




class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_point_cloud_node')


        # information from original OMD dataset
        # 569.031286352279, 568.8228728028029, 338.766074706977, 221.6640023621621
        # 618.3587036132812, 618.5924072265625, 328.9866333007812, 237.7507629394531
        self.K1 = np.array([[618.3587036132812, 0, 328.9866333007812],
                   [0, 618.5924072265625, 237.7507629394531],
                   [0, 0, 1]])
        # self.D1 = np.array([0.08992404268726646, -0.19299690318419144, -0.00989017616746251, 0.0032348666428874432])
        self.D1 = np.array([0, 0, 0, 0])
        self.image_path1 = "/root/data/omm/swinging_4_unconstrained/rgbd/000000_aligned_depth.png"
        self.depth_image1 = cv2.imread(self.image_path1, cv2.IMREAD_UNCHANGED)
        self.depth_image1 = self.depth_image1.astype("float64")

        # information from omd-demo
        self.K2 = np.array([[618.3587036132812, 0,328.9866333007812 ],
                   [0, 618.5924072265625 , 237.7507629394531],
                   [0, 0, 1]])
        self.D2 = np.array([0, 0, 0, 0])
        self.image_path2 = "/root/data/demo-omd/depth/000000.png"
        self.depth_image2 = cv2.imread(self.image_path2, cv2.IMREAD_UNCHANGED)
        self.depth_image2 = self.depth_image2.astype("float64")



        self.pub1 = self.create_publisher(PointCloud2, '/point_cloud1', 10)
        self.pub2 = self.create_publisher(PointCloud2, '/point_cloud2', 10)

        self.bridge = CvBridge()

        self.publish_rate = 5
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

    def timer_callback(self):
        self.depth_callback1(self.depth_image1)
        self.depth_callback2(self.depth_image2)

    def depth_callback1(self, depth_image):
        if self.K1 is not None and self.D1 is not None:
            undistorted_depth_image = depth_image
            # undistorted_depth_image /= 1000.0
            # undistorted_depth_image /= 1000.0
            undistorted_depth_image = self.undistort_image(undistorted_depth_image, self.K1, self.D1)
            height, width = undistorted_depth_image.shape
            for v in range(height):
                for u in range(width):
                    z = undistorted_depth_image[v, u]
                    if z == 0:
                        continue
                    else:
                        undistorted_depth_image[v, u] = 387.5744 / (undistorted_depth_image[v, u] / 1000.0)
            point_cloud = self.convert_depth_to_point_cloud(undistorted_depth_image, self.K1)
            self.pub1.publish(point_cloud)

    def depth_callback2(self, depth_image):
        if self.K2 is not None and self.D2 is not None:
            undistorted_depth_image = self.undistort_image(depth_image, self.K2, self.D2)
            height, width = undistorted_depth_image.shape
            for v in range(height):
                for u in range(width):
                    z = undistorted_depth_image[v, u]
                    if z == 0:
                        continue
                    else:
                        undistorted_depth_image[v, u] = 387.5744 / (undistorted_depth_image[v, u] / 1000.0)

            point_cloud = self.convert_depth_to_point_cloud(undistorted_depth_image, self.K2)
            self.pub2.publish(point_cloud)

    def undistort_image(self, image, K, D):
        h, w = image.shape[:2]
        print(K)
        print(D)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
        print(new_K)
        undistorted_image = cv2.undistort(image, K, D, None, new_K)
        return undistorted_image

    def convert_depth_to_point_cloud(self, depth_image, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = depth_image.shape
        points = []

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                if z == 0:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

        points = np.array(points)
        return self.create_point_cloud_msg(points)

    def create_point_cloud_msg(self, points):
        msg = PointCloud2()
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = np.asarray(points, np.float32).tobytes()
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
