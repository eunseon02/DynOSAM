#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

from typing import Callable

class ImageRecorderNode(Node):
    def __init__(self, image_callback:Callable = None):
        super().__init__('image_recorder_node')

        self.image_callback = image_callback

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('video_file_name', 'output.avi')
        self.declare_parameter('output_folder', './')

        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.video_file_name = self.get_parameter('video_file_name').get_parameter_value().string_value
        self.output_folder = self.get_parameter('output_folder').get_parameter_value().string_value


        if not os.path.exists(self.output_folder):
            self.get_logger().fatal(f'Specified output folder {self.output_folder} does not exist!')
            self.destroy_node()
            rclpy.try_shutdown()
            return

        self.output_path = os.path.join(self.output_folder, self.video_file_name)
        self.get_logger().info(f'Recording video to: {self.output_path}')

        # Initialize CVBridge and video writer attributes
        self.bridge = CvBridge()
        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        self.fps = 30  # Set an approximate frame rate

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_topic_callback,
            10
        )

        self.get_logger().info(f'Subscribed to image topic: {self.image_topic}')

    def image_topic_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.image_callback is not None:
                cv_image = self.image_callback(cv_image)
                assert cv_image is not None

            # Initialize video writer if not already initialized
            if self.video_writer is None:
                self.frame_height, self.frame_width = cv_image.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    (self.frame_width, self.frame_height)
                )

            # Write the frame to the video file
            self.video_writer.write(cv_image)
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    def on_shutdown(self):
        # Release the video writer if it was initialized
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'Video saved to: {self.output_path}')
        else:
            self.get_logger().info('No video was recorded.')


def main(args=None):
    rclpy.init(args=args)

    node = ImageRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')
        node.on_shutdown()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


# To modify params run as
# run dynosam_utils record_image_topic.py --ros-args -p param:=value
if __name__ == '__main__':
    main()
