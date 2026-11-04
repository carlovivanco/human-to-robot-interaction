import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np


class ImageSubscriber:

    def __init__(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)  # start with a black img
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('image_subscriber')
        self.node.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.node.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        executor = SingleThreadedExecutor()  # own executor: the global one can only be spun once per process
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()
        self.bridge = CvBridge()
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth(self, x, y):
        """
        Returns depth at (x,y): the nearest valid value in a 15x15 patch, since tools are thin and the centre pixel
        often lands on the table behind them
        """
        patch = self.depth_image[max(y - 7, 0):y + 8, max(x - 7, 0):x + 8]
        patch = patch[np.isfinite(patch) & (patch > 0.25)]  # below 25 cm is a sensor glitch: nothing is that close to the camera
        return patch.min() if patch.size else self.depth_image[y, x]

if __name__ == '__main__':
    ImageSubscriber()
    threading.Event().wait()  # callbacks run in the spin thread
