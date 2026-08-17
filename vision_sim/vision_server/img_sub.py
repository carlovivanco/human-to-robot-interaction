import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np


class ImageSubscriber:

    def __init__(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)  # start with a black img
        #rospy.init_node('image_subscriber', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.bridge = CvBridge()
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth(self, x, y):
        """
        Returns depth value at point (x,y) cordinates
        """
        depth = self.depth_image[y, x]
        return depth

    def run_listener(self):
        rospy.spin()


if __name__ == '__main__':
    img_sub = ImageSubscriber()
    img_sub.run_listener()
