import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, StopActionGoal
from franka_msgs.msg import FrankaState
from tf.transformations import quaternion_from_matrix
from sensor_msgs.msg import JointState


class Panda:
    def __init__(self, home_position=[-0.05, 0.0, 0.5], home_orientation=[0, 0, 0, 0]):
        """
        Initialize robot class, defines home position in world coordinates
        home_position: list of size 7
                position.x
                position.y
                position.z
                orientation.x
                orientation.y
                orientation.z
                orientation.w
        """
        rospy.init_node("panda_class_node")
        rospy.Rate(10)
        self.tcp_pub = rospy.Publisher(
            "/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)
        self.tcp_pub_msg = PoseStamped()

        self.move_gripper = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal, queue_size=10)
        self.move_gripper_msg = MoveActionGoal()
        self.grasp_gripper = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal, queue_size=10)
        self.grasp_gripper_msg = GraspActionGoal()
        self.stop_gripper = rospy.Publisher("/franka_gripper/stop/goal", StopActionGoal, queue_size=10)
        self.stop_gripper_msg = StopActionGoal()

        self.tcp_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState, self.position_state_callback, queue_size=10)
        self.tcp_sub_msg = FrankaState()

        self.gripper_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self.gripper_state_callback, queue_size=10)
        self.gripper_sub_msg = JointState()

        # variables for converting img coordinates

        # camera intrinsics (modify based on calibration)
        self.K = np.array([[554.256, 0.0, 320.0],    # fx, 0 , cx
                            [0.0, 554.256, 240.0],    # 0, fy, cy
                            [0.0, 0.0, 1.0]])          # 0, 0, 1

        self.theta_x = 0  # radians rotated on X of camera
        self.R_x = np.array([[1, 0, 0],
                              [0, np.cos(self.theta_x), -np.sin(self.theta_x)],  # rotation on Y axis
                              [0, np.sin(self.theta_x), np.cos(self.theta_x)]])

        self.theta_y = 0.7  # radians rotated on Y of camera
        self.R_y = np.array([[np.cos(self.theta_y), 0, np.sin(self.theta_y)],  # rotation on Y axis
                              [0, 1, 0],
                              [-np.sin(self.theta_y), 0, np.cos(self.theta_y)]])

        self.theta_z = np.pi  # radians rotated on Z of camera
        self.R_z = np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0],  # rotation on Y axis
                              [np.sin(self.theta_z), np.cos(self.theta_z), 0],
                              [0, 0, 1]])

        self.R = np.dot(self.R_z, np.dot(self.R_y, self.R_x))

        self.T = np.array([0.5, 0.0, 0.85])  # camera position from world origin

        # [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        self.position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]
        self.x_offset = -0.35  # robot is placed at a -0.35 X offset
        self.z_offset = 0.0  # offset from gripper tip to gripper origin
        self.pose_tolerance = 0.02  # 2cm pose tolerance
        self.gripper_tolerance = 0.0  # 1cm gripper width tolerance

        self.grasp_gripper_msg.goal.epsilon.inner = 0.005
        self.grasp_gripper_msg.goal.epsilon.outer = 0.005
        self.grasp_gripper_msg.goal.speed = 0.1  # speed when grasping/closing gripper
        self.grasp_gripper_msg.goal.force = 20.0  # force when grasping/closing gripper

        self.move_gripper_msg.goal.speed = 0.1  # speed when opening gripper

        home_position = self.convert_to_robot(home_position)
        self.home_position = (home_position, home_orientation)
        self.status = 'idle'

        self.tcp_pub_msg.header.frame_id = "panda_link0"
        self.tcp_pub_msg.pose.position.x = self.home_position[0][0]
        self.tcp_pub_msg.pose.position.y = self.home_position[0][1]
        self.tcp_pub_msg.pose.position.z = self.home_position[0][2]
        self.tcp_pub_msg.pose.orientation.x = self.home_position[1][0]
        self.tcp_pub_msg.pose.orientation.y = self.home_position[1][1]
        self.tcp_pub_msg.pose.orientation.z = self.home_position[1][2]
        self.tcp_pub_msg.pose.orientation.w = self.home_position[1][3]
        rospy.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)

        self.wait_for_pose(home_position)

    def convert_to_robot(self, pose):
        """
        Convert world coordinates into robot coordinates for controller input
        """
        pose[0] -= self.x_offset  # adjust X coordinates
        pose[2] -= self.z_offset  # adjust Z coordinates
        return pose

    def convert_to_world(self, pose):
        """
        Convert robot coordinates into world coordinates for user
        """
        pose[0] += self.x_offset  # adjust X coordinates

        return pose

    def reset(self):
        """
        Resets gripper and returns robot to [0,0,0] position.
        """
        # move robot to home position
        self.tcp_pub_msg.header.frame_id = "panda_link0"
        self.tcp_pub_msg.pose.position.x = self.home_position[0][0]
        self.tcp_pub_msg.pose.position.y = self.home_position[0][1]
        self.tcp_pub_msg.pose.position.z = self.home_position[0][2]
        self.tcp_pub_msg.pose.orientation.x = self.home_position[1][0]
        self.tcp_pub_msg.pose.orientation.y = self.home_position[1][1]
        self.tcp_pub_msg.pose.orientation.z = self.home_position[1][2]
        self.tcp_pub_msg.pose.orientation.w = self.home_position[1][3]
        rospy.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)

        # open gripper
        self.stop_gripper.publish(self.stop_gripper_msg)
        self.move_gripper_msg.goal.width = 0.08
        rospy.sleep(0.2)
        self.move_gripper.publish(self.move_gripper_msg)

    def return_to_home(self):
        """
        Returns to home, no action
        """
        self.tcp_pub_msg.header.frame_id = "panda_link0"
        self.tcp_pub_msg.pose.position.x = self.home_position[0][0]
        self.tcp_pub_msg.pose.position.y = self.home_position[0][1]
        self.tcp_pub_msg.pose.position.z = self.home_position[0][2]
        self.tcp_pub_msg.pose.orientation.x = self.home_position[1][0]
        self.tcp_pub_msg.pose.orientation.y = self.home_position[1][1]
        self.tcp_pub_msg.pose.orientation.z = self.home_position[1][2]
        self.tcp_pub_msg.pose.orientation.w = self.home_position[1][3]
        rospy.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)
        self.wait_for_pose(self.home_position[0])

    def move_to_pose(self, pose, orientation=[0.0, 0.0, 0.0, 0.0]):
        """
        Moves robot's end-effector to defined pose and orientation.
        Input only robot coordinates
        """
        self.tcp_pub_msg.header.frame_id = "panda_link0"
        self.tcp_pub_msg.pose.position.x = pose[0]
        self.tcp_pub_msg.pose.position.y = pose[1]
        self.tcp_pub_msg.pose.position.z = pose[2]
        self.tcp_pub_msg.pose.orientation.x = orientation[0]
        self.tcp_pub_msg.pose.orientation.y = orientation[1]
        self.tcp_pub_msg.pose.orientation.z = orientation[2]
        self.tcp_pub_msg.pose.orientation.w = orientation[3]
        rospy.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)

        self.wait_for_pose(pose)

    def pick_at(self, pose, clearance=0.15, pick_width=0.03):
        """
        Moves to a specific world coordinate, grips, and then raises to clearance level (default 15cm)
        """
        pose = self.convert_to_robot(pose)

        self.release()
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose)
        pose[2] -= clearance  # lower clearance distance to Z coordinate
        self.move_to_pose(pose)
        self.grasp(pick_width)
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose)

    def release_at(self, pose, clearance=0.15):
        """
        Moves to a specific world coordinate, lowers to clearance level (default 15cm) and then releases gripper
        """
        pose = self.convert_to_robot(pose)

        self.grasp()
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose)
        pose[2] -= clearance  # lower clearance distance to Z coordinate
        self.move_to_pose(pose)
        self.release()
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose)

    def grasp(self, distance=0.03):
        """
        Closes gripper to clearance (default 3cm) distance to grasp an object.
        """
        if self.gripper_state > distance:
            self.grasp_gripper_msg.goal.width = distance
            rospy.sleep(0.2)
            self.grasp_gripper.publish(self.grasp_gripper_msg)
            print('waiting for gripper')
            self.wait_for_gripper(distance)
            print('done for gripper')

    def release(self, distance=0.08):
        """
        Stops grasp and opens gripper to clearance (default 8cm) distance.
        """
        if self.gripper_state < distance:
            # self.stop_gripper.publish(self.stop_gripper_msg)
            # rospy.sleep(0.2)
            self.move_gripper_msg.goal.width = distance
            self.move_gripper_msg.goal.speed = 0.1
            rospy.sleep(0.2)
            self.move_gripper.publish(self.move_gripper_msg)
            self.wait_for_gripper(distance)

    def get_status(self):
        """
        Returns robot status
        """
        return self.status

    def gripper_state_callback(self, msg):
        self.gripper_state = np.round(msg.position[0] * 2, 2)  # assumes simmetry, multiply by 2 to get total width opening

    def get_gripper_state(self):
        """
        Returns gripper's opening width in mts.
        """
        return self.gripper_state

    def position_state_callback(self, msg):
        x_pos = msg.O_T_EE[12]
        y_pos = msg.O_T_EE[13]
        z_pos = msg.O_T_EE[14]
        transform_matrix = np.array(msg.O_T_EE).reshape(4, 4, order='F')

        position = np.round([x_pos, y_pos, z_pos], 2)
        quaternion = np.array(np.round(quaternion_from_matrix(transform_matrix), 2))
        self.current_position = (position, quaternion)

    def get_position(self):
        """
        Returns world coordinates current position as tuple:
              -array of size 3 for positions
              -list of size 4 for orientations
        """
        return self.current_position

    def wait_for_pose(self, pose):
        """
        Loop until current_position = published pose with defined tolerance
        """
        while not (np.allclose(self.current_position[0], pose, atol=self.pose_tolerance)):
            # loop
            pass

    def wait_for_gripper(self, width):
        """
        Loop until gripper is set
        """
        while not (np.allclose(self.gripper_state, width, atol=self.gripper_tolerance)):
            # loop
            pass

    def cam_to_world(self, center):
        """
        Converts X,Y,Z coordinates from camera into world coordinates from robot input
        """
        u, v, Z = center

        fx, fy = self.K[0, 0], self.K[1, 1]  # focal lengths
        cx, cy = self.K[0, 2], self.K[1, 2]  # principal point

        Xc = Z
        Yc = -(u - cx) * Z / fx
        Zc = -(v - cy) * Z / fy

        camera_coords = np.array([Xc, Yc, Zc])
        # camera_coords = np.array([[Xc], [Yc], [0.44]]) # depth harcoded for now

        world_coords = self.R @ camera_coords + self.T
        # world_coords[2] -= 0.03 # offset distance for a better gripper position

        return world_coords

    def run_listener(self):
        rospy.spin()


if __name__ == '__main__':
    panda = Panda()
    panda.run_listener()
