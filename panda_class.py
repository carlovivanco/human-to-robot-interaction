import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

ROBOT = "fr3"  # franka_description robot_type: the Panda ("fer") fails Gazebo's inertia check
BASE_FRAME = f"{ROBOT}_link0"
TCP_FRAME = f"{ROBOT}_hand_tcp"
FINGER_JOINT = f"{ROBOT}_finger_joint1"


class Panda:
    def __init__(self, home_position=[-0.05, 0.0, 0.5], home_orientation=[1, 0, 0, 0]):
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
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("panda_class_node")
        self.tcp_pub = self.node.create_publisher(
            PoseStamped, "/cartesian_impedance_example_controller/equilibrium_pose", 10)
        self.tcp_pub_msg = PoseStamped()

        # sim gripper: position controller on the finger joint (see launch/demo.launch.py)
        self.gripper_pub = self.node.create_publisher(
            Float64MultiArray, "/gripper_position_controller/commands", 10)
        self.gripper_sub = self.node.create_subscription(
            JointState, "/joint_states", self.gripper_state_callback, 10)
        self.gripper_state = 0.0

        # end-effector pose comes from TF (robot_state_publisher), same in sim and on the robot
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.current_position = (np.zeros(3), np.zeros(4))
        self.node.create_timer(0.02, self.position_state_callback)

        # callbacks must keep running while the main thread busy-waits
        executor = SingleThreadedExecutor()  # own executor: the global one can only be spun once per process
        executor.add_node(self.node)
        threading.Thread(target=executor.spin, daemon=True).start()

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
        self.z_offset = 0.02  # the depth gives the top of the tool: grasp 2 cm below it (the fingertips stop on the table at TCP z~0.21)
        self.pose_tolerance = 0.02  # 2cm pose tolerance
        self.gripper_tolerance = 0.0  # 1cm gripper width tolerance

        home_position = self.convert_to_robot(home_position)
        self.home_position = (home_position, home_orientation)
        self.status = 'idle'

        self.tcp_pub_msg.header.frame_id = BASE_FRAME
        self.tcp_pub_msg.pose.position.x = float(self.home_position[0][0])
        self.tcp_pub_msg.pose.position.y = float(self.home_position[0][1])
        self.tcp_pub_msg.pose.position.z = float(self.home_position[0][2])
        self.tcp_pub_msg.pose.orientation.x = float(self.home_position[1][0])
        self.tcp_pub_msg.pose.orientation.y = float(self.home_position[1][1])
        self.tcp_pub_msg.pose.orientation.z = float(self.home_position[1][2])
        self.tcp_pub_msg.pose.orientation.w = float(self.home_position[1][3])
        time.sleep(0.2)
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
        self.tcp_pub_msg.header.frame_id = BASE_FRAME
        self.tcp_pub_msg.pose.position.x = float(self.home_position[0][0])
        self.tcp_pub_msg.pose.position.y = float(self.home_position[0][1])
        self.tcp_pub_msg.pose.position.z = float(self.home_position[0][2])
        self.tcp_pub_msg.pose.orientation.x = float(self.home_position[1][0])
        self.tcp_pub_msg.pose.orientation.y = float(self.home_position[1][1])
        self.tcp_pub_msg.pose.orientation.z = float(self.home_position[1][2])
        self.tcp_pub_msg.pose.orientation.w = float(self.home_position[1][3])
        time.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)

        # open gripper
        self.set_gripper(0.08)

    def return_to_home(self):
        """
        Returns to home, no action
        """
        self.tcp_pub_msg.header.frame_id = BASE_FRAME
        self.tcp_pub_msg.pose.position.x = float(self.home_position[0][0])
        self.tcp_pub_msg.pose.position.y = float(self.home_position[0][1])
        self.tcp_pub_msg.pose.position.z = float(self.home_position[0][2])
        self.tcp_pub_msg.pose.orientation.x = float(self.home_position[1][0])
        self.tcp_pub_msg.pose.orientation.y = float(self.home_position[1][1])
        self.tcp_pub_msg.pose.orientation.z = float(self.home_position[1][2])
        self.tcp_pub_msg.pose.orientation.w = float(self.home_position[1][3])
        time.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)
        self.wait_for_pose(self.home_position[0])

    def move_to_pose(self, pose, orientation=[1.0, 0.0, 0.0, 0.0]):
        """
        Moves robot's end-effector to defined pose and orientation.
        Input only robot coordinates
        """
        self.tcp_pub_msg.header.frame_id = BASE_FRAME
        self.tcp_pub_msg.pose.position.x = float(pose[0])
        self.tcp_pub_msg.pose.position.y = float(pose[1])
        self.tcp_pub_msg.pose.position.z = self._nudge_z = float(pose[2])
        self._nudged = False
        self.tcp_pub_msg.pose.orientation.x = float(orientation[0])
        self.tcp_pub_msg.pose.orientation.y = float(orientation[1])
        self.tcp_pub_msg.pose.orientation.z = float(orientation[2])
        self.tcp_pub_msg.pose.orientation.w = float(orientation[3])
        time.sleep(0.2)
        self.tcp_pub.publish(self.tcp_pub_msg)

        self.wait_for_pose(pose)

    def pick_at(self, pose, clearance=0.15, pick_width=0.0, yaw=0.0):  # 0: command fully closed so the fingers squeeze the tool
        """
        Moves to a specific world coordinate, grips, and then raises to clearance level (default 15cm).
        yaw: rotation (rad, about world z) of the tool's long axis from world x; the gripper turns to close across it.
        """
        pose = self.convert_to_robot(pose)
        q = [np.cos(yaw / 2), np.sin(yaw / 2), 0.0, 0.0]  # Rz(yaw) * the pointing-down home orientation (1, 0, 0, 0)

        self.release()
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose, q)
        pose[2] -= clearance  # lower clearance distance to Z coordinate
        self.move_to_pose(pose, q)
        self.grasp(pick_width)
        pose[2] += clearance  # add clearance distance to Z coordinate
        self.move_to_pose(pose, q)

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
            self.set_gripper(distance)
            print('waiting for gripper')
            self.wait_for_gripper(distance)
            print('done for gripper')

    def release(self, distance=0.08):
        """
        Stops grasp and opens gripper to clearance (default 8cm) distance.
        """
        if self.gripper_state < distance:
            self.set_gripper(distance)
            self.wait_for_gripper(distance)

    def get_status(self):
        """
        Returns robot status
        """
        return self.status

    def set_gripper(self, width):
        """
        Commands total gripper opening in mts (each finger travels width / 2).
        """
        time.sleep(0.2)
        self.gripper_pub.publish(Float64MultiArray(data=[width / 2]))
        # in Gazebo a still arm leaves the fingers frozen (a closing command is not applied until the arm moves):
        # a 1 cm nudge (up, then back down on the next call so it is never a repeated target) gets them going
        m = self.tcp_pub_msg
        if m.header.frame_id:
            self._nudged = not getattr(self, '_nudged', False)
            m.pose.position.z = self._nudge_z + (0.01 if self._nudged else 0.0)
            self.tcp_pub.publish(m)

    def gripper_state_callback(self, msg):
        if FINGER_JOINT in msg.name:
            # assumes simmetry, multiply by 2 to get total width opening
            self.gripper_state = np.round(msg.position[msg.name.index(FINGER_JOINT)] * 2, 2)

    def get_gripper_state(self):
        """
        Returns gripper's opening width in mts.
        """
        return self.gripper_state

    def position_state_callback(self):
        try:
            t = self.tf_buffer.lookup_transform(BASE_FRAME, TCP_FRAME, Time())
        except Exception:  # TF not available yet
            return
        p, q = t.transform.translation, t.transform.rotation
        self.current_position = (np.round([p.x, p.y, p.z], 2), np.round([q.x, q.y, q.z, q.w], 2))

    def get_position(self):
        """
        Returns world coordinates current position as tuple:
              -array of size 3 for positions
              -list of size 4 for orientations
        """
        return self.current_position

    def wait_for_pose(self, pose, timeout=20.0):
        """
        Loop until current_position = published pose with defined tolerance
        """
        t0 = last = time.time()
        while not (np.allclose(self.current_position[0], pose, atol=self.pose_tolerance)):
            if time.time() - last > 0.5:  # the first message can be lost while ROS 2 discovers the subscriber
                self.tcp_pub.publish(self.tcp_pub_msg)
                last = time.time()
            if time.time() - t0 > timeout:
                print(f"warning: pose {np.round(pose, 2)} not reached after {timeout:.0f}s, continuing")
                return
            time.sleep(0.01)

    def wait_for_gripper(self, width, timeout=5.0):
        """
        Loop until the gripper is set or has stopped moving (a grasped tool stops the fingers short of the target width)
        """
        t0 = last_change = time.time()
        last = self.gripper_state
        while not np.allclose(self.gripper_state, width, atol=self.gripper_tolerance):
            now = time.time()
            if self.gripper_state != last:
                last, last_change = self.gripper_state, now
            if now - last_change > 0.5 or now - t0 > timeout:
                return
            time.sleep(0.01)

    def cam_to_world(self, center):
        """
        Converts u, v, depth from the camera (scalars or equal-length arrays) into world coordinates
        """
        u, v, Z = center

        fx, fy = self.K[0, 0], self.K[1, 1]  # focal lengths
        cx, cy = self.K[0, 2], self.K[1, 2]  # principal point

        camera_coords = np.array([Z, -(u - cx) * Z / fx, -(v - cy) * Z / fy])
        return (self.R @ camera_coords).T + self.T

    def tool_position(self, box, depth_image, table_z=0.2, max_z=0.5, thickness=0.06):
        """
        World position of a detected tool from the depth pixels of its box: the highest height layer with a
        meaningful number of pixels is the top of the tool (stray pixels of the arm do not count, the table and a palm
        below it are lower), and the pixels within `thickness` under it are the tool, also when it lies tilted on a
        palm. x, y = their centroid, z = their mean height (the top at the centroid). None if there are too few pixels.
        Returns (position, yaw): yaw = angle of the tool's long axis from world x (PCA of those pixels, folded to
        (-pi/2, pi/2]; 0 if the tool is not clearly elongated).
        """
        x1, y1, x2, y2 = (int(c) for c in box)
        v, u = np.mgrid[max(y1, 0):y2, max(x1, 0):x2]
        Z = depth_image[v, u].astype(float)
        ok = np.isfinite(Z) & (Z > 0)
        pts = self.cam_to_world((u[ok], v[ok], Z[ok]))
        pts = pts[(pts[:, 2] > table_z + 0.008) & (pts[:, 2] < max_z)]
        if len(pts) < 20:
            return None
        edges = np.arange(table_z + 0.008, max_z + 0.008, 0.008)
        hist, edges = np.histogram(pts[:, 2], bins=edges)
        top = edges[np.nonzero(hist >= max(20, 0.2 * hist.max()))[0].max() + 1]
        tool = pts[pts[:, 2] > top - thickness]
        # z only from the upper 2 cm: on a tilted palm the 6 cm band includes palm pixels and would pull z down into it
        z = tool[tool[:, 2] > top - 0.02, 2].mean()
        if z > table_z + 0.1:  # resting on a hand: grasp 1.2 cm deeper (found by trial; at the usual depth the fingers only graze it)
            z -= 0.012
        xy = tool[:, :2]
        w, v = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
        yaw = np.arctan2(v[1, 1], v[0, 1]) if w[1] > 1.5 ** 2 * w[0] else 0.0
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        return np.array([*xy.mean(axis=0), z]), yaw

    def shutdown(self):
        self.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    panda = Panda()
    threading.Event().wait()  # callbacks run in the spin thread
