"""Cartesian target -> smooth joint position commands for the simulated FR3.

Gazebo's Cartesian impedance controller oscillates, so the arm is position controlled instead. This node keeps the
topic the impedance controller used, so panda_class.py is unchanged:

  /cartesian_impedance_example_controller/equilibrium_pose (PoseStamped, fr3_link0 frame)  ->  IK on the URDF
  -> /arm_position_controller/commands (Float64MultiArray).

The TCP follows a straight line to the target at MAX_LIN m/s (MAX_ANG rad/s for the orientation); every tick solves a
few IK iterations from the previous solution. The arm first goes to the ready pose in joint space. Targets the arm
cannot reach (IK does not converge at the target) are ignored with a warning.
"""
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

BASE, TIP = "fr3_link0", "fr3_hand_tcp"
Q_HOME = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])  # Franka "ready" pose: tool pointing down
MAX_SPEED = 1.5  # rad/s per joint, only for the first move to the ready pose
MAX_LIN, MAX_ANG = 0.6, 2.0  # TCP speed: m/s and rad/s
RATE = 50


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rot_axis(axis, q):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def rotvec(R):
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-6:
        return v / 2
    if np.pi - th < 1e-3:  # half turn: sin(th) vanishes, take the axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        axis *= np.sign(v) + (v == 0)
        return th * axis / np.linalg.norm(axis)
    return th / (2 * np.sin(th)) * v


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Chain:
    """FK/IK over the URDF joints between BASE and TIP."""

    def __init__(self, urdf):
        joints = {j.find("child").get("link"): j for j in ET.fromstring(urdf).iter("joint") if j.find("child") is not None}
        path, link = [], TIP
        while link != BASE:
            path.append(joints[link])
            link = joints[link].find("parent").get("link")
        self.joints = path[::-1]
        self.names = [j.get("name") for j in self.joints if j.get("type") in ("revolute", "continuous")]
        self.lo = np.array([float(j.find("limit").get("lower")) for j in self.joints if j.get("name") in self.names])
        self.hi = np.array([float(j.find("limit").get("upper")) for j in self.joints if j.get("name") in self.names])
        self.steps = []
        for j in self.joints:
            o = j.find("origin")
            A = np.eye(4)
            A[:3, :3] = rot_rpy(*map(float, o.get("rpy", "0 0 0").split()))
            A[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
            moving = j.get("name") in self.names
            self.steps.append((A, np.array(list(map(float, j.find("axis").get("xyz").split()))) if moving else None))

    def fk(self, q):
        T, i = np.eye(4), 0
        for A, axis in self.steps:
            T = T @ A
            if axis is not None:
                R = np.eye(4)
                R[:3, :3] = rot_axis(axis, q[i])
                T, i = T @ R, i + 1
        return T

    def ik(self, p, R, q0, iters=150, lam=0.02):
        q = np.clip(q0, self.lo, self.hi)
        for _ in range(iters):
            T = self.fk(q)
            e = np.concatenate([p - T[:3, 3], rotvec(R @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 3e-3:
                break
            J = np.empty((6, len(q)))
            for k in range(len(q)):
                d = np.zeros(len(q))
                d[k] = 1e-6
                T2 = self.fk(q + d)
                J[:, k] = np.concatenate([(T2[:3, 3] - T[:3, 3]), rotvec(T2[:3, :3] @ T[:3, :3].T)]) / 1e-6
            Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
            dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ (0.3 * (Q_HOME - q))  # null space keeps the elbow near ready
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        T = self.fk(q)
        return q, np.linalg.norm(T[:3, 3] - p), np.linalg.norm(rotvec(R @ T[:3, :3].T))


class IKNode:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("ik_node")
        self.chain = self.q = self.q_cmd = self.p_cmd = self.R_cmd = self.p_goal = self.R_goal = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(String, "/robot_description", self.on_description, latched)
        self.node.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.node.create_subscription(PoseStamped, "/cartesian_impedance_example_controller/equilibrium_pose",
                                      self.on_target, 10)
        self.pub = self.node.create_publisher(Float64MultiArray, "/arm_position_controller/commands", 10)
        self.node.create_timer(1 / RATE, self.tick)

    def on_description(self, msg):
        self.chain = Chain(msg.data)

    def on_joints(self, msg):
        if self.chain and all(n in msg.name for n in self.chain.names):
            self.q = np.array([msg.position[msg.name.index(n)] for n in self.chain.names])

    def on_target(self, msg):
        if self.p_cmd is None:  # still going to the ready pose
            return
        p, o = msg.pose.position, msg.pose.orientation
        p, R = np.array([p.x, p.y, p.z]), quat_to_R(o.x, o.y, o.z, o.w)
        if np.linalg.norm(p - self.p_goal) < 1e-3 and np.linalg.norm(rotvec(R @ self.R_goal.T)) < 1e-3:
            return  # Panda republishes the same target until it is reached: do not solve it again
        q, perr, rerr = self.chain.ik(p, R, self.q_cmd)  # reachable at the end point?
        if perr > 0.01 or rerr > 0.05:
            self.node.get_logger().warn(f"target ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) unreachable "
                                        f"(err {perr * 1000:.0f} mm, {np.degrees(rerr):.0f} deg), ignored")
            return
        self.p_goal, self.R_goal = p, R

    def tick(self):
        if self.q is None:
            return
        if self.q_cmd is None:  # start from where the arm is
            self.q_cmd = self.q.copy()
        if self.p_cmd is None:  # joint-space move to the ready pose, then switch to Cartesian
            step = MAX_SPEED / RATE
            self.q_cmd = self.q_cmd + np.clip(Q_HOME - self.q_cmd, -step, step)
            if np.abs(Q_HOME - self.q_cmd).max() < 1e-3:
                T = self.chain.fk(self.q_cmd)
                self.p_cmd, self.R_cmd = T[:3, 3], T[:3, :3]
                self.p_goal, self.R_goal = self.p_cmd.copy(), self.R_cmd.copy()
        else:
            d = self.p_goal - self.p_cmd
            n = np.linalg.norm(d)
            p = self.p_goal if n < MAX_LIN / RATE else self.p_cmd + d / n * MAX_LIN / RATE
            w = rotvec(self.R_goal @ self.R_cmd.T)
            a = np.linalg.norm(w)
            R = self.R_goal if a < MAX_ANG / RATE else rot_axis(w / a, MAX_ANG / RATE) @ self.R_cmd
            q, perr, rerr = self.chain.ik(p, R, self.q_cmd, iters=8)
            if perr < 0.005 and rerr < 0.03:  # otherwise hold: never send a jump
                self.p_cmd, self.R_cmd, self.q_cmd = p, R, q
        self.pub.publish(Float64MultiArray(data=self.q_cmd.tolist()))


if __name__ == "__main__":
    n = IKNode()
    rclpy.spin(n.node)
