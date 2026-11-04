"""Test bench for the real SO-101 arm (LeRobot) with ordinary RGB cameras, no depth.

Run it with the LeRobot environment (conda activate lerobot), from the repo root:

  python3 scripts/so101_demo.py selftest                              # kinematics check, no hardware
  python3 scripts/so101_demo.py calibrate --port /dev/ttyACM0 --cam 0  # once: image pixels -> table plane
  python3 scripts/so101_demo.py pick      --port /dev/ttyACM0 --cam 0  # click an object (and a place); no AI needed
  python3 scripts/so101_demo.py run       --port /dev/ttyACM0 --cam 0 --yolo vision_sim/models/<weights>.pt

Without depth, a pixel is turned into a table position with a homography learnt by `calibrate`: you touch the
table with the gripper tip at a few spots and click each spot in the camera image. Objects are assumed to lie on
the table, and the handover hand is assumed to be at table height in the image (it is released --handover-z above).
Only the fixed (environment) camera is used. The wrist camera is not needed.
"""
import argparse
import json
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.expanduser("~/Desktop/Robotics/Prentice AI/assets/so101/so101_new_calib.urdf")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CHAIN = JOINTS + ["gripper_frame_joint"]
DOWN = np.array([0.0, 0.0, -1.0])
FPS = 30


def die(msg):
    sys.exit(f"error: {msg}")


# ---- kinematics (numpy only: LeRobot's placo does not import with numpy 2 / eigenpy 3.12) --------------------

def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class Kin:
    """FK/IK of the SO-101 straight from its URDF. Joint angles in radians, base_link frame."""

    def __init__(self, urdf=URDF):
        if not os.path.isfile(urdf):
            die(f"URDF not found: {urdf} (use --urdf)")
        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint") if j.find("parent") is not None}
        self.origin = []
        for n in CHAIN:
            o = joints[n].find("origin")
            A = np.eye(4)
            A[:3, :3] = rot_rpy(*map(float, o.get("rpy").split()))
            A[:3, 3] = list(map(float, o.get("xyz").split()))
            self.origin.append(A)
        lim = [joints[n].find("limit") for n in JOINTS]
        self.lo = np.array([float(l.get("lower")) for l in lim])
        self.hi = np.array([float(l.get("upper")) for l in lim])

    def fk(self, q):
        T = np.eye(4)
        for i, A in enumerate(self.origin):
            T = T @ A
            if i < len(JOINTS):
                c, s = np.cos(q[i]), np.sin(q[i])
                T = T @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return T  # gripper tip frame; its z axis is the approach direction

    def _f(self, q, w):
        T = self.fk(q)
        return np.concatenate([T[:3, 3], w * T[:3, 2]])

    def ik(self, target, q0, down_weight=0.3, iters=200, lam=0.02):
        """Damped least squares on tip position + (softly) a vertical approach. Returns q, pos_err_m, tilt_deg."""
        goal = np.concatenate([target, down_weight * DOWN])
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        for _ in range(iters):
            e = goal - self._f(q, down_weight)
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            J = np.stack([(self._f(q + d, down_weight) - self._f(q - d, down_weight)) / 2e-6
                          for d in np.eye(len(q)) * 1e-6], axis=1)
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e), self.lo, self.hi)
        T = self.fk(q)
        return q, float(np.linalg.norm(T[:3, 3] - target)), float(np.degrees(np.arccos(np.clip(T[:3, 2] @ DOWN, -1, 1))))


def cmd_selftest(a):
    kin, rng = Kin(a.urdf), np.random.default_rng(0)
    for _ in range(50):  # IK must recover the position of random reachable poses
        q = rng.uniform(kin.lo, kin.hi) * 0.6
        _, err, _ = kin.ik(kin.fk(q)[:3, 3], q + rng.normal(0, 0.1, 5), down_weight=0.0)
        assert err < 1e-3, f"IK round trip off by {err * 1000:.1f} mm"
    print("IK round trip OK (50 random poses, < 1 mm)")
    q0 = np.array([0, -1.0, 1.0, 0.5, 0.0])
    for x in (0.12, 0.18, 0.24):
        q, err, tilt = kin.ik(np.array([x, 0.0, 0.01]), q0)
        print(f"table point x={x:.2f} y=0 z=0.01: pos err {err * 1000:.1f} mm, tilt from vertical {tilt:.0f} deg")


# ---- hardware ---------------------------------------------------------------------------------------------

class Cam(threading.Thread):
    """Plain RGB camera. get_depth() exists only because VisionClass asks for it."""

    def __init__(self, index):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, self.frame = self.cap.read()
        if not ok:
            die(f"cannot read camera {index} (see `lerobot-find-cameras opencv`)")
        self.start()

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame

    def get_depth(self, x, y):
        return 0


class Arm:
    def __init__(self, a):
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            die("lerobot not found: conda activate lerobot")
        self.robot = SO101Follower(SO101FollowerConfig(
            port=a.port, id=a.robot_id, use_degrees=True, max_relative_target=a.max_step))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            die(f"arm '{a.robot_id}' is not calibrated: lerobot-calibrate --robot.type=so101_follower "
                f"--robot.port={a.port} --robot.id={a.robot_id}")
        self.kin, self.speed = Kin(a.urdf), a.speed
        self.grip_open, self.grip_closed = a.grip_open, a.grip_closed
        self.q, self.grip = self.read()
        self.home = (self.q.copy(), self.grip)  # the pose the arm starts in is the safe pose

    def read(self):
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS]), obs["gripper.pos"]

    def move(self, q_deg, grip):
        """Straight joint-space interpolation at --speed deg/s."""
        q0, g0 = self.read()
        steps = max(1, int(max(np.abs(q_deg - q0).max(), abs(grip - g0)) / (self.speed / FPS)))
        for k in range(1, steps + 1):
            t = k / steps
            action = {f"{j}.pos": float(q0[i] + t * (q_deg[i] - q0[i])) for i, j in enumerate(JOINTS)}
            action["gripper.pos"] = float(g0 + t * (grip - g0))
            self.robot.send_action(action)
            time.sleep(1 / FPS)
        self.q, self.grip = q_deg, grip

    def plan(self, points, roll=0.0):
        """IK for a list of tip positions in order; refuses unreachable or too tilted poses before anything moves."""
        out, q = [], np.radians(self.q)
        for p in points:
            q, err, tilt = self.kin.ik(np.asarray(p, float), q)
            if err > 0.006 or tilt > 35:
                die(f"point {np.round(p, 3)} not reachable (err {err * 1000:.0f} mm, tilt {tilt:.0f} deg)")
            out.append(np.append(np.degrees(q[:4]), roll))
        return out

    def close(self):
        self.move(*self.home)
        self.robot.disconnect()


def pix2xy(H, u, v):
    return cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), H)[0, 0].astype(float)


def load_calib(path):
    if not os.path.isfile(path):
        die(f"{path} not found: run `calibrate` first")
    c = json.load(open(path))
    return np.array(c["H"], np.float32), c["table_z"]


# ---- motions ------------------------------------------------------------------------------------------------

def pick(arm, xy, z0, a, roll=0.0):
    above, at = [*xy, z0 + a.clearance], [*xy, z0 + a.grasp_z]
    q_above, q_at = arm.plan([above, at], roll)
    arm.move(q_above, arm.grip_open)
    arm.move(q_at, arm.grip_open)
    arm.move(q_at, arm.grip_closed)
    time.sleep(0.6)
    arm.move(q_above, arm.grip_closed)


def place(arm, xy, z, a, roll=0.0):
    above, at = [*xy, z + a.clearance], [*xy, z]
    q_above, q_at = arm.plan([above, at], roll)
    arm.move(q_above, arm.grip_closed)
    arm.move(q_at, arm.grip_closed)
    arm.move(q_at, arm.grip_open)
    time.sleep(0.4)
    arm.move(q_above, arm.grip_open)


# ---- subcommands --------------------------------------------------------------------------------------------

def cmd_calibrate(a):
    arm, cam = Arm(a), Cam(a.cam)
    print("Torque goes OFF in 3 s: SUPPORT THE ARM so it does not fall.")
    time.sleep(3)
    arm.robot.bus.disable_torque()
    px, xyz = [], []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            T = arm.kin.fk(np.radians(arm.read()[0]))
            px.append((x, y)), xyz.append(T[:3, 3])
            print(f"point {len(px)}: pixel ({x},{y}) <-> tip {np.round(T[:3, 3], 3)} m")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    print("Touch the table with the gripper tip, then click that spot in the image. Use >= 6 spread-out spots.\n"
          "Keys: s = solve and save, u = undo last point, q = quit")
    while True:
        img = cam.frame.copy()
        for i, p in enumerate(px):
            cv2.circle(img, p, 5, (0, 255, 0), -1), cv2.putText(img, str(i + 1), p, 0, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibrate", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and px:
            px.pop(), xyz.pop()
        elif k == ord("q"):
            break
        elif k == ord("s"):
            if len(px) < 4:
                print("need at least 4 points")
                continue
            P, X = np.array(px, np.float32), np.array(xyz)
            H, _ = cv2.findHomography(P, X[:, :2].astype(np.float32), 0)
            err = [np.linalg.norm(pix2xy(H, *p) - x[:2]) * 1000 for p, x in zip(P, X)]
            json.dump({"H": H.tolist(), "table_z": float(X[:, 2].mean())}, open(a.calib, "w"))
            print(f"saved {a.calib}: table z {X[:, 2].mean():.3f} m, reprojection error mean {np.mean(err):.1f} mm, "
                  f"max {np.max(err):.1f} mm")
    arm.robot.disconnect()


def click_once(cam, title):
    pt = []
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, lambda e, x, y, *_: pt.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while not pt:
        cv2.imshow(title, cam.frame)
        cv2.waitKey(30)
    cv2.destroyWindow(title)
    return pt[0]


def cmd_pick(a):
    H, z0 = load_calib(a.calib)
    arm, cam = Arm(a), Cam(a.cam)
    try:
        xy = pix2xy(H, *click_once(cam, "click the OBJECT"))
        dest = pix2xy(H, *click_once(cam, "click where to PLACE it (any point)"))
        pick_plan = arm.plan([[*xy, z0 + a.clearance], [*xy, z0 + a.grasp_z]], a.roll)
        arm.plan([[*dest, z0 + a.clearance], [*dest, z0 + a.grasp_z]], a.roll)
        print(f"object at {np.round(xy, 3)} m, place at {np.round(dest, 3)} m; first target joints (deg): "
              f"{np.round(pick_plan[0], 1)}")
        input("Enter = move the arm (Ctrl+C to abort) ")
        pick(arm, xy, z0, a, a.roll)
        place(arm, dest, z0 + a.grasp_z, a, a.roll)
    finally:
        arm.close()


def cmd_run(a):
    H, z0 = load_calib(a.calib)
    sys.path.insert(0, ROOT)
    from vision_sim.cli.combine import SpeechRecognition
    from vision_sim.vision_server.main import VisionClass
    arm, cam = Arm(a), Cam(a.cam)
    command = SpeechRecognition()
    detector = VisionClass(a.yolo, a.gesture, show_camera=True, img=cam)
    command.start()
    detector.start()
    print("Say 'panda, pass me the <tool>'. Only 'pass' is supported on the real arm. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.2)
            tool, action = command.tool, command.action
            if tool is None:
                continue
            command.tool = command.action = None
            obj = next((o for o in getattr(detector, "predicted_classes", []) if o["class"] == tool), None)
            if obj is None or action != "pass":
                print(f"{'no ' + tool + ' in view' if obj is None else 'action ' + str(action) + ' not supported'}")
                continue
            pick(arm, pix2xy(H, *obj["center"][:2]), z0, a)
            print("Waiting for an open palm...")
            t0 = time.time()
            while time.time() - t0 < a.hand_timeout:
                hand = getattr(detector, "center_hand", None)
                if hand and detector.predicted_gesture["gesture"] == "Open_Palm":
                    place(arm, pix2xy(H, *hand), z0 + a.handover_z, a)
                    break
                time.sleep(0.1)
            else:
                print("no open palm, putting the tool back is NOT implemented: it stays in the gripper")
            arm.move(*arm.home)
    finally:
        command.stop(), detector.stop()
        arm.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("selftest", cmd_selftest), ("calibrate", cmd_calibrate), ("pick", cmd_pick), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--urdf", default=URDF)
        if name == "selftest":
            continue
        p.add_argument("--port", default="/dev/ttyACM0")
        p.add_argument("--robot-id", default="my_awesome_follower_arm", help="LeRobot calibration id of the follower")
        p.add_argument("--cam", type=int, default=0, help="index of the FIXED environment camera")
        p.add_argument("--calib", default=os.path.join(ROOT, "so101_calib.json"))
        p.add_argument("--speed", type=float, default=25, help="max joint speed, deg/s")
        p.add_argument("--max-step", type=float, default=10, help="LeRobot safety clip per command, deg")
        p.add_argument("--grip-open", type=float, default=50, help="gripper 0-100")
        p.add_argument("--grip-closed", type=float, default=8)
        p.add_argument("--clearance", type=float, default=0.06, help="hover height above the grasp point, m")
        p.add_argument("--grasp-z", type=float, default=0.012, help="tip height above the table when grasping, m")
        p.add_argument("--roll", type=float, default=0.0, help="wrist roll, deg: jaws must close across the thin side")
        p.add_argument("--handover-z", type=float, default=0.10, help="release height above the table, m")
        p.add_argument("--hand-timeout", type=float, default=30)
        p.add_argument("--yolo")
        p.add_argument("--gesture", default=os.path.join(ROOT, "vision_sim/models/gesture_recognizer.task"))
    a = ap.parse_args()
    if a.cmd == "run" and not a.yolo:
        die("run needs --yolo <weights>: they must be trained on REAL images of your tools")
    a.fn(a)


if __name__ == "__main__":
    main()
