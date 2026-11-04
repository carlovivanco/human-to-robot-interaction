"""Build a YOLO dataset from the running Gazebo demo (needs `ros2 launch sim_ros2/demo.launch.py`).

Moves the tool models to random poses with `gz service set_pose`, grabs the demo camera image and labels each tool
from its known box geometry (parsed from world.sdf) projected through the camera model. Two kinds of images:
tools lying on the table, and (--hand-share) one tool resting on the open hand of the demo, where the pose is read
back from Gazebo after it settles, since the palm is tilted. Nothing is placed under the hand.

    export GZ_PARTITION=dataset ROS_DOMAIN_ID=77      # in BOTH terminals (launch and this script)
    DEMO_HEADLESS=1 ros2 launch sim_ros2/demo.launch.py
    python3 sim_ros2/make_dataset.py --n 600 --out vision_sim/dataset

The isolated partition is mandatory: `set_pose` reaches every Gazebo with a world called "demo" on the machine, so
without it the script scrambles the tools of any demo you are running at the same time.
"""
import argparse
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["Hammer", "Screwdriver", "Wrench", "Pliers"]  # = SpeechRecognition.tools; sdf model names are lower case
# camera model, keep in sync with Panda.K / Panda.R / Panda.T in panda_class.py
FX = FY = 554.256
CX, CY, W, H = 320.0, 240.0, 640, 480
T = np.array([0.5, 0.0, 0.85])
TH_Y, TH_Z = 0.7, np.pi
RY = np.array([[np.cos(TH_Y), 0, np.sin(TH_Y)], [0, 1, 0], [-np.sin(TH_Y), 0, np.cos(TH_Y)]])
RZ = np.array([[np.cos(TH_Z), -np.sin(TH_Z), 0], [np.sin(TH_Z), np.cos(TH_Z), 0], [0, 0, 1]])
R = RZ @ RY
TABLE_TOP = 0.2
X_RANGE, Y_RANGE, MIN_SEP = (-0.08, 0.16), (-0.28, 0.28), 0.2  # inside the robot's reach and the camera's view
HAND_XY, HAND_CLEAR = np.array([0.06, 0.0]), (0.14, 0.085)  # hand centre in world.sdf, half-size of the free zone under it
ON_HAND_Z = 0.35  # drop height over the palm; the tool settles on the collision plate


def project(points):
    """world points (N,3) -> pixel (N,2), depth (N)"""
    c = (points - T) @ R  # camera frame: x forward, y left, z up
    return np.stack([CX - c[:, 1] * FX / c[:, 0], CY - c[:, 2] * FY / c[:, 0]], axis=1), c[:, 0]


def tool_corners():
    """model-local box corners per tool, read from world.sdf (visual poses are translation only)"""
    out = {}
    for m in ET.parse(os.path.join(HERE, "world.sdf")).getroot().iter("model"):
        if m.get("name").capitalize() not in CLASSES:
            continue
        pts = []
        for v in m.iter("visual"):
            xyz = np.array(v.find("pose").text.split()[:3], float)
            half = np.array(v.find("geometry/box/size").text.split(), float) / 2
            pts += [xyz + half * np.array(s) for s in np.ndindex(2, 2, 2) for s in [np.array(s) * 2 - 1]]
        out[m.get("name").capitalize()] = np.array(pts)
    assert set(out) == set(CLASSES), f"missing tool models in world.sdf: {set(CLASSES) - set(out)}"
    return out


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def label(corners, pos, Rm):
    """YOLO line fields (cx, cy, w, h normalised) or None if the tool is not clearly visible"""
    world = corners @ Rm.T + pos
    uv, depth = project(world)
    if (depth <= 0.1).any():
        return None
    lo, hi = uv.min(0), uv.max(0)
    clo, chi = np.clip(lo, 0, [W, H]), np.clip(hi, 0, [W, H])
    full, seen = np.prod(hi - lo), np.prod(chi - clo)
    if (chi - clo).min() < 8 or seen < 0.9 * full:
        return None
    return [*((clo + chi) / 2 / [W, H]), *((chi - clo) / [W, H])]


def yaw_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def read_poses():
    """{model name: (position, rotation matrix)} of every model, from Gazebo's pose stream"""
    out = subprocess.run(["gz", "topic", "-e", "-t", "/world/demo/pose/info", "-n", "1"], capture_output=True, text=True,
                         timeout=20).stdout
    poses = {}
    for blk in out.split("pose {")[1:]:
        name = re.search(r'name: "([^"]+)"', blk)
        parts = {k: re.search(k + r" \{([^}]*)\}", blk) for k in ("position", "orientation")}
        if not name or not all(parts.values()):
            continue
        # protobuf text omits zero fields
        vec = {k: {c: float(v) for c, v in re.findall(r"([xyzw]): ([-0-9.e]+)", m.group(1))} for k, m in parts.items()}
        p, o = vec["position"], vec["orientation"]
        poses.setdefault(name.group(1), (np.array([p.get(c, 0.0) for c in "xyz"]),
                                         quat_to_R(o.get("x", 0.0), o.get("y", 0.0), o.get("z", 0.0), o.get("w", 0.0))))
    return poses


def set_pose(name, x, y, z, yaw):
    req = f'name: "{name}" position {{x: {x} y: {y} z: {z}}} orientation {{z: {np.sin(yaw / 2)} w: {np.cos(yaw / 2)}}}'
    subprocess.run(["gz", "service", "-s", "/world/demo/set_pose", "--reqtype", "gz.msgs.Pose",
                    "--reptype", "gz.msgs.Boolean", "--timeout", "2000", "--req", req],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="vision_sim/dataset")
    ap.add_argument("--hand-share", type=float, default=0.4, help="share of the images with one tool resting on the hand")
    ap.add_argument("--settle", type=float, default=0.8, help="seconds for the tools to settle after set_pose")
    a = ap.parse_args()
    if not os.environ.get("GZ_PARTITION") or not os.environ.get("ROS_DOMAIN_ID"):
        raise SystemExit("set GZ_PARTITION and ROS_DOMAIN_ID (same values as the launch terminal), see the file header")

    # the optical axis must hit the image centre
    uv, _ = project((T + R @ [1.0, 0, 0])[None]); assert np.allclose(uv, [[CX, CY]])

    rclpy.init()
    node = rclpy.create_node("make_dataset")
    bridge, frame, fresh = CvBridge(), [None], threading.Event()

    def on_image(msg):
        frame[0] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        fresh.set()
    node.create_subscription(Image, "/camera/color/image_raw", on_image, 1)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    corners = tool_corners()
    for split in ("train", "val"):
        os.makedirs(f"{a.out}/images/{split}", exist_ok=True)
        os.makedirs(f"{a.out}/labels/{split}", exist_ok=True)

    kept = 0
    for i in range(a.n):
        hand_cls = CLASSES[i % len(CLASSES)] if random.random() < a.hand_share else None  # every tool gets its turn
        picked = [hand_cls] if hand_cls else []
        picked += random.sample([c for c in CLASSES if c != hand_cls], random.randint(0 if hand_cls else 1, 2))
        table = []  # (class, x, y, yaw)
        for cls in picked:
            if cls == hand_cls:
                continue
            for _ in range(50):
                p = (random.uniform(*X_RANGE), random.uniform(*Y_RANGE))
                if abs(p[0] - HAND_XY[0]) < HAND_CLEAR[0] and abs(p[1] - HAND_XY[1]) < HAND_CLEAR[1]:
                    continue  # nothing under the hand
                if all(np.hypot(p[0] - q[1], p[1] - q[2]) >= MIN_SEP for q in table):
                    table.append((cls, *p, random.uniform(0, 2 * np.pi)))
                    break
            else:
                break
        moves = [(cls.lower(), x, y, TABLE_TOP + 0.002, yaw) for cls, x, y, yaw in table]
        if hand_cls:
            hx, hy = HAND_XY + [random.uniform(-0.03, 0.03), random.uniform(-0.02, 0.02)]
            moves.append((hand_cls.lower(), hx, hy, ON_HAND_Z, random.uniform(-0.4, 0.4) + random.choice([0, np.pi])))
        used = {t[0] for t in table} | ({hand_cls} if hand_cls else set())
        moves += [(cls.lower(), 1.5, 0.0, 0.5, 0.0) for cls in set(CLASSES) - used]  # out of view
        with ThreadPoolExecutor(4) as ex:  # each `gz service` call costs about a second
            list(ex.map(lambda m: set_pose(*m), moves))
        time.sleep(a.settle + (1.0 if hand_cls else 0.0))
        fresh.clear()
        if not fresh.wait(5):
            raise SystemExit("no image on /camera/color/image_raw, is the demo launch running?")

        lines = []
        for cls, x, y, yaw in table:
            box = label(corners[cls], np.array([x, y, TABLE_TOP]), yaw_R(yaw))
            if not box:
                break  # a visible tool without a label would teach the model that it is background
            else:
                lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in box))
        if hand_cls:
            pos, Rm = read_poses().get(hand_cls.lower(), (None, None))
            if pos is None or pos[2] < 0.3 or np.linalg.norm(pos[:2] - HAND_XY) > 0.12:
                continue  # it slid off the palm
            box = label(corners[hand_cls], pos, Rm)
            if not box:
                continue
            lines.append(f"{CLASSES.index(hand_cls)} " + " ".join(f"{v:.6f}" for v in box))
        if not lines or len(lines) < len(table) + bool(hand_cls):
            continue
        split = "val" if i % 10 == 0 else "train"
        cv2.imwrite(f"{a.out}/images/{split}/{i:05d}.jpg", frame[0])
        open(f"{a.out}/labels/{split}/{i:05d}.txt", "w").write("\n".join(lines) + "\n")
        kept += 1
        print(f"\r{i + 1}/{a.n} kept {kept}", end="", flush=True)

    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n")
        f.writelines(f"  {k}: {c}\n" for k, c in enumerate(CLASSES))
    print(f"\nwrote {kept} images to {a.out}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
