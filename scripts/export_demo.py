#!/usr/bin/env python3
"""Export portfolio assets (hri.mp4 / hri.jpg) from a REAL recording of the demo.

Subcommands
  convert   Re-encode an existing recording to 1280x720, 30 fps, H.264 yuv420p,
            +faststart, no audio, < 8 MB, plus a 1280x720 JPG (quality 85).
  annotate  Run the repo's own vision pipeline (VisionClass.process_frame) offline
            over a recorded camera feed (video file or rosbag), write the annotated
            video, then convert it.

Nothing is synthesised: if the input or the models are missing the script fails.
Record the demo with scripts/record_demo.sh.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DEFAULT = os.path.join(ROOT, "portfolio_export")
W, H, FPS = 1280, 720, 30
MAX_BYTES = 8 * 1000 * 1000
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"


def die(msg):
    sys.exit(f"error: {msg}")


def need(tool):
    if shutil.which(tool) is None:
        die(f"'{tool}' not found in PATH")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def probe(path):
    out = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    return json.loads(out)


FIT = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
       f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")


def encode(src, dst, start, duration, crf):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", src, "-t", str(duration),
         "-vf", f"{FIT},fps={FPS},format=yuv420p", "-an",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(crf),
         "-movflags", "+faststart", dst])


def write_jpg(video, dst, at):
    png = tempfile.mkstemp(suffix=".png")[1]
    try:
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", video,
             "-frames:v", "1", png])
        try:
            from PIL import Image
            Image.open(png).convert("RGB").save(dst, "JPEG", quality=85)
        except ImportError:
            try:
                import cv2
                cv2.imwrite(dst, cv2.imread(png), [cv2.IMWRITE_JPEG_QUALITY, 85])
            except ImportError:
                die("need Pillow or opencv-python to write the JPG at quality 85 "
                    "(pip install pillow)")
    finally:
        os.remove(png)


def verify(mp4, jpg):
    info = probe(mp4)
    v = [s for s in info["streams"] if s["codec_type"] == "video"]
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    v = v[0]
    num, den = map(int, v["avg_frame_rate"].split("/"))
    fps = num / den
    dur = float(info["format"]["duration"])
    size = os.path.getsize(mp4)
    with open(mp4, "rb") as f:
        head = f.read(65536)
    faststart = 0 <= head.find(b"moov") < (head.find(b"mdat") if b"mdat" in head else 1 << 30)
    checks = {
        "1280x720": (v["width"], v["height"]) == (W, H),
        "30 fps": abs(fps - FPS) < 0.01,
        "h264": v["codec_name"] == "h264",
        "yuv420p": v["pix_fmt"] == "yuv420p",
        "no audio": not a,
        "5-10 s": 5 <= dur <= 10.05,
        "< 8 MB": size < MAX_BYTES,
        "faststart": faststart,
    }
    from_jpg = probe(jpg)["streams"][0]
    checks["jpg 1280x720"] = (from_jpg["width"], from_jpg["height"]) == (W, H)
    print(f"{mp4}: {dur:.2f}s, {size / 1e6:.2f} MB")
    for k, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        die("output does not meet the spec")


def cmd_convert(a):
    need("ffmpeg"), need("ffprobe")
    if not os.path.isfile(a.input):
        die(f"input not found: {a.input}")
    if not 5 <= a.duration <= 10:
        die("--duration must be between 5 and 10 seconds")
    total = float(probe(a.input)["format"].get("duration") or 0)
    if total and a.start + a.duration > total + 0.05:
        die(f"recording is {total:.1f}s; start+duration ({a.start + a.duration:.1f}s) exceeds it")
    os.makedirs(a.out, exist_ok=True)
    mp4, jpg = os.path.join(a.out, "hri.mp4"), os.path.join(a.out, "hri.jpg")
    for crf in range(a.crf, 37, 3):
        encode(a.input, mp4, a.start, a.duration, crf)
        if os.path.getsize(mp4) < MAX_BYTES:
            break
        print(f"crf {crf} gives {os.path.getsize(mp4) / 1e6:.1f} MB, retrying")
    at = a.frame_at if a.frame_at is not None else a.duration / 2
    write_jpg(mp4, jpg, at)
    verify(mp4, jpg)


# ---- annotate -------------------------------------------------------------

def stub_ros():
    """Let vision_sim import without ROS; the ImageSubscriber is never instantiated."""
    for name in ("rospy", "cv_bridge", "sensor_msgs", "sensor_msgs.msg"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                m = types.ModuleType(name)
                m.Image = object
                m.CvBridge = object
                m.Subscriber = lambda *a, **k: None
                sys.modules[name] = m


class FrameDepth:
    """Stand-in for ImageSubscriber: only get_depth() is used by process_frame."""
    def __init__(self):
        self.depth = None

    def get_depth(self, x, y):
        if self.depth is None:
            return 0
        h, w = self.depth.shape[:2]
        return self.depth[min(max(y, 0), h - 1), min(max(x, 0), w - 1)]


def frames_from_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        die(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, None, fps
    cap.release()


def frames_from_bag(path):
    try:
        import rosbag
        from cv_bridge import CvBridge
    except ImportError:
        die("--bag needs a ROS environment (source /opt/ros/noetic/setup.bash)")
    bridge = CvBridge()
    depth = None
    stamps = []
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(topics=[COLOR_TOPIC, DEPTH_TOPIC]):
            if topic == DEPTH_TOPIC:
                depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            elif depth is not None:
                stamps.append(t.to_sec())
                yield bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), depth, None
    if len(stamps) < 2:
        die(f"no color+depth frames found in {path}")


def cmd_annotate(a):
    need("ffmpeg"), need("ffprobe")
    for p, label in ((a.yolo, "--yolo"), (a.gesture, "--gesture")):
        if not os.path.isfile(p):
            die(f"{label} model not found: {p}")
    if bool(a.input) == bool(a.bag):
        die("give exactly one of --input or --bag")
    src = a.input or a.bag
    if not os.path.isfile(src):
        die(f"recording not found: {src}")
    import cv2
    stub_ros()
    sys.path.insert(0, ROOT)
    from vision_sim.vision_server.main import VisionClass

    vc = VisionClass.__new__(VisionClass)  # skip __init__: it subscribes to ROS topics
    from ultralytics import YOLO
    vc.yolo_model = YOLO(a.yolo)
    vc.gesture_model_path = a.gesture
    recognizer = vc.initialize_gesture_recognizer()
    vc.img = FrameDepth()

    frames = frames_from_bag(src) if a.bag else frames_from_video(src)
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    annotated = os.path.join(raw, "annotated.mp4")
    writer, n = None, 0
    fps = a.fps
    stats = {"frames": 0, "frames_with_hand": 0, "frames_with_tool": 0, "gestures": {}}
    for frame, depth, vfps in frames:
        fps = vfps or fps
        vc.img.depth = depth
        ts = int(n * 1000 / fps)
        out, objs, tip, center, gesture, _ = vc.process_frame(frame, recognizer, vc.yolo_model, ts)
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(annotated, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)
        n += 1
        stats["frames"] += 1
        stats["frames_with_hand"] += center is not None
        stats["frames_with_tool"] += bool(objs)
        if gesture["gesture"]:
            stats["gestures"][gesture["gesture"]] = stats["gestures"].get(gesture["gesture"], 0) + 1
    if writer is None:
        die("no frames processed")
    writer.release()
    # descriptive counts only (no ground truth => these are NOT accuracy figures)
    with open(os.path.join(a.out, "annotate_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"annotated {n} frames -> {annotated}; counts in annotate_stats.json")
    a.input = annotated
    cmd_convert(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", default=OUT_DEFAULT, help="output dir (default: portfolio_export/)")
        sp.add_argument("--start", type=float, default=0.0, help="start second in the recording")
        sp.add_argument("--duration", type=float, default=8.0, help="clip length, 5-10 s")
        sp.add_argument("--frame-at", type=float, default=None,
                        help="second inside the clip for hri.jpg (default: middle)")
        sp.add_argument("--crf", type=int, default=23, help="starting x264 CRF")

    c = sub.add_parser("convert", help="re-encode a recording to the portfolio spec")
    c.add_argument("--input", required=True)
    common(c)
    c.set_defaults(fn=cmd_convert)

    an = sub.add_parser("annotate", help="run the repo's vision pipeline on a recorded camera feed")
    an.add_argument("--input", help="recorded camera video")
    an.add_argument("--bag", help="rosbag with %s and %s" % (COLOR_TOPIC, DEPTH_TOPIC))
    an.add_argument("--yolo", default=os.path.join(ROOT, "vision_sim/models/tools-100-epochs.pt"))
    an.add_argument("--gesture", default=os.path.join(ROOT, "vision_sim/models/gesture_recognizer.task"))
    an.add_argument("--fps", type=float, default=FPS, help="fps when the source has none (bag)")
    common(an)
    an.set_defaults(fn=cmd_annotate)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
