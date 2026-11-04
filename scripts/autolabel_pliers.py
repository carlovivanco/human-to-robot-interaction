"""Box the pliers photos fetched by fetch_real_tools.py with YOLOE-26 (open-vocabulary, text prompt "pliers").

  YOLO_AUTOINSTALL=False python3 scripts/autolabel_pliers.py --out vision_sim/dataset_real

Keeps only photos where a box reaches --conf, and writes them as class 3 (Pliers) into images/ + labels/.
Review a few with any YOLO viewer: auto-labels are good, not perfect.
"""
import argparse
import glob
import os
import random
import shutil

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # ultralytics otherwise pip-installs (numpy 2) into the venv
from ultralytics import YOLOE  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision_sim", "models")
PLIERS_CLASS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vision_sim/dataset_real")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--weights", default="yoloe-26s-seg.pt")
    a = ap.parse_args()
    random.seed(0)

    cwd = os.getcwd()
    os.chdir(MODELS)  # ultralytics downloads the weights and the text encoder into the cwd
    model = YOLOE(a.weights)
    model.set_classes(["pliers"], model.get_text_pe(["pliers"]))
    os.chdir(cwd)

    kept = total = 0
    for path in sorted(glob.glob(f"{a.out}/pliers_unlabeled/*.jpg")):
        total += 1
        boxes = model.predict(path, conf=a.conf, verbose=False)[0].boxes
        if len(boxes) == 0:
            continue
        split = "val" if random.random() < 0.1 else "train"
        name = os.path.basename(path)[:-4]
        shutil.copy(path, f"{a.out}/images/{split}/{name}.jpg")
        with open(f"{a.out}/labels/{split}/{name}.txt", "w") as f:
            for x, y, w, h in boxes.xywhn.tolist():
                f.write(f"{PLIERS_CLASS} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        kept += 1
    print(f"boxed {kept} of {total} pliers photos")


if __name__ == "__main__":
    main()
