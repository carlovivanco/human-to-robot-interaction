"""Download REAL photos of the four tools from Open Images V7 and write them as a YOLO dataset.

  python3 scripts/fetch_real_tools.py --per-class 400 --out vision_sim/dataset_real

Hammer / Screwdriver / Wrench come with human-drawn boxes (bbox annotations CC BY 4.0, images mostly CC BY 2.0).
Open Images has no boxes for "Pliers": the photos tagged pliers are saved to <out>/pliers_unlabeled and boxed by
`scripts/autolabel_pliers.py`. Only the standard library is needed.
"""
import argparse
import csv
import io
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CLASSES = ["Hammer", "Screwdriver", "Wrench", "Pliers"]  # = SpeechRecognition.tools
MIDS = {"/m/03l9g": 0, "/m/01bms0": 1, "/m/01j5ks": 2}  # boxed classes
PLIERS = {"/m/02qtr5", "/m/06y28c", "/m/0cqs9q", "/m/03npntq", "/m/0cqt3q", "/m/03npplq", "/m/0bmc8vs"}
BASE = "https://storage.googleapis.com/openimages"
BBOX = f"{BASE}/v6/oidv6-train-annotations-bbox.csv"
LABELS = f"{BASE}/v7/oidv7-train-annotations-human-imagelabels.csv"
IMG = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"


def rows(url):
    return csv.DictReader(io.TextIOWrapper(urllib.request.urlopen(url, timeout=60), encoding="utf-8"))


def boxed_images(per_class):
    """{image_id: [(cls, xmin, xmax, ymin, ymax)]} until every class has per_class images; rows are sorted by image."""
    out, count, cur, cur_id = {}, [0, 0, 0], [], None

    def flush():
        if cur:
            out[cur_id] = list(cur)
            for c in {b[0] for b in cur}:
                count[c] += 1
            print(f"\rscanning boxes: images per class {count}", end="", flush=True)

    for r in rows(BBOX):
        if r["ImageID"] != cur_id:
            flush()
            if min(count) >= per_class:
                return out
            cur, cur_id = [], r["ImageID"]
        # real, single, non-drawing instances only; a class already full is skipped to keep the balance
        if r["LabelName"] in MIDS and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0" \
                and count[MIDS[r["LabelName"]]] < per_class:
            cur.append((MIDS[r["LabelName"]], *(float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))))
    flush()
    return out


def pliers_images(n):
    ids = []
    for r in rows(LABELS):
        if r["LabelName"] in PLIERS and r["Confidence"] == "1":
            ids.append(r["ImageID"])
            print(f"\rscanning pliers: {len(ids)}", end="", flush=True)
            if len(ids) >= n:
                break
    return ids


def download(args):
    image_id, path = args
    if os.path.exists(path):
        return True
    try:
        urllib.request.urlretrieve(IMG.format(image_id), path)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--out", default="vision_sim/dataset_real")
    a = ap.parse_args()
    random.seed(0)
    for d in ("images/train", "images/val", "labels/train", "labels/val", "pliers_unlabeled"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)

    boxes = boxed_images(a.per_class)
    print(f"\n{len(boxes)} images with boxes")
    jobs = [(i, f"{a.out}/images/{'val' if random.random() < 0.1 else 'train'}/{i}.jpg") for i in boxes]
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(download, jobs))
    kept = 0
    for (i, path), good in zip(jobs, ok):
        if not good:
            continue
        lab = path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        with open(lab, "w") as f:
            for c, x0, x1, y0, y1 in boxes[i]:
                f.write(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")
        kept += 1
    print(f"wrote {kept} labelled images")

    ids = pliers_images(a.per_class)
    with ThreadPoolExecutor(16) as ex:
        n = sum(ex.map(download, [(i, f"{a.out}/pliers_unlabeled/{i}.jpg") for i in ids]))
    print(f"\n{n} pliers photos in {a.out}/pliers_unlabeled (no boxes yet)")
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n")
        f.writelines(f"  {k}: {c}\n" for k, c in enumerate(CLASSES))


if __name__ == "__main__":
    main()
