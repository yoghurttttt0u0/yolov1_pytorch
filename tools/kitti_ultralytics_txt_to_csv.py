"""
Convert Ultralytics YOLO format (txt) to VOC-like CSV format for the KITTI dataset.
"""

import os
from pathlib import Path
import csv
import yaml
from PIL import Image


def load_class_mapping(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    names = data['names']
    
    # ensure keys are int
    class_map = {int(k): v for k, v in names.items()}
    return class_map



def yolo_to_voc(xc, yc, w, h, img_w, img_h):
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    xmin = xc - w / 2
    ymin = yc - h / 2
    xmax = xc + w / 2
    ymax = yc + h / 2

    # clamp
    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymax = max(0, min(ymax, img_h - 1))

    return int(xmin), int(ymin), int(xmax), int(ymax)


def convert_one(image_path, label_path, output_csv, class_map):
    img = Image.open(image_path)
    img_w, img_h = img.size

    rows = []

    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])

                xmin, ymin, xmax, ymax = yolo_to_voc(
                    xc, yc, w, h, img_w, img_h
                )

                class_name = class_map.get(class_id, "unknown")

                rows.append([
                    class_name,
                    xmin, ymin, xmax, ymax
                ])

    # write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "object",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ])
        writer.writerows(rows)


def convert_split(images_dir, labels_dir, targets_dir, class_map):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    targets_dir = Path(targets_dir)

    targets_dir.mkdir(parents=True, exist_ok=True)

    images = list(images_dir.glob("*.*"))

    for i, img_path in enumerate(images):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        output_csv = targets_dir / f"{stem}.csv"

        convert_one(img_path, label_path, output_csv, class_map)

        if i % 500 == 0:
            print(f"Processed {i}/{len(images)}")


def main():
    base = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\kitti"

    yaml_path = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\kitti\kitti.yaml"

    class_map = load_class_mapping(yaml_path)

    # TRAIN
    convert_split(
        images_dir=os.path.join(base, "train", "images"),
        labels_dir=os.path.join(base, "labels", "train"),
        targets_dir=os.path.join(base, "train", "targets"),
        class_map=class_map
    )

    # VAL
    convert_split(
        images_dir=os.path.join(base, "val", "images"),
        labels_dir=os.path.join(base, "labels", "val"),
        targets_dir=os.path.join(base, "val", "targets"),
        class_map=class_map
    )


if __name__ == "__main__":
    main()