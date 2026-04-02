"""Remove COCO entries with no objects to avoid training issues.

Some COCO images have no annotated objects; these samples are deleted along with their CSV labels.
"""

import os
import csv

img_dir = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\yolov1_pytorch\data\COCO\test\images" 
annot_dir = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\yolov1_pytorch\data\COCO\test\targets"

removed = 0

for file in os.listdir(annot_dir):
    if not file.endswith(".csv"):
        continue

    annot_path = os.path.join(annot_dir, file)
    img_path = os.path.join(img_dir, file.replace(".csv", ".jpg"))  # If it's .png, change here

    with open(annot_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None) 
        has_object = any(True for _ in reader)

    if not has_object:
        print(f"Removing empty sample: {file}")
        os.remove(annot_path)

        if os.path.exists(img_path):
            os.remove(img_path)

        removed += 1

print(f"\nTotal removed: {removed}")