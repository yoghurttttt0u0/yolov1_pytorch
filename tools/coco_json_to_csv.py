"""Convert COCO JSON annotations to per-image CSV labels and optionally copy images.

- Reads COCO train annotations and maps category IDs to names.
- Converts COCO bbox format [x,y,w,h] to [xmin,ymin,xmax,ymax].
- Writes one CSV per image with valid boxes.
"""

import json
import os
import csv
import shutil
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# COCO train annotation JSON path
json_path = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\COCO\instances_train2017.json"

# output root folder path
image_dir = BASE_DIR / "data" / "COCO" / "train" / "images"
output_target_dir = BASE_DIR / "data" / "COCO" / "train" / "targets"

# Set to None to process the full dataset
# Change this to an integer like 1000 if you want a limited run
max_images = None

# If True, copy source images into the output folder
copy_images = False

# If True, skip images that end up with no valid boxes
skip_empty_images = False


os.makedirs(output_target_dir, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

print("JSON loaded successfully.")
print(f"Number of images in JSON: {len(images)}")
print(f"Number of annotations in JSON: {len(annotations)}")
print(f"Number of categories in JSON: {len(categories)}")


image_id_to_info = {img["id"]: img for img in images}
category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

anns_by_image = defaultdict(list)
for ann in annotations:
    anns_by_image[ann["image_id"]].append(ann)

print("Mappings built successfully.")

processed_count = 0
written_csv_count = 0
copied_image_count = 0
skipped_no_valid_bbox = 0
skipped_missing_image = 0

for img in images:
    if max_images is not None and processed_count >= max_images:
        break

    image_id = img["id"]
    file_name = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]

    ann_list = anns_by_image.get(image_id, [])
    valid_rows = []

    for ann in ann_list:
        bbox = ann.get("bbox", None)

        if bbox is None or len(bbox) != 4:
            continue

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            continue

        # Convert COCO bbox [x, y, w, h] to [xmin, ymin, xmax, ymax]
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_width), xmax)
        ymax = min(float(img_height), ymax)

        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))

        xmin = max(0, min(xmin, img_width))
        ymin = max(0, min(ymin, img_height))
        xmax = max(0, min(xmax, img_width))
        ymax = max(0, min(ymax, img_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        category_id = ann["category_id"]
        class_name = category_id_to_name[category_id]

        valid_rows.append([class_name, xmin, ymin, xmax, ymax])

    if len(valid_rows) == 0 and skip_empty_images:
        skipped_no_valid_bbox += 1
        continue

    src_img_path = os.path.join(image_dir, file_name)
    dst_img_path = os.path.join(image_dir, file_name)

    if not os.path.exists(src_img_path):
        print(f"[WARNING] Source image not found: {src_img_path}")
        skipped_missing_image += 1
        continue

    if copy_images:
        shutil.copy2(src_img_path, dst_img_path)
        copied_image_count += 1

    csv_name = os.path.splitext(file_name)[0] + ".csv"
    csv_path = os.path.join(output_target_dir, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(valid_rows)

    written_csv_count += 1
    processed_count += 1

    if processed_count % 1000 == 0:
        print(f"[INFO] Processed {processed_count} images...")


print("\n=== DONE ===")
print(f"Images processed successfully : {processed_count}")
print(f"CSV files written             : {written_csv_count}")
print(f"Images copied                 : {copied_image_count}")
print(f"Images skipped (no valid box) : {skipped_no_valid_bbox}")
print(f"Images skipped (missing file) : {skipped_missing_image}")
print(f"Output target folder          : {output_target_dir}")
print("\n=== COCO categories ===")
for cat in sorted(categories, key=lambda x: x["id"]):
    print(f'{cat["id"]}: {cat["name"]}')