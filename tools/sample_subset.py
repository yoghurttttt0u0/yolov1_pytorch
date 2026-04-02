"""
This script picks a subset of COCO train/val/test samples and copies images+labels to a reduced dataset.
"""

from pathlib import Path
from collections import defaultdict
import csv
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent 

SRC_TRAIN_IMG_DIR = BASE_DIR / "data" / "COCO" / "train" / "images"
SRC_TRAIN_TARGET_DIR = BASE_DIR / "data" / "COCO" / "train" / "targets"
SRC_TEST_IMG_DIR = BASE_DIR / "data" / "COCO" / "test" / "images"
SRC_TEST_TARGET_DIR = BASE_DIR / "data" / "COCO" / "test" / "targets"

DATASET_DIR = BASE_DIR / "data" / "COCOsubset"

OUT_TRAIN_IMG = DATASET_DIR / "train" / "images"
OUT_TRAIN_TGT = DATASET_DIR / "train" / "targets"

OUT_VAL_IMG = DATASET_DIR / "val" / "images"
OUT_VAL_TGT = DATASET_DIR / "val" / "targets"

OUT_TEST_IMG = DATASET_DIR / "test" / "images"
OUT_TEST_TGT = DATASET_DIR / "test" / "targets"

TRAIN_SIZE = 10000
VAL_SIZE = 1000
TEST_SIZE = 2000

RARE_CLASSES = {
    "toaster",
    "hair drier",
    "scissors",
    "microwave",
    "parking meter",
    "bear",
}

RARE_MIN_COUNTS = {
    "train": 40,
    "val": 8,
    "test": 8,
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CLASS_NAMES = [
        "person",         # 0   (original COCO id: 1)
        "bicycle",        # 1   (original COCO id: 2)
        "car",            # 2   (original COCO id: 3)
        "motorcycle",     # 3   (original COCO id: 4)
        "airplane",       # 4   (original COCO id: 5)
        "bus",            # 5   (original COCO id: 6)
        "train",          # 6   (original COCO id: 7)
        "truck",          # 7   (original COCO id: 8)
        "boat",           # 8   (original COCO id: 9)
        "traffic light",  # 9   (original COCO id: 10)
        "fire hydrant",   # 10  (original COCO id: 11)
        "stop sign",      # 11  (original COCO id: 13)
        "parking meter",  # 12  (original COCO id: 14)
        "bench",          # 13  (original COCO id: 15)
        "bird",           # 14  (original COCO id: 16)
        "cat",            # 15  (original COCO id: 17)
        "dog",            # 16  (original COCO id: 18)
        "horse",          # 17  (original COCO id: 19)
        "sheep",          # 18  (original COCO id: 20)
        "cow",            # 19  (original COCO id: 21)
        "elephant",       # 20  (original COCO id: 22)
        "bear",           # 21  (original COCO id: 23)
        "zebra",          # 22  (original COCO id: 24)
        "giraffe",        # 23  (original COCO id: 25)
        "backpack",       # 24  (original COCO id: 27)
        "umbrella",       # 25  (original COCO id: 28)
        "handbag",        # 26  (original COCO id: 31)
        "tie",            # 27  (original COCO id: 32)
        "suitcase",       # 28  (original COCO id: 33)
        "frisbee",        # 29  (original COCO id: 34)
        "skis",           # 30  (original COCO id: 35)
        "snowboard",      # 31  (original COCO id: 36)
        "sports ball",    # 32  (original COCO id: 37)
        "kite",           # 33  (original COCO id: 38)
        "baseball bat",   # 34  (original COCO id: 39)
        "baseball glove", # 35  (original COCO id: 40)
        "skateboard",     # 36  (original COCO id: 41)
        "surfboard",      # 37  (original COCO id: 42)
        "tennis racket",  # 38  (original COCO id: 43)
        "bottle",         # 39  (original COCO id: 44)
        "wine glass",     # 40  (original COCO id: 46)
        "cup",            # 41  (original COCO id: 47)
        "fork",           # 42  (original COCO id: 48)
        "knife",          # 43  (original COCO id: 49)
        "spoon",          # 44  (original COCO id: 50)
        "bowl",           # 45  (original COCO id: 51)
        "banana",         # 46  (original COCO id: 52)
        "apple",          # 47  (original COCO id: 53)
        "sandwich",       # 48  (original COCO id: 54)
        "orange",         # 49  (original COCO id: 55)
        "broccoli",       # 50  (original COCO id: 56)
        "carrot",         # 51  (original COCO id: 57)
        "hot dog",        # 52  (original COCO id: 58)
        "pizza",          # 53  (original COCO id: 59)
        "donut",          # 54  (original COCO id: 60)
        "cake",           # 55  (original COCO id: 61)
        "chair",          # 56  (original COCO id: 62)
        "couch",          # 57  (original COCO id: 63)
        "potted plant",   # 58  (original COCO id: 64)
        "bed",            # 59  (original COCO id: 65)
        "dining table",   # 60  (original COCO id: 67)
        "toilet",         # 61  (original COCO id: 70)
        "tv",             # 62  (original COCO id: 72)
        "laptop",         # 63  (original COCO id: 73)
        "mouse",          # 64  (original COCO id: 74)
        "remote",         # 65  (original COCO id: 75)
        "keyboard",       # 66  (original COCO id: 76)
        "cell phone",     # 67  (original COCO id: 77)
        "microwave",      # 68  (original COCO id: 78)
        "oven",           # 69  (original COCO id: 79)
        "toaster",        # 70  (original COCO id: 80)
        "sink",           # 71  (original COCO id: 81)
        "refrigerator",   # 72  (original COCO id: 82)
        "book",           # 73  (original COCO id: 84)
        "clock",          # 74  (original COCO id: 85)
        "vase",           # 75  (original COCO id: 86)
        "scissors",       # 76  (original COCO id: 87)
        "teddy bear",     # 77  (original COCO id: 88)
        "hair drier",     # 78  (original COCO id: 89)
        "toothbrush"      # 79  (original COCO id: 90)
    ] 

def parse_annotation(csv_path):
    class_counts = defaultdict(int)
    classes = set()
    num_boxes = 0

    with open(csv_path,"r",newline="",encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            row = [x.strip() for x in row if x.strip() != ""]
            if len(row) ==0:
                continue

            # skip the header
            if row[0].lower() in ["object", "class", "label"]:
                continue

            #  object,xmin,ymin,xmax,ymax
            if len(row) != 5:
                continue

            cls = row[0]
            class_counts[cls] += 1
            classes.add(cls)
            num_boxes += 1

        
        if num_boxes == 0:
            dominant_class = None
        else:
            dominant_class = max(class_counts, key=class_counts.get)

        if num_boxes <= 2:
            density_bin = "low"
        elif num_boxes <= 5:
            density_bin = "mid"
        else:
            density_bin = "high"

    return {
        "classes": classes,
        "class_counts": dict(class_counts),
        "num_boxes": num_boxes,
        "dominant_class": dominant_class,
        "density_bin": density_bin,
    }

def build_image_index(image_dir):
    print("Building image index...")

    image_map = {}
    image_files = list(image_dir.glob("*"))

    for i, p in enumerate(image_files):
        if i % 5000 == 0:
            print(f"Indexed {i}/{len(image_files)} images")

        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_map[p.stem] = p

    print(f"Image index built: {len(image_map)} images loaded\n")
    return image_map

def collect_image_metadata(image_dir, target_dir, image_map):
    metadata_list = []

    csv_files = sorted(target_dir.glob("*.csv"))

    for i, csv_path in enumerate(csv_files):
        if i % 2000 == 0:
            print(f"Processing {i}/{len(csv_files)} CSV files")

        stem = csv_path.stem
        image_path = image_map.get(stem, None)

        if image_path is None:
            continue

        ann_info = parse_annotation(csv_path)

        item = {
            "id": stem,
            "image_path": image_path,
            "csv_path": csv_path,
            "classes": ann_info["classes"],
            "class_counts": ann_info["class_counts"],
            "num_boxes": ann_info["num_boxes"],
            "dominant_class": ann_info["dominant_class"],
            "density_bin": ann_info["density_bin"],
        }

        metadata_list.append(item)

    print(f"Finished metadata collection from {target_dir}")
    print(f"Collected {len(metadata_list)} valid image-csv pairs")

    return metadata_list

def compute_box_class_distribution(metadata_list):
    counts = defaultdict(int)
    total_boxes = 0

    for item in metadata_list:
        for cls, cnt in item["class_counts"].items():
            counts[cls] += cnt
            total_boxes += cnt

    percentages = {}
    for cls in CLASS_NAMES:
        if total_boxes > 0:
            percentages[cls] = counts[cls] / total_boxes
        else:
            percentages[cls] = 0.0

    return {
        "counts": dict(counts),
        "percentages": percentages,
        "total_boxes": total_boxes,
    }

def compute_density_distribution(metadata_list):
    density_counts = {"low": 0, "mid": 0, "high": 0}
    total_images = len(metadata_list)

    for item in metadata_list:
        density_counts[item["density_bin"]] += 1

    density_percentages = {}
    for k in ["low", "mid", "high"]:
        if total_images > 0:
            density_percentages[k] = density_counts[k] / total_images
        else:
            density_percentages[k] = 0.0

    return {
        "counts": density_counts,
        "percentages": density_percentages,
        "total_images": total_images,
    }

def reserve_rare_images(metadata_list, rare_classes, min_count, selected_ids=None, split_name="split"):
    if selected_ids is None:
        selected_ids = set()
    else:
        selected_ids = set(selected_ids)

    print(f"\n===== Reserving rare-class images for {split_name} =====")

    for rare_cls in sorted(rare_classes):
        candidates = [item for item in metadata_list if rare_cls in item["classes"]]
        random.shuffle(candidates)

        picked = 0
        for item in candidates:
            if item["id"] not in selected_ids:
                selected_ids.add(item["id"])
                picked += 1
            if picked >= min_count:
                break

        print(
            f"Rare class '{rare_cls}': "
            f"available={len(candidates)}, picked={picked}, target={min_count}"
        )

    print(f"Total reserved rare images in {split_name}: {len(selected_ids)}")
    return selected_ids

def stratified_fill_by_dominant_class(
    metadata_list,
    target_size,
    selected_ids,
    reference_box_dist,
    split_name="split"
):
    selected_ids = set(selected_ids)

    remaining = [
        item for item in metadata_list
        if item["id"] not in selected_ids and item["dominant_class"] is not None
    ]

    groups = defaultdict(list)
    for item in remaining:
        groups[item["dominant_class"]].append(item)

    total_remaining = len(remaining)
    need = target_size - len(selected_ids)

    print(f"\n===== Stratified fill for {split_name} =====")
    print(f"Already selected: {len(selected_ids)}")
    print(f"Need more: {need}")
    print(f"Remaining usable images: {total_remaining}")

    if need <= 0:
        return selected_ids

    quotas = {}
    remainders = []
    assigned = 0

    for cls in CLASS_NAMES:
        available = len(groups.get(cls, []))
        raw_quota = reference_box_dist.get(cls, 0.0) * need
        quota = min(available, int(raw_quota))
        quotas[cls] = quota
        assigned += quota

        if available > quota:
            remainders.append((raw_quota - int(raw_quota), cls))

    leftover = need - assigned
    remainders.sort(reverse=True)

    for frac, cls in remainders:
        if leftover <= 0:
            break
        available = len(groups.get(cls, []))
        current = quotas[cls]
        if current < available:
            quotas[cls] += 1
            leftover -= 1

    print(f"Quota assigned before fallback: {sum(quotas.values())}/{need}")

    for cls in CLASS_NAMES:
        items = groups.get(cls, [])
        if not items:
            continue
        random.shuffle(items)
        take = quotas[cls]
        for item in items[:take]:
            selected_ids.add(item["id"])

    if len(selected_ids) < target_size:
        still_remaining = [item for item in metadata_list if item["id"] not in selected_ids]
        random.shuffle(still_remaining)

        print(f"Fallback random fill: need {target_size - len(selected_ids)} more images")

        for item in still_remaining:
            selected_ids.add(item["id"])
            if len(selected_ids) >= target_size:
                break

    print(f"Final selected for {split_name}: {len(selected_ids)}")
    return selected_ids

def get_selected_items(metadata_list, selected_ids):
    return [item for item in metadata_list if item["id"] in selected_ids]

def adjust_density_distribution(
    selected_items,
    candidate_pool,
    target_size,
    reference_density_dist,
    split_name="split",
    max_swaps=200
):
    print(f"\n===== Density adjustment for {split_name} =====")

    selected_ids = {item["id"] for item in selected_items}

    target_counts = {
        "low": int(reference_density_dist["low"] * target_size),
        "mid": int(reference_density_dist["mid"] * target_size),
        "high": int(reference_density_dist["high"] * target_size),
    }

    current_groups = {"low": [], "mid": [], "high": []}
    for item in selected_items:
        current_groups[item["density_bin"]].append(item)

    pool_groups = {"low": [], "mid": [], "high": []}
    for item in candidate_pool:
        if item["id"] not in selected_ids:
            pool_groups[item["density_bin"]].append(item)

    for k in ["low", "mid", "high"]:
        random.shuffle(current_groups[k])
        random.shuffle(pool_groups[k])

    print("Target density counts:", target_counts)
    print("Current density counts:", {k: len(v) for k, v in current_groups.items()})

    swaps_done = 0

    while swaps_done < max_swaps:
        over_bin = None
        under_bin = None

        for k in ["low", "mid", "high"]:
            if len(current_groups[k]) > target_counts[k]:
                over_bin = k
                break

        for k in ["low", "mid", "high"]:
            if len(current_groups[k]) < target_counts[k] and len(pool_groups[k]) > 0:
                under_bin = k
                break

        if over_bin is None or under_bin is None:
            break

        remove_item = current_groups[over_bin].pop()
        add_item = pool_groups[under_bin].pop()

        selected_ids.remove(remove_item["id"])
        selected_ids.add(add_item["id"])

        pool_groups[over_bin].append(remove_item)
        current_groups[under_bin].append(add_item)
        swaps_done += 1

    print(f"Density swaps done: {swaps_done}")
    print("Adjusted density counts:", {k: len(v) for k, v in current_groups.items()})

    return selected_ids

def summarize_selection(selected_items, split_name="split"):
    print(f"\n===== Summary: {split_name} =====")

    num_images = len(selected_items)
    total_boxes = 0
    class_counts = defaultdict(int)
    density_counts = {"low": 0, "mid": 0, "high": 0}

    for item in selected_items:
        total_boxes += item["num_boxes"]
        density_counts[item["density_bin"]] += 1
        for cls, cnt in item["class_counts"].items():
            class_counts[cls] += cnt

    print(f"Images: {num_images}")
    print(f"Total boxes: {total_boxes}")
    print(f"Avg boxes/image: {total_boxes / num_images:.2f}" if num_images > 0 else "Avg boxes/image: 0")

    print("\nDensity distribution:")
    for k in ["low", "mid", "high"]:
        pct = (density_counts[k] / num_images * 100) if num_images > 0 else 0.0
        print(f"{k:10s}: {density_counts[k]:6d} ({pct:6.2f}%)")

    print("\nTop 15 classes by bbox count:")
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, cnt in sorted_counts[:15]:
        pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0.0
        print(f"{cls:20s}: {cnt:8d} ({pct:6.2f}%)")


def copy_one_file(item, out_img_dir, out_target_dir):
    shutil.copy2(
        item["image_path"],
        out_img_dir / item["image_path"].name
    )
    shutil.copy2(
        item["csv_path"],
        out_target_dir / item["csv_path"].name
    )

def copy_selected_files_fast(selected_items, out_img_dir, out_target_dir, split_name="split", num_workers=8):
    print(f"\n===== Copying {split_name} (multithread) =====")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_target_dir.mkdir(parents=True, exist_ok=True)

    total = len(selected_items)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for item in selected_items:
            futures.append(
                executor.submit(copy_one_file, item, out_img_dir, out_target_dir)
            )

        for i, future in enumerate(as_completed(futures), start=1):
            future.result() 
            if i % 500 == 0 or i == total:
                print(f"{split_name}: {i}/{total} copied")

    print(f"{split_name} copying finished. Total: {total}")

def main():

    print("\n===== BUILD TRAIN METADATA =====")
    train_image_map = build_image_index(SRC_TRAIN_IMG_DIR)
    train_meta = collect_image_metadata(SRC_TRAIN_IMG_DIR, SRC_TRAIN_TARGET_DIR, train_image_map)

    print("\n===== BUILD TEST METADATA =====")
    test_image_map = build_image_index(SRC_TEST_IMG_DIR)
    test_meta = collect_image_metadata(SRC_TEST_IMG_DIR, SRC_TEST_TARGET_DIR, test_image_map)

    train_box_dist = compute_box_class_distribution(train_meta)
    train_density_dist = compute_density_distribution(train_meta)

    test_box_dist = compute_box_class_distribution(test_meta)
    test_density_dist = compute_density_distribution(test_meta)

    print("\nTrain total boxes:", train_box_dist["total_boxes"])
    print("Test total boxes:", test_box_dist["total_boxes"])

    # sample train set
    train_selected_ids = reserve_rare_images(
        train_meta,
        RARE_CLASSES,
        RARE_MIN_COUNTS["train"],
        split_name="train"
    )

    train_selected_ids = stratified_fill_by_dominant_class(
        train_meta,
        TRAIN_SIZE,
        train_selected_ids,
        train_box_dist["percentages"],
        split_name="train"
    )

    train_items = get_selected_items(train_meta, train_selected_ids)

    train_selected_ids = adjust_density_distribution(
        train_items,
        train_meta,
        TRAIN_SIZE,
        train_density_dist["percentages"],
        split_name="train"
    )

    train_items = get_selected_items(train_meta, train_selected_ids)
    summarize_selection(train_items, "train")

    # sample val set
    remaining_train_meta = [item for item in train_meta if item["id"] not in train_selected_ids]

    val_selected_ids = reserve_rare_images(
        remaining_train_meta,
        RARE_CLASSES,
        RARE_MIN_COUNTS["val"],
        split_name="val"
    )

    val_selected_ids = stratified_fill_by_dominant_class(
        remaining_train_meta,
        VAL_SIZE,
        val_selected_ids,
        train_box_dist["percentages"],
        split_name="val"
    )

    val_items = get_selected_items(remaining_train_meta, val_selected_ids)

    val_selected_ids = adjust_density_distribution(
        val_items,
        remaining_train_meta,
        VAL_SIZE,
        train_density_dist["percentages"],
        split_name="val"
    )

    val_items = get_selected_items(remaining_train_meta, val_selected_ids)
    summarize_selection(val_items, "val")

    test_selected_ids = reserve_rare_images(
        test_meta,
        RARE_CLASSES,
        RARE_MIN_COUNTS["test"],
        split_name="test"
    )

    test_selected_ids = stratified_fill_by_dominant_class(
        test_meta,
        TEST_SIZE,
        test_selected_ids,
        test_box_dist["percentages"],
        split_name="test"
    )

    test_items = get_selected_items(test_meta, test_selected_ids)

    test_selected_ids = adjust_density_distribution(
        test_items,
        test_meta,
        TEST_SIZE,
        test_density_dist["percentages"],
        split_name="test"
    )

    test_items = get_selected_items(test_meta, test_selected_ids)
    summarize_selection(test_items, "test")

    copy_selected_files_fast(
        train_items,
        OUT_TRAIN_IMG,
        OUT_TRAIN_TGT,
        split_name="train"
    )

    copy_selected_files_fast(
        val_items,
        OUT_VAL_IMG,
        OUT_VAL_TGT,
        split_name="val"
    )

    copy_selected_files_fast(
        test_items,
        OUT_TEST_IMG,
        OUT_TEST_TGT,
        split_name="test"
    )


if __name__ == "__main__" :
    main()