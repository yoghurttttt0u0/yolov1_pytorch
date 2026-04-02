"""Analyze and validate CSV object detection labels for train/val/test splits.

- Count class occurrences based on CLASS_NAMES.
- Detect invalid rows, unknown classes, and bad bounding boxes.
- Print summary by split and train/test comparison.
"""

from pathlib import Path
from collections import defaultdict
import csv

BASE_DIR = Path(__file__).resolve().parent.parent 

DATASET_DIR = BASE_DIR / "data" / "COCO"
TRAIN_TARGET_DIR = DATASET_DIR / "train" / "targets"
TEST_TARGET_DIR = DATASET_DIR / "test" / "targets"


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

def analyze_split(target_dir: Path, split_name: str):
    counts = defaultdict(int)
    unknown_classes = defaultdict(int)
    total_boxes = 0
    total_files = 0
    bad_rows = []

    csv_files = sorted(target_dir.glob("*.csv"))

    print(f"\n===== {split_name.upper()} =====")
    print(f"Target dir: {target_dir}")
    print(f"Found {len(csv_files)} csv files")

    for csv_file in csv_files:
        total_files += 1
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            for line_idx, row in enumerate(reader, start=1):
                if not row:
                    continue

                row = [x.strip() for x in row if x.strip() != ""]
                if len(row) == 0:
                    continue

                # Skip the header
                if row[0].lower() in ["object", "class", "label"]:
                    continue

                # check the number of columns
                if len(row) != 5:
                    bad_rows.append((str(csv_file), line_idx, row, f"Expected 5 columns, got {len(row)}"))
                    continue

                cls = row[0]
                total_boxes += 1

                if cls in CLASS_NAMES:
                    counts[cls] += 1
                else:
                    unknown_classes[cls] += 1

                try:
                    xmin = float(row[1])
                    ymin = float(row[2])
                    xmax = float(row[3])
                    ymax = float(row[4])

                    if xmax <= xmin or ymax <= ymin:
                        bad_rows.append((str(csv_file), line_idx, row, "Invalid bbox: xmax<=xmin or ymax<=ymin"))
                except ValueError:
                    bad_rows.append((str(csv_file), line_idx, row, "BBox columns are not numeric"))

    print(f"Total files: {total_files}")
    print(f"Total boxes: {total_boxes}")

    print("\nClass distribution:")
    for cls in CLASS_NAMES:
        count = counts[cls]
        if total_boxes > 0:
            percentage = (count / total_boxes) * 100
        else:
            percentage = 0.0

        print(f"{cls:20s}: {count:8d} ({percentage:6.2f}%)")

    print("\nUnknown classes:")
    if unknown_classes:
        for cls, cnt in sorted(unknown_classes.items(), key=lambda x: (-x[1], x[0])):
            print(f"{cls:20s}: {cnt}")
    else:
        print("None √ ")

    print(f"\nBad rows: {len(bad_rows)}")
    if bad_rows:
        print("First 20 bad rows:")
        for file_path, line_idx, row, reason in bad_rows[:20]:
            print(f"{file_path} | line {line_idx} | {reason} | row={row}")

    return {
        "counts": dict(counts),
        "unknown_classes": dict(unknown_classes),
        "bad_rows": bad_rows,
        "total_files": total_files,
        "total_boxes": total_boxes,
    }


def compare_train_test(train_stats, test_stats):
    print("\n===== TRAIN / TEST COMPARISON =====")
    print(f"{'Class':20s} {'Train':>10s} {'Test':>10s} {'Test/Train':>12s}")

    for cls in CLASS_NAMES:
        train_count = train_stats["counts"].get(cls, 0)
        test_count = test_stats["counts"].get(cls, 0)

        if train_count == 0:
            ratio = "N/A"
        else:
            ratio = f"{test_count / train_count:.3f}"

        print(f"{cls:20s} {train_count:10d} {test_count:10d} {ratio:>12s}")


def main():
    print("Checking csv labels by split...")

    print("\nDataset class definition:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{i}: {cls}")

    train_stats = analyze_split(TRAIN_TARGET_DIR, "train")
    test_stats = analyze_split(TEST_TARGET_DIR, "test")
    compare_train_test(train_stats, test_stats)


if __name__ == "__main__":
    main()