from pathlib import Path
from collections import defaultdict
import csv

BASE_DIR = Path(__file__).resolve().parent.parent 

DATASET_DIR = BASE_DIR / "data" / "KITTI"
TRAIN_TARGET_DIR = DATASET_DIR / "train" / "targets"
TEST_TARGET_DIR = DATASET_DIR / "test" / "targets"


CLASS_NAMES = [
    "car",
    "van", 
    "truck", 
    "pedestrian", 
    "Person_sitting", 
    "cyclist", 
    "tram",
    "misc"]

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
        print(f"{cls:20s}: {counts[cls]}")

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
    print("Checking KITTI csv labels by split...")

    print("\nDataset class definition:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{i}: {cls}")

    train_stats = analyze_split(TRAIN_TARGET_DIR, "train")
    test_stats = analyze_split(TEST_TARGET_DIR, "test")
    compare_train_test(train_stats, test_stats)


if __name__ == "__main__":
    main()