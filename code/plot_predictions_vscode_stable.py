import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import PIL.Image as Image
import torch as th
import torchvision.transforms.functional as fT
from torchvision.utils import draw_bounding_boxes

from dataset import VOC_Detection
from evaluate import postprocessing
from model import YOLOv1

# Model Hyperparameters
S = 7
B = 2
D = 448

BASE_DIR = Path(__file__).resolve().parent.parent

# Trained Model Path
TRAINED_MODEL_WEIGHTS = BASE_DIR / "checkpoints" / "kitti_finetuned_model_weights_30.pt"

# KITTI Dataset Directory
PASCAL_VOC_DIR_PATH = BASE_DIR / "data" / "KITTI"

# Save Image Path
ASSETS_DIR = BASE_DIR / "assets_KITTI"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Compute Device (use a GPU if available)
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# Postprocessing Hyperparameters
PROB_THRESHOLD = 0.15
NMS_THRESHOLD = 0.6
CONF_MODE = "objectness"

# OpenCV window name
WINDOW_NAME = "YOLOv1 KITTI Predictions"

# OpenCV key codes (waitKeyEx)
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_ESC = 27
KEY_Q_LOWER = ord("q")
KEY_Q_UPPER = ord("Q")
KEY_S_LOWER = ord("s")
KEY_S_UPPER = ord("S")
KEY_A_LOWER = ord("a")
KEY_A_UPPER = ord("A")
KEY_D_LOWER = ord("d")
KEY_D_UPPER = ord("D")

COLORS = {}

def hex_to_bgr(hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return (rgb[2], rgb[1], rgb[0])

def annotate_img(img: Image.Image, bboxes: th.Tensor) -> Image.Image:
    """
    Annotate the given image based on the given bounding boxes.

    The bounding box is plotted for each object of the image and the corresponding
    label is also written inside the box.
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for bb in bboxes:
        cls = int(bb[0])
        conf = float(bb[1].item())
        x1, y1, x2, y2 = map(int, bb[2:])

        # Prevent coordinate out-of-bounds
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        label_name = VOC_Detection.index2label[cls]
        label = f"{label_name} {conf:.2f}"

        color = hex_to_bgr(VOC_Detection.label_clrs[cls])

        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Adaptive font size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.7, img.shape[1] / 1600))
        font_thickness = 1

        (tw, th_text), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Tags are placed above the box by default; If it doesn't fit, put it at the top of the box
        text_y1 = y1 - th_text - baseline - 6
        text_y2 = y1
        text_x1 = x1
        text_x2 = x1 + tw + 10

        if text_y1 < 0:
            text_y1 = y1
            text_y2 = y1 + th_text + baseline + 6

        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (text_x1, text_y1),
            (text_x2, text_y2),
            color,
            -1
        )
        alpha = 0.85
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        text_org = (text_x1 + 5, text_y2 - baseline - 3)
        cv2.putText(
            img,
            label,
            text_org,
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def predict_and_annotate(model: YOLOv1, img: Image.Image) -> Image.Image:
    """
    Run prediction on one image and return the annotated PIL image.
    """
    w, h = img.size
    x = fT.normalize(
        fT.to_tensor(fT.resize(img, (D, D))),
        mean=[0.4549, 0.4341, 0.4010],
        std=[0.2703, 0.2672, 0.2808],
    ).unsqueeze(0).to(DEVICE)

    with th.no_grad():
        y = model(x)

    bboxes_pred = postprocessing(
        y,
        prob_threshold=PROB_THRESHOLD,
        conf_mode=CONF_MODE,
        nms_threshold=NMS_THRESHOLD,
    )

    # After postprocessing, the bounding box coordinates are scaled for a (D x D) image.
    if bboxes_pred.numel() != 0:
        bboxes_pred[:, [2, 4]] *= w / D
        bboxes_pred[:, [3, 5]] *= h / D

    return annotate_img(img, bboxes_pred)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to an OpenCV BGR numpy array.
    """
    rgb = np.array(img)
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_image_name(dataset: VOC_Detection, index: int) -> str:
    """
    Try to retrieve the current image file name from common dataset attributes.
    Falls back to a generic name if the path list is not exposed.
    """
    candidate_attrs = (
        "img_paths",
        "image_paths",
        "imgs",
        "images",
        "img_files",
        "image_files",
        "samples",
    )

    for attr in candidate_attrs:
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            try:
                item = values[index]
            except Exception:
                continue

            if isinstance(item, (str, Path)):
                return Path(item).name

            if isinstance(item, (tuple, list)) and len(item) > 0:
                first = item[0]
                if isinstance(first, (str, Path)):
                    return Path(first).name

    return f"image_{index}"


def add_status_bar(
    img_bgr: np.ndarray,
    index: int,
    total: int,
    image_name: str,
) -> np.ndarray:
    """
    Add usage instructions and the current image file name under the displayed image.
    """
    h, w = img_bgr.shape[:2]
    bar_h = 56
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = img_bgr
    canvas[h:] = (24, 24, 24)

    # Left side: Picture number
    left_text = f"Image {index + 1}/{total}"
    cv2.putText(
        canvas,
        left_text,
        (14, h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    # Middle: File name
    file_text = f"File: {image_name}"
    cv2.putText(
        canvas,
        file_text,
        (14, h + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 235, 180),
        1,
        cv2.LINE_AA,
    )

    # right side: Operation instructions
    help_text = "A/Prev   D/Next   S Save   Q/Esc Quit"
    text_size, _ = cv2.getTextSize(
        help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    text_w = text_size[0]
    cv2.putText(
        canvas,
        help_text,
        (w - text_w - 14, h + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return canvas


def show_image(
    index: int,
    total: int,
    annotated_img: Image.Image,
    image_name: str,
) -> None:
    """
    Show one annotated image in an OpenCV window.
    """
    img_bgr = pil_to_bgr(annotated_img)
    vis = add_status_bar(img_bgr, index, total, image_name)

    max_w = 1400
    max_h = 900

    h, w = vis.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    if scale < 1.0:
        vis = cv2.resize(
            vis,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    cv2.imshow(WINDOW_NAME, vis)


def setup_evaluation() -> Tuple[YOLOv1, VOC_Detection]:
    """
    Instantiate the model and the KITTI test dataset.
    """
    model = YOLOv1(S=S, B=B, C=VOC_Detection.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(trained_model_weights)
    model.eval()

    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH, split="test")
    return model, test_dataset


def save_image(img: Image.Image, index: int) -> str:
    """
    Save the current annotated image and return the path.
    """
    path = ASSETS_DIR / f"annot_img_{index}.jpg"
    img.save(path)
    return str(path)


def main() -> None:
    """
    Browse prediction results interactively in a stable OpenCV window.

    Controls:
    - Left arrow or A: previous image
    - Right arrow or D: next image
    - S: save current image
    - Q or Esc: quit
    """
    model, test_dataset = setup_evaluation()

    if len(test_dataset) == 0:
        raise RuntimeError("The test dataset is empty.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)

    index = 0

    while True:
        pil_img, _, image_name = test_dataset[index]
        # image_name = get_image_name(test_dataset, index)
        annot_img = predict_and_annotate(model, pil_img)
        show_image(index, len(test_dataset), annot_img, image_name)

        key = cv2.waitKeyEx(0)

        if key in (KEY_Q_LOWER, KEY_Q_UPPER, KEY_ESC):
            break

        if key in (KEY_LEFT, KEY_A_LOWER, KEY_A_UPPER):
            index = (index - 1) % len(test_dataset)
            continue

        if key in (KEY_RIGHT, KEY_D_LOWER, KEY_D_UPPER):
            index = (index + 1) % len(test_dataset)
            continue

        if key in (KEY_S_LOWER, KEY_S_UPPER):
            saved_path = save_image(annot_img, index)
            print(f"Saved: {saved_path}", flush=True)
            continue

        # Ignore unsupported keys and stay on the current image.
        print(f"Ignored key code: {key}", flush=True)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
