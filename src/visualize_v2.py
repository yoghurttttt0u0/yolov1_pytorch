"""
OpenCV-based prediction viewer as an alternative to matplotlib.
Some team members have display issues with matplotlib, so this script uses OpenCV for more consistent behavior.
The old version using matplotlib is kept in tools/visualize.py.
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import PIL.Image as Image
import torch as th
import torchvision.transforms.functional as fT
from torchvision.utils import draw_bounding_boxes

from dataset import DetectionDataset
from evaluate import postprocessing
from model import YOLOv1

# Model Hyperparameters
S = 7
B = 2
D = 448

BASE_DIR = Path(__file__).resolve().parent.parent

# Trained Model Path
TRAINED_MODEL_WEIGHTS = BASE_DIR / "checkpoints" / "kitti_model_weights.pt.pt" 

# KITTI Dataset Directory
PASCAL_VOC_DIR_PATH = BASE_DIR / "data" / "KITTI"

# Save Image Path
ASSETS_DIR = BASE_DIR / "results" / "assets_KITTI"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# Postprocessing Hyperparameters
PROB_THRESHOLD = 0.15
NMS_THRESHOLD = 0.6
CONF_MODE = "objectness"

WINDOW_NAME = "YOLOv1 Predictions"

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
KEY_G_LOWER = ord("g")
KEY_G_UPPER = ord("G")
KEY_P_LOWER = ord("p")
KEY_P_UPPER = ord("P")

COLORS = {}

def hex_to_bgr(hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return (rgb[2], rgb[1], rgb[0])

def annotate_img(
    img: Image.Image,
    bboxes: th.Tensor,
    show_conf: bool = True,
    label_prefix: str = "",
    use_dashed: bool = False,
) -> Image.Image:
    """
    Annotate the given image based on the given bounding boxes.

    If show_conf=True, bbox format should be:
        [class_id, conf, x1, y1, x2, y2]

    If show_conf=False, bbox format should be:
        [class_id, x1, y1, x2, y2]
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for bb in bboxes:
        cls = int(bb[0])

        if show_conf:
            conf = float(bb[1].item())
            x1, y1, x2, y2 = map(int, bb[2:])
        else:
            conf = None
            x1, y1, x2, y2 = map(int, bb[1:])

        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        label_name = DetectionDataset.index2label[cls]
        if show_conf:
            label = f"{label_prefix}{label_name} {conf:.2f}"
        else:
            label = f"{label_prefix}{label_name}"

        color = hex_to_bgr(DetectionDataset.label_clrs[cls])

        thickness = 2

        if use_dashed:
            gap = 8
            # top
            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
            # bottom
            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
            # left
            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
            # right
            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(0.7, img.shape[1] / 1600))
        font_thickness = 1

        (tw, th_text), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

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

def overlay_ground_truth(
    img: Image.Image,
    target: th.Tensor,
) -> Image.Image:
    """
    Overlay ground-truth bounding boxes on the image.

    Expected target format: [class_id, x1, y1, x2, y2]
    """
    if target is None:
        return img

    if not isinstance(target, th.Tensor):
        target = th.tensor(target)

    if target.numel() == 0:
        return img

    if target.ndim == 1:
        target = target.unsqueeze(0)

    return annotate_img(
        img,
        target,
        show_conf=False,
        label_prefix="GT: ",
        use_dashed=True,
    )

def predict_and_annotate(
    model: YOLOv1,
    img: Image.Image,
    gt_target: th.Tensor = None,
    show_gt: bool = False,
) -> Image.Image:
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

    if bboxes_pred.numel() != 0:
        bboxes_pred[:, [2, 4]] *= w / D
        bboxes_pred[:, [3, 5]] *= h / D

    annot_img = annotate_img(img, bboxes_pred, show_conf=True)

    if show_gt:
        annot_img = overlay_ground_truth(annot_img, gt_target)

    return annot_img


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to an OpenCV BGR numpy array.
    """
    rgb = np.array(img)
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_image_name(dataset: DetectionDataset, index: int) -> str:
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
    show_gt: bool,
    show_pred: bool,
) -> np.ndarray:
    """
    Add usage instructions and the current image file name under the displayed image.
    """
    h, w = img_bgr.shape[:2]
    bar_h = 56
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = img_bgr
    canvas[h:] = (24, 24, 24)

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

    gt_text = f"GT: {'ON' if show_gt else 'OFF'}"
    cv2.putText(
        canvas,
        gt_text,
        (220, h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 220, 255) if show_gt else (160, 160, 160),
        2,
        cv2.LINE_AA,
    )

    pred_text = f"Pred: {'ON' if show_pred else 'OFF'}"
    cv2.putText(
        canvas,
        pred_text,
        (340, h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 255, 120) if show_pred else (160, 160, 160),
        2,
        cv2.LINE_AA,
    )

    help_text = "A/Prev   D/Next   G Toggle GT   P Toggle Pred   S Save   Q/Esc Quit"
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
    show_gt: bool,
    show_pred: bool,
) -> None:
    """
    Show one annotated image in an OpenCV window.
    """
    img_bgr = pil_to_bgr(annotated_img)
    vis = add_status_bar(
    img_bgr,
    index,
    total,
    image_name,
    show_gt,
    show_pred,
)

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


def setup_evaluation() -> Tuple[YOLOv1, DetectionDataset]:
    """
    Instantiate the model and the KITTI test dataset.
    """
    model = YOLOv1(S=S, B=B, C=DetectionDataset.C).to(DEVICE)
    trained_model_weights = th.load(TRAINED_MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(trained_model_weights)
    model.eval()

    test_dataset = DetectionDataset(root_dir=PASCAL_VOC_DIR_PATH, split="test")
    return model, test_dataset


def save_image(img: Image.Image, index: int) -> str:
    """
    Save the current annotated image and return the path.
    """
    path = ASSETS_DIR / f"annot_img_{index}.jpg"
    img.save(path)
    return str(path)

def render_image(
    model,
    pil_img,
    target,
    show_pred=True,
    show_gt=False,
):
    """
    Render image with optional prediction and GT overlays.
    """
    base_img = pil_img.copy()

    if show_pred:
        base_img = predict_and_annotate(
            model,
            base_img,
            gt_target=None,
            show_gt=False
        )

    if show_gt:
        base_img = overlay_ground_truth(base_img, target)

    return base_img

def main() -> None:
    """
    Browse prediction results interactively in a stable OpenCV window.

    Controls:
    - Left arrow or A: previous image
    - Right arrow or D: next image
    - G: toggle ground-truth bounding boxes
    - S: save current image
    - Q or Esc: quit
    """
    model, test_dataset = setup_evaluation()

    if len(test_dataset) == 0:
        raise RuntimeError("The test dataset is empty.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)

    index = 0
    show_gt = False
    show_pred = True

    while True:
        pil_img, target = test_dataset[index]
        image_name = get_image_name(test_dataset, index)

        annot_img = render_image(
            model,
            pil_img,
            target,
            show_pred=show_pred,
            show_gt=show_gt,
        )
        show_image(
            index,
            len(test_dataset),
            annot_img,
            image_name,
            show_gt,
            show_pred,
        )

        key = cv2.waitKeyEx(0)

        if key in (KEY_Q_LOWER, KEY_Q_UPPER, KEY_ESC):
            break

        if key in (KEY_LEFT, KEY_A_LOWER, KEY_A_UPPER):
            index = (index - 1) % len(test_dataset)
            continue

        if key in (KEY_RIGHT, KEY_D_LOWER, KEY_D_UPPER):
            index = (index + 1) % len(test_dataset)
            continue

        if key in (KEY_G_LOWER, KEY_G_UPPER):
            show_gt = not show_gt
            continue

        if key in (KEY_P_LOWER, KEY_P_UPPER):
            show_pred = not show_pred
            continue

        if key in (KEY_S_LOWER, KEY_S_UPPER):
            saved_path = save_image(annot_img, index)
            print(f"Saved: {saved_path}", flush=True)
            continue

        print(f"Ignored key code: {key}", flush=True)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
