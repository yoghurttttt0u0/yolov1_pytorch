from PIL import Image
import cv2
import numpy as np
import os


def resize_to_448(img: Image.Image, size: int = 448) -> Image.Image:
    return img.resize((size, size), Image.Resampling.BILINEAR)


def draw_grid(img: Image.Image, S: int = 7):
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_np.shape[:2]

    for i in range(1, S):
        x = int(i * w / S)
        cv2.line(img_np, (x, 0), (x, h), (0,255,0), 1)

    for i in range(1, S):
        y = int(i * h / S)
        cv2.line(img_np, (0, y), (w, y), (0,255,0), 1)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_np)


# ===== MAIN =====
img_path = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\yolov1_pytorch\data\KITTI\test\images\001530.png"

print("Loading image...")
pil_img = Image.open(img_path)

print("Processing...")
img_resized = resize_to_448(pil_img)
img_with_grid = draw_grid(img_resized, S=7)

save_path = r"D:\course-resource\Machine_Learning_for_Data_Science\YOLOv1\grid.jpg"

print("Saving...")
img_with_grid.save(save_path)

print("Saved to:", save_path)
print("File exists:", os.path.exists(save_path))

img_with_grid.show()