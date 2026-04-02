"""
Dataset loader for VOC, KITTI, and COCO object detection.
"""

import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List
import colorsys

class DetectionDataset(Dataset):
    """
    A custom Dataset for the VOC Detection data. An index number (starting from 0) and a color is assigned to each of
    the labels of the dataset.
    """

    def _generate_distinct_colors(n: int):
        """
        Generate n visually distinct hex colors.
        Uses evenly spaced hues in HSV space with moderate/high saturation and value.
        """
        colors = []
        for i in range(n):
            h = i / n
            s = 0.72
            v = 0.92
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append("#{0:02x}{1:02x}{2:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            ))
        return colors
    
    ########################################## VOC ##############################################
    # C = 20

    # index2label = ["person",
    #                "bird", "cat", "cow", "dog", "horse", "sheep",
    #                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    #                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

    # label2index = {label: index for index, label in enumerate(index2label)}

    # label_clrs = ["#ff0000",
    #               "#2e8b57", "#808000", "#800000", "#000080", "#2f4f4f", "#ffa500",
    #               "#00ff00", "#ba55d3", "#00fa9a", "#00ffff", "#0000ff", "#f08080", "#ff00ff",
    #               "#1e90ff", "#ffff54", "#dda0dd", "#ff1493", "#87cefa", "#ffe4c4"]


    ########################################## KITTI ############################################
    # C = 8

    # index2label = ["car",
    #                "van", "truck", "pedestrian", "Person_sitting", "cyclist", "tram",
    #                "misc"]

    # label2index = {label: index for index, label in enumerate(index2label)}

    # label_clrs = ["#ff0000",
    #               "#2e8b57", "#808000", "#000080", 
    #               "#00ff00", "#ba55d3", "#00ffff",  "#f08080"]
    

    ########################################## COCO #############################################
    C = 80
    index2label = [
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
    
    label2index = {label: index for index, label in enumerate(index2label)}

    label_clrs = _generate_distinct_colors(C)

    def _find_image_path(self, pid: str) -> str:
        possible_exts = [".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".PNG"]

        for ext in possible_exts:
            candidate = os.path.join(self.img_dir, f"{pid}{ext}")
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"No image found for pid '{pid}' in {self.img_dir}. "
            f"Tried extensions: {possible_exts}"
        )

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None) -> None:
        """ Initialize the VOC_Detection Dataset object.

        :param root_dir: The root directory of the dataset (this directory contains two directories 'train/' and
                         'test/'.
        :param split: The split of the dataset ('train' or 'test')
        :param transforms: The transforms that are applied to the images (x) and their corresponding targets (y).
        """

        assert split == 'train' or split == 'test' or split == 'val'
        split_dir = os.path.join(root_dir, split)

        self.img_dir = os.path.join(split_dir, "images")
        self.annot_dir = os.path.join(split_dir, "targets")
        self.pseudonyms = [filename[:-4] for filename in os.listdir(self.annot_dir)]

        self.transforms = transforms

    def __len__(self) -> int:
        """
        Return the total number of instances of the dataset.

        :return: total instances of the dataset
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx: int) -> Tuple[Union[th.Tensor, Image.Image], th.Tensor]:
        """
        Given an index number in range [0, dataset's length) , return the corresponding image and target of the dataset.
        If transforms is defined, the images and their targets are first transformed and then return by the function.

        :param idx: The given index number
        :return: The (x,y)-pair of the image and the target
        """
        pid = self.pseudonyms[idx]
        img_path = self._find_image_path(pid)
        
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')

        img = Image.open(img_path).convert("RGB")
        target = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader) # Remove the header
            for row in csv_reader:
                target.append([self.label2index[row[0]]] + [int(row[i]) for i in range(1, 5)])
        target = th.Tensor(target)

        try:
            if self.transforms is not None:
                img, target = self.transforms((img, target))
        except Exception as e:
            print("\n===== ERROR SAMPLE FOUND =====")
            print(f"idx       : {idx}")
            print(f"pid       : {pid}")
            print(f"img_path  : {img_path}")
            print(f"annot_path: {annot_path}")
            print(f"target.shape before transforms: {target.shape}")
            print(f"target before transforms:\n{target}")
            raise e


        return img, target
    
    def get_image_name(self, idx: int) -> str:
        return f"{self.pseudonyms[idx]}.png"
    
    