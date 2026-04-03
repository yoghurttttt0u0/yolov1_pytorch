# YOLOv1 Reproduction

## 1. Project Overview

This project reproduces the **YOLOv1 object detection model** based on the original paper:

> You Only Look Once: Unified, Real-Time Object Detection

The work includes:

- Training YOLOv1 on **PASCAL VOC**
- Fine-tuning on **KITTI dataset**
- Fine-tuning on a **custom COCO subset**
- Evaluating performance using mAP and class-wise AP
- Visualising predictions results

------

## 2. Repository Structure

```text
.
├── src/                         # Core implementation
│   ├── dataset.py 					# Dataset loading and parsing
│   ├── train.py					# Training pipeline
│   ├── evaluate.py					# Evaluation (mAP, AP)
│   ├── model.py					# Model architecture
│   ├── loss.py						# Loss function
│   ├── transforms.py				# Data augementation and preprocessing transforms
│   └── visualize_v2.py				# Prediction visualisation
│
├── tools/                      # Data processing utilities
│   ├── coco_json_to_csv.py
│   ├── sample_subset.py
│   ├── analyze_labels.py
│   └── ...
│
├── data/                      # Datasets (not included in the repository)
│   ├── VOC_Detection/
│   ├── KITTI/
│   └── coco_subset/
│
├── checkpoints/                  # Saved models and checkpoints
│
├── results/                   	  # Results for different datasets (loss curve, AP)
│   ├── kitti/                 
│   ├── coco_subset/ 
│   └── voc/
│
├── run_different_datasets_guide.md     # Guide for switching datasets
├── requirements.txt
└── README.md
```

------

## 3. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Tested environments：

- NVIDIA RTX5060 GPU

- **NVIDIA A100(NB: I m not sure if it is the GPU you used in Colab )**

------

## 4. Important Note

To reproduce results:

1. Prepare dataset (VOC / KITTI / COCO subset)
2. Update parameters according to:
   `HOW_TO_SWITCH_DATASETS.md`
3. Then, run:
   - `train.py`
   - `evaluate.py`
   - `visualize_v2.py`

## 4. Data Preparation

Due to size limitations, **datasets (PASCAL VOC, KITTI, COCO subset) and trained models are not included in this repository**. 

### 4.1 Datasets

Please download and organise them as follows:

```text
./data/                  
├── VOC_Detection/
│   ├── train/
│   │   ├── images/
│   │   └── targets/
│   └── test/
│       ├── images/
│       └── targets/
├── KITTI/
│   └── ...
└── coco_subset/
    └── ...
```

Each dataset should follow the structure:

- `train/`  and `test/`
- each containing:
  - `images/`
  - `targets/`

**Download links:**

- PASCAL VOC:

  [VOC_Detection.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucesxy9_ucl_ac_uk/IQC2M1BoE5K_QLjENLfiY4KbAUPNUZttzomcjPFZ4BiIM08?e=C3cpeX)

- KITTI: 

  [KITTI.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucesxy9_ucl_ac_uk/IQDAPGPaSxTqQZkd6tzHP-nZAXclcmCzsgg-74eikumoFh8?e=s0cjtW)

- COCO subset: 

  (Please provide your actual download link here)

After downloading, unzip the datasets and place them under the `data/` directory.



### 4.2 Trained Models

Fine-tuned models are also not included due to file size constraints.

Please download them and place them in the `checkpoints/` directory.

**Model download links:**

- Fine-tuned on KITTI:

  [kitti_finetune_model_weights.pt](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucesxy9_ucl_ac_uk/IQBiZmgYA_BsQYd0wGOvZDYrAcl14N94Slfi-5tkiL8REBs?e=GZewiJ)

- Fine-tuned on COCO subset:

  (link here)

------

## 5. How to Run

**Training**

```bash
python src/train.py
```



**Evaluation**

```bash
python src/evaluate.py
```



**Visualisation**

```bash
python src/evaluate.py
```

Controls:

- `A / D` → previous / next image
- `G` → toggle ground truth
- `P` → toggle predictions
- `S` → save image

------

## 6. Switching Between Datasets

To run experiments on different datasets (VOC, KITTI, COCO subset), you need to manually modify several parts of the code.

👉 Please refer to:
`run_different_datasets_guide.md` 

This document explains:

- dataset settings
- training settings
- evaluation settings
- visualisation settings

