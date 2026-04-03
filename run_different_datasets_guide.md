# How to Switch Between Different Datasets

This document explains which parts of the code need to be modified when running the project on different datasets, including **VOC**, **KITTI**, and **COCO subset**.  
Due to time limitations, we did not replace these manual settings with configuration files or command-line arguments. Instead, the required changes are documented here for reproducibility.

Below, **KITTI** is used as an example.

---

## 1. Training the Model

### `dataset.py`

Update the **class definitions** for the target dataset.

For example, when switching to **KITTI**, enable the KITTI class definition code and disable the class definition code for VOC and COCO.

![Class definition example](docs\images\image-20260402231346582.png)

---

###  `train.py`

1. Change `DATASET_DIR` to the correct dataset folder.

![Dataset path example](docs\images\image-20260402233247947.png)

2. Change `LOAD_MODEL` depending on the experiment:

- **Training on VOC**: `LOAD_MODEL = 'pretrain'`
- **Fine-tuning on KITTI or COCO**: `LOAD_MODEL = 'voc'`
- **Resuming from a checkpoint**: `LOAD_MODEL = 'train'`

![Weight loading mode example](docs\images\image-20260402232237473.png)

3. Update the **checkpoint** and **model weight paths**.

![image-20260403034107085](docs\images\image-20260403034107085.png)

4. Update the **hyperparameters** for the selected dataset.

For example, when switching to **KITTI**, uncomment the KITTI hyperparameter settings and comment out the hyperparameter settings for the other two datasets.

![Training strategy example](docs\images\image-20260402231619542.png)

---

## 2. Evaluating the Model

### `evaluate.py`

Update:

- the dataset path
- the model weights path

![Evaluation example](docs\images\image-20260402233442074.png)

---

## 3. Visualising Predictions

### `visualize_v2.py`

Update:

- the model path
- the dataset path
- the output path for saved images

![Visualisation example](docs\images\image-20260402234000889.png)

---

## 4. Recommended Workflow for Changing Datasets

1. Update the **class definitions** in `dataset.py`.
2. Update the **dataset path** in `train.py`, `evaluate.py`, and `visualize_v2.py`.
3. Update the **model loading mode**.
4. Update the **model path**.
5. Update the training **hyperparameters**.
6. Update the **output file names** so that results from different datasets are stored separately.

---

## 5. Important Note

When switching datasets, the most important requirement is **consistency**:

- the class definitions must match the annotations
- the dataset path must match the selected dataset
- the model weights must match the training stage
- the output names should be separated by dataset to avoid overwriting files
