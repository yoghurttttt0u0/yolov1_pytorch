## Dataset

Due to its large size, the `data/` folder is **excluded from the repository** via `.gitignore`. the expected folder structure is:

```
data/
└── VOC_Detection/
    ├── train/
    │   ├── images/   # .jpg files
    │   └── targets/  # .csv annotation files
    └── test/
        ├── images/   # .jpg files
        └── targets/  # .csv annotation files
```

You must provide this dataset locally before running the code.

## Code Modifications

Only minimal changes have been made to the original repository. I have updated **global variables related to file paths** (e.g. path to the VOC dataset, path to the pretrained model). No major changes were made to the model architecture or training logic.

