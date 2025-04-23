# Curriculum learning for stroke lesion segmentation

This repository is linked to the article Difficulty Metrics Study for Curriculum-Based Deep Learning in the Context of Stroke Lesion Segmentation [10.1109/ISBI53787.2023.10230836](https://ieeexplore.ieee.org/abstract/document/10230836), which proposes a curriculum learning framework for stroke lesion segmentation based on several difficulty metrics. The curriculum learning paradigm origns from [Curriculum Learning](https://dl.acm.org/doi/abs/10.1145/1553374.1553380), Y. Bengio et alAll is based on 2D U-Net architecture but can be adapted to other architectures.

# Data organization

The data should already be separated between the groups of difficulty before launching training. Proposed organization for this codes:

```
data/
├── image/
│   ├── 1/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 2/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 3/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 4/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 5/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   └── 6/
│       ├── train/
│       │   └── img1.png
│       └── validation/
│           └── img2.png
├── ref/
│   ├── 1/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 2/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 3/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 4/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   ├── 5/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   └── validation/
│   │       └── img2.jpg
│   └── 6/
│       ├── train/
│       │   └── img1.png
│       └── validation/
│           └── img2.png
```

The difficulty is decreasing with the number of the group: 1 contrains all data and 6 only the easiest images. The number of groups can be adjusted but it will need code modifications in `main_val_CL.py` and `train_val_CL.py`.

Two proposed metrics based on images caracteristics are the lesion area and fisher ratio. These features can be calculated with the script `fisher_area_measure.py`, based on the 3D volumes and assuming that the 3D volumes are separated in 2D images for training with [med2image](https://github.com/FNNDSC/med2image) and the following commands:

```
# medical images
med2image -i directory/image.nii.gz -d /output/dir
# reference masks
med2image -i directory/image.nii.gz -d /output/dir -t png
```

# Training

Training can be performed with a command line like 

```
python /path/to/data/ /path/to/output/ lr /optiional/weights -> /path/to/output/log.txt
```
