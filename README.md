# KCOC: Ordinal-Aware Logit Adjustment and Balanced Representation Learning for Diabetic Retinopathy Grading

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![BSPC 2026](https://img.shields.io/badge/BSPC-2026-2f6f9f.svg)](https://doi.org/10.1016/j.bspc.2026.110522)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)

## News

- **[2026/07]** The paper was published in *Biomedical Signal Processing and Control*. [[Paper]](https://doi.org/10.1016/j.bspc.2026.110522)

KCOC is a two-stage framework for diabetic retinopathy (DR) grading. It combines k-positive contrastive representation learning with an ordinal-aware classifier to address class imbalance and the ordered relationships between disease grades.

This repository is the official implementation of [*Ordinal-aware logit adjustment and balanced representation learning for diabetic retinopathy grading*](https://doi.org/10.1016/j.bspc.2026.110522), published in *Biomedical Signal Processing and Control*, Volume 123, Article 110522 (2026).

![KCOC Framework](assets/KCOC_framework.png)

## Highlights

- **k-positive contrastive learning:** Uses a fixed number of same-grade positive samples to reduce representation bias toward majority classes.
- **Dynamic dictionary queue:** Maintains a large and continuously updated set of contrastive features independently of the mini-batch size.
- **Momentum teacher encoder:** Provides stable key representations through exponential moving average updates.
- **Ordinal-aware Cross-Entropy:** Introduces distance-aware margins between DR grades through a simple logit adjustment.
- **Efficient inference:** Discards the projection head and teacher encoder after pretraining, leaving only a standard classification backbone.
- **Multi-dataset evaluation:** Evaluated on APTOS2019, Messidor-2, and DDR with ResNet-50, DenseNet-121, and ViT-S/16 backbones.

## Method

KCOC consists of two stages.

### 1. k-Positive Contrastive Representation Learning


### 2. Ordinal-Aware Classifier Training

![Conceptual overview of the Ordinal-aware Cross-Entropy (OCE) framework](assets/OCE_framework.png)


## Project Structure

```text
KCOC/
|-- ordinal_classification/
|   |-- train_kcoc.py          # Training, classifier fitting, and evaluation
|   |-- preprocess_fundus.py   # FOV cropping, CLAHE, and resizing
|   |-- make_folds.py          # Stratified cross-validation splits
|   |-- data_augmentation/     # Image augmentation utilities
|   |-- data_manager/          # Dataset and sampling utilities
|   |-- loss/                  # OCE and baseline losses
|   |-- moco/                  # MoCo and KCL implementations
|   |-- moco_models/           # CNN backbone implementations
|   `-- models/                # Vision Transformer implementations
|-- assets/                    # README figures
|-- CITATION.cff               # Citation metadata
|-- requirements.txt           # Python dependencies
|-- LICENSE                    # MIT License
`-- README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.9+
- A CUDA-capable GPU is recommended
- PyTorch matching the local CUDA driver

### Install Dependencies

```bash
git clone https://github.com/liluhu0/KCOC.git
cd KCOC

python -m venv .venv
source .venv/bin/activate  # Linux or macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

## Usage

### 1. Prepare Data Manifests

APTOS2019, Messidor-2, and DDR use CSV manifests with `path` and `label` columns. Paths may be absolute or relative to the manifest. Labels must be integers from 0 to 4.

```csv
path,label
images/0001.png,0
images/0002.png,2
```

For APTOS2019 and Messidor-2, generate five stratified folds:

```bash
python ordinal_classification/make_folds.py \
  --manifest all.csv \
  --output splits/aptos \
  --folds 5 \
  --seed 591884092
```

For DDR, prepare manifests following the official train, validation, and test splits.

### 2. Preprocess Fundus Images

The preprocessing pipeline crops the retinal field of view, applies CLAHE, resizes the image, and writes a new manifest.

```bash
python ordinal_classification/preprocess_fundus.py \
  --manifest all.csv \
  --output data/aptos_clahe \
  --size 512
```

### 3. Contrastive Pretraining

```bash
python ordinal_classification/train_kcoc.py pretrain \
  --train-csv splits/aptos/train_fold0.csv \
  --output outputs/aptos/fold0 \
  --backbone resnet50 \
  --image-size 224 \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.016 \
  --queue-size 1024 \
  --num-positive 64 \
  --momentum 0.999 \
  --temperature 0.07 \
  --balanced-sampling
```

### 4. Ordinal-Aware Classifier Training

```bash
python ordinal_classification/train_kcoc.py finetune \
  --train-csv splits/aptos/train_fold0.csv \
  --val-csv splits/aptos/test_fold0.csv \
  --pretrained outputs/aptos/fold0/pretrain_best.pth \
  --output outputs/aptos/fold0 \
  --backbone resnet50 \
  --image-size 224 \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.1 \
  --beta 4 \
  --balanced-sampling
```

### 5. Evaluation

```bash
python ordinal_classification/train_kcoc.py evaluate \
  --data-csv splits/aptos/test_fold0.csv \
  --checkpoint outputs/aptos/fold0/finetune_best.pth \
  --backbone resnet50 \
  --image-size 224
```

The evaluation command reports accuracy, balanced accuracy, quadratic weighted Kappa, macro F1, weighted F1, macro recall, macro precision, and multiclass AUC. For 512 x 512 inputs, reduce the batch size according to available GPU memory.

## Datasets and Model Weights

The datasets, data splits, and trained checkpoints are not redistributed in this repository. See [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md) for model-weight release options. The published article is available from the [publisher via DOI](https://doi.org/10.1016/j.bspc.2026.110522).

## Citation

If you find this work useful, please cite:

```bibtex
@article{LI2026110522,
  title    = {Ordinal-aware logit adjustment and balanced representation learning for diabetic retinopathy grading},
  journal  = {Biomedical Signal Processing and Control},
  volume   = {123},
  pages    = {110522},
  year     = {2026},
  issn     = {1746-8094},
  doi      = {10.1016/j.bspc.2026.110522},
  author   = {Luhu Li and Xuya Liu and Xinguo Hou and Li Chen and Ziyu Wang and Qingfeng Ding and Shujun Fu},
  keywords = {Diabetic retinopathy, Imbalanced classification, Ordinal labels, Contrastive learning, Inter-class correlation}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party source files retain their original copyright notices.

## Acknowledgments

This implementation builds upon the open-source PyTorch, torchvision, timm, and Momentum Contrast communities. We thank the maintainers and contributors of these projects.
