# CNN Character Classifier (Assignment 2 Part 2)

## Overview
This repository contains my implementation of a scratch-built Convolutional Neural Network (CNN) that classifies 28×28 colour images into 36 alphanumeric classes (digits 0-9 and uppercase A-Z).

*Notebook*: `CNN_Implementation.ipynb`

The project was developed for an Intro to Machine Learning coursework and achieves **91.8 % test accuracy**.

---

## 1) Problem & Data
* **Task** – Multi-class classification (36 classes).
* **Dataset** – Custom image dataset organised in `datasets/cnn_dataset/` using the standard `ImageFolder` layout:
  ```text
  └── cnn_dataset
      ├── 0
      │   ├── img0.png
      │   └── ...
      ├── 1
      └── ...
      └── Z
  ```
* **Pre-processing**
  * Images resized to 28×28.
  * Normalised with dataset-wide mean/std.
---

## 2) Model
```
Input 3×28×28 → [Conv32-BN-ReLU] → MaxPool
                 → [Conv64-BN-ReLU] → MaxPool
                 → Conv128-BN-ReLU
                 → Conv256-BN-ReLU → MaxPool
                 → Flatten (256×3×3)
                 → Dropout(0.5) → FC-512-ReLU → FC-36 → Softmax
```
---

## 3) Training Pipeline
* `torchvision.transforms` for augmentation & normalisation.
* `DataLoader` with `batch_size = 32` and shuffling.
* **Loss** – `nn.CrossEntropyLoss()`.
* **Optimiser** – Adam (lr = 1e-3).
* **Regularisation / Tricks**
  * Batch Normalisation after every conv.
  * Dropout(0.5) before fully-connected layer.
  * Gradient accumulation (4 steps) variant for memory efficiency.
  * Early stopping (patience = 3) variant.
* **Scheduler** – *none* (fixed lr performed best during short training window).
---

## 4) Experiments & Results
| Setup | Test Accuracy |
|-------|---------------|
| Baseline CNN | **91.80 %** |
| + Early Stopping | 91.54 % |
| + Grad. Accumulation | 91.33 % |
| 2-Fold CV (val) | 90.75 % avg |

Additional metrics on the best model:
* Precision = 0.922
* Recall = 0.918
* F1-score = 0.918

The notebook visualises learning curves, confusion matrix, and per-class ROC curves.

---

## 5) Repository Structure
```
.
├── CNN_Implementation.ipynb  # main notebook
├── datasets/                 # image data (not included in repo)
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```

---

## 6) Quick-start
1. Clone the repo and create a Python ≥ 3.9 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in `datasets/cnn_dataset/` following the `ImageFolder` structure.
4. Launch the notebook and run all cells:
   ```bash
   jupyter notebook CNN_Implementation.ipynb
   ```

The best model weights are saved automatically to `best_model_part3.pth`.

---

## 7) License
This project is released under the MIT License – see `LICENSE` for details.

## 8) Acknowledgements
University assignment context as per `fall24_cse574_d_Assignment_2.pdf`.
