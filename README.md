# SARNets (Building Damage Assessment on BRIGHT Dataset)

###### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - by [Mykyta Kyselov (TheMegistone4Ever)](https://github.com/TheMegistone4Ever).

SARNets is a high-performance Python application for semantic segmentation and building damage assessment, utilizing the
BRIGHT dataset and U-Net architecture. This project provides a complete pipeline, from automated dataset downloading and
extraction to training a deep learning model with advanced loss functions (Lovasz-Softmax, Focal Loss) and performing
inference with visual overlays.

The project is designed to analyze Synthetic Aperture Radar (SAR) imagery to classify building damage into four
categories: No Damage (0), Minor (1), Major (2), and Destroyed (3).

## Table of Contents

- [1. Getting Started](#1-getting-started)
    - [1.1. Project Overview](#11-project-overview)
    - [1.2. Prerequisites](#12-prerequisites)
    - [1.3. Installation](#13-installation)
- [2. Usage](#2-usage)
    - [2.1. Data Preparation](#21-data-preparation)
    - [2.2. Training](#22-training)
    - [2.3. Evaluation & Inference](#23-evaluation--inference)
- [3. Project Structure](#3-project-structure)
- [4. Configuration](#4-configuration)
- [5. License](#5-license)

## 1. Getting Started

### 1.1. Project Overview

This repository contains PyTorch scripts to train and evaluate a U-Net model on the BRIGHT dataset. The system is
capable of handling multi-temporal data (pre-event and post-event SAR images) to determine the extent of damage caused
by natural disasters.

### 1.2. Prerequisites

Before you begin, ensure you have the following installed:

- An NVIDIA GPU (strongly recommended for training)
- Python 3.10+
- PyTorch (CUDA support recommended)
- `aria2c` (optional, for faster dataset downloading)

### 1.3. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheMegistone4Ever/SARNets.git
   cd SARNets
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 2. Usage

### 2.1. Data Preparation

The project includes a script to automatically download the BRIGHT dataset from HuggingFace and extract it.

Run the `download_extract.py` script:

```bash
python download_extract.py
```

This will create a `dataset/bright_dataset` directory containing `pre-event`, `post-event`, and `target` folders.

### 2.2. Training

To train the model with default settings (defined in `model/utils.py`), run:

```bash
python train.py
```

To perform a grid search over different hyperparameters (Batch Size, Input Size, Learning Rate, Loss Functions), run:

```bash
python train_extended.py
```

*Note: Grid search results are logged to `grid_search_metrics.csv`.*

### 2.3. Evaluation & Inference

To evaluate the best trained model and generate visualization results:

1. Ensure a trained model checkpoint exists in `checkpoints/`.
2. Run the test script:
   ```bash
   python test.py
   ```

The script will calculate IoU metrics per class and save visualization images (SAR image overlaid with prediction masks)
to the `test_results/` directory.

## 3. Project Structure

```
.
├── .gitattributes          # Specifies Git attributes
├── .gitignore              # Lists files to ignore
├── dataset/                # Dataset handling scripts
│   ├── bright_dataset.py   # Custom Dataset class
│   └── download_extract.py # Data downloader
├── model/                  # Model definition and logic
│   ├── loss.py             # Custom loss functions (Lovasz, Focal, Combined)
│   ├── u_net.py            # U-Net architecture implementation
│   └── utils.py            # Configuration and utility functions
├── distributions.py        # Script to plot data histograms
├── LICENSE.md              # Project license
├── requirements.txt        # Python dependencies
├── test.py                 # Inference and evaluation script
├── train.py                # Main training loop
└── train_extended.py       # Hyperparameter grid search script
```

## 4. Configuration

Key parameters can be adjusted in `model/utils.py` and `train.py`:

- **General Settings**:
    - `BATCH_SIZE`: Training batch size.
    - `INPUT_SIZE`: Image resolution for resizing (e.g., 372).
    - `LEARNING_RATE`: Initial learning rate for the optimizer.
    - `NUM_EPOCHS`: Maximum number of training epochs.

- **Channel Mask (`CHANNEL_MASK`)**:
  A binary string of length 6 that determines which input channels are fed into the network. It corresponds to the
  sequence: `[Pre-R, Pre-G, Pre-B, Post-R, Post-G, Post-B]`.
    - `111111`: Uses all 6 channels (Pre-event and Post-event imagery).
    - `000111`: Uses only the Post-event imagery (3 channels).
    - `100100`: Uses only the Red channel from Pre-event and Red channel from Post-event.

  *Modifying this allows for flexible experimentation with different input modalities.*

## 5. License

The project is licensed under the [CC BY-NC 4.0 License](LICENSE.md).