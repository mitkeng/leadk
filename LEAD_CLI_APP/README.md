# LEAD: Latent Embedded Adversarial Dual-discriminator CLI Application

LEAD is a deep-learning multi-task framework designed to accurately predict drug targets for novel compounds. The architecture features a **Dynamic Autoencoder (with multi-decoder subspace reconstruction)** and a **Dual-Discriminator Shield Agent** that calibrates classification confidence and manages out-of-distribution (OOD) risk in novel chemical spaces.

---

## 🛠️ GitHub Repository & Quick Start Instructions

To clone, set up, and deploy LEAD locally or on a remote cluster, follow these steps:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/lead-classifier.git
cd lead-classifier
```

### Step 2: Establish the Virtual Environment
It is highly recommended to use an isolated conda or virtualenv space to prevent dependency collisions.
```bash
# Creating and activating conda environment
conda create -n lead_env python=3.10 -y
conda activate lead_env
```
Or using virtualenv:
```bash
python3 -m venv lead_env
source lead_env/bin/activate
```

### Step 3: Install Required Dependencies
Ensure dependencies are updated and installed securely:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📖 Command Line Manual (CLI User Guide)

This application is split into two primary modular pipelines: **Training** (`train.py`) and **Inference** (`predict.py`).

### 1. Training Pipeline (`train.py`)
Use this module to retrain your autoencoder networks and adversarial discriminators on custom tabular feature profiles.

#### Positional & Optional CLI Arguments:
*   `--train_pos` (Required): String. Path to the CSV file representing active/positive compound configurations.
*   `--train_neg` (Required): String. Path to the CSV file representing negative/unlabelled background configurations.
*   `--save_dir` (Optional): String. Output folder to write model checkpoints and transforms (Default: `models`).
*   `--epochs` (Optional): Integer. Maximum number of training epochs (Default: `250`).
*   `--model_lr` (Optional): Float. Adjust learning rate parameters for model gradients (Default: `0.0003`).
*   `--decoders` (Optional): Integer. The number of independent reconstruction decoders for your feature space (Default: `5`).
*   `--adv_weight` (Optional): Float. Adversarial classification loss regularization scale parameter (Default: `0.03`).
*   `--seed` (Optional): Integer. Anchor global seed parameters for perfect result reproduction (Default: `42`).

#### Training CLI Command Example:
```bash
python train.py \
  --train_pos "Cancer_Drug_Train_Sanitized.csv" \
  --train_neg "Cancer_Drug_Test_Negative.csv" \
  --save_dir "models" \
  --epochs 250 \
  --model_lr 0.0003 \
  --decoders 5
```

---

### 2. Inference & Target Screening Pipeline (`predict.py`)
Predict kinase target affinities for newly designed target molecules with active dual-discriminator logit shielding.

#### Positional & Optional CLI Arguments:
*   `--input` (Required): String. Path to input screening CSV containing custom descriptor values.
*   `--checkpoint` (Optional): String. Target model parameters checkpoint path (Default: `models/lead_checkpoint.pth`).
*   `--transforms` (Optional): String. Saved preprocessing transform configurations path (Default: `models/transforms.pkl`).
*   `--output` (Optional): String. Filepath to store the prediction log results (Default: `predictions_output.csv`).
*   `--penalty_max` (Optional): Float. Maximum temperature penalty factor allowed by the Shield Agent (Default: `6.0`).

#### Inference CLI Command Example:
```bash
python predict.py \
  --input "Cancer_Drug_Test_Strictly_Novel.csv" \
  --checkpoint "models/lead_checkpoint.pth" \
  --output "Test_Prediction_Logs.csv" \
  --penalty_max 6.0
