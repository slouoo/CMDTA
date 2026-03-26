
# CMDTA: Cross-Modal Deep Learning Framework for Drug-Target Affinity Prediction

![Python 3.8](https://img.shields.io/badge/Python-3.8.20-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2Bcu124-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**CMDTA** is a novel, advanced deep learning framework designed for accurately predicting Drug-Target Affinity (DTA). By uniquely leveraging both **1D sequence information** (SMILES for drugs, amino acid sequences for proteins) and **3D structural information** (ligand 2D graphs and protein 3D pockets) through a parallel dual-path architecture, CMDTA provides a comprehensive and precise model of drug-target interactions.

## 🧬 Framework Architecture
CMDTA extracts multi-scale features using cutting-edge graph and sequence neural networks:
- **1D Sequence Path:** Utilizes Bidirectional LSTMs enhanced with Residual Attention mechanisms to capture contextual dependencies in drug SMILES and protein sequences.
- **2D/3D Structural Path:** Employs Graph Isomorphism Networks (GIN) for 2D molecular graphs of drugs and Geometric Vector Perceptrons (GVP) to rigorously model the 3D atomic coordinates of protein binding pockets.
- **Cross-Modal Fusion:** A carefully designed fully connected neural network integrates textual and structural embeddings to output highly accurate continuous affinity values (e.g., Kd, Ki).

![CMDTA Framework](CMDTA_Framework.png)

## 📁 Repository Structure

```text
CMDTA/
├── Dataset/             # Directory for placing Davis and KIBA datasets (CSV and PDB files)
├── Model/               # Directory where trained model checkpoints (.pt) are saved
├── Vocab/               # Pre-built vocabulary files (.pkl) for tokenizing SMILES and Proteins
├── gvp/                 # Source code for Geometric Vector Perceptron (GVP) modules
├── build_vocab.py       # Script for building natural language vocabularies
├── dataset.py           # Custom PyTorch Geometric Dataset classes for handling multi-modal data
├── main.py              # Main script for training, evaluation, and logging
├── model.py             # Model architecture definitions (CMDTA, GINConv, SeqEncoder)
├── utils.py             # Helper functions for data parsing, graph extraction, and metrics
└── README.md
````

## 🛠️ Installation & Setup

We recommend using [Anaconda](https://www.anaconda.com/) or Miniconda to manage your environment.

**1. Clone the repository:**

```bash
git clone [https://github.com/slouoo/CMDTA.git](https://github.com/slouoo/CMDTA.git)
cd CMDTA
```

**2. Create and activate a conda environment:**

```bash
conda create -n cmdta python=3.8.20
conda activate cmdta
```

**3. Install dependencies:**

```bash
pip install torch==2.4.1+cu124 --extra-index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install torch-geometric==2.6.1 torch-cluster==1.6.3 torch-scatter==2.1.2 torch-sparse==0.6.18
pip install scikit-learn==1.3.2 scipy==1.10.1 pandas==2.0.3 networkx==3.1 atom3d==0.2.6
pip install rdkit==2022.9.5
```

## 📊 Dataset Preparation

The model is evaluated on two standard benchmark datasets: **Davis** and **KIBA**. Due to file size constraints, the cleaned datasets and the target 3D structures (PDB files) are hosted externally.

1.  Download the datasets from our [Google Cloud Drive](https://drive.google.com/file/d/1osd9GRS1itQUi8e3NzBlZdnndllfIGxV/view?usp=sharing).
2.  Extract the downloaded archive.
3.  Place the sequence CSV files (e.g., `kiba_dataset_cleaned.csv` or `davis_dataset_cleaned.csv`) in the root directory.
4.  Place the 3D structure files in the corresponding `Dataset/<dataset_name>/protein/` directories.

## 🚀 Training and Evaluation

You can train the CMDTA model by running the `main.py` script. The script performs k-fold style runs across multiple random seeds to ensure robust evaluation.

```bash
python main.py
```

**Configurations (editable in `main.py`):**

  - `dataset_name`: Choose between `'kiba'` or `'davis'`.
  - `LR`: Learning rate (default: 1e-3).
  - `batch_size`: Default is 64.
  - `NUM_EPOCHS`: Default is 200 (includes early stopping).

**Outputs:**

  - **Model Checkpoints:** Saved in the `./Model/` directory (e.g., `best_model_kiba_seed42.pt`).
  - **Evaluation Metrics:** A summary CSV file (e.g., `kiba_result_nf.csv`) will be generated containing the Mean Squared Error (MSE), Concordance Index (CI), and $r_m^2$ metrics across all random seeds.

## 📄 License

This project is licensed under the terms of the [MIT License](https://www.google.com/search?q=LICENSE).

```
```
