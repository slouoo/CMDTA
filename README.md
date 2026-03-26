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
