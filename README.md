# CMDTA: an advanced cross-modal deep learning framework for enhanced prediction of drug-target affinity
CMDTA is a novel deep learning framework for accurately predicting drug-target affinity (DTA). By uniquely leveraging both 1D sequence and 3D structural information through a parallel dual-path architecture, CMDTA provides a more comprehensive and precise model of molecular interactions.

## Framework
![CMDTA Framework](CMDTA_Framework.png)

## File list
- Dataset: The Folder contains the 3D structures of all targets in benchmarks.
- Vocab: The Folder contains the vocabulary files.
- gvp: The Folder contains gvp-gnn model.
- built_vocab.py: The file is used to build the vocabulary files.
- dataset.py: The code about dataset.
- model.py: The code about CMDTA model.
- main.py: The code about training the CMDTA model.
- utils.py: The code about utils.


## Requirements
- networkx==3.1
- numpy==1.24.3
- pandas==1.5.3
- rdkit==2022.03.2
- scikit_learn==1.3.0
- scipy==1.10.1
- torch==1.12.1
- torch_geometric==2.3.1
- tqdm==4.65.0
