# CMDTA: an advanced cross-modal deep learning framework for enhanced prediction of drug-target affinity
A deep learning framework to predict drug-target affinity


## Framework
![CMDTA Framework](CMDTA_Framework.png)

## File list
- Model: The Folder contains the trained model of DMFF-DTA.
- Vocab: The Folder contains the vocabulary files.
- built_vocab.py: The file is used to build the vocabulary files.
- dataset.py: The code about dataset.
- model.py: The code about model.
- main_warm_up.py: The code about training the warm-up model.
- main.py: The code about training the DMFF-DTA model.
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
