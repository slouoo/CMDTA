import sys
import os
import csv
import torch
import numpy as np
import pandas as pd
import atom3d.util.formats as format 
from build_vocab import WordVocab
from model import CMDTA
from dataset import DTADataset
from torch.utils.data import DataLoader, random_split
from utils import smiles_to_graph, featurize_as_graph, collate_fn, train, calculate_metrics_and_return, predicting, process_sequence

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Config
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
LR = 1e-3
NUM_EPOCHS = 200
dataset_name = 'kiba'  # or davis
batch_size = 64
seq_len = 540
tar_len = 1000
pocket_root = f'./Dataset/{dataset_name}/protein/'
csv_path = f'{dataset_name}_result_nf.csv'

# 确保输出目录存在
os.makedirs('./Model', exist_ok=True)

# Data Load

df = pd.read_csv(f'./{dataset_name}_dataset_cleaned.csv')


# 检查是否已经有处理好的文件，如果有，跳过繁重的预处理
processed_file = f'processed/processed_data_{dataset_name}.pt'
has_processed_data = os.path.exists(processed_file)

smiles_list = df['drug_smiles'].unique()
target_seq = df.set_index('target_key')['target_sequence'].to_dict()

# 加载词汇表
drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

# 预处理数据容器
drug_data = {}
target_data = {}



print(f"Preparing features for {dataset_name}...")

# drug data
for sm in smiles_list:
    emb, length = process_sequence(sm, drug_vocab, seq_len)
    graph = smiles_to_graph(sm) 
    drug_data[sm] = {'emb': emb, 'len': length, 'graph': graph}

# protein data
for target_key, seq in target_seq.items():
    emb, actual_len = process_sequence(seq, target_vocab, tar_len)
    
    if dataset_name == 'kiba':
        pocket_path = os.path.join(pocket_root, f"AF-{target_key}-F1-model_v4.pdb")
    else:
        pocket_path = os.path.join(pocket_root, f"{target_key}.pdb")

取
    if not has_processed_data and os.path.exists(pocket_path):
        protein_df = format.bp_to_df(format.read_pdb(pocket_path))
        graph = featurize_as_graph(protein_df) 
    else:
        graph = None 

    target_data[target_key] = {'emb': emb, 'len': actual_len, 'graph': graph}

# 构建数据集
dataset = DTADataset(
    root='./',
    path=f'./{dataset_name}_dataset_cleaned.csv',
    dataset_name=dataset_name, 
    smiles_emb={sm: data['emb'] for sm, data in drug_data.items()},
    target_emb={k: data['emb'] for k, data in target_data.items()},
    smiles_len={sm: data['len'] for sm, data in drug_data.items()},
    target_len={k: data['len'] for k, data in target_data.items()},
    mol_graphs_dict={sm: data['graph'] for sm, data in drug_data.items()},
    pro_graphs_dict={k: data['graph'] for k, data in target_data.items()}
)

seeds = [42, 123, 789, 555, 999]
metrics_list = []
patience = 30 
min_delta = 1e-4

for seed_idx, seed in enumerate(seeds):
    print(f"=============== Experiment with Seed {seed} ===============")
    
    # 严格的复现性设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_mse = float('inf')
    best_ci = 0
    best_rm2 = 0
    best_epoch = 0
    early_stop_counter = 0

    total_size = len(dataset)
    test_size = total_size // 6 
    train_size = total_size - test_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             collate_fn=collate_fn, shuffle=True, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            collate_fn=collate_fn, shuffle=False)
    
    model = CMDTA(embedding_dim=256, lstm_dim=128, hidden_dim=256, dropout_rate=0.2,
                  n_heads=8, bilstm_layers=2, protein_vocab=26,
                  smile_vocab=45, device=device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    model_file_name = f"./Model/best_model_{dataset_name}_seed{seed}.pt" # 模型保存名也带上 dataset_name
    
    print(f"Training set size: {train_size}, Test set size: {test_size}")
    
    for epoch in range(NUM_EPOCHS):
        train(model, train_loader, optimizer, epoch, device)
        G, P = predicting(model, test_loader, device)
        val_cindex, val_rm2, val_mse = calculate_metrics_and_return(G, P)
        
        if val_mse < best_mse - min_delta:
            best_mse = val_mse
            best_ci = val_cindex
            best_rm2 = val_rm2
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print(f'result improved at epoch {best_epoch}; best_mse {best_mse:.4f}; best_ci {best_ci:.4f}; best_rm2 {best_rm2:.4f}')
            early_stop_counter = 0
        else:
            print(f'current epoch: {epoch + 1} No result improved')
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        scheduler.step()    
    
    metrics_list.append({'seed': seed, 'cindex': best_ci, 'rm2': best_rm2, 'mse': best_mse})
    print(f"=============== End Seed {seed} Experiment ===============\n")

# Statistics
cindex_values = np.array([m['cindex'] for m in metrics_list])
rm2_values = np.array([m['rm2'] for m in metrics_list])
mse_values = np.array([m['mse'] for m in metrics_list])

stats = {
    'cindex': {'mean': cindex_values.mean(), 'std': cindex_values.std()},
    'rm2': {'mean': rm2_values.mean(), 'std': rm2_values.std()},
    'mse': {'mean': mse_values.mean(), 'std': mse_values.std()}
}

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Seed', 'CIndex', 'RM2', 'MSE'])
    for metric in metrics_list:
        writer.writerow([metric['seed'], f"{metric['cindex']:.4f}", f"{metric['rm2']:.4f}", f"{metric['mse']:.4f}"])
    writer.writerow([])  
    writer.writerow(['Metric', 'Mean', 'Std'])
    writer.writerow(['CIndex', f"{stats['cindex']['mean']:.4f}", f"{stats['cindex']['std']:.4f}"])
    writer.writerow(['RM2', f"{stats['rm2']['mean']:.4f}", f"{stats['rm2']['std']:.4f}"])
    writer.writerow(['MSE', f"{stats['mse']['mean']:.4f}", f"{stats['mse']['std']:.4f}"])
