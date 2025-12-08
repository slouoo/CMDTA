import numpy as np
import networkx as nx
import rdkit.Chem as Chem
from math import sqrt
import torch
import torch.nn.functional as F
from torch_geometric import data as DATA
import torch_cluster
from torch_geometric.data import Batch
from torch_geometric.data import Data as PyGData

def set_gpu(data, device):

    return [g.to(device) for g in data]

def calculate_metrics_and_return(Y, P, dataset='kiba'):
    cindex = get_ci(Y, P)
    rm2 = get_rm2(Y, P)
    mse = get_mse(Y, P)
    return cindex, rm2, mse

def train(model, train_loader, optimizer, epoch, device):
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    loss_fn = torch.nn.MSELoss()
    
    for batch_idx, data in enumerate(train_loader):
        data = set_gpu(data, device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.float(), data[-1].squeeze().float())
        loss.backward()
        optimizer.step()

def predicting(model, loader, device):
    model.eval()
    total_preds = []
    total_labels = []
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data = set_gpu(data, device)            
            output = model(data)
            total_preds.append(output.cpu())
            total_labels.append(data[-1].squeeze().cpu())
            
    return torch.cat(total_labels).numpy().flatten(), torch.cat(total_preds).numpy().flatten()

# --- Metrics Functions ---
def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z if z != 0 else 0
    return ci

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / float(down))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

# --- Graph Features ---
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    return DATA.Data(x=torch.Tensor(features),
                     edge_index=torch.LongTensor(edge_index).transpose(1, 0))

_amino_acids = lambda x: {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7,
    'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}.get(x, 20)

def _normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16):
    # Fix: 使用输入 Tensor D 的 device，而不是写死 cpu
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

@torch.no_grad()
def featurize_as_graph(protein, num_rbf=16, device='cpu'):
    # device 参数现在会被正确使用
    name = protein['name']
    
    # 提取坐标
    coords_dict = {atom_name: protein[protein.name == atom_name][['x', 'y', 'z']].to_numpy() 
                   for atom_name in ['N', 'CA', 'C', 'O']}
    
    max_len = min([c.shape[0] for c in coords_dict.values()])
    coords = np.stack([coords_dict[k][:max_len] for k in ['N', 'CA', 'C', 'O']], axis=1)
    
    # 转为 Tensor 并移动到指定 device
    coords = torch.from_numpy(coords).float().to(device)
    
    seq = torch.as_tensor([_amino_acids(a) for a in protein[protein.name == 'CA']['resname'][:max_len]],
                            dtype=torch.long).to(device)
    
    mask = torch.isfinite(coords.sum(dim=(1,2)))
    coords[~mask] = float('inf') # 使用 python float inf 兼容性更好
    
    X_ca = coords[:, 1]
    edge_index = torch_cluster.radius_graph(X_ca, r=5.0)
    
    # 这里的函数调用将自动继承 device
    pos_embeddings = positional_embeddings(edge_index) 
    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)
    
    dihedrals = get_dihedrals(coords)                     
    orientations = get_orientations(X_ca)
    sidechains = get_sidechains(coords)
    
    node_s = dihedrals
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    
    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
    
    return DATA.Data(x=X_ca, seq=seq, node_s=node_s, node_v=node_v,
                     edge_s=edge_s, edge_v=edge_v, edge_index=edge_index)

def get_dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]

    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features

def positional_embeddings(edge_index, num_embeddings=16, period_range=[2, 1000]):
    # Fix: 使用 edge_index 的 device
    device = edge_index.device
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * sqrt(1 / 3) - perp * sqrt(2 / 3)
    return vec 

def collate_fn(batch):
    # 分子图批处理
    mol_batch = Batch.from_data_list([PyGData(x=d.x, edge_index=d.edge_index) for d in batch])
    
    # 蛋白质图批处理
    pro_batch = Batch.from_data_list([PyGData(
        x=d.pro_x, edge_index=d.pro_edge_index, node_s=d.pro_node_s,
        node_v=d.pro_node_v, edge_s=d.pro_edge_s, edge_v=d.pro_edge_v
    ) for d in batch])
    
    labels = torch.stack([d.y for d in batch])
    smiles_emb = torch.stack([d.smiles for d in batch]) 
    protein_emb = torch.stack([d.protein for d in batch])
    smiles_lengths = torch.tensor([d.smiles_lengths for d in batch], dtype=torch.long)
    protein_lengths = torch.tensor([d.protein_lengths for d in batch], dtype=torch.long) 
    
    return mol_batch, pro_batch, smiles_emb, protein_emb, smiles_lengths, protein_lengths, labels

def process_sequence(sequence, vocab, max_len):
    tokens = []
    flag = 0
    while flag < len(sequence):
        if flag+1 < len(sequence) and sequence[flag:flag+2] in vocab.stoi:
            tokens.append(vocab.stoi[sequence[flag:flag+2]])
            flag += 2
        else:
            tokens.append(vocab.stoi.get(sequence[flag], vocab.unk_index))
            flag += 1
    
    max_content = max_len - 2
    tokens = tokens[:max_content] if len(tokens) > max_content else tokens
    processed = [vocab.sos_index] + tokens + [vocab.eos_index]
    actual_len = len(tokens)
    
    if len(processed) < max_len:
        processed += [vocab.pad_index] * (max_len - len(processed))
    else:
        processed = processed[:max_len]
    
    return torch.tensor(processed), actual_len
