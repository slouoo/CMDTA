import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data as PyGData

class DTADataset(InMemoryDataset):
    def __init__(self,
                 root, 
                 path, 
                 smiles_emb, 
                 target_emb, 
                 smiles_len,
                 target_len,
                 mol_graphs_dict,
                 pro_graphs_dict):
        
        # 存储所有必要参数
        self.path = path
        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len
        self.mol_graphs_dict = mol_graphs_dict
        self.pro_graphs_dict = pro_graphs_dict
        
        # 提前读取数据
        self.df = pd.read_csv(self.path)
        
        super(DTADataset, self).__init__(root)
        
        # 自动加载处理后的数据
        self.data = torch.load(self.processed_paths[0]) if os.path.exists(self.processed_paths[0]) else None
        if not self.data:
            self.process()
            torch.save(self.data, self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # 不需要原始文件

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        pass  # 不需要下载

    def process(self):
        self.data = []
        for i in tqdm(range(len(self.df))):  # 修复变量名
            row = self.df.iloc[i]
            sm = row['drug_smiles']
            target = row['target_key']
            
            # 获取分子图数据
            mol_graph = self.mol_graphs_dict[sm]
            
            # 获取蛋白质图数据
            pro_graph = self.pro_graphs_dict[target]
            
            # 构建数据对象
            data = PyGData(
                y=torch.FloatTensor([row['affinity']]),
                # 分子图特征
                x=mol_graph.x,
                edge_index=mol_graph.edge_index,
                # 蛋白质图特征
                pro_x=pro_graph.x,
                pro_node_s=pro_graph.node_s,  
                pro_node_v=pro_graph.node_v,
                pro_edge_s=pro_graph.edge_s,
                pro_edge_v=pro_graph.edge_v,
                pro_edge_index=pro_graph.edge_index,

                smiles=self.smiles_emb[sm],
                protein=self.target_emb[target],
                smiles_lengths=self.smiles_len[sm],
                protein_lengths=self.target_len[target]
            )
            self.data.append(data)

        # 应用预处理
        if self.pre_filter is not None:
            self.data = [d for d in self.data if self.pre_filter(d)]
            
        if self.pre_transform is not None:
            self.data = [self.pre_transform(d) for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]