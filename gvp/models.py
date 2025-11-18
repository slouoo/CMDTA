import torch
import torch.nn as nn
from . import GVP, LayerNorm, GVPConvLayer
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import global_add_pool

class GVPModel(nn.Module):
    def __init__(self, node_in_dim = (6, 3) , node_h_dim= (256, 32), 
                 edge_in_dim= (32, 1), edge_h_dim=(32, 1), num_layers=3, drop_rate=0.1):
        
        super().__init__()
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
        
        self.ln = nn.LayerNorm(node_h_dim[0])
        

        self.readout = nn.Sequential(
            nn.Linear(node_h_dim[0], 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256))
        
        
    def forward(self, protein, batch=None):
        
        h_V = (protein.node_s, protein.node_v)
        edge_index = protein.edge_index
        h_E = (protein.edge_s, protein.edge_v)
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        out = self.ln(out)
        
        out = global_add_pool(out, protein.batch)
               
        out = self.readout(out)
        
        return out

