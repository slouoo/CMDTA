
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gvp.models import GVPModel
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU

# GIN
class GINConvNet(nn.Module):
    def __init__(self, num_features_xd=78, output_dim=128, num_layers=5, dropout=0.2):
        super().__init__()
        dim = 32
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 创建多层GIN卷积
        for _ in range(num_layers):
            nn_seq = Sequential(Linear(num_features_xd if len(self.convs)==0 else dim, dim), 
                              ReLU(), Linear(dim, dim))
            self.convs.append(GINConv(nn_seq))
            self.bns.append(nn.BatchNorm1d(dim))
        
        self.fc = Linear(dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(F.relu(x))
        
        x = global_add_pool(x, batch)
        return self.dropout(F.relu(self.fc(x)))
    
#
class ResidualAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(ResidualAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.tensor(0.1)) 
        
    def forward(self, x, masks):
        
        residual = x.mean(dim=1)
        query = self.query(x).transpose(1, 2) # (B,heads,seq_len)
        value = x # (B,seq_len,hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query) # (B,heads,seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e) # (B,heads,seq_len)

        out = torch.matmul(a, value) # (B,heads,seq_len) * (B,seq_len,hidden_dim) = (B,heads,hidden_dim)
        out = torch.mean(out, dim=1).squeeze() # (B,hidden_dim)
        out = out + self.alpha*residual
        return out, a





class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, lstm_dim=128, 
                 num_layers=2, dropout=0.2, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.fc = Linear(emb_dim, lstm_dim)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.ln = nn.LayerNorm(lstm_dim*2)
        self.attention = ResidualAttention(lstm_dim*2, n_heads)

    def forward(self, x, lengths):
        x = self.embed(x)  # (B, seq_len, emb_dim)
        x = self.fc(x)      # (B, seq_len, lstm_dim)
        x, _ = self.lstm(x) # (B, seq_len, lstm_dim*2)
        x = self.ln(x)
        mask = self._generate_mask(x, lengths, self.n_heads)
        out, _ =  self.attention(x, mask)

        return out, x, mask
    
    @staticmethod
    def _generate_mask(tensor, lengths, n_heads):
        batch_size, max_len = tensor.size(0), tensor.size(1)
        mask = torch.ones(batch_size, max_len)
        for i, length in enumerate(lengths):
            if length < max_len:
                mask[i, length:] = 0
        return mask.unsqueeze(1).expand(-1, n_heads, -1).to(tensor.device)
    


class CMDTA(nn.Module):
    def __init__(self, embedding_dim=256,
                 lstm_dim = 128,
                 hidden_dim = 256,
                 dropout_rate = 0.2,
                 n_heads = 8,
                 bilstm_layers=2,
                 protein_vocab=26,
                 smile_vocab=45,
                 device = 'cuda:0'
                 ):
        
        super(CMDTA, self).__init__()
                
        # gnn metheds
        self.GIN = GINConvNet()
        self.GVP = GVPModel()
        self.graph_fc = nn.Sequential(
                            nn.Linear(256+128, 256), 
                            nn.ReLU(),
                            nn.Dropout(dropout_rate),
                            nn.LayerNorm(256)
                        ) 
        
        self.SmilesEncoder = SeqEncoder(smile_vocab, embedding_dim, lstm_dim, bilstm_layers, dropout_rate, n_heads)
        self.ProteinEncoder = SeqEncoder(protein_vocab, embedding_dim, lstm_dim, bilstm_layers, dropout_rate, n_heads)

        # fusion
        self.out_attentions = ResidualAttention(hidden_dim, n_heads)
        
        self.fusion_nn = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*6),
            nn.ReLU(),
            nn.Dropout(dropout_rate),       
            nn.Linear(hidden_dim*6, hidden_dim*4)
        )

        self.fusion_nn2 = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),       
            nn.Linear(hidden_dim*8, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, data):
        
        # data unzip
        smiles_graph_feat, pocket_graph_feat, smiles_seq_feat, protein_seq_feat, smi_lens, prot_lens = data[:-1]

        # graph embedding
        smi_graph = self.GIN(smiles_graph_feat) # B * 128
        pocket_graph = self.GVP(pocket_graph_feat) # B * 256
        combined_graph = torch.cat([smi_graph, pocket_graph], dim=1)
        combined_graph = self.graph_fc(combined_graph) # B * 256
        
        # sequence embedding 
        smiles_seq_feat_out,smiles_seq_feat_hid, smiles_mask = self.SmilesEncoder(smiles_seq_feat,smi_lens)
        pro_seq_feat_out,pro_seq_feat_hid, protein_mask = self.ProteinEncoder(protein_seq_feat,prot_lens)

        # smiles and proteins
        out_cat = torch.cat((smiles_seq_feat_hid, pro_seq_feat_hid), dim=1)  # B * tar_len+seq_len * (lstm_dim*2)
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * head * tar_len+seq_len
        out_cat, _ = self.out_attentions(out_cat, out_masks)
        
        # prediction layer
        cat_out = torch.cat([smiles_seq_feat_out, pro_seq_feat_out, out_cat, combined_graph], dim=-1)  
        dta_value = self.fusion_nn(cat_out)
        dta_value = dta_value + cat_out
        dta_value = self.fusion_nn2(dta_value).squeeze()

        return dta_value