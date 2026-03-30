import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from model.ref_layers import OutputBlock
from script.ref_utility import calc_gso

class DCRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, K):
        super(DCRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = K
        
        # ============================================================
        # [核心修复] 修改 num_supports 为 1
        # ============================================================
        # 因为我们使用的是对称归一化矩阵 (sym_renorm_adj)，只有一个图结构
        # 之前的代码设为 2 是为了处理有向图 (forward + backward)
        self.num_supports = 1 
        
        self.gate_linear = nn.Linear( (self.num_supports * K + 1) * (input_dim + hidden_dim), 2 * hidden_dim )
        self.cand_linear = nn.Linear( (self.num_supports * K + 1) * (input_dim + hidden_dim), hidden_dim )

    def _diffusion(self, x, supports):
        # x: (Batch, Nodes, Features)
        out = [x]
        for support in supports:
            x0 = x
            # (Nodes, Nodes) @ (Batch, Nodes, Features) -> (Batch, Nodes, Features)
            x1 = torch.einsum('nm, bnc -> bmc', support, x0)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = 2 * torch.einsum('nm, bnc -> bmc', support, x1) - x0
                out.append(x2)
                x1, x0 = x2, x1
        return torch.cat(out, dim=-1)

    def forward(self, x, h, supports):
        inp = torch.cat([x, h], dim=-1)
        diff_out = self._diffusion(inp, supports)
        gates = self.gate_linear(diff_out)
        r, u = torch.split(gates, self.hidden_dim, dim=-1)
        r = torch.sigmoid(r)
        u = torch.sigmoid(u)
        h_reset = r * h
        inp_r = torch.cat([x, h_reset], dim=-1)
        diff_out_r = self._diffusion(inp_r, supports)
        c = torch.tanh(self.cand_linear(diff_out_r))
        new_h = u * h + (1.0 - u) * c
        return new_h

class DCRNN_Class_Model(nn.Module):
    def __init__(self, args, n_vertex, device):
        super(DCRNN_Class_Model, self).__init__()
        self.device = device
        self.n_vertex = n_vertex
        self.hidden_dim = 64
        self.K = 2 
        
        self.output_dim = 1
        input_dim = 1 + args.n_static_features
        
        self.dcrnn_cell = DCRNNCell(input_dim, self.hidden_dim, self.K)
        
        self.output_block = OutputBlock(
            Ko=1, 
            last_block_channel=self.hidden_dim, 
            channels=[64, 32], 
            end_channel=self.output_dim, 
            n_vertex=n_vertex, 
            act_func='glu', 
            bias=True, 
            droprate=0.3
        )
        
        self.supports = None

    def _calculate_random_walk_matrix(self, edge_index):
        # 使用 ref_utility 计算对称归一化矩阵
        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
                            shape=(self.n_vertex, self.n_vertex))
        gso = calc_gso(adj, 'sym_renorm_adj') 
        gso = sp.coo_matrix(gso)
        values = gso.data
        indices = np.vstack((gso.row, gso.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = gso.shape
        tensor_gso = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device).to_dense()
        
        # 返回列表，包含这唯一的一个矩阵
        return [tensor_gso]

    def forward(self, x, edge_index, grouping_matrix):
        if self.supports is None:
            with torch.no_grad():
                self.supports = self._calculate_random_walk_matrix(edge_index)
        
        # 维度对齐
        B, D1, D2, D3 = x.shape
        target_time = 12 
        target_nodes = self.n_vertex
        if D1 == target_time:
            if D3 == target_nodes: x = x.permute(0, 1, 3, 2)
        elif D2 == target_time:
            if D1 == target_nodes: x = x.permute(0, 2, 1, 3)
            elif D3 == target_nodes: x = x.permute(0, 2, 3, 1)
        elif D3 == target_time:
            if D2 == target_nodes: x = x.permute(0, 3, 2, 1)
            elif D1 == target_nodes: x = x.permute(0, 3, 1, 2)
            
        batch_size, seq_len, num_nodes, _ = x.shape
        
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :, :] 
            h = self.dcrnn_cell(x_t, h, self.supports)
            
        h_reshaped = h.permute(0, 2, 1).unsqueeze(2)
        logits = self.output_block(h_reshaped) 
        logits = logits.squeeze(2).squeeze(1) 
        
        return logits