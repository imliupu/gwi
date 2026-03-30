import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from model.DCRNN_Class_Model import DCRNNCell
from script.ref_utility import calc_gso


class DCRNN_AE_Model(nn.Module):
    """
    Graph-temporal autoencoder for anomaly detection.
    Input/Output shape: (B, T, N, F)
    """

    def __init__(self, input_dim: int, n_vertex: int, device: torch.device, hidden_dim: int = 64, k_order: int = 2):
        super().__init__()
        self.device = device
        self.n_vertex = n_vertex
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_order = k_order

        self.encoder_cell = DCRNNCell(input_dim=input_dim, hidden_dim=hidden_dim, K=k_order)
        self.decoder_cell = DCRNNCell(input_dim=input_dim, hidden_dim=hidden_dim, K=k_order)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

        self.supports = None

    def _calculate_support(self, edge_index: torch.Tensor):
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
            shape=(self.n_vertex, self.n_vertex),
        )
        gso = calc_gso(adj, 'sym_renorm_adj')
        gso = sp.coo_matrix(gso)
        indices = np.vstack((gso.row, gso.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(gso.data)
        tensor_gso = torch.sparse.FloatTensor(i, v, torch.Size(gso.shape)).to(self.device).to_dense()
        return [tensor_gso]

    def _build_support_if_needed(self, edge_index: torch.Tensor):
        if self.supports is None:
            with torch.no_grad():
                self.supports = self._calculate_support(edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x: (B, T, N, F)
        return: x_hat (B, T, N, F)
        """
        self._build_support_if_needed(edge_index)
        batch_size, seq_len, num_nodes, _ = x.shape

        # Encode
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
        for t in range(seq_len):
            h = self.encoder_cell(x[:, t, :, :], h, self.supports)

        # Decode (autoregressive)
        x_hat_steps = []
        dec_h = h
        prev = torch.zeros(batch_size, num_nodes, self.input_dim, device=x.device)
        for _ in range(seq_len):
            dec_h = self.decoder_cell(prev, dec_h, self.supports)
            pred = self.out_proj(dec_h)
            x_hat_steps.append(pred)
            prev = pred

        x_hat = torch.stack(x_hat_steps, dim=1)
        return x_hat

    @staticmethod
    def reconstruction_error(x_hat: torch.Tensor, x: torch.Tensor):
        """Returns per-sample error and per-node error."""
        diff = torch.abs(x_hat - x)
        # sample-level: mean over T,N,F -> (B,)
        sample_err = diff.mean(dim=(1, 2, 3))
        # node-level: mean over T,F -> (B,N)
        node_err = diff.mean(dim=(1, 3))
        return sample_err, node_err
