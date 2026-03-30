import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from script.ref_utility import calc_gso


class GraphChebConv(nn.Module):
    """Chebyshev-style graph convolution on (B, C, T, N)."""

    def __init__(self, c_in: int, c_out: int, k_order: int):
        super().__init__()
        self.k_order = k_order
        self.theta = nn.Parameter(torch.randn(k_order, c_in, c_out) * 0.1)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor, supports):
        # x: (B, C, T, N)
        out = 0.0
        for k in range(self.k_order):
            a_k = supports[k]  # (N, N)
            # propagate along node dimension
            xk = torch.einsum("bctn,nm->bctm", x, a_k)
            out = out + torch.einsum("bctn,co->botn", xk, self.theta[k])
        out = out + self.bias.view(1, -1, 1, 1)
        return out


class STConvBlock(nn.Module):
    """Temporal Conv -> Graph Conv -> Temporal Conv with residual."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int, k_t: int, k_order: int, dropout: float):
        super().__init__()
        pad_t = k_t // 2
        self.tconv1 = nn.Conv2d(c_in, c_hidden, kernel_size=(k_t, 1), padding=(pad_t, 0))
        self.gconv = GraphChebConv(c_hidden, c_hidden, k_order)
        self.tconv2 = nn.Conv2d(c_hidden, c_out, kernel_size=(k_t, 1), padding=(pad_t, 0))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, supports):
        # x: (B, C, T, N)
        h = F.silu(self.tconv1(x))
        h = F.silu(self.gconv(h, supports))
        h = self.tconv2(h)
        h = self.bn(h + self.res(x))
        return self.dropout(F.silu(h))


class STGCN_AE_Model(nn.Module):
    """
    STGCN-like autoencoder for graph-temporal reconstruction.
    Input / output: (B, T, N, F)
    """

    def __init__(
        self,
        input_dim: int,
        n_vertex: int,
        device: torch.device,
        hidden_dim: int = 64,
        k_order: int = 2,
        k_t: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.n_vertex = n_vertex
        self.input_dim = input_dim
        self.k_order = k_order

        self.enc1 = STConvBlock(input_dim, hidden_dim, hidden_dim, k_t, k_order, dropout)
        self.enc2 = STConvBlock(hidden_dim, hidden_dim, hidden_dim, k_t, k_order, dropout)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1)),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.dec1 = STConvBlock(hidden_dim, hidden_dim, hidden_dim, k_t, k_order, dropout)
        self.dec2 = STConvBlock(hidden_dim, hidden_dim, hidden_dim, k_t, k_order, dropout)
        self.out_proj = nn.Conv2d(hidden_dim, input_dim, kernel_size=(1, 1))

        self.supports = None

    def _calculate_support(self, edge_index: torch.Tensor):
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())),
            shape=(self.n_vertex, self.n_vertex),
        )
        gso = calc_gso(adj, "sym_renorm_adj")
        base = sp.coo_matrix(gso)

        dense = torch.tensor(base.toarray(), dtype=torch.float32, device=self.device)
        supports = [torch.eye(self.n_vertex, device=self.device)]
        for _ in range(1, self.k_order):
            supports.append(torch.matmul(supports[-1], dense))
        return supports

    def _build_support_if_needed(self, edge_index: torch.Tensor):
        if self.supports is None:
            with torch.no_grad():
                self.supports = self._calculate_support(edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # x: (B, T, N, F) -> (B, F, T, N)
        self._build_support_if_needed(edge_index)
        z = x.permute(0, 3, 1, 2)

        h1 = self.enc1(z, self.supports)
        h2 = self.enc2(h1, self.supports)
        h = self.bottleneck(h2)

        d1 = self.dec1(h, self.supports)
        d2 = self.dec2(d1 + h1, self.supports)
        out = self.out_proj(d2)

        # back to (B, T, N, F)
        return out.permute(0, 2, 3, 1)

    @staticmethod
    def reconstruction_error(x_hat: torch.Tensor, x: torch.Tensor):
        diff = torch.abs(x_hat - x)
        sample_err = diff.mean(dim=(1, 2, 3))
        node_err = diff.mean(dim=(1, 3))
        return sample_err, node_err
