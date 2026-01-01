from typing import Tuple, Any
import torch
import torch.nn as nn
import dhg
from dgl.nn.pytorch.conv import DGNConv
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 num_dgnn: int,
                 num_hgnn: int,
                 use_bn: bool = False) -> None:
        super().__init__()
        self.mapping_layer = feature_mapping_mlp(2816, in_channels)
        self.layers = nn.ModuleList()
        self.num_dgnn = num_dgnn
        self.num_hgnn = num_hgnn
        if self.num_dgnn == 1:
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir2-av', 'dir2-dx', 'sum'], ['identity', 'amplification'], 2.5))
        else:
            # input layer
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], ['identity', 'amplification'], 2.5))
            # hidden layer
            for i in range(self.num_dgnn - 1):
                self.layers.append(
                    DGNConv(hid_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], ['identity', 'amplification'],
                            2.5))

        if self.num_hgnn == 1:
            self.layers.append(HGNN(hid_channels, out_channels, use_bn=use_bn, is_last=True))
        else:
            for i in range(self.num_hgnn - 1):
                self.layers.append(HGNN(hid_channels, hid_channels, use_bn=use_bn))
            self.layers.append(HGNN(hid_channels, out_channels, use_bn=use_bn, is_last=True))

    def forward(self, m_emb: torch.Tensor, g, hg_pos: dhg.Hypergraph, hg_neg: dhg.Hypergraph) -> Any:
        mapping_features = self.mapping_layer(m_emb)
        for i in range(self.num_dgnn):
            mapping_features = self.layers[i](g, mapping_features, eig_vec=g.ndata['eig'])
        X1 = mapping_features
        X2 = mapping_features
        for i in range(self.num_dgnn, self.num_dgnn + self.num_hgnn):
            X1, Y_pos = self.layers[i](X1, hg_pos)
            X2, Y_neg = self.layers[i](X2, hg_neg)
        return X1, X2, Y_pos, Y_neg


class HGNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> Tuple[Any, Any]:
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        Y = hg.v2e(X, aggr="mean")
        X = hg.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X, Y


class Classifier(nn.Module):
    def __init__(
            self,
            node_embedding,
            metabolite_count,
            bottle_neck,
            **args):
        super().__init__()

        self.node_embedding = node_embedding
        self.is_tensor_embedding = isinstance(node_embedding, torch.Tensor)
        if self.is_tensor_embedding:
            self.node_embedding = self.node_embedding.to(args.get('device', 'cuda'))
        elif node_embedding is None:
            n_nodes = metabolite_count
            self.node_embedding = torch.randn(n_nodes, bottle_neck).to(args.get('device', 'cuda'))
            self.is_tensor_embedding = True

        # Simple MLP scoring layer (W * y + b) as per Eq 10 in NVM description
        self.score_layer = nn.Linear(bottle_neck, 1)

    def set_node_embedding(self, new_embedding):
        self.node_embedding = new_embedding
        self.is_tensor_embedding = isinstance(new_embedding, torch.Tensor)

    def forward(self, x, return_recon=False):
        # x: [batch_size, seq_len] containing node indices
        x = x.long()

        # Create mask for valid nodes (assuming 0 is padding)
        mask = (x != 0).float().unsqueeze(-1)

        # Get node embeddings
        if self.is_tensor_embedding:
            sz_b, len_seq = x.shape
            x_flat = x.view(-1)
            embedded_x = self.node_embedding[x_flat]
            embedded_x = embedded_x.view(sz_b, len_seq, -1)
        else:
            embedded_x = self.node_embedding(x)

        # Mean Pooling (Eq 9)
        # Sum embeddings across the sequence dimension and divide by the number of valid nodes
        sum_embeddings = torch.sum(embedded_x * mask, dim=1)
        sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9) # Avoid division by zero
        mean_embedding = sum_embeddings / sum_mask

        # Scoring (Eq 10)
        output = self.score_layer(mean_embedding)
        output = torch.sigmoid(output)

        return output


def feature_mapping_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )