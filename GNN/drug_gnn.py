import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DrugSideEffectGNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.lin(x))
