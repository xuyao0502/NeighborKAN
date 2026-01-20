# ---------------------------
# two stage KAN
# ---------------------------

import torch
import torch.nn as nn

class KANLayer(nn.Module):
    """
    Simple KAN-style layer:
      linear -> tanh -> basis combination (sin on linear proj) -> final linear
    This is a practical KAN-like block suitable for (x,y)->(u,v) mapping.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=64, num_basis=32):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        # basis projection
        self.basis_proj = nn.Linear(hidden_dim, num_basis, bias=True)
        # final projection from basis to output
        self.final = nn.Linear(num_basis, out_dim, bias=True)

        # init
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.basis_proj.weight)
        nn.init.zeros_(self.basis_proj.bias)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        # x: [N, in_dim]
        h = self.act(self.lin(x))            # [N, hidden_dim]
        b = torch.sin(self.basis_proj(h))    # [N, num_basis]
        out = self.final(b)                  # [N, out_dim]
        return out

class KAN_MLP_pre(nn.Module):
    """
    KAN-based MLP: stack of KANLayer blocks.
    Output scaled by max_disp using tanh to keep pixel-range control.
    """
    def __init__(self, in_dim=2, out_dim=2, hidden_dim=64, num_layers=4, num_basis=32, max_disp=20.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(KANLayer(in_dim, hidden_dim, hidden_dim, num_basis))
            elif i == num_layers - 1:
                layers.append(KANLayer(hidden_dim, out_dim, hidden_dim, num_basis))
            else:
                layers.append(KANLayer(hidden_dim, hidden_dim, hidden_dim, num_basis))
        self.layers = nn.ModuleList(layers)
        self.max_disp = float(max_disp)

    def forward(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)
        out = torch.tanh(x) * self.max_disp
        return out

class KAN_MLP(nn.Module):
    """
    KAN-based MLP: stack of KANLayer blocks.
    Output scaled by max_disp using tanh to keep pixel-range control.
    """
    def __init__(self, in_dim=10, out_dim=2, hidden_dim=64, num_layers=4, num_basis=32, max_disp=20.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(KANLayer(in_dim, hidden_dim, hidden_dim, num_basis))
            elif i == num_layers - 1:
                layers.append(KANLayer(hidden_dim, out_dim, hidden_dim, num_basis))
            else:
                layers.append(KANLayer(hidden_dim, hidden_dim, hidden_dim, num_basis))
        self.layers = nn.ModuleList(layers)
        self.max_disp = float(max_disp)

    def forward(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)
        out = torch.tanh(x) * self.max_disp
        return out
