import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALAR = 12
GRID_SIZE = 7

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_scalar=N_SCALAR, grid_size=GRID_SIZE):
        super(DQN, self).__init__()
        self.n_scalar = n_scalar
        self.grid_size = grid_size

        # CNN branch for the local grid
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 32 * grid_size * grid_size

        # FC branch for scalar features
        self.scalar_fc = nn.Sequential(
            nn.Linear(n_scalar, 64),
            nn.ReLU(),
        )

        # Combined head
        self.head = nn.Sequential(
            nn.Linear(conv_out_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        scalar = x[:, :self.n_scalar]
        grid = x[:, self.n_scalar:].view(-1, 1, self.grid_size, self.grid_size)

        conv_out = self.conv(grid).flatten(1)
        scalar_out = self.scalar_fc(scalar)
        return self.head(torch.cat([conv_out, scalar_out], dim=1))


class EnsembleDQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_networks=5):
        super(EnsembleDQN, self).__init__()
        self.networks = nn.ModuleList([DQN(n_observations, n_actions) for _ in range(n_networks)])

    def forward(self, x):
        outputs = torch.stack([net(x) for net in self.networks])
        return outputs.mean(dim=0)
