import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        d = 128
        self.layer1 = nn.Linear(n_observations, d)
        self.layer2 = nn.Linear(d, d)
        self.layer3 = nn.Linear(d, n_actions)

        self.dropout = nn.Dropout(p=0.1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = self.dropout(x)  # Apply dropout to prevent overfitting
        return self.layer3(x)
    
class EnsembleDQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_networks=5):
        super(EnsembleDQN, self).__init__()
        self.networks = nn.ModuleList([DQN(n_observations, n_actions) for _ in range(n_networks)])

        self.dropout = nn.Dropout(p=0.2)  # Dropout layer to prevent overfitting

    def forward(self, x):
        outputs = [net(x) for net in self.networks]
        outputs = [self.dropout(out) for out in outputs]
        return torch.stack(outputs).mean(dim=0)