import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class QNetwork(nn.Module):
    """Simple Q network."""

    def __init__(self, state_size, action_size, layer_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): number of actions
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingQNetwork(nn.Module):
    def __init__(self,state_size, action_size, layer_size, seed):
        super(DuelingQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc_value = nn.Linear(layer_size, layer_size)
        self.fc_adv = nn.Linear(layer_size, layer_size)

        self.value = nn.Linear(layer_size, 1)
        self.adv = nn.Linear(layer_size, action_size)

    def forward(self, state):
        y = torch.relu(self.fc1(state))
        value = torch.relu(self.fc_value(y))
        adv = torch.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

