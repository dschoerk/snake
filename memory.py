from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class TensorReplayMemory:
    """Replay buffer stored as contiguous GPU tensors for fast sampling."""

    def __init__(self, capacity, obs_size, device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros(capacity, obs_size, device=device)
        self.actions = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros(capacity, obs_size, device=device)
        self.dones = torch.ones(capacity, dtype=torch.bool, device=device)  # init as done
        self.pos = 0
        self.size = 0

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push a batch of transitions. All inputs are tensors on device."""
        n = states.shape[0]
        if n == 0:
            return
        end = self.pos + n
        if end <= self.capacity:
            self.states[self.pos:end] = states
            self.actions[self.pos:end, 0] = actions
            self.rewards[self.pos:end] = rewards
            self.next_states[self.pos:end] = next_states
            self.dones[self.pos:end] = dones
        else:
            # Wrap around
            first = self.capacity - self.pos
            self.states[self.pos:] = states[:first]
            self.actions[self.pos:, 0] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_states[self.pos:] = next_states[:first]
            self.dones[self.pos:] = dones[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest, 0] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_states[:rest] = next_states[first:]
            self.dones[:rest] = dones[first:]
        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """Sample a batch. Returns (states, actions, rewards, next_states, non_final_mask)."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        non_final_mask = ~self.dones[idx]
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            non_final_mask,
        )

    def __len__(self):
        return self.size

    def save(self, filename):
        torch.save({
            'states': self.states[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'rewards': self.rewards[:self.size].cpu(),
            'next_states': self.next_states[:self.size].cpu(),
            'dones': self.dones[:self.size].cpu(),
            'pos': self.pos,
            'size': self.size,
        }, filename)

    def load(self, filename):
        data = torch.load(filename, weights_only=True)
        n = data['size']
        self.states[:n] = data['states'].to(self.device)
        self.actions[:n] = data['actions'].to(self.device)
        self.rewards[:n] = data['rewards'].to(self.device)
        self.next_states[:n] = data['next_states'].to(self.device)
        self.dones[:n] = data['dones'].to(self.device)
        self.pos = data['pos']
        self.size = n
