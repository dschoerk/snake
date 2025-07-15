from collections import deque, namedtuple
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

        # shuffle the memory to ensure randomness
        random.shuffle(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)