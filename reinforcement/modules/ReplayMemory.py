import random
from collections import namedtuple, deque

def create_transition():
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    return Transition

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        transition = create_transition()
        self.memory.append(transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)