import random
from collections import namedtuple, deque

def transition_pet():
    Transition = namedtuple('Transition', ('emotion', 'action', 'reward'))
    return Transition

def transition_first():
    Transition = namedtuple('Transition', ('emotion', 'action'))
    return Transition

def transition_second():
    Transition = namedtuple('Transition', ('reward'))
    return Transition

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.memory_first = deque([], maxlen=capacity)
        self.memory_second = deque([], maxlen=capacity)

    def push_first(self, *args):
        """Save a transition"""
        transition = transition_first()
        self.memory_first.append(transition(*args))
        print(f"push first : {transition(*args)}")
        if len(self.memory_second) != 0:
            first = self.memory_first.popleft()
            second = self.memory_second.popleft()
            transition = transition_pet()
            self.memory.append(transition(first.emotion, first.action, second.reward))

    def push_second(self, *args):
        """Save a transition"""
        transition = transition_second()
        self.memory_second.append(transition(*args))
        print(f"push second : {transition(*args)}")
        # queue 매칭 되면 memory에 추가하기
        if len(self.memory_first) != 0:
            first = self.memory_first.popleft()
            second = self.memory_second.popleft()
            transition = transition_pet()
            self.memory.append(transition(first.emotion, first.action, second.reward))

    def length(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)