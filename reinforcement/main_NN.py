import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .modules.Action_queue import ReplayMemory, transition_pet
from .model.DQN import DQN



class Reinforcement():
    def __init__(self):
        # 순서는 어케 해야할까 (init)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.TAU = 0.5
        self.LR = 0.2

        # 1. target, policy 선언
        self.n_observations = 5 # emotion 5개
        self.n_actions = 5 # 행동은 10개 정도
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprise"]
        self.reward_labels = ["negative", "nothing", "positive"]
        self.routine = 4
        self.batch_size = 4

        # 2. Replay Memory 만들어 두기
        self.Transition = transition_pet()
        self.memory = ReplayMemory(10000)

    # train (method 1 private)
    def __optimize_model(self):
        optimizer = optim.SGD(self.policy_net.parameters(), lr=self.LR)
        # batch 수만큼 가져오기
        transitions = self.memory.sample(self.batch_size)
        print("학습 시작")
        batch = self.Transition(*zip(*transitions))
        
        for idx, (emotion_batch, action_batch, reward_batch) in enumerate(zip(batch.emotion, batch.action, batch.reward)):
            print(f"emotion : {emotion_batch}")
            print(f"reward : {reward_batch}")
            # forward
            state_action_values = self.policy_net(emotion_batch)

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, reward_batch.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            optimizer.step()
            print(f"{idx}번째 학습 성공")

        # 5. target에 복사
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def first_process(self, emotion):
        # 3. emotion과 action 뽑아내서 저장 시키기
        # face에서 받은 값을 input에 설정 (method 2 public)
        emotion_index = torch.tensor(self.emotion_labels.index(emotion))
        emotioning = torch.nn.functional.one_hot(emotion_index, num_classes=len(self.emotion_labels)).float().to(self.device)

        actioning = torch.max(self.target_net(emotioning), dim=0)[1].item()
        actioning_tensor = torch.nn.functional.one_hot(torch.tensor(actioning), num_classes=self.n_actions).to(self.device)

        self.memory.push_first(emotioning, actioning_tensor)

        return actioning

    def second_process(self, reward):
        # hand에서 받은 값을 rewarding에 설정 (method 3 public)
        rewarding = torch.tensor([self.reward_labels.index(reward)-1], device=self.device) # 받은 값 넣기
        self.memory.push_second(rewarding)

        # 데이터가 4의 배수로 채워질 때 학습한다.
        if self.memory.length() % self.routine == 0 and self.memory.length() != 0:
            self.__optimize_model()
