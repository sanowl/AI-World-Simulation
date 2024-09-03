import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AgentNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AgentNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x, hidden

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = AgentNN(state_size, 128, action_size).to(self.device)
        self.target_net = AgentNN(state_size, 128, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.hidden = None

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            action_values, self.hidden = self.policy_net(state, self.hidden)
            return action_values.max(1)[1].item()

    def learn(self):
        if len(self.memory) < 64:
            return

        experiences = self.memory.sample(64)
        batch = Experience(*zip(*experiences))

        state_batch = torch.FloatTensor(batch.state).unsqueeze(1).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        q_values, _ = self.policy_net(state_batch, None)
        q_values = q_values.gather(1, action_batch)

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch, None)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
