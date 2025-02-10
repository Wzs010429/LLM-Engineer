from torch import nn
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

class Actor:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * 2
        std = self.softplus(self.fc_std(x)) + 1e-3 ## Avoiding zero standard deviation

        return mean, std
    
    def select_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(state)
            normal_dist = torch.distributions.Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-2, 2)

        return action


class Critic:
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)

        return value



class ReplayMemory:
    def __init__(self, batch_size):
        self.BATCH_SIZE = batch_size
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []

    def add_memo(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)


    def sample(self):
        num_state = len(self.state_cap)
        batch_start_point = np.range(0, num_state, self.BATCH_SIZE)

        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)

        batches = [memory_indices[i:i+self.BATCH_SIZE] for i in batch_start_point]

        return (np.array(self.state_cap), 
                np.array(self.action_cap), 
                np.array(self.reward_cap), 
                np.array(self.value_cap),  
                np.array(self.done_cap), 
                batches)

    
    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        



class PPOAgent: