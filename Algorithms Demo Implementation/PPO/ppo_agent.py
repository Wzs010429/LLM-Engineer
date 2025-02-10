from torch import nn
import torch
import numpy as np
from torch.distributions import Normal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

class Actor(nn.Module):  # Inherit from nn.Module
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


class Critic(nn.Module):  # Inherit from nn.Module
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
        batch_start_point = np.arange(0, num_state, self.BATCH_SIZE)

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
    def __init__(self, state_dim, batch_size, action_dim):
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 3e-4
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.EPOCHS = 10
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)

        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)

        self.replay_buffer = ReplayMemory(batch_size)

        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

        for epoch_i in range(self.EPOCHS):
            memo_states, memo_actions, memo_rewards, memo_values, memo_dones, memo_batches = self.replay_buffer.sample()
            T = len(memo_rewards)
            memo_Advantage = np.zeros(T, dtype=np.float32)  ## Corrected np.zeroes to np.zeros

            for t in range(T):
                discount = 1
                a_t = 0
                for k in range(t, T-1):
                    a_t +=  memo_rewards[k] + self.GAMMA * memo_values[k+1] * (1-int(memo_dones[k])) - memo_values[k]
                    discount *= self.GAMMA * self.LAMBDA
                memo_Advantage[t] = a_t
            
            with torch.no_grad():
                memo_advantages_tensor = torch.tensor(memo_Advantage).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_values).to(device)

            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

            for batch in memo_batches:
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                    old_pi = Normal(old_mu, old_sigma)  ## Corrected torch.dis to Normal
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])
                mu, sigma =self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])


                ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
                surr1 = ratio * memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * memo_advantages_tensor[batch]
                actor_loss = -torch.min(surr1, surr2).mean()
                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]
                batch_old_values = self.critic(memo_states_tensor[batch])
                critic_loss = nn.MSELoss()(batch_old_values, batch_returns)  ## Corrected nn.MSELoss to nn.MSELoss()
                       
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()



    def save_policy(self):
        torch.save(self.actor.state_dict(), "ppo_policy_pendulum_v1.para")