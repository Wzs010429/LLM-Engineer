import gym
import os
import time
import torch
import numpy as np
from ppo_agent import PPOAgent

scenario = "Pendulum-v1"
env = gym.make(scenario)

NUM_EPISODES = 3000
NUM_STEPS = 200

# Directory to save the policy
current_dir = os.path.dirname(os.path.realpath(__file__))
model_file = current_dir + "/models/"
timestamp = time.strftime('%Y%m%d%H%M%S')


# Initialize the agent
# Input: state space 
# Output: action space

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
BATCH_SIZE = 25
UPDATE_EVERY = 50

REWARD_BUFFER = np.empty(shape=NUM_EPISODES)

agent = PPOAgent(STATE_DIM, BATCH_SIZE, ACTION_DIM)  ## Corrected initialization


## Training loop
best_reward = -2000

for episode_i in range(NUM_EPISODES):  
    # Reset the environment
    state, info = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEPS):
        action, value = agent.get_action(state)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        done = True if step_i == (NUM_STEPS-1) else done
        agent.replay_buffer.add_memo(state, action, reward, value, done)

        state = observation

        if (step_i+1) % UPDATE_EVERY == 0 or done:
            agent.update()   ## TODO: Implement update function in PPOAgent class

    if episode_reward >= -100 and episode_reward > best_reward:
        ## Save the policy
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model_file + f"ppo_actor_{timestamp}.pth")  ## Corrected save path
        print(f"best reward: {best_reward} at episode {episode_i}")

    REWARD_BUFFER[episode_i] = episode_reward  ## Moved outside the if condition
    print(f"Episode {episode_i}, Reward: {round(episode_reward, 2)}")  ## Corrected indentation

env.close()

np.savetxt(current_path + f'/ppo_reward_{timestamp}.txt', REWARD_BUFFER)