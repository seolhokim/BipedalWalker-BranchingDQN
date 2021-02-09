import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import time
import random

from utils import ReplayBuffer
from agent import BQN

use_tensorboard = True
if use_tensorboard : 
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

import gym
env = gym.make("BipedalWalker-v3")

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print('observation space : ', env.observation_space)
print('action space : ', env.action_space)
print(env.action_space.low, env.action_space.high)

action_scale = 6
learning_rate = 0.0001
batch_size = 64
gamma = 0.99

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device)
memory = ReplayBuffer(100000,action_space,device)
real_action = np.linspace(-1.,1., action_scale)

for n_epi in range(2000):
    state = env.reset()
    done = False
    score = 0.0
    while not done:
        #env.render()
        epsilon = max(0.01, 0.3 - 0.01*(n_epi/200))
        if epsilon > random.random():
            action = random.sample(range(0,(action_scale)),4)
        else:
            action_prob = agent.action(torch.tensor(state).float().reshape(1,-1).to(device))
            action = [int(x.max(1)[1]) for x in action_prob]
        next_state, reward, done, info = env.step(np.array([real_action[x] for x in action]))
        
        score += reward
        done = 0 if done == False else 1
        memory.put((state,action,reward,next_state, done))
        if memory.size()>5000:
            agent.train_mode(n_epi, memory, batch_size, gamma, use_tensorboard,writer)
        state = next_state
    if use_tensorboard:
        writer.add_scalar("reward", score, n_epi)
    #time.sleep(1)
    print("epi : ",n_epi,", score : ",score)
