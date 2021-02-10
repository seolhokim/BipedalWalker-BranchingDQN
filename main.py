import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import time
import random
import argparse

from utils import ReplayBuffer
from agent import BQN

import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size(default : 64)')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma (default : 0.99)')
parser.add_argument('--action_scale', type=int, default=6, help='action scale between -1 ~ +1')

parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 1)')
args = parser.parse_args()

use_tensorboard = args.tensorboard
action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma

if use_tensorboard : 
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None
os.makedirs('./model_weights', exist_ok=True)


env = gym.make("BipedalWalker-v3")
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print('observation space : ', env.observation_space)
print('action space : ', env.action_space)
print(env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale), learning_rate, device)
if args.load != 'no':
    agent.load_state_dict(torch.load('./model_weights/'+args.load))
memory = ReplayBuffer(100000,action_space,device)
real_action = np.linspace(-1.,1., action_scale)

for n_epi in range(2000):
    state = env.reset()
    done = False
    score = 0.0
    while not done:
        if args.render :
            env.render()
        epsilon = max(0.01, 0.9 - 0.01*(n_epi/10))
        if epsilon > random.random():
            action = random.sample(range(0,(action_scale)),4)
        else:
            action_prob = agent.action(torch.tensor(state).float().reshape(1,-1).to(device))
            action = [int(x.max(1)[1]) for x in action_prob]
        next_state, reward, done, info = env.step(np.array([real_action[x] for x in action]))
        
        score += reward
        done = 0 if done == False else 1
        memory.put((state,action,reward,next_state, done))
        if (memory.size()>5000) and (args.train):
            agent.train_mode(n_epi, memory, batch_size, gamma, use_tensorboard,writer)
        state = next_state
    if use_tensorboard:
        writer.add_scalar("reward", score, n_epi)
    if (n_epi % args.save_interval == 0) and (n_epi > 0):
        torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
    if (n_epi % args.print_interval == 0):
        print("epi : ",n_epi,", score : ",score)
