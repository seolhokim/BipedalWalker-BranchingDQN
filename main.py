import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import collections
import time
import random

use_tensorboard = True
if use_tensorboard : 
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    
class ReplayBuffer():
    def __init__(self,buffer_limit,action_space):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for i in range(self.action_space)]

        for transition in mini_batch:
            state, actions,reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(device),\
               actions_lst ,torch.tensor(reward_lst).to(device),\
                torch.tensor(next_state_lst, dtype=torch.float).to(device),\
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self,state_space : int, action_num : int,action_scale : int):
        super(QNetwork,self).__init__()
        self.linear_1 = nn.Linear(state_space,state_space*20)
        self.linear_2 = nn.Linear(state_space*20,state_space*10)
        
        self.actions = [nn.Sequential(nn.Linear(state_space*10,state_space*5),
              nn.ReLU(),
              nn.Linear(state_space*5,action_scale)
              ) for _ in range(action_num)]

        self.actions = nn.ModuleList(self.actions)

        self.value = nn.Sequential(nn.Linear(state_space*10,state_space*5),
              nn.ReLU(),
              nn.Linear(state_space*5,1)
              )
        
    def forward(self,x):
        x = F.relu(self.linear_1(x))
        encoded = F.relu(self.linear_2(x))
        actions = [x(encoded) for x in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            actions[i] += value
        return actions

class BQN(nn.Module):
    def __init__(self,state_space : int, action_num : int,action_scale : int):
        super(BQN,self).__init__()

        self.q = QNetwork(state_space, action_num,action_scale).to(device)
        self.target_q = QNetwork(state_space, action_num,action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam([\
                                    {'params' : self.q.linear_1.parameters(),'lr': learning_rate / (action_num+2)},\
                                    {'params' : self.q.linear_2.parameters(),'lr': learning_rate / (action_num+2)},\
                                    {'params' : self.q.value.parameters(), 'lr' : learning_rate/ (action_num+2)},\
                                    {'params' : self.q.actions.parameters(), 'lr' : learning_rate},\
                                    ])
        self.update_freq = 1000
        self.update_count = 0
    def action(self,x):
        return self.q(x)
    
    def train_mode(self,n_epi):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        
        done_mask = torch.abs(done_mask-1)
        
        cur_actions = self.q(state)
        cur_actions = [x.gather(1,actions[idx].reshape(-1,1).long()) for idx, x in enumerate(cur_actions)]
        
        target_cur_actions = self.target_q(next_state)
        target_action_max_q = [x.max(-1)[0].reshape(batch_size,1) for x in target_cur_actions]
        target_action = [reward + done_mask * gamma * x for x in target_action_max_q]
        
        loss = [(cur_actions[idx]- target_action[idx])**2 for idx in range(len(cur_actions))]
        if use_tensorboard:
            for idx in range(len(cur_actions)):
                writer.add_scalar("Loss/action_"+str(idx), loss[idx], n_epi)
        
        loss = torch.stack(loss).sum(0).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if (self.update_count % self.update_freq == 0) and (self.update_count > 0):
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())
        return loss

import gym
env = gym.make("BipedalWalker-v3")

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print('observation space : ', env.observation_space)
print('action space : ', env.action_space)
print(env.action_space.low, env.action_space.high)

action_scale = 6
learning_rate = 0.001
batch_size = 64
gamma = 0.99
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = BQN(state_space,action_space,(action_scale)).cuda()
else : 
    agent = BQN(state_space,action_space,(action_scale))
memory = ReplayBuffer(100000,action_space)
real_action = np.linspace(-1.,1., action_scale)

for n_epi in range(2000):
    state = env.reset()
    done = False
    time_step = 0
    score = 0.0
    while not done:
        #env.render()
        epsilon = max(0.01, 0.3 - 0.01*(n_epi/200))
        if epsilon > random.random():
            action = random.sample(range(0,int(action_scale)),4)
        else:
            action_prob = agent.action(torch.tensor(env.observation_space.sample()).to(device))
            action = [int(x.max(1)[1]) for x in action_prob]
        next_state, reward, done, info = env.step(np.array([real_action[x] for x in action]))
        score += reward
        done = 0 if done == False else 1
        memory.put((state,action,reward,next_state, done))
        if memory.size()>5000:
            agent.train_mode(n_epi)
        state = next_state
        time_step += 1
        if use_tensorboard:
            writer.add_scalar("reward", score, n_epi)

    time.sleep(1)
    print("epi : ",n_epi,", score : ",score)
