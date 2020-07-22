import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [] * count

        for transition in mini_batch:
            state, actions,reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(len(actions_lst)):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x,dtype= torch.float).to(device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(device), 
               actions_lst ,\
                torch.tensor(reward_lst).to(device), \
                torch.tensor(next_state_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)



learning_rate = 0.0001
class BQN(nn.Module):
    def __init__(self,action_num,input_length,action_scale,encodding_dim,head_num):
        super(BQN,self).__init__()

        self.q = QNetwork(action_num,input_length,action_scale,encodding_dim,head_num)
        self.target_q = QNetwork(action_num,input_length,action_scale,encodding_dim,head_num)
        
        for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam([\
                                    {'params' : self.q.encoder.parameters(),'lr': learning_rate / (count+2)},\
                                    {'params' : self.q.value.parameters(), 'lr' : learning_rate},\
                                    {'params' : self.q.actions.parameters(), 'lr' : learning_rate},\
                                    {'params' : self.q.self_attention.parameters(), 'lr' : learning_rate},\
                                    ],\
                                    lr = learning_rate)
    def action(self,x):
        return self.q(x)
    
    def train_mode(self):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        
        done_mask = torch.abs(done_mask-1)
        
        cur_actions = self.q(state)
        cur_actions = [x.reshape(batch_size,-1) for x in cur_actions]
        cur_actions = [x.gather(1,actions[idx].reshape(-1,1)) for idx, x in enumerate(cur_actions)]
        
        target_cur_actions = self.target_q(next_state)
        
        target_action_max_q = [x.max(-1)[0].reshape(batch_size,1) for x in target_cur_actions]
        
        target_action = [reward + done_mask * gamma * x for x in target_action_x_max_q]
        loss = [F.smooth_l1_loss(cur_actions[idx], target_action[idx].detach()) for idx in range(len(cur_actions))]
        loss = sum(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

class QNetwork(nn.Module):
    def __init__(self,action_num : int,input_length : int,action_scale : int,encodding_dim : int,head_num : int):
        super(QNetwork,self).__init__()
        #customize
        self.encoder = nn.Sequential(
                            nn.Conv1d(3,encodding_dim,1),
                            nn.ReLU(),
                            nn.BatchNorm1d(encodding_dim)
                            )
        self.self_attention = nn.MultiheadAttention(encodding_dim,head_num)
        self.actions = [nn.Sequential(nn.Linear(encodding_dim * input_length,128),
              nn.ReLU(),
              nn.Linear(128,action_scale)
              ) for _ in range(count)]

        self.actions = nn.ModuleList(self.actions)

        self.value = nn.Sequential(nn.Linear(encodding_dim * input_length,128),
              nn.ReLU(),
              nn.Linear(128,1)
              )
        
    def forward(self,x):
        encoded = self.encoder(x)
        encoded = encoded.transpose(1,0).transpose(0,2)
        encoded, attn_output_weights = self.self_attention(encoded,encoded,encoded)
        encoded = encoded.transpose(0,2).transpose(0,1)
        encoded = encoded.reshape(-1, (encoded.size()[1] * encoded.size()[2]))
        actions = [x(encoded) for x in self.actions]
        
        value = self.value(encoded)
        
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max().reshape(-1,1).detach()
            actions[i] += value
        
        return actions
