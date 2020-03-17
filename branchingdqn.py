import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, action_x_lst, action_y_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], [], [], []

        for transition in mini_batch:
            state, action_x, action_y, reward, next_state, done_mask = transition
            state_lst.append(state)
            action_x_lst.append([action_x])
            action_y_lst.append([action_y])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(state_lst, dtype=torch.float).to(device), torch.tensor(action_x_lst).to(device), \
                 torch.tensor(action_y_lst).to(device),torch.tensor(reward_lst).to(device), \
                torch.tensor(next_state_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self,image_width,image_height,action_x_dim,action_y_dim):
        super(QNetwork,self).__init__()
        #customize
        self.encoder = nn.Sequential(
                            nn.Conv2d(3,32,3, stride= 1 , padding = 1),
                            nn.LeLU(),
                            nn.Conv2d(32,64,5, stride= 1 , padding = 2),
                            nn.ReLU(),
                            nn.Conv2d(32,1,1, stride= 1 , padding = 0),
                            nn.LeLU()
                            )
        
        
        self.value = nn.Sequential(
                            nn.Linear(image_width * image_height,1024),
                            nn.ReLU(),
                            nn.Linear(1024,128),
                            nn.ReLU(),
                            nn.Linear(64,1)
                            )
        self.action_x = nn.Sequential(
                            nn.Linear(image_width * image_height,1024),
                            nn.ReLU(),
                            nn.Linear(1024,128),
                            nn.ReLU(),
                            nn.Linear(64,action_x_dim)
                            )
        self.action_y = nn.Sequential(
                            nn.Linear(image_width * image_height,1024),
                            nn.ReLU(),
                            nn.Linear(1024,128),
                            nn.ReLU(),
                            nn.Linear(64,action_y_dim)
                            )
        
    def forward(self,x):
        encoding = self.encoder(x)
        
        value = self.value(encoding)
        
        action_x = self.action_x(encoding)
        action_x = action_x - action_x.max(-1)[0].reshape(-1,1) #.detach()
        action_x = action_x + value
        
        action_y = self.action_y(encoding)
        action_y = action_y - action_y.max(-1)[0].reshape(-1,1) #.detach()
        action_y = action_y + value
        
        return action_x, action_y

learning_rate = 0.0001
class BQN(nn.Module):
    def __init__(self):
        super(BQN,self).__init__()

        self.q = QNetwork()
        self.target_q = QNetwork()
        
        for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
            target_param.data.copy_(param.data)
        
        self.optimizer = optim.Adam(self.q.parameters(),lr = learning_rate)

    def action(self,x):
        return self.q(x)
    
    def train_mode(self):
        state, action_x, action_y, reward, next_state, done_mask = memory.sample(batch_size)
        #raise Exception()
        done_mask = torch.abs(done_mask-1)
        
        action_x_q_prob, action_y_q_prob = self.q(state)

        action_x_q = action_x_q_prob.reshape(batch_size,-1)
        action_x_q = action_x_q.gather(1,action_x.reshape(-1,1))
        
        action_y_q = action_y_q_prob.reshape(batch_size,-1)
        action_y_q = action_y_q.gather(1,action_y.reshape(-1,1))
        
        
        target_action_x_q, target_action_y_q = self.target_q(next_state)
        
        target_action_x_max_q = target_action_x_q.max(-1)[0].reshape(batch_size,1)
        target_action_y_max_q = target_action_y_q.max(-1)[0].reshape(batch_size,1)
        
        
        
        target_action_x = reward + done_mask * gamma * target_action_x_max_q
        target_action_y = reward + done_mask * gamma * target_action_y_max_q
        

        loss = F.smooth_l1_loss(action_x_q, target_action_x.detach())

        loss += F.smooth_l1_loss(action_y_q, target_action_y.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss