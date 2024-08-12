import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from collections import deque, namedtuple

#part1- building the AI

#network class
#architecture of our neural network
#the brain of AI
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed = 42) -> None: #means nothing will be returened 
        super(Network, self).__init__() #activates the inheritance
        self.seed = torch.manual_seed(seed) #generates the randome vector
        self.fc1 = nn.Linear(state_size, 64) #representing the first input layer between the full connected layer
        self.fc2 = nn.Linear(64, 64) #2nd fully connected layer between 1st and full connected layer
        self.fc3 = nn.Linear(64, action_size) 
