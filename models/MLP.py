#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F 
from torchvision import datasets
import torchvision.transforms as transforms


# In[ ]:


class MLP(torch.nn.Module):  
    def __init__(self,n_feature,n_label):
        super(MLP,self).__init__()  
        self.fc1 = torch.nn.Linear(n_feature,512) 
        self.fc2 = torch.nn.Linear(512,128) 
        self.fc3 = torch.nn.Linear(128,n_label)   
        
        self.n_feature = n_feature
        
    def forward(self,din):
        din = din.view(-1,self.n_feature)      
        #dropout = torch.nn.Dropout(p=0.2)
        #din = dropout(din)
        dout = F.relu(self.fc1(din))  
        dout = F.relu(self.fc2(dout))
        dout = torch.sigmoid(self.fc3(dout))
        return dout

