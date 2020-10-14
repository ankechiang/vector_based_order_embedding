import torch
import torch.nn as nn
import torch.nn.functional as func

class Order_Loss(torch.nn.Module):

def __init__(self):
    super(Order_Loss,self).__init__()
    
def forward(self,x,y):
	torch.pow(func.pairwise_distance(x,y), 2)
    ((input - target) ** 2).sum()

	self:add(nn.CSubTable())
	self:add(nn.AddConstant(params.eps))
	self:add(nn.ReLU())
	self:add(nn.Power(2))
	self:add(nn.Sum(2))

    return totloss

