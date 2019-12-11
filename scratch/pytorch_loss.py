"""Play with custom losses."""
import numpy
import torch
import torch.nn as nn
import torchvision
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(246)

def custom_loss(y, y_mask):
    return (y * y_mask).sum()

# create a simple network
fc1 = nn.Linear(6, 3)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(6, 6)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(6, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

criterion = nn.MSELoss(size_average=False)

net = Net()

learning_rate = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
print(net)

def my_mse(y, y_mask, y_pred):
    return torch.mean(y * y_mask - y_pred * y_mask, 1).sum()

inputs = torch.autograd.Variable(torch.randn(4, 6))

mask = torch.autograd.Variable(torch.Tensor(
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
), requires_grad=False)
out_data = torch.randn(4, 3)
# outputs = torch.autograd.Variable(torch.randn(4, 3))
import pdb; pdb.set_trace()
for t in range(5000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = net(inputs)
    if t == 0:
        print(y_pred)
    outputs = torch.autograd.Variable(out_data.clone(), requires_grad=False)
    # Compute and print loss.
    # loss = criterion(y_pred, outputs)
    # import pdb; pdb.set_trace()
    temp = torch.mul(outputs, mask) - torch.mul(y_pred, mask)
    loss = torch.mean(torch.abs(temp * temp), 1).sum()
    # loss = torch.mean(outputs * mask - y_pred * mask, 1).sum()
    # import pdb; pdb.set_trace()
    if t % 100 == 0:
        print(t, loss.data[0])

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

import pdb; pdb.set_trace()