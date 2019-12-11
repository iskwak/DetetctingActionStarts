"""Test batch_first for RNN's."""
from __future__ import print_function, division
import torch
import torch.autograd as autograd
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import gflags
import numpy as np
# import flags.lstm_flags as lstm_flags

class RNNTest(nn.Module):
    def __init__(self, batch_first):
        self.fc1 = nn.Linear(10, 5)
        self.relu1 = nn.ReLU()

        self.batch_first = False
        self.lstm = nn.LSTM(5, 3, 2, batch_first=batch_first)

    def forward(self, x, hidden):
        if self.batch_first is True:
            batch_idx = 1
        else:
            batch_idx = 0

        batch_size = x.size(batch_idx)

        # x is either: batch, seq, feats
        # or is: seq, batch, feats
        # seq is hardcoded to be 4
        # batch is batch_size
        # feat is 5.
        embeds = x.view(-1, 5)

        embeds = self.relu1(self.fc1(embeds))

        if self.batch_first is True:
            embeds = embeds.view(-1, 4, 3)
        else:
            embeds = embeds.view(-1, batch_size, 3)
        
        lstm_out, hidden = self.lstm(embeds, hidden)

        return lstm_out

    def init_hidden(self, batch_size, use_cuda=False):
        """Initializes the hidden and memory cell with given batch size."""
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # The tuple reprensents hidden cell and memory cell.
        hidden = [
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False),
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False)
        ]
        # hidden = [
        #     autograd.Variable(torch.zeros(
        #         self.n_layers, batch_size, self.hidden_dim),
        #         requires_grad=False)
        #     for i in range(len(self.input_dims))
        # ]
        if use_cuda is not None and use_cuda >= 0:
            hidden = [init_hid.cuda() for init_hid in hidden]

        hidden = tuple(hidden)
        return hidden
