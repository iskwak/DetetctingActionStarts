"""Hantman hungarian pytorch."""
from __future__ import print_function, division
import torch
import torch.autograd as autograd
import torch.nn as nn


class HantmanHungarianConcat(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, input_dims=[64], hidden_dim=64,
                 output_dim=6):
        super(HantmanHungarianConcat, self).__init__()
        # input_dims is a list of dimensions. The network will concatenate
        # the inputs into one giant vector/matrix as input to the network.
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # output dim/number of classes.
        self.output_dim = output_dim

        # network layers
        self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        self.relu1 = nn.ReLU()

        # The LSTM takes in embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to label space.
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, FLAGS, batch_size):
        """Initializes the hidden and memory cell with given batch size."""
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # The list reprensents hidden cell and memory cell.
        hidden = [
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False),
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False)
        ]

        # put the hidden values on the gpu.
        if FLAGS.cuda_device != -1:
            hidden = [init_hid.cuda() for init_hid in hidden]

        # The LSTM expects a tuple, convert the list to at tuple.
        hidden = tuple(hidden)
        return hidden

    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        # inputs dimensions should be:
        # num_layers, minibatch_size, feature/input dimension.
        batch_size = inputs[0].size(1)
        # Concatenate everything on the feature dimension.
        both = torch.cat(inputs, dim=2)

        # "Flatten" the inputs into (num_layers * minibatch_size, feature dims).
        # This is necessary because the first layer to the network is a fully
        # connected layer (not expecting a sequence).
        embeds = both.view(-1, sum(self.input_dims))
        embeds = self.fc1(embeds)
        embeds = self.relu1(embeds)

        # Convert the representation back into sequence forme.
        embeds = embeds.view(-1, batch_size, self.hidden_dim)

        # The lstm module takes care of the recurrence.
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Again, to pass the lstm outputs into a fully connected layer, convert
        # the lstm outputs into non sequence form.
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))

        # To make the future sequence based loss calculations easier, change
        # the view back into sequence form.
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden
