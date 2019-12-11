
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class LSTMTest(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTMTest, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # The tuple reprensents hidden cell and memory cell.
        hidden = (
            autograd.Variable(
                torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                requires_grad=False).cuda(),
            autograd.Variable(
                torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                requires_grad=False).cuda()
        )
        return hidden

    def forward(self, seq):
        self.hidden = self.init_hidden()

        embeds = self.fc1(seq.view(-1, seq.size(2)))
        embeds = embeds.view(-1, self.batch_size, self.hidden_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, self.batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out

    # def repackage_hidden(h):
    #     """Wraps hidden states in new Variables, to detach them from their history."""
    #     if type(h) == Variable:
    #         return Variable(h.data)
    #     else:
    #         return tuple(repackage_hidden(v) for v in h)

input_dim = 6
hidden_dim = 4
output_dim = 3
batch_size = 2
seq_len = 10
lstm = LSTMTest(input_dim, hidden_dim, output_dim, batch_size)
lstm.cuda()

moo_data = torch.randn(seq_len, batch_size, input_dim)


# fc1 = nn.Linear(input_dim, hidden_dim)

# beep = []
# for i in range(batch_size):
#     beep.append(fc1(moo[:, i, :]))

# boop = fc1(moo.view(-1, moo.size(2)))

out_data = torch.randn(10, 2, 3)

criterion = nn.MSELoss(size_average=False).cuda()

learning_rate = 1e-3
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for t in range(50000):
    # Forward pass: compute predicted y by passing x to the model.
    outputs = autograd.Variable(out_data, requires_grad=False).cuda()
    moo = autograd.Variable(moo_data, requires_grad=True).cuda()

    y_pred = lstm(moo)

    # Compute and print loss.
    loss = criterion(y_pred, outputs)
    # import pdb; pdb.set_trace()
    # temp = torch.mul(outputs, mask) - torch.mul(y_pred, mask)
    # loss = torch.mean(torch.abs(temp * temp), 1).sum()
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