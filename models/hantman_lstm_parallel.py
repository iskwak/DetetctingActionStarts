"""Hantman hungarian pytorch."""
from __future__ import print_function,division
import torch
import torch.autograd as autograd
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import gflags
import numpy as np
# import flags.lstm_flags as lstm_flags

gflags.DEFINE_string(
    "hantman_perframeloss", "MSE", "Perframe loss")
gflags.DEFINE_float(
    "hantman_perframe_weight", 1.0, "Perframe weighting")
gflags.DEFINE_float(
    "hantman_struct_weight", 1.0, "Structured weighting")
gflags.DEFINE_float("hantman_tp", 2.0, "Weight for true positives.")
gflags.DEFINE_float("hantman_fp", 1.0, "Weight for false positives.")
gflags.DEFINE_float("hantman_fn", 20.0, "Weight for false negatives.")
# gflags.ADOPT_module_key_flags(lstm_flags)


class HantmanHungarianConcat(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, opts=None, input_dims=[64], hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianConcat, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        self.opts = opts
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        self.relu1 = nn.ReLU()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

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

    # def forward(self, side, front, hidden):
    def forward(self, inputs):
        """Forward pass for the network."""
        batch_size = inputs[0].size(1)
        hidden = self.init_hidden(batch_size, use_cuda=True)
        both = torch.cat(inputs, dim=2)

        embeds = both.view(-1, sum(self.input_dims))
        # embeds = self.bn1(embeds)
        embeds = self.fc1(embeds)
        # embeds = self.fc1(both.view(-1, sum(self.input_dims)))
        embeds = self.relu1(embeds)

        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden


class HantmanHungarianSum(nn.Module):
    def __init__(self, opts=None, input_dims=[64], hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianSum, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        self.opts = opts
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        # self.fc1 = nn.Linear(sum(input_dim), hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for input_dim in input_dims]
        )
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size, use_cuda=None):
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
        if use_cuda is not None and use_cuda >= 0:
            hidden = [init_hid.cuda() for init_hid in hidden]

        hidden = tuple(hidden)
        return hidden

    # def forward(self, side, front, hidden):
    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        batch_size = inputs[0].size(1)

        embed_sum = None
        for feat, layer in zip(inputs, self.fcs):
            if embed_sum is None:
                embed_sum = layer(feat.view(-1, feat.size(2)))
            else:
                embed_sum = embed_sum + layer(feat.view(-1, feat.size(2)))
        # import pdb; pdb.set_trace()
        embeds = self.relu1(embed_sum)
        embeds = self.bn1(embeds)

        # re-expand the network outputs 6
        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden


def structured_loss(predict, frame_mask, tp_weight, fp_weight, num_false_neg):
    """Create the structured loss."""
    tp_weight = torch.autograd.Variable(torch.Tensor(tp_weight), requires_grad=False).cuda()
    fp_weight = torch.autograd.Variable(torch.Tensor(fp_weight), requires_grad=False).cuda()

    tp_cost = tp_weight * -predict * frame_mask
    fp_cost = fp_weight * predict * frame_mask

    tp_cost = tp_cost.sum(2).sum(0).view(-1)
    fp_cost = fp_cost.sum(2).sum(0).view(-1)

    fn_cost = torch.autograd.Variable(
        torch.Tensor(num_false_neg), requires_grad=False).cuda()

    return tp_cost, fp_cost, fn_cost


def perframe_loss(predict, mask, labels, pos_mask, neg_mask):
    """Get the perframe loss."""
    cost = predict * mask - labels * mask
    cost = cost * cost
    cost = (cost * pos_mask + cost * neg_mask)
    cost = cost.sum(2).sum(0).view(-1)
    return cost


def get_perframe_weight(opts, weight, t):
    decay_rate = 0.9
    decay_step = 500
    if opts["flags"].anneal is True:
        weight = np.float32(
            weight * (decay_rate**np.floor(t / decay_step)))
        # prevent underflow?
        weight = max(0.0001, weight)
        return weight
    else:
        return np.float32(weight)


def combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost):
    """Combine costs."""
    perframe_lambda = get_perframe_weight(
        opts, opts["flags"].hantman_perframe_weight, step)

    # package the weights... needs to be redone otherwise autograd gets confused
    # by repeat variables.... maybe? need to double check.
    tp_lambda = torch.autograd.Variable((torch.Tensor(
        [opts["flags"].hantman_tp]
    )), requires_grad=False).cuda().expand(tp_cost.size())
    fp_lambda = torch.autograd.Variable(torch.Tensor(
        [opts["flags"].hantman_fp]
    ), requires_grad=False).cuda().expand(fp_cost.size())
    fn_lambda = torch.autograd.Variable(torch.Tensor(
        [opts["flags"].hantman_fn]
    ), requires_grad=False).cuda().expand(fn_cost.size())
    perframe_lambda = torch.autograd.Variable(torch.Tensor(
        [float(perframe_lambda)]
    ), requires_grad=False).cuda().expand(perframe_cost.size())
    struct_lambda = torch.autograd.Variable(torch.Tensor(
        [opts["flags"].hantman_struct_weight]
    ), requires_grad=False).cuda().expand(tp_cost.size())

    tp_cost = tp_cost * tp_lambda * struct_lambda
    fp_cost = fp_cost * fp_lambda * struct_lambda
    fn_cost = fn_cost * fn_lambda * struct_lambda
    struct_cost = tp_cost + fp_cost + fn_cost

    perframe_cost = perframe_cost * perframe_lambda

    total_cost = perframe_cost + struct_cost

    return total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost


def create_pos_neg_masks(labels, pos_weight, neg_weight):
    """Create pos/neg masks."""
    temp = labels.data
    pos_mask = (temp > 0.9).float() * pos_weight
    pos_mask = torch.autograd.Variable(pos_mask, requires_grad=False)
    neg_mask = (temp < 0.000001).float() * neg_weight.expand(temp.size())
    # neg_mask = (temp < 0.9).float() * neg_weight.expand(temp.size())
    neg_mask = torch.autograd.Variable(neg_mask, requires_grad=False)

    return pos_mask, neg_mask


class HantmanHungarianBidirConcat(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, opts=None, input_dims=[64], hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianBidirConcat, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        self.opts = opts
        self.input_dims = input_dims
        # https://discuss.pytorch.org/t/trying-to-understand-behavior-of-bidirectional-lstm/4343
        self.hidden_dim = hidden_dim // 2
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        self.fc1 = nn.Linear(sum(self.input_dims), self.hidden_dim)
        self.relu1 = nn.ReLU()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, bidirectional=True
        )

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size, use_cuda=False):
        """Initializes the hidden and memory cell with given batch size."""
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # The tuple reprensents hidden cell and memory cell.
        hidden = [
            autograd.Variable(
                torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False),
            autograd.Variable(
                torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim),
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

    # def forward(self, side, front, hidden):
    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        batch_size = inputs[0].size(1)
        both = torch.cat(inputs, dim=2)

        embeds = both.view(-1, sum(self.input_dims))
        embeds = self.bn1(embeds)
        embeds = self.fc1(embeds)
        embeds = self.bn2(embeds)
        embeds = self.relu1(embeds)

        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        # import pdb; pdb.set_trace()
        return out, hidden


class HantmanHungarianImageConcat(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, opts=None, input_dims=[64], hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianImageConcat, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        self.opts = opts
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        # self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(100352 // 2, hidden_dim)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

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

    # def forward(self, side, front, hidden):
    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        # import pdb; pdb.set_trace()
        (seq_len, batch_size, dims) = inputs[0].size()

        img1s = []
        img2s = []
        for i in range(batch_size):
            img1 = inputs[0][:, i, :].contiguous().view(seq_len, 1, 224, 224)
            img1s.append(img1.contiguous().view(seq_len, 1, -1))

            # img2 = inputs[1][:, i, :].contiguous().view(seq_len, 1, 224, 224)
            # img2s.append(img2.contiguous().view(seq_len, 1, -1))

        seq1 = torch.cat(img1s, dim=1)
        both = seq1
        # seq2 = torch.cat(img2s, dim=1)
        # both = torch.cat([seq1, seq2], dim=2)
        # import pdb; pdb.set_trace()
        embeds = self.fc1(both.view(-1, 100352 // 2))
        embeds = self.relu1(embeds)
        # embeds = self.bn1(embeds)

        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden


class HantmanHungarianImageConcat2(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, opts=None, input_dims=[64], hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianImageConcat2, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        self.opts = opts
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        # self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(100352, hidden_dim)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

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

    # def forward(self, side, front, hidden):
    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        # import pdb; pdb.set_trace()
        (seq_len, batch_size, dims) = inputs[0].size()

        img1s = []
        img2s = []
        for i in range(batch_size):
            img1 = inputs[0][:, i, :].contiguous().view(seq_len, 1, 224, 224)
            img1s.append(img1.contiguous().view(seq_len, 1, -1))

            img2 = inputs[1][:, i, :].contiguous().view(seq_len, 1, 224, 224)
            img2s.append(img2.contiguous().view(seq_len, 1, -1))

        seq1 = torch.cat(img1s, dim=1)
        both = seq1
        # seq2 = torch.cat(img2s, dim=1)
        # both = torch.cat([seq1, seq2], dim=2)
        # import pdb; pdb.set_trace()
        embeds = self.fc1(both.view(-1, 100352))
        embeds = self.relu1(embeds)
        # embeds = self.bn1(embeds)

        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden
