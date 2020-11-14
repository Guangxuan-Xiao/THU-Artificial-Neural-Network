import torch
from torch import nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        # return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state):
        # flag indicates whether the position is valid. 1 for valid, 0 for invalid.
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output  # stored for next step
        return output, new_state


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_layer = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.hidden_layer = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state
        return torch.randn(batch_size, self.hidden_size, device=device)
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        gate_x = self.input_layer(incoming).squeeze()
        gate_h = self.hidden_layer(state).squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        reset_gate = F.sigmoid(i_r + h_r)
        input_gate = F.sigmoid(i_i + h_i)
        new_gate = F.tanh(i_n + (reset_gate * h_n))
        output = new_gate + input_gate * (state - new_gate)
        return output, output
        # TODO END


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.input_layer = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hidden_layer = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state (which can be a tuple)
        return torch.randn(batch_size, self.hidden_size, device=device), torch.randn(batch_size, self.hidden_size, device=device)
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        h, c = state
        gates = self.input_layer(incoming) + self.hidden_layer(h)
        gates = gates.squeeze()
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        in_gate = F.sigmoid(in_gate)
        forget_gate = F.sigmoid(forget_gate)
        cell_gate = F.sigmoid(cell_gate)
        out_gate = F.sigmoid(out_gate)

        new_c = torch.mul(c, forget_gate) + torch.mul(in_gate, cell_gate)
        new_h = torch.mul(out_gate, F.tanh(new_c))
        return new_h, (new_h, new_c)
        # TODO END
