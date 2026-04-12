import torch
import torch.nn as nn
import numpy as np

class ReservoirCell(nn.Module):
    """
    author: Prof. Claudio Gallicchio (c.gallicch)   (originally in Keras/TF)

    edited and implemented in Pytorch by Dr. Corrado Baccheschi (Bakko000)
    """
    # builds a reservoir as a hidden dynamical layer for a recurrent neural network

    def __init__(self, input_size, units,
                 input_scaling=1.0, bias_scaling=1.0,
                 spectral_radius=0.99,
                 leaky=1, activation=torch.tanh,
                 **kwargs):
        super().__init__()
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky  # leaking rate
        self.activation = activation

        # build the recurrent weight matrix
        # uses circular law to determine the values of the recurrent weight matrix
        # rif. paper
        # Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
        # "Fast spectral radius initialization for recurrent neural networks."
        # INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))

        # build the recurrent weight matrix
        self.recurrent_kernel = nn.Parameter(
            torch.empty(self.units, self.units).uniform_(-value, value),
            requires_grad=False
        )

        # build the input weight matrix
        self.kernel = nn.Parameter(
            torch.empty(input_size, self.units).uniform_(-self.input_scaling, self.input_scaling),
            requires_grad=False
        )

        # initialize the bias
        self.bias = nn.Parameter(
            torch.empty(self.units).uniform_(-self.bias_scaling, self.bias_scaling),
            requires_grad=False
        )

    def forward(self, inputs, states):
        prev_output = states[0]

        input_part = torch.matmul(inputs, self.kernel)

        state_part = torch.matmul(prev_output, self.recurrent_kernel)

        if self.activation is not None:  # più il leaky è basso più conserva la storia precedente
            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]



import copy

class BidirectionalReservoir(nn.Module):
    """
    Bidirectional implementation for reservoir
    author: Dr. Corrado Baccheschi (Bakko000)
    """
    def __init__(self, cell, units):
        super().__init__()
        self.cell_forward = copy.deepcopy(cell)

        self.cell_backward = ReservoirCell(
            input_size=cell.kernel.shape[0],
            units=units,
            input_scaling=cell.input_scaling,
            bias_scaling=cell.bias_scaling,
            spectral_radius=cell.spectral_radius,
            leaky=cell.leaky,
            activation=cell.activation
        )
        self.units = units

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            output: Concatenated forward and backward outputs (batch, seq_len, units*2)
            states: List containing final forward and backward states
        """
        batch_size, seq_len, _ = x.shape

        # Forward pass
        forward_outputs = []
        forward_state = [torch.zeros(batch_size, self.units, device=x.device)]

        for t in range(seq_len):
            output, forward_state = self.cell_forward(x[:, t, :], forward_state)
            forward_outputs.append(output)

        forward_outputs = torch.stack(forward_outputs, dim=1)

        # Backward pass
        backward_outputs = []
        backward_state = [torch.zeros(batch_size, self.units, device=x.device)]

        for t in range(seq_len - 1, -1, -1):
            output, backward_state = self.cell_backward(x[:, t, :], backward_state)
            backward_outputs.append(output)

        backward_outputs = torch.stack(backward_outputs[::-1], dim=1)

        # Concatenate forward and backward
        output = torch.cat([forward_outputs, backward_outputs], dim=-1)

        return output, [forward_state, backward_state]
