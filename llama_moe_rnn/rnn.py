import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomGRUCell(nn.Module):
    def __init__(self, config):
        super().__init__(config=config)
        self.input_size = config.moe_router_rnn_hidden_size
        self.hidden_size = config.moe_router_rnn_hidden_size

        self.weight_ir = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hr = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_ir = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hr = nn.Parameter(torch.empty(self.hidden_size))

        self.weight_iz = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hz = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_iz = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hz = nn.Parameter(torch.empty(self.hidden_size))
        
        self.weight_in = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hn = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_in = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hn = nn.Parameter(torch.empty(self.hidden_size))

        
        self.config = config
        self.init_weights()
        
    def init_weights(self):
        for weight in self.parameters():
            self.config.init_method(weight)
        
    def forward(self, x, h=None):

        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

        r = torch.sigmoid(F.linear(x, self.weight_ir, self.bias_ir) + F.linear(h, self.weight_hr, self.bias_hr))
        z = torch.sigmoid(F.linear(x, self.weight_iz, self.bias_iz) + F.linear(h, self.weight_hz, self.bias_hz))
        n = torch.tanh(F.linear(x, self.weight_in, self.bias_in) + r * F.linear(h, self.weight_hn, self.bias_hn))
        h_new = (1 - z) * n + z * h

        return h_new




class CustomRNNCell(nn.Module):
    def __init__(self, config):
        super().__init__(config=config)
        self.input_size = config.moe_router_rnn_hidden_size
        self.hidden_size = config.moe_router_rnn_hidden_size

        self.weight_in = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hn = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_in = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hn = nn.Parameter(torch.empty(self.hidden_size))

        self.config = config
        self.init_weights()
        
    def init_weights(self):
        for weight in self.parameters():
            self.config.init_method(weight)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        h_new = torch.tanh(F.linear(x, self.weight_in, self.bias_in) + F.linear(h, self.weight_hn, self.bias_hn))
        return h_new
