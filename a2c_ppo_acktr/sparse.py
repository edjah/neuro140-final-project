import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseNNModule(nn.Module):
    def __init__(self, weight_shape, bias_shape=None, gain=np.sqrt(2)):
        super().__init__()
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.mask_version = 1
        self.gain = gain

        n = np.prod(weight_shape)
        self._weight = nn.Parameter(torch.Tensor(n))
        self.register_buffer('_mask_weight', torch.ones(n, dtype=torch.int8))

        if self.bias_shape is not None:
            m = np.prod(bias_shape)
            self._bias = nn.Parameter(torch.Tensor(m))
            self.register_buffer('_mask_bias', torch.ones(m, dtype=torch.int8))

        self.reset_parameters()

    def reset_parameters(self):
        weight_view = self._weight.view(self.weight_shape)
        nn.init.orthogonal_(weight_view, gain=self.gain)

        if self.bias_shape is not None:
            nn.init.constant_(self._bias, 0)

    def update_mask_version(self, version):
        # update the mask versions
        self.mask_version = version

    def reinit_pruned_weights(self):
        saved_weight = self._weight[self._mask_weight != 0].clone()
        saved_bias = None
        if self.bias_shape is not None:
            saved_bias = self._bias[self._mask_bias != 0].clone()

        self.reset_parameters()

        with torch.no_grad():
            self._weight[self._mask_weight != 0] = saved_weight
            self._mask_weight[self._mask_weight == 0] = self.mask_version

            if self.bias_shape is not None:
                self._bias[self._mask_bias != 0] = saved_bias
                self._mask_bias[self._mask_bias == 0] = self.mask_version

    @property
    def weight(self):
        return self.mask_weight * self._weight.view(self.weight_shape)

    @property
    def bias(self):
        if self.bias_shape is None:
            return None
        return self.mask_bias * self._bias.view(self.bias_shape)

    @property
    def mask_bias(self):
        if self.bias_shape is None:
            return None

        vals = (self._mask_bias > 0) & (self._mask_bias <= self.mask_version)
        return vals.view(self.bias_shape)

    @property
    def mask_weight(self):
        vals = (self._mask_weight > 0) & (self._mask_weight <= self.mask_version)
        return vals.view(self.weight_shape)


class SparseConv2d(SparseNNModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, gain=np.sqrt(2)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        bias_shape = (out_channels,) if bias else None
        super().__init__(weight_shape, bias_shape, gain)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class SparseLinear(SparseNNModule):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2)):
        self.in_features = in_features
        self.out_features = out_features

        weight_shape = (out_features, in_features)
        bias_shape = (out_features,) if bias else None
        super().__init__(weight_shape, bias_shape, gain)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
