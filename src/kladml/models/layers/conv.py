
import torch
import torch.nn as nn

class CausalConv1d(torch.nn.Conv1d):
    """
    1D Convolution that respects causality (no future information leakage).
    """
    def __init__(self,
                  in_channels,
                  out_channels,
                  kernel_size,
                  stride=1,
                  dilation=1,
                  groups=1,
                  bias=True):
        self.__padding = (kernel_size - 1) * dilation
 
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
 
    def forward(self, input):
        result = super().forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
