import torch
import torch.nn as nn 

class LinearNorm(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(input_dim, output_dim,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)

class UNetDown(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, batchnorm=True, dropout=0.0, kernel=5, stride=2, padding=2, dilation=1):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(input_size, output_size, kernel, stride, padding, dilation, bias=False)]
        if normalize:
            if batchnorm:
                layers.append(nn.BatchNorm2d(output_size))
            else:
                layers.append(nn.InstanceNorm2d(output_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, batchnorm=True, dropout=0.0, kernel=5, stride=2, padding=2, dilation=1):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(input_size, output_size, kernel, stride, padding, dilation=dilation, bias=False)]
        if normalize:
            if batchnorm:
                layers.append(nn.BatchNorm2d(output_size))
            else:
                layers.append(nn.InstanceNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x