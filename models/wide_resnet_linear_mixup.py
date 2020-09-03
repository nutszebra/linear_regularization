import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
try:
    from .prototype import NN
except:
    from prototype import NN


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(NN):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.conv1 = Conv2d(in_planes, planes, (3, 3), 1, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, (3, 3), stride, 1)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        # self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.conv2.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv2))
        # self.conv2.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv2, constant=0))
        if self.shortcut is not None:
            self.shortcut.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.shortcut))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.shortcut is None:
            out += x
        else:
            out += self.shortcut(x)
        return out


class Wide_ResNet(NN):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, alpha=1.0, beta=1.0, base=16, name=None):
        super(Wide_ResNet, self).__init__()
        self.in_planes, self.num_classes = base, num_classes
        self.alpha, self.beta = alpha, beta

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [base, base * k, base * k * 2, base * k * 4]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes, bias=True)
        self.name = name

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        [layer.weight_initialization() for layer in layers]
        return nn.Sequential(*layers)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.linear.weight.data = torch.FloatTensor(np.random.uniform(-0.1, 0.1, self.linear.weight.data.shape))
        self.linear.bias.data = torch.FloatTensor(NN.bias_initialization(self.linear, constant=0.0))

    def mixup(self, x):
        beta = np.random.beta(self.beta, self.beta)
        return beta * x + (1.0 - beta) * self.flip(x, 0), beta

    @staticmethod
    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, x, t=None, mixup=True):

        if (self.training is True) and (mixup is True):
            x, beta = self.mixup(x)
            beta = torch.Tensor([beta]).to(x.device)
        else:
            beta = torch.Tensor([0.0]).to(x.device)

        out = self.conv1(x)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = F.relu(self.bn1(out3))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training is True or (mixup is False):
            return out, out3, out2, out1, x, beta
        else:
            return out

    @staticmethod
    def cosine(z1, z2, eps=1.0e-8):
        return ((z1 * z2) / (torch.norm(z1, dim=1) * torch.norm(z2, dim=1) + eps).unsqueeze(1)).sum(1).mean()

    @staticmethod
    def l2loss(z1, z2, eps=1.0e-8):
        z1_norm = z1 / (torch.norm(z1, dim=1) + eps).view(-1, 1)
        z2_norm = z2 / (torch.norm(z2, dim=1) + eps).view(-1, 1)
        loss = ((z1_norm - z2_norm) ** 2).sum(1).mean()
        return loss

    def calc_loss(self, y, t, reduction='elementwise_mean'):
        if self.training is True:
            p, out3, out2, out1, x, beta = y
            t2 = self.flip(t.view(-1, int(p.shape[0] / beta.shape[0])), 1).view(-1)
            beta = beta.unsqueeze(1).repeat(1, int(p.shape[0] / beta.shape[0])).view(-1)
            loss = (beta * F.cross_entropy(p, t, reduction='none') + (1.0 - beta) * F.cross_entropy(p, t2, reduction='none')).mean()
            x1, x2 = beta.view(-1, 1, 1, 1) * x, (1.0 - beta.view(-1, 1, 1, 1)) * x
            x_mix = x1 + x2
            self.eval()
            _p, _out3, _out2, _out1, _x, _beta = self(torch.cat((x1, x2, x_mix), 0), mixup=False)
            self.train()
            loss_linear = 0.0
            import IPython
            IPython.embed()
            _out3_1, _out3_2, _out3_mix = _out3[:x.shape[0]], _out3[x.shape[0]: int(x.shape[0] * 2)], _out3[int(x.shape[0] * 2):]
            loss_linear = loss_linear + self.alpha * F.mse_loss(_out3_1 + _out3_2, _out3_mix) 
            _out2_1, _out2_2, _out2_mix = _out2[:x.shape[0]], _out2[x.shape[0]: int(x.shape[0] * 2)], _out2[int(x.shape[0] * 2):]
            loss_linear = loss_linear + self.alpha * F.mse_loss(_out2_1 + _out2_2, _out2_mix) 
            _out1_1, _out1_2, _out1_mix = _out1[:x.shape[0]], _out1[x.shape[0]: int(x.shape[0] * 2)], _out1[int(x.shape[0] * 2):]
            loss_linear = loss_linear + self.alpha * F.mse_loss(_out1_1 + _out1_2, _out1_mix) 
            loss = loss + loss_linear
        else:
            loss = F.cross_entropy(y, t)
        return loss
