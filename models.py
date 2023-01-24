import torch
import torch.nn as nn
import torch.nn.functional as F
from hypernet import Hypernet
from hypernet_prob import HypernetProb

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x



class FNNet(nn.Module):

    def __init__(self, input_dim, interm_dim=100, output_dim=10):
        super(FNNet, self).__init__()

        self.input_dim = input_dim
        self.dp1 = torch.nn.Dropout(0.2)
        self.dp2 = torch.nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, interm_dim)
        self.fc2 = nn.Linear(interm_dim, interm_dim)
        self.fc3 = nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc3(x)
        return x

    def embed(self, x):
        x = self.dp1(F.relu(self.fc1(x.view(-1, self.input_dim))))
        x = self.dp2(F.relu(self.fc2(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNet, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.dp3(self.fc1(x)))
        return x

class ConvNetNoDropout(nn.Module):
    def __init__(self, output_dim, maxpool=True, ifnormalize=False, base_hid=32):
        super(ConvNetNoDropout, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.ifnormalize = ifnormalize
        self.normalize = Normalize((0.1307,), (0.3081,))

    def forward(self, x, return_feat=False):
        if self.ifnormalize:
            x = self.normalize(x)
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

class ConvNetHyper(Hypernet):
    def __init__(self, output_dim, maxpool=True, base_hid=32, *args, **kwargs):
        super(ConvNetHyper, self).__init__(*args, **kwargs)
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def extract_feat(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

    def predict(self, feat):
        out = self.fc2(feat)
        return out

class ConvNetHyperProb(HypernetProb):
    def __init__(self, output_dim, maxpool=True, base_hid=32, *args, **kwargs):
        super(ConvNetHyperProb, self).__init__(*args, **kwargs)
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def extract_feat(self, x):
        x = F.relu(self.conv1(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.fc1(x))
        return x

    def predict(self, feat):
        out = self.fc2(feat)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_dim=3, planes=64):
        super(ResNet, self).__init__()
        self.planes = planes

        self.conv1 = nn.Conv2d(input_dim, self.planes,  kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, self.planes, self.planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.planes, 2*self.planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 2*self.planes, 4*self.planes, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 4*self.planes, 8*self.planes, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*self.planes * block.expansion, num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(inplanes, planes, stride))
            inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.embed(x)
        out = self.linear(out)
        return out

    def embed(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(input_dim=3, planes=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_dim=input_dim, planes=planes)

# class ScoreNet(torch.nn.Module):
#     def __init__(self, input=10, hidden=100, output=1):
#         super(ScoreNet, self).__init__()
#         self.linear1 = nn.Linear(input, hidden)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden, output)
#         # torch.nn.init.xavier_uniform(self.linear1.weight)
#         # torch.nn.init.xavier_uniform(self.linear2.weight)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         out = self.linear2(x)
#         # return out.flatten()
#         return torch.sigmoid(out)

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, ifbn=False):
        super(HiddenLayer, self).__init__()
        self.ifbn = ifbn
        self.fc = nn.Linear(input_size, output_size)
        if ifbn:
            self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        out = self.fc(x)
        if self.ifbn:
            out = self.bn(out)
        return self.relu(out)


class ScoreNet(nn.Module):
    def __init__(self,input=10, hidden=100, num_layers=1, ifbn=False, activation="sigmoid"):
        super(ScoreNet, self).__init__()
        # self.normalize = Normalize()
        self.activation = activation
        self.first_hidden_layer = HiddenLayer(input, hidden, ifbn)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden, hidden, ifbn) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        # x = self.normalize(x)
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            return torch.relu(x)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class LeNet(nn.Module):
    def __init__(self, input_dim=1, ifnormalize=True):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.ifnormalize = ifnormalize
        self.normalize = Normalize((0.1307,), (0.3081,))

    def forward(self, x):
        if self.ifnormalize:
            x = self.normalize(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

class LogisticRegressionProb(HypernetProb):
    def __init__(self, input_dim, out_dim=10, *args, **kwargs):
        super(LogisticRegressionProb, self).__init__(*args, **kwargs)
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, out_dim))
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)

    def forward(self, x):
        out = x @ self.fc
        return out

class LogisticRegression(Hypernet):
    def __init__(self, input_dim, out_dim=10, *args, **kwargs):
        super(LogisticRegression, self).__init__(*args, **kwargs)
        self.fc = torch.nn.Parameter(torch.zeros(input_dim, out_dim))
        self.num_params = len([p for p in self.parameters()])
        self.init_wdecay(self.weight_decay_type, self.weight_decay_init)


    def forward(self, x):
        out = x @ self.fc
        return out