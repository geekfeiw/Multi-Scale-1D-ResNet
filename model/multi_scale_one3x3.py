import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3_1(nn.Module):
    expansion = 1

    def __init__(self, inplanes3_1, planes, stride=1, downsample=None):
        super(BasicBlock3x3_1, self).__init__()
        self.conv1 = conv3x3(inplanes3_1, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock3x3_2(nn.Module):
    expansion = 1

    def __init__(self, inplanes3_2, planes, stride=1, downsample=None):
        super(BasicBlock3x3_2, self).__init__()
        self.conv1 = conv3x3(inplanes3_2, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock3x3_3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3_3, planes, stride=1, downsample=None):
        super(BasicBlock3x3_3, self).__init__()
        self.conv1 = conv3x3(inplanes3_3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3_1 = 64
        self.inplanes3_2 = 64
        self.inplanes3_3 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_11 = self._make_layer3_1(BasicBlock3x3_1, 64, layers[0], stride=2)
        self.layer3x3_12 = self._make_layer3_1(BasicBlock3x3_1, 128, layers[1], stride=2)
        self.layer3x3_13 = self._make_layer3_1(BasicBlock3x3_1, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3_1 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.layer3x3_21 = self._make_layer3_2(BasicBlock3x3_2, 64, layers[0], stride=2)
        self.layer3x3_22 = self._make_layer3_2(BasicBlock3x3_2, 128, layers[1], stride=2)
        self.layer3x3_23 = self._make_layer3_2(BasicBlock3x3_2, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3_2 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.layer3x3_31 = self._make_layer3_3(BasicBlock3x3_3, 64, layers[0], stride=2)
        self.layer3x3_32 = self._make_layer3_3(BasicBlock3x3_3, 128, layers[1], stride=2)
        self.layer3x3_33 = self._make_layer3_3(BasicBlock3x3_3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3_3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)



        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256*3, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3_1(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3_1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3_1, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3_1, planes, stride, downsample))
        self.inplanes3_1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3_1, planes))

        return nn.Sequential(*layers)

    def _make_layer3_2(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3_2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3_2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3_2, planes, stride, downsample))
        self.inplanes3_2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3_2, planes))

        return nn.Sequential(*layers)


    def _make_layer3_3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3_3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3_3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3_3, planes, stride, downsample))
        self.inplanes3_3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3_3, planes))

        return nn.Sequential(*layers)



    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_11(x0)
        x = self.layer3x3_12(x)
        x = self.layer3x3_13(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3_1(x)

        y = self.layer3x3_21(x0)
        y = self.layer3x3_22(y)
        y = self.layer3x3_23(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool3_2(y)

        z = self.layer3x3_31(x0)
        z = self.layer3x3_32(z)
        z = self.layer3x3_33(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool3_3(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.squeeze()
        # out = self.drop(out)
        out1 = self.fc(out)

        return out1, out







