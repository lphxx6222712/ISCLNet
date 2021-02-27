import torch
import torch.nn as nn
from utils.attention import PAM_Module, CAM_Module
import os
import cv2
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



def draw_features(x,savename):
    width, height, channel = x.shape[2], x.shape[3], x.shape[1]
    savepath = os.path.join('/home/hxx/Documents/hxx_code/pytorch/Multi-modal/CBAM.PyTorch-master/logs/heatmaps',savename)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    x = x.cpu().numpy()
    for i in range(channel):
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        cv2.imwrite((savepath+'/'+'%s.jpg'%i),img)
        print("{}/{}".format(i,width*height))


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=9, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        #self.avgpool = nn.AvgPool2d(7,stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #draw_features(x, 'resnet18_OCT')
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)#.squeeze()
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch, x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

class resnet18_with_position_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18_with_position_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        PA = self.pam_attention(x,x,x)
        #draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(PA)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch, x

class resnet18_with_2position_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18_with_2position_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)
        self.pam_attention2 = PAM_Module(128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        PA = self.pam_attention(x,x,x)
        #draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(PA)
        PA2 = self.pam_attention2(x,x,x)
        x = self.layer3(PA2)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch, x

class resnet18_with_3position_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18_with_3position_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)
        self.pam_attention2 = PAM_Module(128)
        self.pam_attention3 = PAM_Module(256)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        PA = self.pam_attention(x,x,x)
        #draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(PA)
        PA2 = self.pam_attention2(x,x,x)
        x = self.layer3(PA2)
        PA3 = self.pam_attention3(x,x,x)
        x = self.layer4(PA3)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch, x

class resnet18_with_4position_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18_with_4position_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)
        self.pam_attention2 = PAM_Module(128)
        self.pam_attention3 = PAM_Module(256)
        self.pam_attention4 = PAM_Module(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        PA = self.pam_attention(x,x,x)
        #draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(PA)
        PA2 = self.pam_attention2(x,x,x)
        x = self.layer3(PA2)
        PA3 = self.pam_attention3(x,x,x)
        x = self.layer4(PA3)
        PA4 = self.pam_attention4(x,x,x)
        x = self.avgpool(PA4)
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch, x
class resnet18_with_channel_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18_with_channel_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)
        self.cam_attention = CAM_Module(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #PA = self.pam_attention(x,x,x)
        #draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cam_attention(x,x,x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        cls_branch = self.cls(x)#.squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch


class plain_cat_feature_map_resnet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(plain_cat_feature_map_resnet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes1 = 64
        self.inplanes2 = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv11 = nn.Conv2d(3, self.inplanes1, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.conv21 = nn.Conv2d(3, self.inplanes2, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn = norm_layer(self.inplanes1)


        self.layer11 = self._make_layer1(block, 64, layers[0])
        self.layer12 = self._make_layer1(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer13 = self._make_layer1(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer14 = self._make_layer1(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.layer21 = self._make_layer2(block, 64, layers[0])
        self.layer22 = self._make_layer2(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer23 = self._make_layer2(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer24 = self._make_layer2(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)

        self.conv_fundus = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)

        self.conv_OCT = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)




        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7,stride=1)

        self.cls_OCT = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )

        self.cls_fundus = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9)
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )

        self.cls_fusion = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(16384,9),
            #nn.Conv2d(1024, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)


        self.pam_attention2 = nn.Sequential(
            PAM_Module(128)
        )

        self.pam_attention3 = nn.Sequential(
            PAM_Module(256)
        )

        self.pam_attention4 = nn.Sequential(
            PAM_Module(512)
        )

        self.cam_attention = nn.Sequential(
            CAM_Module(1280)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes1, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes1 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes1, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes2, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes2 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes2, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x1, x2):
        x1 = self.conv11(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer11(x1)
        PA1 = self.pam_attention(x1, x1, x1)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x1 = self.layer12(PA1) #128
        #PA2 = self.pam_attention2(x)
        x1 = self.layer13(x1)
        # = self.pam_attention3(x)
        x1 = self.layer14(x1)
        #PA4 = self.pam_attention4(x)
        x1 = self.avgpool(x1)
        logits_fundus = self.cls_fundus(x1.view(x1.size(0),-1))#.squeeze()

        x2 = self.conv21(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer21(x2)
        #PA1 = self.pam_attention(x)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x2 = self.layer22(x2) #128
        #PA2 = self.pam_attention2(x)
        x2 = self.layer23(x2)
        # = self.pam_attention3(x)
        x2 = self.layer24(x2)
        #PA4 = self.pam_attention4(x)
        x2 = self.avgpool(x2)
        logits_OCT = self.cls_OCT(x2.view(x2.size(0),-1))#.squeeze()

        cls = torch.cat([x1,x2],dim=1)#.squeeze()
        logits = self.cls_fusion(cls.view(cls.size(0),-1))#.squeeze()

        return  logits, logits_fundus, logits_OCT









class plain_vote_feature_map_resnet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(plain_vote_feature_map_resnet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes1 = 64
        self.inplanes2 = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv11 = nn.Conv2d(3, self.inplanes1, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.conv21 = nn.Conv2d(3, self.inplanes2, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn = norm_layer(self.inplanes1)


        self.layer11 = self._make_layer1(block, 64, layers[0])
        self.layer12 = self._make_layer1(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer13 = self._make_layer1(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer14 = self._make_layer1(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.layer21 = self._make_layer2(block, 64, layers[0])
        self.layer22 = self._make_layer2(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer23 = self._make_layer2(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer24 = self._make_layer2(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)

        self.conv_fundus = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)

        self.conv_OCT = nn.Conv2d(512, 512, kernel_size=1, stride=2,
                               bias=False)




        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7,stride=1)

        self.cls_OCT = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9),
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )

        self.cls_fundus = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9),
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )

        self.cls_fusion = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(8192,9),
            #nn.Conv2d(512, 9, 1),
            #nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = PAM_Module(64)


        self.pam_attention2 = nn.Sequential(
            PAM_Module(128)
        )

        self.pam_attention3 = nn.Sequential(
            PAM_Module(256)
        )

        self.pam_attention4 = nn.Sequential(
            PAM_Module(512)
        )

        self.cam_attention = nn.Sequential(
            CAM_Module(1280)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer1(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes1, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes1, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes1 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes1, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes2 != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes2, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes2, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes2 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes2, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x1, x2):
        x1 = self.conv11(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer11(x1)
        PA1 = self.pam_attention(x1, x1, x1)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x1 = self.layer12(PA1) #128
        #PA2 = self.pam_attention2(x)
        x1 = self.layer13(x1)
        # = self.pam_attention3(x)
        x1 = self.layer14(x1)
        #PA4 = self.pam_attention4(x)
        x1 = self.avgpool(x1)
        logits_fundus = self.cls_fundus(x1.view(x1.size(0),-1))#.squeeze()

        x2 = self.conv21(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer21(x2)
        #PA1 = self.pam_attention(x)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x2 = self.layer22(x2) #128
        #PA2 = self.pam_attention2(x)
        x2 = self.layer23(x2)
        # = self.pam_attention3(x)
        x2 = self.layer24(x2)
        #PA4 = self.pam_attention4(x)
        x2 = self.avgpool(x2)
        logits_OCT = self.cls_OCT(x2.view(x2.size(0),-1))

        logits = .5*logits_fundus+.5*logits_OCT
        return  logits, logits_fundus, logits_OCT












def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class resnet34_with_position_attention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet34_with_position_attention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 9, 1),
            nn.AdaptiveMaxPool2d(1),
            #nn.Sigmoid()
        )
        self.pam_attention = nn.Sequential(
            PAM_Module(64)
        )

        self.pam_attention2 = nn.Sequential(
            PAM_Module(128)
        )

        self.pam_attention3 = nn.Sequential(
            PAM_Module(256)
        )

        self.pam_attention4 = nn.Sequential(
            PAM_Module(512)
        )

        self.cam_attention = nn.Sequential(
            CAM_Module(1280)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        PA1 = self.pam_attention(x)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x = self.layer2(PA1) #128
        #PA2 = self.pam_attention2(x)
        x = self.layer3(x)
        # = self.pam_attention3(x)
        x = self.layer4(x)
        #PA4 = self.pam_attention4(x)
        x = self.avgpool(x)


        cls_branch = self.cls(x).squeeze()

        # x = torch.flatten(x, 1)

        # x = self.fc(x)

        return cls_branch


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
