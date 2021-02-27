import torch
import torch.nn as nn
from utils.attention import PAM_Module, CAM_Module
from model.ResNet import BasicBlock,Bottleneck,conv1x1

class ROI_guided_OCT(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ROI_guided_OCT, self).__init__()
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
        self.conv21 = nn.Conv2d(4, self.inplanes2, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn = norm_layer(self.inplanes1)


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


    def forward(self, ROI, x):
        #x = torch.mul(x,ROI)
        x = torch.cat([x,ROI],dim=1)
        x = self.conv21(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer21(x)
        PA1 = self.pam_attention(x,x,x)
        #draw_features(x, 'resnet34_with_spatial_attention_Resblock1')
        #draw_features(PA1, 'resnet34_with_spatial_attention_attention_after_Resblock1')
        x = self.layer22(PA1) #128
        #PA2 = self.pam_attention2(x)
        x = self.layer23(x)
        # = self.pam_attention3(x)
        x = self.layer24(x)
        #PA4 = self.pam_attention4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        logits_OCT = self.cls_OCT(x)#.squeeze()
        return  logits_OCT, x