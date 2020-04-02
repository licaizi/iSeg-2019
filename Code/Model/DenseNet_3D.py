import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from Config.config import model_params


class BNIN(nn.Module):
    def __init__(self, planes, half_bn=0.6):
        super(BNIN, self).__init__()
        half1 = int(planes * half_bn)
        self.half = half1
        half2 = planes - half1
        self.half2 = half2
        if half1 > 0:
            self.BN = nn.BatchNorm3d(half1)
        if half2 > 0:
            self.IN = nn.InstanceNorm3d(half2, affine=True)

    def forward(self, x):
        out = None
        if self.half > 0:
            split = torch.split(x, [self.half, self.half2], 1)
            out1 = self.BN(split[0].contiguous())
            out2 = self.IN(split[1].contiguous())
            out = torch.cat((out1, out2), 1)
        elif self.half == 0:
            out = self.IN(x)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Bottleneck, self).__init__()
        self.dropout = dropout

        self.batch_norm1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.batch_norm2 = nn.BatchNorm3d(out_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, input):
        out = self.conv1(self.relu1(self.batch_norm1(input)))
        out = self.conv2(self.relu2(self.batch_norm2(out)))
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, training=self.training)
        out = torch.cat([input, out], dim=1)
        return out


class BottleneckBNIN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, half_bn=0.6):
        super(BottleneckBNIN, self).__init__()
        self.dropout = dropout

        self.batch_norm1 = BNIN(in_channels, half_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.batch_norm2 = nn.BatchNorm3d(out_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, input):
        out = self.conv1(self.relu1(self.batch_norm1(input)))
        out = self.conv2(self.relu2(self.batch_norm2(out)))
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, training=self.training)
        out = torch.cat([input, out], dim=1)
        return out


class DenseBlockDoubleBN(nn.Sequential):
    def __init__(self, *args):
        super(DenseBlockDoubleBN, self).__init__(*args)

    def forward(self, input, domain='S'):
        for module in self._modules.values():
            input = module(input, domain)
        return input


class Transition(nn.Module):
    def __init__(self, inchannel, reduction):
        super(Transition, self).__init__()
        outchannel = int(math.floor(inchannel * reduction))

        self.batch_norm1 = nn.BatchNorm3d(inchannel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inchannel, outchannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')

        self.batch_norm2 = nn.BatchNorm3d(outchannel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_down = nn.Conv3d(outchannel, outchannel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        init.kaiming_normal_(self.conv_down.weight, nonlinearity='relu')

    def forward(self, input):
        out = self.conv1(self.relu1(self.batch_norm1(input)))
        out = self.conv_down(self.relu2(self.batch_norm2(out)))
        return out


class Transition_BNIN(nn.Module):
    def __init__(self, inchannel, reduction, half_bn=0.6):
        super(Transition_BNIN, self).__init__()
        outchannel = int(math.floor(inchannel * reduction))

        self.batch_norm1 = BNIN(inchannel, half_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inchannel, outchannel, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')

        self.batch_norm2 = nn.BatchNorm3d(outchannel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_down = nn.Conv3d(outchannel, outchannel, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        init.kaiming_normal_(self.conv_down.weight, nonlinearity='relu')

    def forward(self, input):
        out = self.conv1(self.relu1(self.batch_norm1(input)))
        out = self.conv_down(self.relu2(self.batch_norm2(out)))
        return out


class DenseNet_3D(nn.Module):
    def __init__(self, ver='ver1', half_bn=0.6):
        super(DenseNet_3D, self).__init__()
        self.ver = ver
        if self.ver in ['ver1_bnin_ver1']:
            self.init_v1_bnin_ver1(half_bn=half_bn)

    def init_v1_bnin_ver1(self, half_bn=0.6):    # BNIN every two conv layers
        print('init_v1_bnin_ver1, self.ver=', self.ver)
        first_output = model_params['first_output']
        nchannels = first_output
        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        padding = (1, 1, 1)

        self.growth_rate = model_params['growth_rate']
        self.reduction = model_params['reduction']
        self.nb_classes = model_params['nb_classes']
        self.dropout = model_params['dropout']

        self.conv1a = nn.Conv3d(2, nchannels, kernel_size, stride=stride, padding=padding, bias=True)
        init.kaiming_normal_(self.conv1a.weight, nonlinearity='relu')
        init.constant_(self.conv1a.bias, -0.1)
        self.bnorm1a = BNIN(nchannels, half_bn)
        self.relu1a = nn.ReLU(inplace=True)

        self.conv1b = nn.Conv3d(nchannels, nchannels, kernel_size, stride=stride, padding=padding, bias=False)
        init.kaiming_normal_(self.conv1b.weight, nonlinearity='relu')
        self.bnorm1b = nn.BatchNorm3d(nchannels)
        self.relu1b = nn.ReLU(inplace=True)

        self.conv1c = nn.Conv3d(nchannels, nchannels, kernel_size, stride=stride, padding=padding, bias=False)
        init.kaiming_normal_(self.conv1c.weight, nonlinearity='relu')
        self.bnorm1c = BNIN(nchannels, half_bn)
        self.relu1c = nn.ReLU(inplace=True)

        self.down1 = nn.Conv3d(nchannels, nchannels, (2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
        init.kaiming_normal_(self.down1.weight, nonlinearity='relu')

        # ============Dense block 2================
        self.dense_block2 = self.__dense_block_bnin_ver2(0, nchannels, self.growth_rate, self.dropout, half_bn=half_bn)
        nchannels += self.growth_rate * model_params['N'][0]
        # ============Dense block 2 end================

        # ============Deconvolution layer 2================
        self.model_deconv_x2 = nn.ConvTranspose3d(nchannels, self.nb_classes, (4, 4, 4), stride=(2, 2, 2),
                                                  padding=(1, 1, 1), bias=False, groups=self.nb_classes)
        init.kaiming_normal_(self.model_deconv_x2.weight, nonlinearity='relu')
        # ============Deconvolution layer 2 end========

        # ============Transition layer 2===============
        self.transition2 = Transition_BNIN(nchannels, self.reduction, half_bn=half_bn)
        nchannels = int(math.floor(nchannels * self.reduction))
        # ============Transition layer 2 end===========

        # ============Dense block 3================
        self.dense_block3 = self.__dense_block_bnin_ver2(1, nchannels, self.growth_rate, self.dropout, half_bn=half_bn)
        nchannels += self.growth_rate * model_params['N'][1]
        # ============Dense block 3 end================

        # ============Deconvolution layer 3================
        self.model_deconv_x4 = nn.ConvTranspose3d(nchannels, self.nb_classes, (6, 6, 6), stride=(4, 4, 4),
                                                  padding=(1, 1, 1), bias=False, groups=self.nb_classes)
        init.kaiming_normal_(self.model_deconv_x4.weight, nonlinearity='relu')
        # ============Deconvolution layer 3 end========

        # ============Transition layer 3===============
        self.transition3 = Transition_BNIN(nchannels, self.reduction, half_bn=half_bn)
        nchannels = int(math.floor(nchannels * self.reduction))
        # ============Transition layer 3 end===========

        # ============Dense block 4================
        self.dense_block4 = self.__dense_block_bnin_ver2(2, nchannels, self.growth_rate, self.dropout, half_bn=half_bn)
        nchannels += self.growth_rate * model_params['N'][2]
        # ============Dense block 4 end================

        # ============Deconvolution layer 4================
        self.model_deconv_x8 = nn.ConvTranspose3d(nchannels, self.nb_classes, (10, 10, 10), stride=(8, 8, 8),
                                                  padding=(1, 1, 1), bias=False, groups=self.nb_classes)
        init.kaiming_normal_(self.model_deconv_x8.weight, nonlinearity='relu')
        # ============Deconvolution layer 4 end========

        # ============Transition layer 4===============
        self.transition4 = Transition_BNIN(nchannels, self.reduction, half_bn=half_bn)
        nchannels = int(math.floor(nchannels * self.reduction))
        # ============Transition layer 4 end===========

        # ============Dense block 5================
        self.dense_block5 = self.__dense_block_bnin_ver2(3, nchannels, self.growth_rate, self.dropout, half_bn=half_bn)
        nchannels += self.growth_rate * model_params['N'][3]
        # ============Dense block 5 end================

        # ============Deconvolution layer 5================
        self.model_deconv_x16 = nn.ConvTranspose3d(nchannels, self.nb_classes, (18, 18, 18), stride=(16, 16, 16),
                                                   padding=(1, 1, 1), bias=False, groups=self.nb_classes)
        init.kaiming_normal_(self.model_deconv_x16.weight, nonlinearity='relu')
        # ============Deconvolution layer 5 end========

        # ============output===========================
        out_concat_channels = self.conv1c.out_channels + self.model_deconv_x2.out_channels \
                              + self.model_deconv_x4.out_channels + self.model_deconv_x8.out_channels \
                              + self.model_deconv_x16.out_channels
        self.out_bnorm_concat = nn.BatchNorm3d(out_concat_channels)
        self.out_relu_concat = nn.ReLU(inplace=True)
        self.model_conv_concate = nn.Conv3d(out_concat_channels, self.nb_classes, (1, 1, 1))
        init.kaiming_normal_(self.model_conv_concate.weight, nonlinearity='relu')
        init.constant_(self.model_conv_concate.bias, 0)

    def forward(self, input):
        if self.ver == 'ver1_bnin_ver1':
            return self.forward_ver1(input)
        else:
            raise RuntimeError("invalid model version '{}'".format(self.ver))

    def forward_ver1(self, input):
        out = self.relu1a(self.bnorm1a(self.conv1a(input)))
        out = self.relu1b(self.bnorm1b(self.conv1b(out)))
        out_x1 = self.conv1c(out)
        out_block1 = self.relu1c(self.bnorm1c(out_x1))
        out = self.down1(out_block1)
        out = self.dense_block2(out)
        out_x2 = self.model_deconv_x2(out)
        out = self.transition2(out)
        out = self.dense_block3(out)
        out_x4 = self.model_deconv_x4(out)
        out = self.transition3(out)
        out = self.dense_block4(out)
        out_x8 = self.model_deconv_x8(out)
        out = self.transition4(out)
        out = self.dense_block5(out)
        out_x16 = self.model_deconv_x16(out)
        out_concat = torch.cat((out_block1, out_x2, out_x4, out_x8, out_x16), dim=1)
        out = self.model_conv_concate(self.out_relu_concat(self.out_bnorm_concat(out_concat)))
        return out

    def __dense_block_bnin_ver2(self, idx, in_channles, growth_rate, dropout, half_bn=0.6):
        layers = []
        for i in range(model_params['N'][idx]):
            if i % 2 == 0:
                layers.append(BottleneckBNIN(in_channles, growth_rate, dropout, half_bn=half_bn))
                in_channles += growth_rate
                continue
            layers.append(Bottleneck(in_channles, growth_rate, dropout))
            in_channles += growth_rate
        return nn.Sequential(*layers)


