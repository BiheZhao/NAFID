import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .basicmodule import BasicModule
from collections import OrderedDict
import math


'''
Multi-Head Attention
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, n_heads, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.conv_q = nn.Conv2d(in_channels, hid_dim, 1, stride=1, padding=0)
        self.conv_k = nn.Conv2d(in_channels, hid_dim, 1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(in_channels, hid_dim, 1, stride=1, padding=0)

        # self.w_q = nn.Linear(hid_dim, hid_dim)
        # self.w_k = nn.Linear(hid_dim, hid_dim)
        # self.w_v = nn.Linear(hid_dim, hid_dim)

        # print(out_channels)
        self.conv_out = nn.Conv2d(hid_dim, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim//n_heads]))

    def forward(self, x, mask=None):
        shortcut = x
        query = self.conv_q(x).permute(0,2,3,1) # bsz*H*W*C
        key   = self.conv_k(x).permute(0,2,3,1)
        value = self.conv_v(x).permute(0,2,3,1)

        bsz = query.shape[0]
        H = query.shape[1]
        W = query.shape[2]

        # Q = self.w_q(query)
        # K = self.w_k(key)
        # V = self.w_v(value)

        Q = query
        K = key
        V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3) # bsz*n*HW*c
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)

        energy = torch.matmul(Q, K.permute(0,1,3,2)) #/ self.scale

        # if mask is not None:
            # energy = energy.masked_fill(mask=self.mask, -1e10)

        # attention = self.do(torch.softmax(energy, dim=-1))
        attention = torch.softmax(energy, dim=1)

        x = torch.matmul(attention, V)

        x = x.permute(0,1,3,2).contiguous()

        x = x.view(bsz, self.n_heads * (self.hid_dim // self.n_heads), H, W)

        x = self.conv_out(x)

        x = self.dropout(x)

        x += shortcut

        # x = F.relu(x)

        return x

'''
xception block
'''

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

'''
Dense Block
'''

class SingleLayer(nn.Module):
    def __init__(self,in_channels,growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,growth_rate,kernel_size=3,padding=1)
    def forward(self,x):
        # out = self.conv(F.relu(self.bn(x)))
        out = self.bn(x)
        out = F.relu(out)
        # out = F.gelu(out)
        out = self.conv(out)
        out = torch.cat([x,out],1)
        return out

class BottleneckLayer(nn.Module):
    def __init__(self,in_channels,growth_rate):
        super().__init__()
        inter_channels = 4*growth_rate
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,inter_channels,kernel_size=1)
        self.bn2   = nn.BatchNorm2d(inter_channels)
        out_channels = growth_rate
        self.conv2 = nn.Conv2d(inter_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.cat([x,out],1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x = self.conv(F.relu(self.bn(x)))
        out = self.bn(x)
        out = F.relu(out)
        # out = F.gelu(out)
        out = self.conv(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self,in_channels,growth_rate,n_layers,reduction=0.5,bottleneck=False):
        super().__init__()
        layers = []
        for i in range(n_layers):
            # print(in_channels)
            if bottleneck:
                layers.append(BottleneckLayer(in_channels, growth_rate))
            else:
                layers.append(SingleLayer(in_channels, growth_rate))
            in_channels += growth_rate
        self.out_channels = math.floor(in_channels*reduction)
        self.dense = nn.Sequential(*layers)
        self.trans = Transition(in_channels, self.out_channels)

    def forward(self,x):
        out = self.dense(x)
        out = self.trans(out)
        return out


'''
feature extraction block
'''

# NABlock dense style

class NABlock(nn.Module):
    def __init__(self, opt, init_channel=3, shortcut=False):
        super().__init__()
        self.shortcut    = shortcut
        self.hid_dim     = opt.hid_dim
        self.n_heads     = opt.n_heads
        self.n_blocks    = len(opt.n_layers)
        self.n_layers    = opt.n_layers
        self.growth_rate = opt.growth_rate
        self.in_channels = 2*opt.growth_rate#opt.in_channels
        self.dropout     = opt.dropout
        self.conv_in = nn.Conv2d(init_channel,self.in_channels,kernel_size=3,padding=1)
        self.trunk, trunk_channels = self._get_trunk(self.n_blocks,self.n_layers,self.growth_rate)
        self.mask , mask_channels  = self._get_mask(self.in_channels,self.hid_dim,self.n_heads,
            self.n_blocks,self.n_layers,self.growth_rate,self.dropout)
        assert trunk_channels == mask_channels

        # if shortcut:
        #     # self.out_channels = self.in_channels + trunk_channels
        #     inter_channels = init_channel + trunk_channels
        # else:
        #     inter_channels = trunk_channels
        # layers = []
        # out_n_layers = [2, 2]
        # n_channels = inter_channels
        # for i in range(len(out_n_layers)):
        #     # print(n_channels)
        #     layers.append(DenseBlock(n_channels, opt.growth_rate, out_n_layers[i], reduction=0.5, bottleneck=False))
        #     n_channels = layers[i].out_channels
        # self.out_dense = nn.Sequential(*layers)
        # self.out_channels = n_channels

        if shortcut:
            # self.out_channels = self.in_channels + trunk_channels
            self.out_channels = init_channel + trunk_channels
        else:
            self.out_channels = trunk_channels

    def _get_trunk(self,n_blocks,n_layers,growth_rate):
        n_channels = self.in_channels
        layers = []
        # layers.append(MultiHeadAttention(in_channels,n_channels,hid_dim,n_heads,dropout=0.5))
        for i in range(n_blocks):
            # print(n_channels)
            layers.append(DenseBlock(n_channels, growth_rate, n_layers[i],reduction=0.5,bottleneck=False))
            n_channels = layers[i].out_channels
        trunk_branch = nn.Sequential(*layers)
        return trunk_branch, n_channels

    def _get_mask(self,in_channels,hid_dim,n_heads,
        n_blocks,n_layers,growth_rate,dropout):
        n_channels = self.in_channels
        layers = []
        layers.append(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1))
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        layers.append(MultiHeadAttention(in_channels,n_channels,hid_dim,n_heads,dropout=dropout))
        for i in range(n_blocks):
            layers.append(DenseBlock(n_channels,growth_rate,n_layers[i],reduction=0.5,bottleneck=False))
            n_channels = layers[i+2].out_channels
        layers.append(nn.Upsample(scale_factor=2,mode='bilinear'))
        # layers.append(nn.Upsample(scale_factor=2,mode='nearest'))
        # layers.append(nn.ConvTranspose2d(n_channels, n_channels, kernel_size=6, stride=2, padding=2))
        mask_branch = nn.Sequential(*layers)
        return mask_branch, n_channels

    def forward(self,x):
        conv_x = self.conv_in(x)
        t = self.trunk(conv_x)
        m = self.mask(conv_x)
        out = t*m
        if self.shortcut:
            out = torch.cat((out,x),1)
        # out = self.out_dense(out)
        return out


'''
ResNeXt
'''

class ResNeXtBottleneck(BasicModule):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.leaky_relu(residual + bottleneck, inplace=True)

class ResNeXt(BasicModule):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, opt, in_channels, spp_level=3):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        self.cardinality = opt.cardinality
        self.depth = opt.depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = opt.base_width
        self.widen_factor = opt.widen_factor
        self.nlabels = opt.nlabels
        # self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor, 512 * self.widen_factor]
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        # self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 2)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        # self.stage_4 = self.block('stage_4', self.stages[3], self.stages[4], 2)
        self.classifier = nn.Linear(self.stages[len(self.stages)-1], opt.nlabels)
        
        # self.spp_layer = SPPLayer(spp_level)
        # self.spp_tail = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(self.num_grids*1024,1024)),
        #     ('fc1_relu', nn.ReLU()),
        #     ('fc2', nn.Linear(1024,2)),
        # ]))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        # init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                # if 'conv' in key:
                    # init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = self.bn_1.forward(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.maxpool.forward(x)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)

        x = self.adaptive_pool(x)
        x = x.view(-1, self.stages[len(self.stages)-1])
        x = self.classifier(x)
        return x

'''
our proposed network
'''

class NAFNet1(BasicModule):
    def __init__(self,opt):
        super().__init__()
        self.get_feature = NABlock(opt)
        n_channels = self.get_feature.out_channels
        self.classifier = ResNeXt(opt,n_channels)
    def forward(self,x):
        feature = self.get_feature(x)
        self.feature_map = feature.detach()
        out = self.classifier(feature)
        return out