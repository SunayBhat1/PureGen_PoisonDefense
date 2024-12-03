from copy import deepcopy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

def load_model(model_name, num_classes=10, eval_bn=False, grad_bn=False,img_size=32):
    if model_name == 'ResNet18': 
        model = ResNet18Style(RN18BasicBlock, [2,2,2,2], num_classes=num_classes, eval_bn=eval_bn,grad_bn=grad_bn,img_size=img_size)
    elif model_name == 'ResNet34':
        model = ResNet18Style(RN18BasicBlock, [3,4,6,3], num_classes=num_classes, eval_bn=eval_bn,grad_bn=grad_bn,img_size=img_size)
    elif model_name == 'ResNet18_HLB':
        model = ResNet18_HLB(img_size=img_size, num_classes=num_classes)
    elif model_name in ['HLB_S', 'HLB_M', 'HLB_L']:
        model = make_hlb_net(model_name)
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(num_classes=num_classes)
    elif model_name == 'DenseNet121':
        model = densenet_cifar(growth_rate=32)
    elif model_name == 'NFNet':
        model = NFNet(BasicBlock, [2, 2, 2], num_classes=num_classes)
    else:
        raise ValueError('Unknown model: {}'.format(model_name))

    return model

################
# ResNet20 #
################
    
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20Style(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet20Style, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

################
# ResNet18 #
################

class eval_BatchNorm(nn.Module):
    def __init__(self, planes, eps=1e-5,grad=False):
        super().__init__()
        self.running_mean = nn.Parameter(torch.zeros(size=[planes]), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(size=[planes]), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(size=[planes]), requires_grad=grad)
        self.bias = nn.Parameter(torch.zeros(size=[planes]), requires_grad=grad)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        x_center = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.eps + self.running_var.view(1, -1, 1, 1))
        return self.weight.view(1, -1, 1, 1) * x_center + self.bias.view(1, -1, 1, 1)

class RN18BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, train_dp=0, test_dp=0, droplayer=0, bdp=0, eval_bn=False
    ):
        # if test_dp > 0: will always keep dp there
        super(RN18BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if eval_bn:
            self.bn1 = eval_BatchNorm(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        if eval_bn:
            self.bn2 = eval_BatchNorm(planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if eval_bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    eval_BatchNorm(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
        self.train_dp = train_dp
        self.test_dp = test_dp

        self.droplayer = droplayer

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if action == 1:
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            if self.test_dp > 0 or (self.training and self.train_dp > 0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)

        out = F.relu(out)
        return out

class ResNet18Style(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        conv1_size=3,
        num_classes=10,
        train_dp=0,
        test_dp=0,
        droplayer=0,
        bdp=0,
        middle_feat_num=1,
        eval_bn=False,
        grad_bn=False,
        img_size=32,
    ):
        super(ResNet18Style, self).__init__()
        self.in_planes = 64
        kernel_size, stride, padding = {3: [3, 1, 1], 7: [7, 2, 3]}[conv1_size]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        
        if eval_bn:
            self.bn1 = eval_BatchNorm(64,grad=grad_bn)
        else:
            self.bn1 = nn.BatchNorm2d(64)

        nblks = sum(num_blocks)
        dl_step = droplayer / nblks

        dl_start = 0
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            train_dp=train_dp,
            test_dp=test_dp,
            dl_start=dl_start,
            dl_step=dl_step,
            bdp=bdp,
            eval_bn=eval_bn,
        )

        dl_start += dl_step * num_blocks[0]
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            train_dp=train_dp,
            test_dp=test_dp,
            dl_start=dl_start,
            dl_step=dl_step,
            bdp=bdp,
            eval_bn=eval_bn,
        )

        dl_start += dl_step * num_blocks[1]
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            train_dp=train_dp,
            test_dp=test_dp,
            dl_start=dl_start,
            dl_step=dl_step,
            bdp=bdp,
            eval_bn=eval_bn,
        )

        dl_start += dl_step * num_blocks[2]
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            train_dp=train_dp,
            test_dp=test_dp,
            dl_start=dl_start,
            dl_step=dl_step,
            bdp=bdp,
            eval_bn=eval_bn,
        )
        img_size = img_size // 2 // 2 // 2 // 4
        self.linear = nn.Linear(512*img_size*img_size, num_classes)

        self.test_dp = test_dp
        self.middle_feat_num = middle_feat_num

    def get_block_feats(self, x):
        feat_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        for nl, layer in enumerate(self.layer4):
            out = layer(out)
            if (
                len(self.layer4) - nl - 1 <= self.middle_feat_num
                and len(self.layer4) - nl - 1 > 0
            ):
                feat_list.append(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

    def set_testdp(self, dp):
        for layer in self.layer1:
            layer.test_dp = dp
        for layer in self.layer2:
            layer.test_dp = dp
        for layer in self.layer3:
            layer.test_dp = dp
        for layer in self.layer4:
            layer.test_dp = dp

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        train_dp=0,
        test_dp=0,
        dl_start=9,
        dl_step=0,
        bdp=0,
        eval_bn=False,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for ns, stride in enumerate(strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    train_dp=train_dp,
                    test_dp=test_dp,
                    droplayer=dl_start + dl_step * ns,
                    bdp=bdp,
                    eval_bn=eval_bn,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def forward(self, x, penu=False, block=False, drop=False, lam=1, rand_index=None, use_linear=False):
        if block:
            return self.get_block_feats(x)

        out = self.penultimate(x)
        if drop:
            mask = torch.argmax(out, dim=0)
            out[mask,:] = 0
        if penu:
            return out
        if use_linear:
            return self.linear(out), out

        if lam < 1:
            out = out * lam + out[rand_index] * (1-lam)
        out = self.linear(out)

        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if "linear" in name]
    
    
#############
# MobileNet #
#############

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, train_dp, test_dp, droplayer=0, bdp=0):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.train_dp = train_dp
        self.test_dp = test_dp

        self.droplayer = droplayer
        self.bdp = bdp

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if self.stride == 1 and action == 1:
            # if stride is not 1, then there is no skip connection. so we keep this layer unchanged
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            if self.test_dp > 0 or (self.training and self.train_dp > 0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)

            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = torch.bernoulli(
                    self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_planes=32, train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.test_dp = test_dp
        self.bdp = bdp

    def _make_layers(self, in_planes, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        layers = []

        # get the total number of blocks
        nblks = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            nblks += num_blocks

        dl_step = droplayer / nblks

        blkidx = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                dl = dl_step * blkidx
                blkidx += 1

                layers.append(Block(in_planes, out_planes, expansion, stride, train_dp=train_dp, test_dp=test_dp,
                                    droplayer=dl, bdp=bdp))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.layers:
            layer.test_dp = dp

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, penu=False):
        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

################
# DenseNet121 #
################


def densenet_cifar(growth_rate=12):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=growth_rate)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, train_dp=0, test_dp=0, bdp=0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        if self.test_dp > 0 or (self.train_dp > 0 and self.training):
            dp = max(self.train_dp, self.test_dp)
            out = F.dropout(out, dp, training=True)
        if self.bdp > 0:
            # each sample will be applied the same mask
            bdp_mask = torch.bernoulli(self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
            out = bdp_mask * out

        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, train_dp=0, test_dp=0, bdp=0):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], train_dp=train_dp, test_dp=test_dp, bdp=bdp)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.test_dp = test_dp

    def _make_dense_layers(self, block, in_planes, nblock, train_dp=0, test_dp=0, bdp=0):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, train_dp=train_dp, test_dp=test_dp, bdp=bdp))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.dense1:
            layer.test_dp = dp
        for layer in self.dense2:
            layer.test_dp = dp
        for layer in self.dense3:
            layer.test_dp = dp
        for layer in self.dense4:
            layer.test_dp = dp

    def penultimate(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)

        return out

    def forward(self, x, penu=False):
        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

################
# NFNet #
################

class VPGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input) * 1.7015043497085571

class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super(WSConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, groups, bias, padding_mode)

        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in',
                             torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight),
                             persistent=False)

    def standardized_weights(self):
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = WSConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gelu = VPGELU()
        self.conv2 = WSConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gelu(out)

        out = self.conv2(out)

        if self.stride != 1 or identity.size(1) != out.size(1):
            identity = nn.functional.conv2d(identity, self.conv1.weight, stride=self.stride, padding=1)

        out += identity
        out = self.gelu(out)

        return out

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling
class NFNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(NFNet, self).__init__()

        self.in_channels = 16

        self.conv1 = WSConv2D(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gelu = VPGELU()

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))#FastGlobalMaxPooling()#
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = self.conv1(x)
        out = self.gelu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'fc' in name]

    def forward(self, x):
        out = self.conv1(x)
        out = self.gelu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


################
# ResNet18_HLB #
################

'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_Basic(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Basic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet_HLB(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,img_size=32):
        super(ResNet_HLB, self).__init__()

        widths = [64, 128, 256, 512]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        img_size = img_size // 2 // 2 // 2 // 4
        self.linear = nn.Linear(widths[3]*img_size*img_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, penu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)

        if penu:
            return pre_out
        
        final = self.linear(pre_out)
        return final

def ResNet18_HLB(**kwargs):
    return ResNet_HLB(BasicBlock_Basic, [2,2,2,2], **kwargs)


#####################
# HyoperLight Bench #
#####################

### Network Components ###
class HLB_Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class HLB_Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class HLB_BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=0.6,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class HLB_Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        # Create an implicit residual via identity initialization
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class HLB_ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out,heavy=False,momentum=0.6):
        super().__init__()
        self.conv1 = HLB_Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = HLB_BatchNorm(channels_out, momentum=momentum)
        self.conv2 = HLB_Conv(channels_out, channels_out)
        self.norm2 = HLB_BatchNorm(channels_out)
        self.activ = nn.GELU()

        self.heavy = heavy
        if heavy:
            self.conv3 = HLB_Conv(channels_out, channels_out)
            self.norm3 = HLB_BatchNorm(channels_out, momentum=momentum)     

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.heavy:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        if self.heavy:
            x = self.activ(x)
            x = self.conv3(x)
            x = self.norm3(x)
        x = self.activ(x) 
        if self.heavy:
            x += x0     
        return x

#### Network Definition ###

def make_hlb_net(hlb_type='HLB_S'):

    hlb_args = {
                'block_sizes': [1,4,4],
                'heavy': False,
                'whitening': {
                    'kernel_size': 2,
                },
                'batchnorm_momentum': 0.7,
                'base_width': 64,
                'scaling_factor': 1/9,
            }
    
    if hlb_type == 'HLB_M':
        hlb_args['block_sizes'] = [2, 6, 6]
        hlb_args['batchnorm_momentum'] = 0.6

    if hlb_type == 'HLB_l':
        hlb_args['heavy'] = True
        hlb_args['block_sizes'] = [1,4,4]
        hlb_args['batchnorm_momentum'] = 0.6
        hlb_args['base_width'] = 128
        
    widths = {
        'block1': (hlb_args['block_sizes'][0] * hlb_args['base_width']), # 64  w/ width at base value
        'block2': (hlb_args['block_sizes'][1] * hlb_args['base_width']), # 256 w/ width at base value
        'block3': (hlb_args['block_sizes'][2] * hlb_args['base_width']), # 256 w/ width at base value
    }
    whiten_conv_width = 2 * 3 * hlb_args['whitening']['kernel_size']**2
    net = nn.Sequential(
        HLB_Conv(3, whiten_conv_width, kernel_size=hlb_args['whitening']['kernel_size'], padding=0),
        HLB_BatchNorm(whiten_conv_width,momentum=hlb_args['batchnorm_momentum']),
        nn.GELU(),
        HLB_ConvGroup(whiten_conv_width, widths['block1'],heavy=hlb_args['heavy'],momentum=hlb_args['batchnorm_momentum']),
        HLB_ConvGroup(widths['block1'],  widths['block2'],heavy=hlb_args['heavy'],momentum=hlb_args['batchnorm_momentum']),
        HLB_ConvGroup(widths['block2'],  widths['block3'],heavy=hlb_args['heavy'],momentum=hlb_args['batchnorm_momentum']),
        nn.MaxPool2d(3),
        HLB_Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        HLB_Mul(hlb_args['scaling_factor']),
    )
    return net
