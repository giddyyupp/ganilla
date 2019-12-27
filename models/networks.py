import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F

import math
import torch.utils.model_zoo as model_zoo
import sys

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], depth=18, fpn_weights=[1.0, 1.0, 1.0, 1.0]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_fpn':
        # Create the model
        if depth == 18:
            net = resnet18(input_nc, output_nc, ngf, fpn_weights, use_dropout=use_dropout, pretrained=False)
            # print "EVET"
            # netG_B2A = resnet18(pretrained=False)
        elif depth == 34:
            net = resnet34(input_nc, output_nc, ngf, fpn_weights, use_dropout=use_dropout, pretrained=False)
            # netG_B2A = resnet34(pretrained=False)
    elif netG == "ablation_model1":
        net = AblationModel1(BasicBlock_Ganilla, [2, 2, 2, 2])
    elif netG == "ablation_model2":
        net = AblationModel2(ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


######## GANILLA #########


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)


class BasicBlock_orj(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_orj, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = conv3x3(planes, planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.rp1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)

        out = self.rp2(out)
        out = self.conv2(out)
        out = self.in2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_Ganilla(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, use_dropout, stride=1):
        super(BasicBlock_Ganilla, self).__init__()
        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.out_planes = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )

            self.final_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.expansion * planes * 2, self.expansion * planes, kernel_size=3, stride=1,
                                        padding=0, bias=False),
                nn.InstanceNorm2d(self.expansion * planes)
            )
        else:
            self.final_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(planes*2, planes, kernel_size=3, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.rp1(x))))
        if self.use_dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(self.rp2(out)))
        inputt = self.shortcut(x)
        catted = torch.cat((out, inputt), 1)
        out = self.final_conv(catted)
        out = F.relu(out)
        return out


class PyramidFeatures_v3(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures_v3, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp1 = nn.ReflectionPad2d(1)
        #self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp2 = nn.ReflectionPad2d(1)
        #self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp3 = nn.ReflectionPad2d(1)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        #self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp4 = nn.ReflectionPad2d(1)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size/2, kernel_size=3, stride=1, padding=0)

        self.P1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp5 = nn.ReflectionPad2d(1)
        self.P1_2 = nn.Conv2d(feature_size, feature_size/2, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)

        P1_x = self.P1_1(P3_upsampled_x)
        P1_upsampled_x = self.P1_upsampled(P1_x)
        P2_x = self.rp5(P1_upsampled_x)
        P2_x = self.P1_2(P2_x)

        return P2_x


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, fpn_weights, feature_size=128):
        super(PyramidFeatures, self).__init__()

        self.sum_weights = fpn_weights #[1.0, 0.5, 0.5, 0.5]

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp1 = nn.ReflectionPad2d(1)
        #self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp2 = nn.ReflectionPad2d(1)
        #self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp3 = nn.ReflectionPad2d(1)
        #self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp4 = nn.ReflectionPad2d(1)
        self.P2_2 = nn.Conv2d(int(feature_size), int(feature_size/2), kernel_size=3, stride=1, padding=0)

        #self.P1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp5 = nn.ReflectionPad2d(1)
        #self.P1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        i = 0
        P5_x = self.P5_1(C5) * self.sum_weights[i]
        P5_upsampled_x = self.P5_upsampled(P5_x)
        #P5_x = self.rp1(P5_x)
        # #P5_x = self.P5_2(P5_x)
        i += 1
        P4_x = self.P4_1(C4) * self.sum_weights[i]
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        #P4_x = self.rp2(P4_x)
        # #P4_x = self.P4_2(P4_x)
        i += 1
        P3_x = self.P3_1(C3) * self.sum_weights[i]
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        #P3_x = self.rp3(P3_x)
        #P3_x = self.P3_2(P3_x)
        i += 1
        P2_x = self.P2_1(C2) * self.sum_weights[i]
        P2_x = P2_x * self.sum_weights[2] + P3_upsampled_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.rp4(P2_upsampled_x)
        P2_x = self.P2_2(P2_x)

        return P2_x


class ResNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, fpn_weights, block, layers, use_dropout):
        self.inplanes = ngf
        super(ResNet, self).__init__()

        # first conv
        self.pad1 = nn.ReflectionPad2d(input_nc)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(ngf)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Output layer
        self.pad3 = nn.ReflectionPad2d(output_nc)
        self.conv2 = nn.Conv2d(64, output_nc, 7)
        self.tanh = nn.Tanh()

        if block == BasicBlock_orj:
            # residuals
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        elif block == BasicBlock_Ganilla:
            # residuals
            self.layer1 = self._make_layer_ganilla(block, 64, layers[0], use_dropout, stride=1)
            self.layer2 = self._make_layer_ganilla(block, 128, layers[1], use_dropout, stride=2)
            self.layer3 = self._make_layer_ganilla(block, 128, layers[2], use_dropout, stride=2)
            self.layer4 = self._make_layer_ganilla(block, 256, layers[3], use_dropout, stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        else:
            print("Block Type is not Correct")
            sys.exit()

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], fpn_weights)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_ganilla(self, block, planes, blocks, use_dropout, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, use_dropout, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        img_batch = inputs

        x = self.pad1(img_batch)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.fpn([x1, x2, x3, x4]) # use all resnet layers

        out = self.pad3(out)
        out = self.conv2(out)
        out = self.tanh(out)

        return out


class AblationModel1(nn.Module):
    def __init__(self, block, layers, ngf=64):
        self.inplanes = 64
        self.ngf = ngf
        super(AblationModel1, self).__init__()

        # first conv
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        if block == BasicBlock_orj:
            # residuals
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        elif block == BasicBlock_Ganilla:
            # residuals
            self.layer1 = self._make_layer_ganilla(block, 64, layers[0])
            self.layer2 = self._make_layer_ganilla(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer_ganilla(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer_ganilla(block, 256, layers[3], stride=2)

            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]

        else:
            print("Block Type is not Correct")

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        n_downsampling = 4

        model = []

        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(256, self.ngf * 2 ** (n_downsampling), kernel_size=3, padding=0)]
        model += [nn.ReLU(inplace=True)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      nn.InstanceNorm2d(int(self.ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.deconv_part = nn.Sequential(*model)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_ganilla(self, block, planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        img_batch = inputs

        x = self.pad1(img_batch)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.deconv_part(x4)

        return out


class AblationModel2(nn.Module):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        self.inplanes = 64
        self.ngf = ngf
        super(AblationModel2, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_nc = 3
        self.output_nc = 3

        fpn_sizes = []

        init_part = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        self.init_part = nn.Sequential(*init_part)
        fpn_sizes.append(ngf)

        n_downsampling = 2
        # for i in range(n_downsampling):
        #     mult = 2 ** i
        mult = 1
        down1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(ngf * mult * 2),
                  nn.ReLU(True)]

        self.down1 = nn.Sequential(*down1)

        fpn_sizes.append(ngf * mult * 2)

        mult = 2
        down2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(ngf * mult * 2),
                  nn.ReLU(True)]
        self.down2 = nn.Sequential(*down2)

        # fpn_sizes.append(ngf * mult * 2)

        flat_part = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            flat_part += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        self.flat_part = nn.Sequential(*flat_part)

        fpn_sizes.append(ngf * mult)

        self.fpn = PyramidFeatures_v3(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        final_part = [nn.ReflectionPad2d(3)]
        final_part += [nn.Tanh()]
        final_part += [nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0)]

        self.final_part = nn.Sequential(*final_part)

    def forward(self, inputs):

        img_batch = inputs

        x = self.init_part(img_batch)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.flat_part(x2)

        out = self.fpn([x, x1, x3])

        out = self.final_part(out)

        return out


def resnet18(input_nc, output_nc, ngf, fpn_weights, use_dropout, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(input_nc, output_nc, ngf, fpn_weights, BasicBlock_Ganilla, [2, 2, 2, 2], use_dropout,  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(input_nc, output_nc, ngf, fpn_weights, use_dropout=False, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(input_nc, output_nc, ngf, fpn_weights, BasicBlock_Ganilla, [3, 4, 6, 3], use_dropout=use_dropout, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


#### ORJ MODELS ######
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
