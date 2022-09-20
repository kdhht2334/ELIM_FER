import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

import pretrainedmodels
import pretrainedmodels.utils as utils

from fabulous.color import fg256

#alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained=None).cuda()
#resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained=None).cuda()
#resnet50 = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained=None).cuda()
alexnet = pretrainedmodels.__dict__['alexnet'](num_classes=1000, pretrained='imagenet').cuda()
resnet  = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet').cuda()
#resnet50 = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet').cuda()
print(fg256("green", 'Successfully loaded INet weights.'))


class Encoder_AL(nn.Module):
    def __init__(self):
        super(Encoder_AL, self).__init__()
        self.features = alexnet._features
        self.avgpool = alexnet.avgpool
#        self.adapool = nn.AdaptiveAvgPool2d((6,6))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.avgpool(x), 1)
#        x = ( torch.flatten(self.avgpool(x), 1) + torch.flatten(self.adapool(x), 1) )/2.
        return x


class Encoder_R18(nn.Module):

    def __init__(self):
        super(Encoder_R18, self).__init__()

        self.conv1 = resnet.conv1
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
#        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = resnet.last_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # shape: [1, 512, 8, 8]

        x = torch.flatten(self.avgpool(x), 1)
#        x = ( torch.flatten(self.avgpool(x), 1) + torch.flatten(self.adapool(x), 1) )/2.
        x = self.last_linear(x)  # [1000]
        return x


class Regressor_AL(nn.Module):

    def __init__(self, no_domain):
        super(Regressor_AL, self).__init__()
        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 256)
        self.lin1 = nn.Linear(256, no_domain)  #TODO(): `bias=False` means we use \phi.
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.bn = nn.BatchNorm1d(no_domain, affine=True)

    def forward(self, x):
        x = self.relu0(self.lin0(self.drop0(x)))
        latent_variable = self.relu1(self.bn(self.lin1(self.drop1(x))))
        return latent_variable


class Regressor_R18(nn.Module):

    def __init__(self, no_domain):
        super(Regressor_R18, self).__init__()
        self.lin0 = nn.Linear(1000, 512)  #TODO(): Do I have to increase this size(256)???
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, no_domain)
        self.bn1  = nn.BatchNorm1d(256, affine=True)
        self.bn2  = nn.BatchNorm1d(no_domain, affine=True)
#        self.lin1 = nn.Linear(512, no_domain)
#        self.bn = nn.BatchNorm1d(no_domain, affine=True)

    def forward(self, x):
        x = F.relu(self.lin0(F.dropout2d(x)))
        x = F.relu(self.bn1(self.lin1(F.dropout2d(x))))
        latent_variable = F.relu(self.bn2(self.lin2(F.dropout2d(x))))
#        latent_variable = F.relu(self.bn(self.lin1(F.dropout2d(x))))
        return latent_variable


class Regressor_R50(nn.Module):

    def __init__(self, no_domain):
        super(Regressor_R50, self).__init__()
        self.lin0 = nn.Linear(1000, 512)  #128
        self.lin1 = nn.Linear(512, no_domain)

        self.bn = nn.BatchNorm1d(no_domain, affine=True)
#        self.va_regressor = nn.Linear(no_domain, 2)

    def forward(self, x):
        x_btl_1 = F.relu(self.lin0(F.dropout2d(x)))
        latent_variable = F.relu(self.bn(self.lin1(F.dropout2d(x_btl_1))))
        return latent_variable


#class ERM_FC(nn.Module):
#
#    def __init__(self, erm_input_dim, erm_output_dim):
#        super(ERM_FC, self).__init__()
#
#        self.erm_input_dim = erm_input_dim
#        self.erm_hidden_dim = self.erm_input_dim * 10
#        self.erm_output_dim = erm_output_dim
#        self.linear1 = nn.Linear(self.erm_input_dim, self.erm_hidden_dim)
#        self.linear2 = nn.Linear(self.erm_hidden_dim, self.erm_hidden_dim)
#        self.linear3 = nn.Linear(self.erm_hidden_dim, self.erm_output_dim)
#        self.bn1 = nn.BatchNorm1d(self.erm_hidden_dim, affine=True)
#        self.bn2 = nn.BatchNorm1d(self.erm_hidden_dim, affine=True)
#
#        self.layer_blocks = nn.Sequential(
#            self.linear1,
#            self.bn1,
#            nn.LeakyReLU(inplace=True),
#            self.linear2,
#            self.bn2,
#            nn.LeakyReLU(inplace=True),
#            self.linear3,
#        )
#
#    def forward(self, inputs):
#        return self.layer_blocks(inputs)


class ERM_FC(nn.Module):

    def __init__(self, erm_input_dim, erm_output_dim):
        super(ERM_FC, self).__init__()

        self.erm_input_dim = erm_input_dim
        self.erm_hidden_dim = self.erm_input_dim * 20  #10
        self.erm_output_dim = erm_output_dim
        self.linear1 = nn.Linear(self.erm_input_dim, self.erm_hidden_dim)
        self.linear2 = nn.Linear(self.erm_hidden_dim, self.erm_output_dim)
        self.bn1 = nn.BatchNorm1d(self.erm_hidden_dim, affine=True)

        self.layer_blocks = nn.Sequential(
            self.linear1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.linear2,
        )

    def forward(self, inputs, train=True):
        if train:
            return self.layer_blocks(inputs)
        else:
            x = self.layer_blocks(inputs)
            x = torch.clamp(x, min=-1., max=1.)
            return F.tanh(x)


class Stn_NN(nn.Module):

    def __init__(self, erm_input_dim, hidden_dim):
        super(Stn_NN, self).__init__()

        self.erm_input_dim = erm_input_dim
        self.hidden_dim = hidden_dim

        self.enc = nn.Linear(self.erm_input_dim, self.hidden_dim)
        self.dec = nn.Linear(self.hidden_dim, self.erm_input_dim)

        self.layer_blocks = nn.Sequential(
            self.enc,
            nn.ReLU(inplace=True),
            self.dec,
        )

    def forward(self, inputs, shift_or_scale):
        if shift_or_scale == 'shift':  # mu
            outputs = self.layer_blocks(inputs)
        elif shift_or_scale == 'scale':  # sigma
            #outputs = F.relu(self.layer_blocks(inputs))
            outputs = self.layer_blocks(inputs)
        return outputs


class Rsc_NN(nn.Module):

    def __init__(self, erm_input_dim, hidden_dim):
        super(Rsc_NN, self).__init__()

        self.erm_input_dim = erm_input_dim
        self.hidden_dim = hidden_dim

        self.enc = nn.Linear(self.erm_input_dim, self.hidden_dim)
        self.reg = nn.Linear(self.hidden_dim, 2)
        self.weight = [nn.Parameter(torch.randn(64, self.hidden_dim)).cuda() for _ in range(2)]

        self.layer_blocks1 = nn.Sequential(
            self.enc, nn.ReLU(inplace=True), self.reg,
        )
        self.layer_blocks2 = nn.Sequential(
            self.enc, nn.ReLU(inplace=True),
        )

    def forward(self, inputs, shift_or_scale):
        if shift_or_scale == 'shift':  # beta
            outputs = F.tanh(self.layer_blocks1(inputs))
            return outputs
        elif shift_or_scale == 'scale':  # gamma
            outputs = torch.zeros(size=(2, 64))
            x = self.layer_blocks2(inputs)  # shape: [self.hidden_dim]
            for i in range(2):
                outputs[i] = F.linear(x, self.weight[i])  # shape: [2, 64]
            outputs = F.sigmoid(outputs)
            return outputs


def encoder_AL():
    encoder = Encoder_AL()
    return encoder
def encoder_R18():
    encoder = Encoder_R18()
    return encoder

def regressor_AL(latent_dim):
    domainregressor = Regressor_AL(latent_dim)  # 350
    return domainregressor
def regressor_R18(latent_dim):
    domainregressor = Regressor_R18(latent_dim)  # 350
    return domainregressor
def regressor_R50(latent_dim):
    domainregressor = Regressor_R50(latent_dim)  # 350
    return domainregressor

def load_ERM_FC(erm_input_dim, erm_output_dim):
    erm_fc = ERM_FC(erm_input_dim, erm_output_dim)
    return erm_fc

def load_Stn_NN(erm_input_dim, hidden_dim):
    stn_nn = Stn_NN(erm_input_dim, hidden_dim)
    return stn_nn
def load_Rsc_NN(erm_input_dim, hidden_dim):
    rsc_nn = Rsc_NN(erm_input_dim, hidden_dim)
    return rsc_nn


if __name__ == "__main__":

    from pytorch_model_summary import summary
#    print(fg256("cyan", summary(Encoder_AL(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
#    print(fg256("cyan", summary(Encoder_R18(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
#    print(fg256("orange", summary(Regressor_Alex(), torch.ones_like(torch.empty(1, 256, 6, 6)), show_input=True)))
#    print(fg256("orange", summary(Domain_regressor(500), torch.ones_like(torch.empty(1, 256, 6, 6)), show_input=True)))
#    print(fg256("white", summary(Regressor_R18(64), torch.ones_like(torch.empty(1, 1000)), show_input=True)))
    print(fg256("yellow", "ERM_FC", summary(ERM_FC(64, 2), torch.ones_like(torch.empty(10, 64)), show_input=True)))
#    print(fg256("cyan", "Regressor_AL", summary(Regressor_AL(64), torch.ones_like(torch.empty(10, 9216)), show_input=True)))
#    print(fg256("green", "Regressor_R18", summary(Regressor_R18(64), torch.ones_like(torch.empty(10, 1000)), show_input=True)))
#    print(fg256("yellow", summary(SPRegressor_light(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
#    print(fg256("green", summary(Variational_regressor(), torch.ones_like(torch.empty(1, 32)), show_input=True)))
