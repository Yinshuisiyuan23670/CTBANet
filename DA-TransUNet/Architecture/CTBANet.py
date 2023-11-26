# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from tkinter import W
from xml.sax.handler import feature_external_ges

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import configs as configs
from .block import ResNetV2
from .block import BANetHead

from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)



def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

class Attention(nn.Module):
    def __init__(self, config, vis, bn_type=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
 
        self.softmax = Softmax(dim=-1)
        self.weights = None
        self.q = nn.Sequential(
            nn.Conv2d(768, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
            nn.Conv2d(384, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
        )
        self.k = nn.Sequential(
            nn.Conv2d(768, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
            nn.Conv2d(384, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
        )
        self.v = nn.Sequential(
            nn.Conv2d(768, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
            nn.Conv2d(384, 384,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(384, bn_type=bn_type),
        )

        self.up = nn.Sequential(
            nn.Conv2d(384, 768,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(768, bn_type=bn_type),
        )


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q,K,V):
       
        q = self.q(Q)
        k = self.k(K)
        v = self.v(V)

        b,h,W = Q.size(0) , K.size(2), V.size(3)

        query  = q.view(b , 384, -1)
        query = query.permute(0,2,1)

        key = k.view(b,384,-1)

        value = v.view(b, 384, -1)
        value = value.permute(0,2,1)
       
        feature_image = torch.matmul(query,key)
        feature_image = (384**-.5)*feature_image
        feature_image = F.softmax(feature_image,dim=-1)
      

        con = torch.matmul(feature_image,value)
        con = con.permute(0,2,1).contiguous()
        con = con.view(b,384,*Q.size()[2:])
        con = self.up(con)
        return con , self.weights

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        
        self.BAblock1 = BANetHead(768, 768)


        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        
        


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)
        x = self.BAblock1(x)
        embeddings = self.dropout(x)
        return embeddings, features

def make_pairs(x):
    """make the int -> tuple 
    """
    return x if isinstance(x, tuple) else (x, x)

class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim, 
            out_channels=dim, 
            kernel_size=make_pairs(kernel_size),
            padding=make_pairs(1),
            groups=dim)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
       


    def forward(self, x):

        for i in range(len(self.convs)):
            x = self.act(self.bns[i](self.convs[i](x)))
            y = x
        y=x+y

        return self.sqe(self.act(self.bn(y)))
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size

        self.convdw1 = ConvDW3x3(768)
        self.convdw2 = ConvDW3x3(768)
        self.convdw3 = ConvDW3x3(768)
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        self.res = ResPath(768,1)

    def forward(self, x):
        Q = self.convdw1(x)
        K = self.convdw2(x)
        V = self.convdw2(x)
        y , weights = self.attn(Q,K,V)
        y = x + y
        y = self.res(y)
        return y, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = hidden_states
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.da = BANetHead(64, 64)
        self.da2 = BANetHead(256, 256)
        self.da3 = BANetHead(512, 512)
            
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if skip.size(1) and x.size(1) == 64:
                skip = self.da(skip) 
            
            if skip.size(1) and x.size(1) == 256:
                skip = self.da2(skip)
                
            if skip.size(1) and x.size(1) == 512:
                skip = self.da3(skip)
                
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features=None):
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class BA_Transformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(BA_Transformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x) 
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)#
        return logits#

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))


            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


