import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from core.Resnet import ResNetV2
import numpy as np
from typing import Tuple, List
from scipy import ndimage
import logging
from os.path import join as pjoin
# from core.attention import Attention as MemEffAttention
from core.attention import Attention, MemEffAttention
from core.MSCAM import MSCAM

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            mlp_drop: float,
    )->None:
        '''
        Mlp in Transformer layer
        config: 
        hidden_size(int): dim of hidden vector
        mlp_dim(int): dim of mlp hidden layer
        mlp_drop(float): dropout rate for mlp
        '''
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(mlp_drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    config:
    hidden_size: int,
    patch_grid: Tuple[int,int],
    patch_size: Tuple[int,int],
    resnet_num_layers: int,
    resnet_width_factor: int,
    img_size: int,
    in_channels: int =3,
    embed_drop: float = 0.0,
    """
    def __init__(
            self,
            hidden_size,
            patch_grid,
            patch_size,
            resnet_num_layers,
            resnet_width_factor,
            img_size: int, 
            in_channels: int =3,
            embed_drop: float = 0.0,
    ):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if patch_grid is not None:   # ResNet
            grid_size = patch_grid
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            # print('n patch:', n_patches)
            self.hybrid = True
        else:
            patch_size = patch_size
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=resnet_num_layers, width_factor=resnet_width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.dropout = Dropout(embed_drop)


    def forward(self, x):
        # print('-------Resnet input------', x.shape) 
        # x shape (B, C, H, W)
        if self.hybrid:
            x, features = self.hybrid_model(x)
            # x shape (B, hidden, n_patches*(1/2), n_patches*(1/2))
        else:
            features = None
        # print('-------Resnet output------', x.shape)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings # (B, n_patches, hidden)
        # print('embeddings shape:', embeddings.shape)
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Block(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            attn_num_heads: int,
            qkv_bias: bool,
            proj_bias: bool,
            attn_drop: float,
            proj_drop: float,
            vis: bool,
            mlp_dim: int,
            mlp_drop: float,
    )->None:
        '''
        config: 
        hidden_size(int): dim of hidden vector
        attn_num_heads(int): num of attention heads
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        attn_drop(float): dropout rate for attention
        proj_drop(float): dropout rate for output projection
        vis(bool): whether to visualize attention weights

        mlp config:
        mlp_dim: int,
        mlp_drop
        '''
        super(Block, self).__init__()
        self.hidden_size = hidden_size 
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, mlp_drop)
        self.attn = MemEffAttention(
            hidden_size, attn_num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, vis
        )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        # print(ROOT)
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            proj_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            qkv_weights = torch.cat([query_weight, key_weight, value_weight], dim=0)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)

            # print('-------load qkv_weights:', qkv_weights.shape)
            # print('-------load qkv bias:', qkv_bias.shape)
            # print('------- qkv weight', self.attn.qkv.weight.shape)
            # print('------- qkv bias:', self.attn.qkv.bias.shape)
            self.attn.qkv.weight.copy_(qkv_weights)
            self.attn.proj.weight.copy_(proj_weight)
            self.attn.qkv.bias.copy_(qkv_bias)
            self.attn.proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
            
            # print('-------load mlp_weight_0:', mlp_weight_0.shape)
            # print('-------load mlp_weight_1:', mlp_weight_1.shape)
            # print('-------load mlp_bias_0:', mlp_bias_0.shape)
            # print('-------load mlp_bias_1:', mlp_bias_1.shape)
            # print('-------ffn weight:', self.ffn.fc1.weight.shape)
            # print('-------ffn bias:', self.ffn.fc1.bias.shape)
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Encoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            attn_num_heads: int,
            qkv_bias: bool,
            proj_bias: bool,
            attn_drop: float,
            attn_proj_drop: float,
            vis: bool,
            mlp_dim: int,
            mlp_drop: float,
    ):
        '''
        config: 
        num_layers: num of transformer blocks

        attention config:
        hidden_size(int): dim of hidden vector
        num_heads(int): num of attention heads
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        attn_drop(float): dropout rate for attention
        attn_proj_drop(float): dropout rate for attention output projection
        vis(bool): whether to visualize attention weights
        
        '''
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, attn_num_heads, qkv_bias, proj_bias, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            patch_gird: Tuple[int,int],
            patch_size: Tuple[int, int],
            resnet_num_layers: Tuple[int],
            resnet_width_factor: int,
            img_size: int,
            in_channels: int,
            attn_num_layers: int,
            attn_num_heads: int,
            qkv_bias: bool,
            proj_bias: bool,
            embed_drop: float,
            attn_drop: float,
            attn_proj_drop: float,
            vis: bool,
            mlp_dim: int,
            mlp_drop: float,
    ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(hidden_size, patch_gird, patch_size, resnet_num_layers, resnet_width_factor, img_size, in_channels, embed_drop)
        self.encoder = Encoder(attn_num_layers, hidden_size, attn_num_heads, qkv_bias, proj_bias, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
        # add class token
        # self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids) # （B, n_patch, hidden）
        # add class token
        # embedding_output = torch.cat([self.class_token.repeat(embedding_output.shape[0], 1, 1), embedding_output], dim=1) # (B, n_patch+1, hidden)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features
    
#----------------------------------------------------------------------
#---------------------------DecoderCup------------------------------
#----------------------------------------------------------------------
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

    def forward(self, x, skip=None):
        # print('---', x.dtype)
        x = self.up(x)
        # print('---', x.dtype)
        # print('--------------', x.shape, skip.shape)
        # print('---skip', skip.dtype)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        # print('---', x.dtype)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            decoder_channels: int,
            n_skip: int,
            skip_channels: Tuple[int],


    ):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if n_skip != 0:
            skip_channels = list(skip_channels)
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.n_skip = n_skip

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        h = w = torch.floor(torch.sqrt(torch.tensor(n_patch))).long()
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
#----------------------------------------------------------------------

class ArtifactRemoverHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        sigmod_layer = nn.Sigmoid()
        super().__init__(conv2d, upsampling, sigmod_layer)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # sigmod_layer = nn.Sigmoid()
        super().__init__(conv2d, upsampling)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_dim, out_dim):
        # flatten (B, 1, in_dim) -> (B, in_dim)
        # flatten = nn.Flatten(start_dim=1, end_dim=-1)

        fc = nn.Linear(in_dim, out_dim)
        # softmax (B, out_dim) -> (B, out_dim)
        # softmax = nn.Softmax(dim=1)
        super().__init__(fc)

class CoordHead(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            gru_hidden_size=256,
            coord_size = 1000,
            coord_dim = 2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride = 2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # self.global_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.global_pool = nn.AvgPool2d(kernel_size=4, stride=4) # for 512x512 input
        self.gru = nn.GRU(32*32, gru_hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.mlp = nn.Linear(out_channels*gru_hidden_size*2, coord_dim*coord_size)
        self.relu = nn.ReLU()
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        # print('---', x.dtype)
        B = x.shape[0]
        features = x
        x_ = self.conv1(x)
        x_ = self.batch_norm_1(x_)
        x_ = self.relu(x_)

        x_ = self.conv2(x_)
        x_ = self.batch_norm_2(x_)
        x_ = self.relu(x_)

        x_ = self.global_pool(x_) # (Batch_size, out_channel, 32, 32)
        x_ = x_.view(B,-1 ,32*32)
        x_, hn = self.gru(x_) # (batch_size, 21, hidden_size)
        x_ = x_.reshape(B, -1)
        x_ = self.mlp(x_) # (batch_size, coord_dim*coord_size)
        return x_, features

class VisionTransformer(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            patch_gird: Tuple[int, int],
            patch_size: Tuple[int, int],
            resnet_num_layers: Tuple[int],
            resnet_width_factor: int,
            in_channels: int,
            attn_num_layers: int,
            attn_num_heads: int,
            mlp_dim: int,
            qkv_bias: bool,
            proj_bias: bool,
            embed_drop: float,
            attn_drop: float,
            attn_proj_drop: float,
            mlp_drop: float,
            decoder_channels: Tuple[int],
            n_skip: int,
            skip_channels: Tuple[int],
            classifier, 
            img_size=224, 
            vis=False
            ):
        '''
        config: 
        hidden_size(int): dim of hidden vector
        patch_gird(Tuple[int,int]): num of patches in one row/column
        patch_size(Tuple[int,int]): size of one patch
        resnet_num_layers Tuple(int): num of resnet layers
        resnet_width_factor(int): width factor of resnet
        in_channels(int): num of input channels
        attn_num_layers(int): num of transformer blocks
        num_heads(int): num of attention heads
        mlp_dim(int): dim of mlp
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        embed_drop(float): dropout rate for embedding
        attn_drop(float): dropout rate for attention
        attn_proj_drop(float): dropout rate for attention output projection
        mlp_drop(float): dropout rate for mlp
        decoder_channels(Tuple[int]): num of decoder channels
        n_skip(int): num of skip connections
        skip_channels(Tuple[int]): the dimensions of skip channels
        classifier(str): type of classifier, "seg" or "clf"
        img_size(int): size of input image
        vis(bool): whether to visualize attention weights
        
        '''
        super(VisionTransformer, self).__init__()
  
        self.classifier = classifier
        self.transformer = Transformer(hidden_size, patch_gird, patch_size, resnet_num_layers, resnet_width_factor, img_size, in_channels, attn_num_layers, attn_num_heads, qkv_bias, proj_bias, embed_drop, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
        self.decoder = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.arts_head = ArtifactRemoverHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        # self.clf_head = ClassificationHead(hidden_size, num_classes)
    def forward(self, x):
        
        x = x.repeat(1,3,1,1)

        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.arts_head(x)
        return logits



class MultiTaskVisionTransformer_3Dwire(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            patch_gird: Tuple[int, int],
            patch_size: Tuple[int, int],
            resnet_num_layers: Tuple[int],
            resnet_width_factor: int,
            in_channels: int,
            attn_num_layers: int,
            attn_num_heads: int,
            mlp_dim: int,
            qkv_bias: bool,
            proj_bias: bool,
            embed_drop: float,
            attn_drop: float,
            attn_proj_drop: float,
            mlp_drop: float,
            decoder_channels: Tuple[int],
            n_skip: int,
            skip_channels: Tuple[int],
            num_view: int,
            coord_dim: int,
            total_num_control_point: int,
            classifier, 
            img_size=224, 
            vis=False
            ):
        '''
        config: 
        hidden_size(int): dim of hidden vector
        patch_gird(Tuple[int,int]): num of patches in one row/column
        patch_size(Tuple[int,int]): size of one patch
        resnet_num_layers Tuple(int): num of resnet layers
        resnet_width_factor(int): width factor of resnet
        in_channels(int): num of input channels
        attn_num_layers(int): num of transformer blocks
        num_heads(int): num of attention heads
        mlp_dim(int): dim of mlp
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        embed_drop(float): dropout rate for embedding
        attn_drop(float): dropout rate for attention
        attn_proj_drop(float): dropout rate for attention output projection
        mlp_drop(float): dropout rate for mlp
        decoder_channels(Tuple[int]): num of decoder channels
        n_skip(int): num of skip connections
        skip_channels(Tuple[int]): the dimensions of skip channels
        num_view(int): number of views
        coord_dim(int): dim of control points
        total_num_control_point(int): total number of control points
        classifier(str): type of classifier, "seg" or "clf"
        img_size(int): size of input image
        vis(bool): whether to visualize attention weights
        
        '''
        super(MultiTaskVisionTransformer_3Dwire, self).__init__()
        self.classifier = classifier
        self.transformer = Transformer(hidden_size, patch_gird, patch_size, resnet_num_layers, resnet_width_factor, img_size, in_channels, attn_num_layers, attn_num_heads, qkv_bias, proj_bias, embed_drop, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
        self.decoder_arts = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        # self.decoder_seg = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.decoder_coord = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)

        # MSCAM
        self.mscam_features_1 = MSCAM(skip_channels[0]*num_view, skip_channels[0])
        self.mscam_features_2 = MSCAM(skip_channels[1]*num_view, skip_channels[1])
        self.mscam_features_3 = MSCAM(skip_channels[2]*num_view, skip_channels[2])
        
        self.mscam_code = MSCAM(hidden_size*num_view, hidden_size)

        self.arts_head = ArtifactRemoverHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        # self.clf_head = ClassificationHead(hidden_size, num_classes)
        # 3D guide head
        self.coord_head = CoordHead(
            in_channels=decoder_channels[-1],
            out_channels= total_num_control_point,
            coord_size = total_num_control_point,
            coord_dim = coord_dim,
            gru_hidden_size=256
        )
    
        self.grid_size = patch_gird
    def forward(self, x):
        
        # x: (B, V, H, W)
        B, V, H, W = x.shape
        x = x.reshape(B*V ,1, H, W).contiguous()
        x = x.repeat(1,3,1,1)
        code, attn_weights, features = self.transformer(x)  # (B*V, n_patch, hidden) features: (B*V, 512, 64, 64), (B*V, 256, 128, 128), (B*V, 64, 256, 256)
        arts_features = self.decoder_arts(code, features)
        # seg_features = self.decoder_seg(code, features)
        seg_features = arts_features
        _, n_patch, hidden = code.shape

        # fused features of different views
        code_fused = code.view(B, V, n_patch, hidden)
        code_fused = code_fused.permute(0, 1, 3, 2)
        code_fused = code_fused.contiguous().view(B, V*hidden, self.grid_size[0], self.grid_size[1]) # (B, V*hidden, 32, 32)
        # Multi-sacle concolutional attention module (MSCAM)
        code_fused = self.mscam_code(code_fused) # (B, hidden, 32, 32)
        code_fused = code_fused.view(B, hidden, -1).permute(0, 2, 1) # (B, n_patches, hidden)

        features_fused = []

        _, channels, f_H, f_W = features[0].shape
        features_fused.append(self.mscam_features_1(features[0].view(B, V*channels, f_H, f_W)))
        _, channels, f_H, f_W = features[1].shape
        features_fused.append(self.mscam_features_2(features[1].view(B, V*channels, f_H, f_W)))
        _, channels, f_H, f_W = features[2].shape
        features_fused.append(self.mscam_features_3(features[2].view(B, V*channels, f_H, f_W)))
        
        coord_features = self.decoder_coord(code_fused, features_fused)

        control_points, _ = self.coord_head(coord_features)
        seg_logits = self.seg_head(seg_features)
        arts_image = self.arts_head(arts_features)
        seg_logits = seg_logits.reshape(B, V, H, W) 
        arts_image = arts_image.reshape(B, V, H, W)
        # clf_logits = self.clf_head(code[:, 0])

        return arts_image, seg_logits, control_points
    
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
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class MultiTaskVisionTransformer_2Dwire(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            patch_gird: Tuple[int, int],
            patch_size: Tuple[int, int],
            resnet_num_layers: Tuple[int],
            resnet_width_factor: int,
            in_channels: int,
            attn_num_layers: int,
            attn_num_heads: int,
            mlp_dim: int,
            qkv_bias: bool,
            proj_bias: bool,
            embed_drop: float,
            attn_drop: float,
            attn_proj_drop: float,
            mlp_drop: float,
            decoder_channels: Tuple[int],
            n_skip: int,
            skip_channels: Tuple[int],
            num_view: int,
            coord_dim: int,
            total_num_control_point: int,
            classifier, 
            img_size=224, 
            vis=False
            ):
        '''
        config: 
        hidden_size(int): dim of hidden vector
        patch_gird(Tuple[int,int]): num of patches in one row/column
        patch_size(Tuple[int,int]): size of one patch
        resnet_num_layers Tuple(int): num of resnet layers
        resnet_width_factor(int): width factor of resnet
        in_channels(int): num of input channels
        attn_num_layers(int): num of transformer blocks
        num_heads(int): num of attention heads
        mlp_dim(int): dim of mlp
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        embed_drop(float): dropout rate for embedding
        attn_drop(float): dropout rate for attention
        attn_proj_drop(float): dropout rate for attention output projection
        mlp_drop(float): dropout rate for mlp
        decoder_channels(Tuple[int]): num of decoder channels
        n_skip(int): num of skip connections
        skip_channels(Tuple[int]): the dimensions of skip channels
        num_view(int): number of views
        coord_dim(int): dim of control points
        total_num_control_point(int): total number of control points
        classifier(str): type of classifier, "seg" or "clf"
        img_size(int): size of input image
        vis(bool): whether to visualize attention weights
        
        '''
        super(MultiTaskVisionTransformer_2Dwire, self).__init__()
        self.classifier = classifier
        self.transformer = Transformer(hidden_size, patch_gird, patch_size, resnet_num_layers, resnet_width_factor, img_size, in_channels, attn_num_layers, attn_num_heads, qkv_bias, proj_bias, embed_drop, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
        self.decoder_arts = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.decoder_coord = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)

        self.arts_head = ArtifactRemoverHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        # self.clf_head = ClassificationHead(hidden_size, num_classes)
        # 3D guide head
        self.coord_head = CoordHead(
            in_channels=decoder_channels[-1],
            out_channels= total_num_control_point,
            coord_size = total_num_control_point,
            coord_dim = coord_dim,
            gru_hidden_size=256
        )
    
        self.grid_size = patch_gird
    def forward(self, x):
        
        # x: (B, V, H, W)
        B, V, H, W = x.shape
        x = x.reshape(B*V ,1, H, W).contiguous()
        x = x.repeat(1,3,1,1)
        code, attn_weights, features = self.transformer(x)  # (B*V, n_patch, hidden) features: (B*V, 512, 64, 64), (B*V, 256, 128, 128), (B*V, 64, 256, 256)
        arts_features = self.decoder_arts(code, features)
        # seg_features = self.decoder_seg(code, features)
        seg_features = arts_features
        _, n_patch, hidden = code.shape

        
        coord_features = self.decoder_coord(code, features)

        control_points, _ = self.coord_head(coord_features)
        seg_logits = self.seg_head(seg_features) 
        arts_image = self.arts_head(arts_features)

        seg_logits = seg_logits.reshape(B, V, H, W)
        arts_image = arts_image.reshape(B, V, H, W)
        # clf_logits = self.clf_head(code[:, 0])

        return arts_image, seg_logits, control_points
    
#-----------------------------------------------------------------------
class MultiTaskVisionTransformer(nn.Module):
    def __init__(
            self, 
            hidden_size: int,
            patch_gird: Tuple[int, int],
            patch_size: Tuple[int, int],
            resnet_num_layers: Tuple[int],
            resnet_width_factor: int,
            in_channels: int,
            attn_num_layers: int,
            attn_num_heads: int,
            mlp_dim: int,
            qkv_bias: bool,
            proj_bias: bool,
            embed_drop: float,
            attn_drop: float,
            attn_proj_drop: float,
            mlp_drop: float,
            decoder_channels: Tuple[int],
            n_skip: int,
            skip_channels: Tuple[int],
            classifier, 
            img_size=224, 
            vis=False
            ):
        '''
        config: 
        hidden_size(int): dim of hidden vector
        patch_gird(Tuple[int,int]): num of patches in one row/column
        patch_size(Tuple[int,int]): size of one patch
        resnet_num_layers Tuple(int): num of resnet layers
        resnet_width_factor(int): width factor of resnet
        in_channels(int): num of input channels
        attn_num_layers(int): num of transformer blocks
        num_heads(int): num of attention heads
        mlp_dim(int): dim of mlp
        qkv_bias(bool): whether to add bias to qkv projections
        proj_bias(bool): whether to add bias to output projection
        embed_drop(float): dropout rate for embedding
        attn_drop(float): dropout rate for attention
        attn_proj_drop(float): dropout rate for attention output projection
        mlp_drop(float): dropout rate for mlp
        decoder_channels(Tuple[int]): num of decoder channels
        n_skip(int): num of skip connections
        skip_channels(Tuple[int]): the dimensions of skip channels
        classifier(str): type of classifier, "seg" or "clf"
        img_size(int): size of input image
        vis(bool): whether to visualize attention weights
        
        '''
        super(MultiTaskVisionTransformer, self).__init__()
        self.classifier = classifier
        self.transformer = Transformer(hidden_size, patch_gird, patch_size, resnet_num_layers, resnet_width_factor, img_size, in_channels, attn_num_layers, attn_num_heads, qkv_bias, proj_bias, embed_drop, attn_drop, attn_proj_drop, vis, mlp_dim, mlp_drop)
        self.decoder_arts = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.decoder_seg = DecoderCup(hidden_size, decoder_channels, n_skip, skip_channels)
        self.arts_head = ArtifactRemoverHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
        )

    def forward_seg(self, x):

        x = x.repeat(1,3,1,1)

        code, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        seg_features = self.decoder_seg(code, features)
        seg_logits = self.seg_head(seg_features)
        return seg_logits
    
    def forward(self, x):
        
        x = x.repeat(1,3,1,1)

        code, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        seg_features = self.decoder_seg(code, features)
        arts_features = self.decoder_arts(code, features)
        seg_logits = self.seg_head(seg_features)
        arts_logits = self.arts_head(arts_features)

        return arts_logits, seg_logits