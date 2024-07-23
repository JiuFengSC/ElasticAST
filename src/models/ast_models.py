# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat



# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,bias=True,dynamic_img_pad=False):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,bias=bias)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, attn_drop=0., proj_drop=0., bias = False):
        super().__init__()
        inner_dim = dim
        dim_head = dim // heads
        self.heads = heads

        # self.q_norm = RMSNorm(heads, dim_head)
        # self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1) # Not sure if this is necessary 
        self.attn_drop = nn.Dropout(attn_drop)

        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)

        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.queries = nn.Parameter(torch.randn(dim))

    def forward(
        self,
        x,
        context = None,
    ):
        B, N, C = x.shape
        queries = repeat(self.queries, 'd -> b n d', n = 1, b = x.shape[0])


        qkv = (self.to_q(queries), *self.to_kv(x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # q = self.q_norm(q)
        # k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='deit384', verbose=True, no_cls=None, factorized_pos=False, drop_compression=1):

        super(ASTModel, self).__init__()
        # assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        # timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.no_cls = no_cls

        if model_size == 'deit224':
            self.v = timm.create_model('deit_base_distilled_patch16_224', pretrained=imagenet_pretrain,embed_layer=PatchEmbed)
            self.deit = True
        elif model_size == 'deit384':
            self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=imagenet_pretrain,embed_layer=PatchEmbed)
            # self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            self.deit = True
        elif model_size == 'vit224':
            self.v = timm.create_model('vit_base_patch16_224', pretrained=imagenet_pretrain,embed_layer=PatchEmbed)
            self.deit = False
        elif model_size == 'vit384':
            print('Using vit384')
            self.v = timm.create_model('vit_base_patch16_384', pretrained=imagenet_pretrain,embed_layer=PatchEmbed)
            self.deit = False
        else:
            raise Exception('Model size error.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))


        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        if self.no_cls == "attn":
            self.attn_pool = Attention(self.original_embedding_dim, heads = 12)
        self.factorized_pos = factorized_pos
        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            if self.deit:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            else:
                new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.

            if self.factorized_pos:
                self.f_dim, self.t_dim = f_dim, t_dim
                self.v.f_pos_embed = nn.Parameter(torch.randn(self.f_dim, self.original_embedding_dim ))
                self.v.t_pos_embed = nn.Parameter(torch.randn(self.t_dim, self.original_embedding_dim ))
            # elif self.no_cls:
            #     self.v.pos_embed = nn.Parameter(new_pos_embed)
            elif self.deit:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            # TODO can use sinusoidal positional embedding instead
            if self.factorized_pos:
                    self.f_dim, self.t_dim = f_dim, t_dim
                    self.v.f_pos_embed = nn.Parameter(torch.randn(self.f_dim, self.original_embedding_dim ))
                    self.v.t_pos_embed = nn.Parameter(torch.randn(self.t_dim, self.original_embedding_dim ))
            # elif self.no_cls:
            #     new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches, self.original_embedding_dim))
            elif self.deit:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 1, self.original_embedding_dim))
            
            if not self.factorized_pos:
                self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        self.drop_compression = drop_compression
        
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]

        x = self.v.patch_embed(x)

        # if self.no_cls:
        #     pass
        # else:

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        if self.deit:
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # if self.factorized_pos:
        #     pos = torch.stack(torch.meshgrid((
        #             torch.arange(8),
        #             torch.arange(64)
        #         ), indexing = 'ij'), dim = -1).to(x.device)
        #     f_indices,t_indices = pos.unbind(dim = -1)
        #     f_pos = self.v.f_pos_embed[f_indices.flatten()]
        #     t_pos = self.v.t_pos_embed[t_indices.flatten()]
        #     x = x + f_pos.unsqueeze(0) + t_pos.unsqueeze(0)
        # else:
        x = x + self.v.pos_embed
        if self.drop_compression > 1.0:
            cls = x[:, 0]
            x = x[:, 1:]
            keep_indices = []
            for i in range(x.shape[0]):
                num_keep = int(x.shape[1] / self.drop_compression) # how many tokens to keep
                keep_indices_i = torch.randn(x.shape[1]).topk(num_keep).indices # random select the tokens to keep
                keep_indices.append(keep_indices_i) # record the indices
            keep_indices = torch.stack(keep_indices)
            x = x[torch.arange(x.shape[0])[:, None], keep_indices]
            x = torch.cat([cls.unsqueeze(1), x], dim=1)
        
        # if self.eval_constant_drop > 1:
        #     cls = x[:, 0]
        #     x = x[:, 1:]
        #     keep_indices = []
        #     for i in range(x.shape[0]):
        #         num_keep = int(x.shape[1] / self.drop_compression) # how many tokens to keep
        #         keep_indices_i = torch.randn(x.shape[1]).topk(num_keep).indices # random select the tokens to keep
        #         keep_indices.append(keep_indices_i) # record the indices
        #     keep_indices = torch.stack(keep_indices)
        #     x = x[torch.arange(x.shape[0])[:, None], keep_indices]
        #     x = torch.cat([cls.unsqueeze(1), x], dim=1)

        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        if self.no_cls == 'avg':
            # x = x.mean(dim=1)
            if self.deit:
                x = torch.mean(x[:, 2:, :], dim=1)
            else:
                x = torch.mean(x[:, 1:, :], dim=1)
        
        elif self.no_cls == 'attn':
            x = self.attn_pool(x).squeeze(1)
        else:
            if self.deit:
                x = (x[:, 0] + x[:, 1]) / 2
            else:
                x = x[:, 0]

        x = self.mlp_head(x)
        return x

if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    input_tdim = 256
    ast_mdl = ASTModel(input_tdim=input_tdim,label_dim=50, audioset_pretrain=True)
    # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    print(test_output.shape)