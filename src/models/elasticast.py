# Some components are borrowed from https://github.com/lucidrains/vit-pytorch

from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from torch.nn import LayerNorm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from timm.models.layers import to_2tuple,trunc_normal_
from torch.cuda.amp import autocast
import random


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

# auto grouping samples

def group_samples_by_max_seq_len(
    samples: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:
    # before inputting to ViT, we need to group samples by their size
    # pre-compute the number of patches per samples, 
    # and then group them by max sequence length
    # return [ [sample1, sample2, ...], [sample3, sample4, ...], ... ]

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for sample in samples:
        assert isinstance(sample, Tensor)

        sample_dims = sample.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, sample_dims) # How many patches in height and width

        sample_seq_len = (ph * pw)
        sample_seq_len = int(sample_seq_len * (1 - calc_token_dropout(*sample_dims))) # How many patches to keep

        assert sample_seq_len <= max_seq_len, f'sample with dimensions {sample_dims} exceeds maximum sequence length'

        if (seq_len + sample_seq_len) > max_seq_len: # If the current group is too long, start a new group, like a new batch
            groups.append(group)
            group = []
            seq_len = 0

        group.append(sample)
        seq_len += sample_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups


class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma


class FeedForward(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,dim,hidden_features=None,out_features=None,act_layer=nn.GELU,bias=True,drop=0.,use_conv=False,):
        super().__init__()
        out_features = out_features or dim
        hidden_features = hidden_features or dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(dim, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, attn_drop=0., proj_drop=0., bias = False):
        super().__init__()
        inner_dim = dim
        dim_head = dim // heads
        self.heads = heads

        # self.q_norm = RMSNorm(heads, dim_head)
        # self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)

        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # q = self.q_norm(q)
        # k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)
        

        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x,mask = None,attn_mask = None):
        for module in self.modules_list:
            x = module(x,mask = mask,attn_mask = attn_mask)
        return x

# some components are based on timm library
class Transformer(nn.Module):
    def __init__(self,dim,num_heads, mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,
                 init_values=None,drop_path=0.,act_layer=nn.GELU,norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim,eps=1e-6)
        self.attn = Attention(dim, heads=num_heads, attn_drop=attn_drop, proj_drop=drop, bias=qkv_bias)
        
        self.ls1 = nn.Identity() # No LayerScale, just to align with timm in terms of structure
        self.drop_path1 = nn.Identity()


        self.norm2 = norm_layer(dim,eps=1e-6)
        self.mlp = FeedForward(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.ls2 = nn.Identity() # No LayerScale, just to align with timm in terms of structure
        self.drop_path2 = nn.Identity()

    def forward(self,x,mask = None,attn_mask = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x),mask = mask, attn_mask = attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class ElasticAST(nn.Module):
    def __init__(self, *, sample_size, patch_size, num_classes, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None, 
                 imagenet_pretrain=False, SSAST_pretrain=False,AST_pretrain=False, avg_pool_tk = False,random_token_dropout=0,eval_token_dropout=0, **kwargs):
        super().__init__()
        sample_height, sample_width = pair(sample_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width
        self.random_token_dropout = random_token_dropout
        if self.random_token_dropout > 0:
            # if random token dropout is enabled, then disable token dropout
            # this will affect the packing process
            token_dropout_prob = 0

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob
    
        elif isinstance(token_dropout_prob, (float, int)) and token_dropout_prob > 0.:
            # assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob
        # calculate patching related stuff

        assert divisible_by(sample_height, patch_size) and divisible_by(sample_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_freq_dim, patch_time_dim = (sample_height // patch_size), (sample_width // patch_size)
        patch_dim = channels * (patch_size ** 2)
        self.channels = channels
        self.patch_size = patch_size
        self.patch_embed = nn.Sequential(
            # LayerNorm(patch_dim,eps=1e-6),
            nn.Linear(patch_dim, dim),
            # LayerNorm(dim,eps=1e-6),
        )

        self.pos_embed_freq = nn.Parameter(torch.randn(patch_freq_dim, dim))
        self.pos_embed_time = nn.Parameter(torch.randn(patch_time_dim, dim))

        self.pos_drop = nn.Dropout(emb_dropout)
        self.blocks = CustomSequential(*[Transformer(dim, heads, drop=dropout, qkv_bias=True)for i in range(depth)])

        self.norm = LayerNorm(dim,eps=1e-6)
        # final attention pooling queries
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))

        self.attn_pool = Attention(dim = dim, heads = heads, bias=True)

        # output to logits
        self.to_latent = nn.Identity()
        self.mlp_norm = LayerNorm(dim,eps=1e-6)
        self.head = nn.Linear(dim, num_classes, bias = True)

        self.avg_pool_tk = avg_pool_tk

        self.eval_token_dropout = eval_token_dropout

        if imagenet_pretrain == True:
            print("Loaded Pretrained ViT Model")
            self.load_state_dict(self.transfer_ImageNetWeights(), strict=False)
            # miss, expect = self.load_state_dict(self.transfer_ImageNetWeights(), strict=False)
            # print(f"Missed keys: {miss}")
            # print(f"Expected keys: {expect}")
        if SSAST_pretrain == True:
            print("Loaded Pretrained SSAST Model")
            miss, expect = self.load_state_dict(self.transfer_SSAST(), strict=False)
            print(f"Missed keys: {miss}")

    def transfer_ImageNetWeights(self):
        load_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=309)
        dim=768
        load_dict = {}

        for i in load_model.state_dict().keys():
            if 'qkv' in i:
                if len(load_model.state_dict()[i].shape) == 1: # if this tensor only has one dimension, then it is a bias
                    load_dict[i.replace('qkv', 'to_q').replace("blocks","blocks.modules_list")] = load_model.state_dict()[i][:dim]
                    load_dict[i.replace('qkv', 'to_kv').replace("blocks","blocks.modules_list")] = load_model.state_dict()[i][dim:]
                else:
                    load_dict[i.replace('qkv', 'to_q').replace("blocks","blocks.modules_list")] = load_model.state_dict()[i][:dim,:]
                    load_dict[i.replace('qkv', 'to_kv').replace("blocks","blocks.modules_list")] = load_model.state_dict()[i][dim:,:]
            else:
                if "blocks" in i:
                    load_dict[i.replace("blocks","blocks.modules_list")] = load_model.state_dict()[i]
                elif 'head.weight' == i or 'head.bias' == i:
                    pass
                elif "patch_embed.proj.weight" == i:
                    proj_weight = torch.sum(load_model.state_dict()["patch_embed.proj.weight"], dim=1).unsqueeze(1)
                    load_dict['patch_embed.0.weight'] = proj_weight.permute(0,2,3,1).reshape(768, -1)
                elif "patch_embed.proj.bias" == i:
                    load_dict['patch_embed.0.bias'] = load_model.state_dict()[i]
                else:
                    load_dict[i] = load_model.state_dict()[i]

        original_num_patches = load_model.patch_embed.num_patches
        oringal_hw = int(original_num_patches ** 0.5)
        original_embedding_dim = load_model.pos_embed.shape[2]

        new_pos_embed = load_dict['pos_embed'][:,1:,:].transpose(1, 2).reshape(1, original_embedding_dim, oringal_hw, oringal_hw)

        t_dim, f_dim = self.pos_embed_time.shape[0], self.pos_embed_freq.shape[0]
        if t_dim <= oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(oringal_hw / 2) - int(t_dim / 2): int(oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(oringal_hw, t_dim), mode='bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(oringal_hw / 2) - int(f_dim / 2): int(oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

        load_dict['pos_embed_time'] = new_pos_embed.mean(dim=-2).squeeze(0).transpose(0,1)
        load_dict['pos_embed_freq'] = new_pos_embed.mean(dim=-1).squeeze(0).transpose(0,1)

        # load_dict['mlp_norm.weight'] = load_dict['norm.weight']
        # load_dict['mlp_norm.bias'] = load_dict['norm.bias']

        ####### Average the weights of the attention layers, to load into the attention pooling layer #######
        layers = 0
        to_q_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_q.weight'])
        to_q_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_q.bias'])
        to_kv_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_kv.weight'])
        to_kv_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_kv.bias'])
        proj_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.proj.weight'])
        proj_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.proj.bias'])

        for i in load_dict.keys():
            if "attn" in i:
                layers += 1
                if "to_q.weight" in i:
                    to_q_weight += load_dict[i]
                elif "to_q.bias" in i:
                    to_q_bias += load_dict[i]
                elif "to_kv.weight" in i:
                    to_kv_weight += load_dict[i]
                elif "to_kv.bias" in i:
                    to_kv_bias += load_dict[i]
                elif "proj.weight" in i:
                    proj_weight += load_dict[i]
                elif "proj.bias" in i:
                    proj_bias += load_dict[i]

        to_q_weight /= layers
        to_q_bias /= layers
        to_kv_weight /= layers
        to_kv_bias /= layers
        proj_weight /= layers
        proj_bias /= layers


        load_dict['attn_pool.to_q.weight'] = to_q_weight
        load_dict['attn_pool.to_q.bias'] = to_q_bias
        load_dict['attn_pool.to_kv.weight'] = to_kv_weight
        load_dict['attn_pool.to_kv.bias'] = to_kv_bias
        load_dict['attn_pool.proj.weight'] = proj_weight
        load_dict['attn_pool.proj.bias'] = proj_bias

        return load_dict

    def transfer_SSAST(self):
        SSAST_pth = torch.load('/home/jfeng/FJ/ElasticAST/egs/SSAST-Base-Patch-400.pth')['model_state']
        pos_emb = SSAST_pth["module.v.pos_embed"] # [1, 514, 768]
        pos_emb = torch.cat((pos_emb[:,0,:].unsqueeze(1),pos_emb[:,2:,:]),dim=1) # remove the distillation token
        SSAST_pth["module.v.pos_embed"] = torch.nn.Parameter(pos_emb)

        SSAST = {}
        for k, v in SSAST_pth.items(): # Adjust the name of dict
            if not k[7:].startswith("v."):
                SSAST[k[7:]] = v
            else:
                SSAST[k[7+2:]] = v

        dim=768
        load_dict = {}

        for i in SSAST.keys():

            if 'qkv' in i:
                if len(SSAST[i].shape) == 1: # if this tensor only has one dimension, then it is a bias
                    load_dict[i.replace('qkv', 'to_q').replace("blocks","blocks.modules_list")] = SSAST[i][:dim]
                    load_dict[i.replace('qkv', 'to_kv').replace("blocks","blocks.modules_list")] = SSAST[i][dim:]
                else:
                    load_dict[i.replace('qkv', 'to_q').replace("blocks","blocks.modules_list")] = SSAST[i][:dim,:]
                    load_dict[i.replace('qkv', 'to_kv').replace("blocks","blocks.modules_list")] = SSAST[i][dim:,:]
            else:
                if "blocks" in i:
                    load_dict[i.replace("blocks","blocks.modules_list")] = SSAST[i]
                elif 'head.weight' == i or 'head.bias' == i:
                    pass
                elif "patch_embed.proj.weight" == i:
                    proj_weight = torch.sum(SSAST["patch_embed.proj.weight"], dim=1).unsqueeze(1)
                    load_dict['patch_embed.0.weight'] = proj_weight.permute(0,2,3,1).reshape(768, -1)
                elif "patch_embed.proj.bias" == i:
                    load_dict['patch_embed.0.bias'] = SSAST[i]
                else:
                    load_dict[i] = SSAST[i]

        new_pos_embed = SSAST["pos_embed"][:,1:,:].transpose(1, 2).reshape(1, -1, 8, 64)


        t_dim, f_dim = self.pos_embed_time.shape[0], self.pos_embed_freq.shape[0]
        if t_dim != 64:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim != 8:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')


        load_dict['pos_embed_time'] = new_pos_embed.mean(dim=-2).squeeze(0).transpose(0,1)
        load_dict['pos_embed_freq'] = new_pos_embed.mean(dim=-1).squeeze(0).transpose(0,1)

        load_dict['mlp_norm.weight'] = load_dict['norm.weight']
        load_dict['mlp_norm.bias'] = load_dict['norm.bias']


        layers = 0
        to_q_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_q.weight'])
        to_q_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_q.bias'])
        to_kv_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_kv.weight'])
        to_kv_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.to_kv.bias'])
        proj_weight = torch.zeros_like(load_dict['blocks.modules_list.0.attn.proj.weight'])
        proj_bias = torch.zeros_like(load_dict['blocks.modules_list.0.attn.proj.bias'])

        for i in load_dict.keys():
            if "attn" in i:
                layers += 1
                if "to_q.weight" in i:
                    to_q_weight += load_dict[i]
                elif "to_q.bias" in i:
                    to_q_bias += load_dict[i]
                elif "to_kv.weight" in i:
                    to_kv_weight += load_dict[i]
                elif "to_kv.bias" in i:
                    to_kv_bias += load_dict[i]
                elif "proj.weight" in i:
                    proj_weight += load_dict[i]
                elif "proj.bias" in i:
                    proj_bias += load_dict[i]

        to_q_weight /= layers
        to_q_bias /= layers
        to_kv_weight /= layers
        to_kv_bias /= layers
        proj_weight /= layers
        proj_bias /= layers


        load_dict['attn_pool.to_q.weight'] = to_q_weight
        load_dict['attn_pool.to_q.bias'] = to_q_bias
        load_dict['attn_pool.to_kv.weight'] = to_kv_weight
        load_dict['attn_pool.to_kv.bias'] = to_kv_bias
        load_dict['attn_pool.proj.weight'] = proj_weight
        load_dict['attn_pool.proj.bias'] = proj_bias

        return load_dict
    
    @property
    def device(self):
        return next(self.parameters()).device

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def avg_pool_tokens(self,data,attn_pool_mask):
        b_num = attn_pool_mask.size(0)
        pack_num = attn_pool_mask.size(2)
        pool = []

        for i in range(b_num):
            for j in range(pack_num):
                if attn_pool_mask[i,0,j].nonzero().squeeze().shape[0] != 0:
                    mask_idx = attn_pool_mask[i,0,j].nonzero().squeeze()
                    pool.append(data[i,mask_idx[0]:mask_idx[-1]+1,:].mean(dim=0))
        return torch.stack(pool)
    
    @autocast()
    def forward(
        self,
        batched_samples: Union[List[Tensor], List[List[Tensor]]], # assume different resolution samples already grouped correctly
        group_samples = True,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)


        if group_samples and self.training:
            batched_samples = group_samples_by_max_seq_len(
                batched_samples,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout,
                max_seq_len = group_max_seq_len
            )
        elif group_samples: # if in eval mode, or if not grouping, then just flatten the list
            batched_samples = group_samples_by_max_seq_len(
                batched_samples,
                patch_size = self.patch_size,
                calc_token_dropout = None,
                max_seq_len = group_max_seq_len
            )

        # process samples into variable lengthed sequences with attention mask
        num_samples = []
        batched_sequences = []
        batched_positions = []
        batched_sample_ids = []

        for samples in batched_samples:
            num_samples.append(len(samples))

            sequences = []
            positions = []
            sample_ids = torch.empty((0,), device = device, dtype = torch.long) # empty tensor to keep track of sample ids

            for sample_id, sample in enumerate(samples):
                assert sample.ndim ==3 and sample.shape[0] == c
                sample_dims = sample.shape[-2:]
                assert all([divisible_by(dim, p) for dim in sample_dims]), f'height and width {sample_dims} of samples must be divisible by patch size {p}'
                ph, pw = map(lambda dim: dim // p, sample_dims)

                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)

                pos = rearrange(pos, 'h w c -> (h w) c') # flatten pos embedding
                seq = rearrange(sample, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p) # make the spec to [num_patches, patch_size*patch_size]

                seq_len = seq.shape[-2]

                if self.random_token_dropout>0 and self.training:
                    token_dropout = random.uniform(0,self.random_token_dropout)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]
                elif has_token_dropout and self.training:
                    token_dropout = self.calc_token_dropout(*sample_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]
                
                elif not self.training and self.eval_token_dropout > 0:
                    # drop some tokens at evaluation time, check its flexibility
                    token_dropout = self.eval_token_dropout
                    num_keep = max(1, int(seq_len * token_dropout))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]
                    
                

                sample_ids = F.pad(sample_ids, (0, seq.shape[-2]), value = sample_id) 
                # pad sample ids to match sequence length, use sample id as token id
                # could also be used to distinguish between different samples
                sequences.append(seq)
                positions.append(pos)

            batched_sample_ids.append(sample_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        # derive key padding mask
        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

        # derive attention mask, and combine with key padding mask from above
        batched_sample_ids = pad_sequence(batched_sample_ids,padding_value=-1) # [0,0,0,1,1,1,1,2,2,2,2,2,...., n,n,n, -1,-1,-1]
        attn_mask = rearrange(batched_sample_ids, 'b i -> b 1 i 1') == rearrange(batched_sample_ids, 'b j -> b 1 1 j')

        attn_mask = attn_mask & rearrange(~key_pad_mask, 'b j -> b 1 1 j')

        # combine patched samples as well as the patched width / height positions for 2d positional embedding
        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # need to know how many samples for final attention pooling
        num_samples = torch.tensor(num_samples, device = device, dtype = torch.long)        
        # to patches
        x = self.patch_embed(patches)        

        # factorized 2d absolute positional embedding
        h_indices, w_indices = patch_positions.unbind(dim = -1)

        h_pos = self.pos_embed_freq[h_indices]
        w_pos = self.pos_embed_time[w_indices]


        x = x + h_pos + w_pos

        # embed dropout
        x = self.pos_drop(x)

        # attention
        x = self.blocks(x,attn_mask=attn_mask)
        x = self.norm(x)

        max_queries = num_samples.amax().item()
        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        # attention pool mask
        sample_id_arange = arange(max_queries)
        attn_pool_mask = rearrange(sample_id_arange, 'i -> i 1') == rearrange(batched_sample_ids, 'b j -> b 1 j')
        attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')
        # Now, the tokens from the same samples are True, others are False

        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

        # attention pool
        if self.avg_pool_tk:
            x = self.avg_pool_tokens(x,attn_pool_mask)
        else:
            x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask)

            x = rearrange(x, 'b n d -> (b n) d') # unpack samples from multiple rows back into one row
            is_samples = sample_id_arange < rearrange(num_samples, 'b -> b 1')
            is_samples = rearrange(is_samples, 'b n -> (b n)')

            x = x[is_samples]

        # project out to logits
        x = self.to_latent(x)
        x = self.mlp_norm(x)
        x = self.head(x)

        return x
