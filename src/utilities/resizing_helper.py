import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import numpy as np
import timm
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from typing import List, Optional
import math
import copy

def divs(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bilinear',
        antialias: bool = True, # Google uses True (implicitly)
        verbose: bool = False,
        pos_embed_prefix=True,
):
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens > 0 and pos_embed_prefix: # TODO: CHECK THIS!!!
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation

    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if verbose:
        # TODO: Implement logging here
        # _logger.info(f'Resized position embedding: {old_size} to {new_size}.')
        pass

    # add back extra (class, etc) prefix tokens
    if num_prefix_tokens > 0 and pos_embed_prefix:
        if verbose:
            print(posemb_prefix.shape, posemb.shape)
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

def get_resize_mat_pinv(
    old_size: List[int],
    new_size: List[int], 
    interpolation: str = 'bilinear',
    antialias: bool = False,
):
    
    import numpy as np
    assert len(old_size) == 2, "Old shape should only be hw"
    assert len(new_size) == 2, "New shape should only be hw"
    
    if tuple(old_size) == tuple(new_size):
        return torch.eye(np.prod(old_size))

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size) # This might be the B mentioned in the paper.

    try:
        resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))
    except:
        resize_mat_pinv = torch.linalg.pinv(torch.Tensor(resize_mat.T))

    return resize_mat_pinv

def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bilinear',
        antialias: bool = False,
        resize_mat_pinv=None,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """

    old_size = patch_embed.shape[-2:]

    if old_size == new_size:
        return patch_embed

    if resize_mat_pinv is None:
        resize_mat_pinv = get_resize_mat_pinv(
            old_size=old_size,
            new_size=new_size,
            interpolation=interpolation,
            antialias=antialias,
        ).detach()

    # new^2 old^w,768 1 old^2 -> 768 1 new^2
    ens = torch.einsum('xk,abk->abx', [
        resize_mat_pinv.to(patch_embed.device),
        patch_embed.reshape(patch_embed.size(0),patch_embed.size(1), -1)
    ]).reshape(patch_embed.size(0), patch_embed.size(1), new_size[0], new_size[1])
    return ens

def vanilla_resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bilinear',
        antialias: bool = True # Google uses True (implicitly)
    ):

    B, C, H, W = patch_embed.shape

    new_size = to_2tuple(new_size)
    old_size = to_2tuple((H, W))
    if new_size == old_size:  # might not both be same container type
        return patch_embed

    # do the interpolation
    patch_embed = F.interpolate(patch_embed, size=new_size, mode=interpolation, antialias=antialias)

    return patch_embed

def get_shape(fstride, tstride, patch_size, input_fdim=128, input_tdim=1024):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(1, 768, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return [f_dim, t_dim]

def resize_weights(weights,ori_patch,new_patch,methods,t_length,s_length):
    """This funtion is different with change_model, which is to transfer the original weights to target shape
    so that it can be loaded by another new model with different patch size. It means the model should keep the dimension the same.
    """
    # Fristly, deal with the pos_embed.
    pos_emb = weights["module.v.pos_embed"] # [1, 513, 768]
    ori_size = get_shape(ori_patch,ori_patch,ori_patch,input_tdim=t_length)
    new_size = get_shape(new_patch,new_patch,new_patch,input_tdim=s_length)

    pos_emb = resample_abs_pos_embed(pos_emb,
                        new_size=new_size,
                        old_size=ori_size,
                        num_prefix_tokens=1,
                        verbose=True)
    weights["module.v.pos_embed"] = pos_emb


    p_weight = weights["module.v.patch_embed.proj.weight"]
    if methods == "PI":
        print("Use PI Resize")
        weights["module.v.patch_embed.proj.weight"] = resample_patch_embed(p_weight,(new_patch,new_patch))
    elif methods == "BL":
        print("Use Bilinear Resize")
        weights["module.v.patch_embed.proj.weight"] = vanilla_resample_patch_embed(p_weight,(new_patch,new_patch))

    return weights
