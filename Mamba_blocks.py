import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from functools import partial

from mamba_ssm.modules.mamba2 import Mamba2
import importlib
import mamba_ssm.modules.block
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

'''d_model: size of the hidden representation
   ssm_cfg: dictionary with mamba config options
   norm_epsilon: small cnt to avoid division by zero in normalization
   rms_norm: if true RMSNorm instead of LayerNorm
   residual_in_fp32: use 32-bit precision for residuals (helps stability)
   fused_add_norm: use fused operation (residual+norm), GPU
   layer_idx: layer #
   device, dtype
'''

# Weight Initialization:
# change n_residuals_per_layers to 2 if we have MLP
def _init_weights(module, n_layer, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):  # checks if module is an instance of nn.Linear
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):  # get attribute
                nn.init.zeros_(module.bias)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# class that prepares the input, the mixer to be used, the parameters and the architecture of the backbone
class MixerModel(nn.Module):
    def __init__(
        self,
        preprocessed_features: int,  # preprocessed input 
        d_model: int,  # size of hidden state
        n_layer: int,  # # of blocks
        d_intermediate: int,  # size of block's MLP (size of the feed-forward)
        ssm_cfg=None,  # dictionary with mamba config options
        norm_epsilon=1e-5,  # small cnt to avoid division by zero in normalization
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        #############################################################################################
        self.input_proj = nn.Linear(preprocessed_features, d_model, **factory_kwargs)  # Extra projection
        #############################################################################################

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        
        # apply applies the given function (partial) to each submodule (layer)
        self.apply(partial(
            _init_weights,
            n_layer=n_layer,
            **(initializer_cfg if initializer_cfg is not None else {}),
            n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP (the mlp inside each block)
        ))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, preprocessed_features, inference_params=None, **mixer_kwargs):
        hidden_states = self.input_proj(preprocessed_features)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


def create_block(d_model, d_intermediate, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, 
                residual_in_fp32=False, fused_add_norm=False, layer_idx=None, device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
        
    # it packs common arguments for pytorch layers
    factory_kwargs = {"device": device, "dtype": dtype}

    # mamba_block, do a copy of config to modify config INSIDE this function only!
    if ssm_cfg is not None:
        ssm_cfg_copy = copy.deepcopy(ssm_cfg)
    else:
        ssm_cfg_copy = {}
    
    # extract me ssm_cfg['layer'] if it exists — or else just use 'Mamba2'
    ssm_layer = ssm_cfg_copy.pop("layer", "Mamba2")
    if ssm_layer not in ["Mamba1", "Mamba2"]:
        raise ValueError(f"INVALID SSM_LAYER")
    
    # with partial I dont create the Mamba block yet, just prepare it with the parameters to be called later.
    mixer_cls = partial(Mamba2 if ssm_layer == "Mamba2" else Mamba2, layer_idx=layer_idx, **ssm_cfg_copy, **factory_kwargs)  # I am forcing Mamba2 always
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    
    block = Block(
        d_model,
        mixer_cls,  # passing a function
        mlp_cls,
        norm_cls=norm_cls,  # passing a class
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        # drop_path=0.1 #########
    )
    block.layer_idx = layer_idx  # dynamic attribute
    return block