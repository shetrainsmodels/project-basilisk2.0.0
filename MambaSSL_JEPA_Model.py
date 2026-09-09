from dataclasses import dataclass, field
from Mamba_blocks import MixerModel
import copy
import torch
import torch.nn as nn

def masking_algorithm_targets(tokens, mask_ratio: float = 0.33, t_l: int = 3, generator = None)-> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: 
    B, Lw, _ =  tokens.shape # [B, 18, 384]

    # ---- SPAN: contiguous timestep t_l = 3, mask shape [B, Lw] ----
    n_blocks = max(1, round(Lw * mask_ratio / t_l))   # how many whole t_l-blocks fit the request
    target_current = n_blocks * t_l

    #create empty mask
    mask_targets = torch.zeros((B, Lw), dtype=torch.bool, device=tokens.device)
    target_blocks = []
    masks = []

    for b in range(B):
        I = set()
        counter = 0
        blocks_b = []

        while counter < target_current:
            valid_start = [] # find valid starting indices

            for index in range(Lw - t_l + 1):
                candidates = set(range(index, index + t_l)) # create candidates
                if not(candidates & I):
                    valid_start.append(index)
            if len(valid_start) == 0:
                raise RuntimeError(f"Cannot place span of {t_l} tokens")
                
            pick_indices = torch.randint(0, len(valid_start), (1,), generator = generator).item() #sample rnd value
            candidates = list(range(valid_start[pick_indices], valid_start[pick_indices] + t_l)) # create candidates

            blocks_b.append(candidates) # store the indices of the masked blocks
            I.update(candidates) # create idx for that window
            counter = counter + t_l
        target_blocks.append(blocks_b)
        mask_targets[b, list(I)] = True # need ALL the masked values for the loss.
    
    blocks = torch.tensor(target_blocks, dtype = torch.long, device = tokens.device)
    for i in range(n_blocks):
        masks.append(blocks[:, i, :]) # A lists of 2 tensors

    context_input = tokens.clone()
    context_input[mask_targets] = 0
    return context_input, mask_targets, masks


@dataclass
class HARMambaConfig:
    conv_kernel: int = 5
    conv_stride: int = 5
    d_model: int = 384
    num_sensor_features: int = 45
    d_intermediate: int = 512
    n_layer: int = 24 # Testing with : 8, 12, 16, 24 
    n_tokens: int = 18
    ssm_cfg: dict = field(default_factory=lambda: {"expand": 4, "layer": "Mamba2", "headdim": 8, "d_ssm": 768, "dt_min": 0.001, "dt_max": 0.1,})
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

### -------------- Same Encoder as MambaSSL masking -------------- ###   
class MambaEncoderModel(nn.Module):
    def __init__(self, config, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        self.kernel_size = config.conv_kernel
        self.stride = config.conv_stride
        self.padding = 0
        
        # HAR INPUT AND BLOCKS
        # CONV1D -> GELU -> Backbone (MAMBA) -> Classifier
        self.convolutional_input = nn.Conv1d(
            in_channels=config.num_sensor_features, 
            out_channels=d_model, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
        )
        self.LN_layer = nn.LayerNorm(d_model) 
        self.act_function = nn.GELU()

        self.backbone = MixerModel(
            preprocessed_features=d_model, 
            d_model=d_model, 
            n_layer=n_layer, 
            d_intermediate=d_intermediate, 
            ssm_cfg=ssm_cfg, 
            rms_norm=rms_norm, 
            initializer_cfg=initializer_cfg, 
            fused_add_norm=fused_add_norm, 
            residual_in_fp32=residual_in_fp32, 
            **factory_kwargs
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)       

    def tokenize(self, input_sensor1):
        '''
        Sensor Input -> [Batch_size, Length_window, num_sensor_features]
        Output ->  [B, L, d_model]
        '''
        assert input_sensor1.size(1) % self.stride == 0, f"window lenght {input_sensor1.size(1)} not divisible by stride {self.stride}"
        x = input_sensor1.transpose(1, 2).contiguous()  # [B, L, Ft] -> [B, Ft, L]
        x = self.convolutional_input(x) # [B, d_model, L]
        x = x.transpose(1, 2)  # [B, d_model, L] -> [B, L, d_model]
        x = self.LN_layer(x) # [B, L, d_model]
        x = self.act_function(x) # [B, L, d_model]        
        return x

    def encode_tokens(self, x, inference_params=None, **mixer_kwargs) -> torch.Tensor:
        '''
        Input -> [B, L, d_model]
        Output -> hidden_states = [B, L, d_model]
        '''
        hidden_states = self.backbone(x, inference_params=inference_params, **mixer_kwargs) # [B, L, d_model]
        return hidden_states
    
    def forward(self, input_sensor1, inference_params=None, **mixer_kwargs) -> torch.Tensor:
        return self.encode_tokens(self.tokenize(input_sensor1), inference_params=inference_params, **mixer_kwargs)

##################################################################### 

class MambaJEPA(nn.Module):
    '''
    Context Encoder: MambaEncoderModel:
    Training mode
    Target Encoder: MambaEncoderModel (Deepcopy of Context Encoder):
    Gradient frozen and not optimizer
    Updated by EMA of Context Encoder weights
    '''
    def __init__(self, config, mask_ratio = 1/2, t_l = 3, num_heads = 6, use_pe: bool = False, drop: bool = False, recon: bool = False, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.mask_ratio = mask_ratio
        self.t_l = t_l
        #context encoder
        self.context_encoder = MambaEncoderModel(config = config, initializer_cfg=initializer_cfg, device=device, dtype=dtype)
        #target encoder
        self.target_encoder = copy.deepcopy(self.context_encoder) #same arch and weights, mandatory
        self.target_encoder.requires_grad_(False) #no grad flow in T.Encoder
        self.use_pe = use_pe
        self.drop = drop
        if use_pe:
            self.pe = nn.Parameter(torch.zeros(1, config.n_tokens, config.d_model, device=device, dtype=dtype))
            nn.init.trunc_normal_(self.pe, std = 0.02)
            self.register_buffer("pe_target", self.pe.detach().clone()) # copy pe of target encoder 
        else:
            self.pe = None
            self.pe_target = None
        #predictor
        self.predictor = PredictorTransformer(config, num_heads = num_heads, drop = drop)
        #reconstruction 
        self.recon = recon
        self.decoder = DecoderModel(config, device=device, dtype=dtype) if recon else None
        if recon:
            self.recon_mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.trunc_normal_(self.recon_mask_token, std = 0.02)

    #  EMA UPDATE FOR TARGET ENCODER
    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for context_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_param.mul_(momentum).add_(context_param, alpha=1.0 - momentum)
        if self.use_pe:
            self.pe_target.mul_(momentum).add_(self.pe, alpha=1.0 - momentum)

    def forward(self, input_sensor1):
        self.target_encoder.eval() # 
        with torch.no_grad():
            tokens_target = self.target_encoder.tokenize(input_sensor1)
            # ADD PE
            if self.use_pe:
                tokens_target = tokens_target + self.pe_target
            target_embeddings = self.target_encoder.encode_tokens(tokens_target) # [B, 18, 384]

        tokens_context = self.context_encoder.tokenize(input_sensor1)
        context_input, mask_targets, target_blocks = masking_algorithm_targets(tokens_context, mask_ratio = self.mask_ratio, t_l = self.t_l, generator=None)

        # ADD PE
        #if self.use_pe:
            #context_input = context_input + self.pe
        # ADD DROPPING and PE
        # -------------------------------------------------
        # RAW / MAE BRANCH
        # -------------------------------------------------
        context_input_recon = None
        if self.recon:
            mask_token = self.recon_mask_token.expand_as(context_input)
            recon_input = torch.where(mask_targets.unsqueeze(-1), mask_token, context_input)
            if self.use_pe:
                recon_input = recon_input + self.pe
            # Full sequence goes through Mamba for reconstruction
            context_input_recon = self.context_encoder.encode_tokens(recon_input)

        # -------------------------------------------------
        # JEPA BRANCH
        # -------------------------------------------------
        jepa_input = context_input

        if self.use_pe:
            jepa_input = context_input + self.pe

        if self.drop:
            # REMOVE masked positions completely for JEPA
            visible_input = jepa_input[~mask_targets].view(mask_targets.size(0), -1, jepa_input.size(2))
        else:
            visible_input = jepa_input

        context_embeddings = self.context_encoder.encode_tokens(visible_input)


        # Predictor
        predictions = self.predictor(target_blocks, context_embeddings, mask_targets) # [2B, 3, 384]
        targets_emb_2B = target_embeddings.repeat(len(target_blocks), 1, 1) # [2B, 18, 384]
        blocks_cat = torch.cat(target_blocks, dim = 0) # [2B, 3]
        index = blocks_cat.unsqueeze(-1).expand(-1, -1, target_embeddings.size(-1))
        targets = torch.gather(targets_emb_2B, 1, index) 

        return target_embeddings, targets, context_embeddings, mask_targets, target_blocks, predictions, context_input_recon

class MLP(nn.Module):
    def __init__(self, config, mlp_ratio = 4):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model//2, (config.d_model//2)*mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear((config.d_model//2)*mlp_ratio, config.d_model//2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn. Module):
    def __init__(self, config, num_heads = 6):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.scale = ((config.d_model//2)//num_heads) ** -0.5
        self.qkv = nn.Linear(config.d_model//2, (config.d_model//2)*3) # 192 -> 576
        self.proj = nn.Linear(config.d_model//2, config.d_model//2)
        
    def forward(self, x):
        B, TK, C = x.shape # [2B, 15 192]
        qkv = self.qkv(x)  # [2B, 15 576]
        qkv = qkv.reshape(B, TK, 3, self.num_heads, C//self.num_heads) # [2B, 15, 3, 6, 32] 6x32 gives 6 score relationships
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, 2B, 6, 15, 32] 
        q, k, v = qkv[0], qkv[1], qkv[2] # each [2B, 6, 15, 32]
        
        attn = q@k.transpose(-2, -1) * self.scale # [15, 32] @ [32, 15] = [15, 15] 
        attn = attn.softmax(dim = -1) 
        x = attn@v # [2B, 6, 15, 32] 
        x = x.transpose(1, 2).reshape(B, TK, C) # [2B, 15, 192]
        x = self.proj(x)
        return x

class Block_Attn(nn.Module):
    def __init__(self, config, num_heads = 6):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model//2)
        self.attn = Attention(config, num_heads)
        self.ln2 = nn.LayerNorm(config.d_model//2)
        self.mlp = MLP(config, mlp_ratio = 4)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PredictorTransformer(nn.Module):
    def __init__(self, config, n_blocks = 4, num_heads = 6, drop: bool = False) -> None:
        super().__init__()
        self.config = config
        self.project_in = nn.Linear(config.d_model, config.d_model//2)
        self.project_out = nn.Linear(config.d_model//2, config.d_model)
        self.predictor_pe = nn.Parameter(torch.zeros(1, config.n_tokens, config.d_model//2))
        nn.init.trunc_normal_(self.predictor_pe, std = 0.02)
        self.learnable_mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model//2))

        self.blocks = nn.ModuleList([Block_Attn(config, num_heads) for _ in range(n_blocks)])
        self.layer_norm = nn.LayerNorm(config.d_model//2) 
        self.drop = drop


    def forward(self, masks, context_embeddings, mask_targets):
        # ---- Take visible context and expand 2B ----
        visible_index = (~mask_targets).nonzero(as_tuple = False)[:, 1].view(context_embeddings.size(0), -1) #what is visible [B, 12]
        if not self.drop:
            visible_index_full = visible_index.unsqueeze(-1).expand(-1, -1, context_embeddings.size(-1)) # [B, 12, 384]
            context_emb_visible = torch.gather(context_embeddings, 1, visible_index_full) # [B, 12, 384]
        else:
            context_emb_visible = context_embeddings

        context_emb_expanded = context_emb_visible.repeat(len(masks), 1, 1) # n of tensors in my masks list [2B, 12, 384]
        # ---- Build Target Blocks, Learnable mask token and PE expand 2B ----
        target_blocks = torch.cat(masks, dim = 0) # [2B, 3]
        predictor_mt = self.learnable_mask_token.repeat(target_blocks.size(0), target_blocks.size(1), 1) #[2B, 3, 192]
        pe_all = self.predictor_pe.repeat(target_blocks.size(0), 1, 1) #[2B, 18, 192]
        # ---- Take PE for visible context ----
        visible_index_2B = visible_index.repeat(len(masks), 1) # [2B, 12]
        visible_index_ctx = visible_index_2B.unsqueeze(-1).expand(-1, -1, pe_all.size(-1)) # [2B, 12, 192]
        pe_visible_context = torch.gather(pe_all, 1, visible_index_ctx) # [2B, 12, 192]
        # ---- Take PE for learnable mask tokens ----
        mask_index_ctx = target_blocks.unsqueeze(-1).repeat(1, 1, pe_all.size(-1)) #[2B, 3, 192] = the positions of the mask
        pe_masks = torch.gather(pe_all, 1, mask_index_ctx) # gather from pe_all the index  = gather the PE of the mask positions
        predictor_mt = predictor_mt + pe_masks # add PE to predictor's learnable mas tokens 

        # Flow, project from 384 -> 192, add PE visible context, concat PE learnable mask tokens
        x = self.project_in(context_emb_expanded)
        x = x + pe_visible_context
        x = torch.cat([x, predictor_mt], dim = 1)
        ##TRANSFORMER
        for block in self.blocks:
            x = block(x)
        #layer norm
        x = self.layer_norm(x)
        x = x[:, context_emb_expanded.size(1):]
        x = self.project_out(x) # [B, 3, 384]
        return x
        
class DecoderModel(nn.Module):
    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()

        self.config = config
        d_model = config.d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.decoder = nn.Sequential(
        nn.Linear(d_model, d_model, **factory_kwargs),
        nn.GELU(),
        nn.LayerNorm(d_model, **factory_kwargs),
        nn.Linear(d_model, self.config.conv_stride * self.config.num_sensor_features, **factory_kwargs)
        )
          
    def forward(self, hidden_states) -> torch.Tensor:
        '''
        Input -> hidden_states = [B, L, d_model]
        Output -> Reconstructed signal [B, L*5, num_sensor_features]
        '''
        B, L, _ = hidden_states.shape
        out = self.decoder(hidden_states)                   # [B, L, 5*num_sensor_features]
        out = out.view(B, L, self.config.conv_stride, -1).reshape(B, L * self.config.conv_stride, -1)   # each token -> its 5 timesteps
        return out   

class MambaDownstreamClassifier(nn.Module):
    def __init__(self, config, num_classes: int, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        
        self.pe = None
        self.encoder = MambaEncoderModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype
        )
        
        d_model = config.d_model
        factory_kwargs = {"device": device, "dtype": dtype}

        self.classifier = nn.Sequential(
            #TEST1: MLP PROBE
            nn.LayerNorm(d_model, **factory_kwargs),
            nn.Linear(d_model, num_classes, **factory_kwargs)
            #nn.Linear(d_model, d_model, **factory_kwargs),
            #nn.ReLU(),
            #nn.LayerNorm(d_model, **factory_kwargs),
            #nn.Dropout(0.2),
            #nn.Linear(d_model, num_classes, **factory_kwargs)
        )

    def forward(self, x):
        x = self.encoder.tokenize(x)
        if self.pe is not None:
            x = x + self.pe
        hidden_states = self.encoder.encode_tokens(x) # [B, L, d_model]
        pooled_features = hidden_states.mean(dim=1)   # [B, d_model]
        logits = self.classifier(pooled_features)
        return logits



        