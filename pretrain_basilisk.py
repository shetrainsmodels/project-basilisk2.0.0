from pyexpat import model
import os, sys
from MambaSSL_JEPA_Model import MambaJEPA, HARMambaConfig
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from data.PAMAP2_data import load_PAM_loco_data, data_split_PAM, make_loaders_PAM
from data.REALWORLD_data import load_REALWORLD_loco_data, data_split_REALWORLD, make_loaders_RW
from data.REALDISP_data import load_REALDISP_data, data_split_Realdisp, make_loaders_REALDISP
from test_mamba import test_model
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from mamba_ssm.modules.mamba2 import Mamba2
import mamba_ssm.modules.mamba2 as mm
from data.preprocessing import fit_labelencoder, Dataset_HAR
from datetime import datetime
import torch.nn.functional as F
import mamba_ssm
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import set_seed
import argparse
from torch.profiler import profile, ProfilerActivity
import time
import math
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import random

print(f"Mamba version: {mamba_ssm.__version__}")
print(f"Path: {mm.__file__}")

parser = argparse.ArgumentParser(description = "J_mamba")
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--fold", type = int, required = True)
parser.add_argument("--lam", type = float, required = True)
args = parser.parse_args()
RUN = f"{args.dataset}_L{HARMambaConfig.n_layer}_lam{args.lam:g}"     # dataset + depth prefixed: OPP_L24_lam0.1 / PAM_L16_lam0 / RD_L24_lam0 (depth read from HARMambaConfig default, never overwrite each other)
os.makedirs(f"JEPA_models_pt/{RUN}", exist_ok = True)
os.makedirs("logs", exist_ok = True)
if args.dataset == "OPP":
    if args.fold in [1, 2, 3, 4]:
        training_files, validation_files, test_files = data_split_OPP(args.fold)
    else:
        raise ValueError(f"Fold must be 1, 2, 3 or 4. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, verbose = True, drill = True)
    make_loaders = make_loaders_OPP
    NUM_SENSOR_FEATURES = 45
    d_model = 384
    d_intermediate = 512
    ssm_cfg = {"expand":4, "layer":"Mamba2", "headdim":8, "d_ssm":768, "dt_min":0.001, "dt_max":0.1}
    num_heads = 6
elif args.dataset == "PAM":
    if args.fold in range(1, 9):
        training_files, validation_files, test_files = data_split_PAM(args.fold)
    else:
        raise ValueError(f"PAM fold must be 1..8. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_PAM_loco_data(training_files, validation_files, test_files, verbose = True, drill = False)
    make_loaders = make_loaders_PAM
    NUM_SENSOR_FEATURES = 27
    #d_model = 160
    #d_intermediate = 320
    #ssm_cfg = {"expand":2, "layer":"Mamba2", "headdim":8, "d_ssm":192, "dt_min":0.001, "dt_max":0.1}
    num_heads = 6

elif args.dataset == "RW":
    NUM_SENSORS = 1                      # HARD-CODED HERE (pretraining decides): 2 = chest + forearm (18 ch), 3 = chest + forearm + upper arm (27 ch). Downstream reads it from the checkpoint.
    if args.fold in range(1, 16):
        training_files, validation_files, test_files = data_split_REALWORLD(args.fold, num_sensors = NUM_SENSORS)
    else:
        raise ValueError(f"RW fold must be 1..15. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_REALWORLD_loco_data(training_files, validation_files, test_files, verbose = True, num_sensors = NUM_SENSORS)
    make_loaders = make_loaders_RW
    NUM_SENSOR_FEATURES = 9 * NUM_SENSORS
    num_heads = 6

elif args.dataset == "RD":
    NUM_SENSORS = 6                      # HARD-CODED HERE (pretraining decides): 6 = OPP + RC (54 ch) / 5 = OPP (45 ch) / 3 = PAM (27 ch); 9 = all is NOT supported (filter_realdisp raises). Downstream reads it from the checkpoint.
    if args.fold in range(1, 18):
        training_files, validation_files, test_files = data_split_Realdisp(args.fold, scenarios = ["self", "ideal"])
    else:
        raise ValueError(f"RD fold must be 1..17. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_REALDISP_data(training_files, validation_files, test_files, NUM_SENSORS, verbose = True)
    make_loaders = make_loaders_REALDISP
    NUM_SENSOR_FEATURES = 9 * NUM_SENSORS
    num_heads = 6

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model_PRETRAIN_JEPA(model, val_loader, device, criterion, lam = 0.0, recon = False):
    '''
    Validation: avg latent prediction loss, with fixed masks each epoch.
    Also returns target-embedding std as a collapse monitor.
    '''    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    emb_stds = []
    mean_coss = []
    
    with torch.random.fork_rng():
        torch.manual_seed(42)
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device, non_blocking = True)
            target_embeddings, targets, _, mask_targets, _, predictions, context_input_recon = model(x_batch)
            loss = criterion(predictions, targets)
            if recon: # same objective as training: lam *jepa + recon 
                reconstruction = model.decoder(context_input_recon) # [B, 90, NUM_SENSOR_FEATURES] (45 OPP / 27 PAM / 18 RW / 54 RD)
                mask = mask_targets.repeat_interleave(model.config.conv_stride, dim = 1) # [B, 18] -> [B, 90]
                loss = lam * loss + criterion(reconstruction[mask], x_batch[mask])

            bsize = x_batch.size(0)
            total_loss += loss.item() * bsize
            total_samples += bsize

            flat_emb = target_embeddings.reshape(-1, target_embeddings.size(-1)) # [B*18, 384]
            #Monitor 1: per ft std across all tkns (collapse -> 0)
            emb_stds.append(flat_emb.std(dim = 0).mean().item())
            #Monitor 2: avg cosine similarity bt tokens (collapse -> 1)
            normed = torch.nn.functional.normalize(flat_emb, dim = -1)
            mean_coss.append(normed.mean(dim = 0).norm().pow(2).item())
 
    return total_loss / total_samples, sum(emb_stds) / len(emb_stds), sum(mean_coss) / len(mean_coss)

# -------------------------------------------- GRAD ANALYSIS ---------------------------------------------------
def get_grad_vector(jepa_loss, raw_loss, layer_param, lam):
    layer_param = [p for p in layer_param if p.requires_grad]
    grad_jepa = torch.autograd.grad(jepa_loss, layer_param, retain_graph = True, allow_unused = True)
    grad_raw = torch.autograd.grad(raw_loss, layer_param, retain_graph = True, allow_unused = True)
    grad_jepa = torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(layer_param, grad_jepa)])
    grad_raw = torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(layer_param, grad_raw)])

    norm_jepa = grad_jepa.norm().item()
    norm_raw = grad_raw.norm().item()
    effective_jepa_norm = lam * norm_jepa

    ratio_l = norm_raw/(effective_jepa_norm + 1e-8)  # STRENGHT
    cos = torch.nn.functional.cosine_similarity(grad_jepa, grad_raw, dim = 0, eps = 1e-8) # SIMILARITY

    return ratio_l, cos.item()
#  ---------------------------------------------------- TRAINING ---------------------------------------------------
for seed in [42, 58, 7, 128, 92]:
    g = set_seed(seed)
    train_loader, val_loader, test_loader, label_encoder = make_loaders(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)

    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HARMambaConfig(num_sensor_features = NUM_SENSOR_FEATURES)
    recon = True
    lam = args.lam
    model = MambaJEPA(config, mask_ratio = 0.33, t_l = 3, num_heads = num_heads, use_pe = False, drop = True, recon = recon)
    model.to(device, non_blocking = True)
    # ----------------------
    if args.dataset in ("PAM", "OPP"):      # small datasets: val loss flat by ~1300 steps (ep ~30 PAM / ~25 OPP) -> L16-PAM / L24-OPP protocol (2026-09-13, Katy)
        num_epochs, patience = 35, 8
    else:                                   # RD, RW: val loss still improving at ep 40, converges ~50-55 (2026-09-11)
        num_epochs, patience = 60, 12
    warmup_Epochs = 5
    lr = 0.0006
    criterion = nn.SmoothL1Loss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]   # excludes frozen target encoder
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: #skip frozen targer encoder
            continue
        if p.ndim <= 1 or name.endswith("pe") or "predictor_pe" in name or "mask_token" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    n_layers = len(model.context_encoder.backbone.layers)        # actual depth of the encoder that is running (8 / 12 / 16 / 24)
    b1, b2 = n_layers // 3, 2 * n_layers // 3                     # thirds: early = [0, b1), mid = [b1, b2), late = [b2, n_layers)
    GROUPS = {   # gradient diagnostic only (first batch of each epoch): early / mid / late = first / middle / last third of the encoder
    "stem": list(model.context_encoder.convolutional_input.parameters()) + list(model.context_encoder.LN_layer.parameters()),
    "early": [p for p in model.context_encoder.backbone.layers[0:b1] for p in p.parameters()],
    "mid":   [p for p in model.context_encoder.backbone.layers[b1:b2] for p in p.parameters()],
    "late":  [p for p in model.context_encoder.backbone.layers[b2:n_layers] for p in p.parameters()] + list(model.context_encoder.backbone.norm_f.parameters()),
    "all":   list(model.context_encoder.parameters())     
    }
    g_ratio_epoch = {}
    g_cos_epoch = {}
    #### Scheduler ####
    def lr_lambda(epoch):
        if epoch < warmup_Epochs:
            return (epoch + 1)/warmup_Epochs
        progress = (epoch - warmup_Epochs)/max(1, num_epochs - warmup_Epochs)
        min_ratio = 1e-6 / lr
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}], lr = lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # EMA momentum schedule: 0.996 -> 1.0 linearly over all steps (I-JEPA style)
    total_steps = num_epochs * len(train_loader)
    m_start, m_end = 0.996, 1.0
    global_step = 0

    #  ----------- TRAINING -----------
    model_name = f"JEPA_models_pt/{RUN}/JEPA_model_{args.dataset}_fold{args.fold}_seed{seed}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    bad_epochs = 0
    with open(f"logs/JEPA_training_{RUN}_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | SEED: {seed}\n")
        log_file.write(f"GROUPS: early layers 0-{b1-1} | mid {b1}-{b2-1} | late {b2}-{n_layers-1} (n_layer = {n_layers})\n")
        log_file.flush()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            total_recon = 0.0
            total_samples = 0
            grad_stat = {name: {"ratio": [], "cos": []} for name in GROUPS}

            loop = tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (x_batch, _) in enumerate(loop):
                x_batch = x_batch.to(device, non_blocking = True)

                optimizer.zero_grad()
                _, targets, context_embeddings, mask_targets, target_blocks, predictions, context_input_recon = model(x_batch)
                loss = criterion(predictions, targets)

                # RECONSTRUCTION
                if recon:
                    reconstruction = model.decoder(context_input_recon) # [B, 90, NUM_SENSOR_FEATURES] (45 OPP / 27 PAM / 18 RW / 54 RD)
                    mask = mask_targets.repeat_interleave(model.config.conv_stride, dim = 1)
                    recon_loss = criterion(reconstruction[mask], x_batch[mask])
                    jepa = loss
                    loss = lam * jepa + recon_loss
                    if batch_idx == 0: # gradient diagnostic: first batch of each epoch, all groups (cost control)
                        for name, layer_level in GROUPS.items():
                            ratio_l, cos = get_grad_vector(jepa, recon_loss, layer_level, lam)
                            grad_stat[name]["ratio"].append(ratio_l);  grad_stat[name]["cos"].append(cos)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm = 1.0, error_if_nonfinite = True)
                optimizer.step()

                momentum = m_start + (m_end - m_start) * (global_step / total_steps)
                model.update_target_encoder(momentum = momentum)
                global_step += 1

                bsize = x_batch.size(0)
                total_loss += loss.item() * bsize
                if recon:
                    total_recon += recon_loss.item() * bsize
                total_samples += bsize
                loop.set_postfix(loss = f"{total_loss/total_samples:.4f}", recon = f"{total_recon/total_samples:.4f}")

            train_loss = total_loss / total_samples
            train_recon = total_recon / total_samples
            for name in GROUPS.keys():
                g_ratio_epoch[name] = sum(grad_stat[name]["ratio"]) / max(1, len(grad_stat[name]["ratio"]))
                g_cos_epoch[name] = sum(grad_stat[name]["cos"]) / max(1, len(grad_stat[name]["cos"]))
            g_str = " | ".join(f"{n}: r={g_ratio_epoch[n]:.3f} c={g_cos_epoch[n]:.3f}" for n in GROUPS)
            val_loss, emb_std, mean_cos = validate_model_PRETRAIN_JEPA(model, val_loader, device, criterion, lam = lam, recon = recon)

            epoch_time = time.time() - epoch_start
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "val_loss": float(val_loss),
                "emb_std": float(emb_std),
                "mean_cos": float(mean_cos),
                "grad_ratio": dict(g_ratio_epoch),
                "grad_cos": dict(g_cos_epoch)
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | tr_recon: {train_recon:.4f} | val_loss: {val_loss:.4f} | emb_std: {emb_std:.4f} | mean_cos: {mean_cos:.4f} | epoch_time: {epoch_time:.2f}s | g_str: {g_str}") 
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | tr_recon: {train_recon:.4f} | val_loss: {val_loss:.4f} | emb_std: {emb_std:.4f} | mean_cos: {mean_cos:.4f} | epoch_time: {epoch_time:.2f}s | g_str: {g_str}\n")
            log_file.flush()

            if val_loss < best_val_loss - 1e-12:
                best_val_loss = float(val_loss)
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
            scheduler.step()

        if best_state is not None:
            torch.save(best_state, model_name)
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")
