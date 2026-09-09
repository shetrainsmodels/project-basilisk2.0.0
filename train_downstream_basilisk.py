import os, sys
# make the shared data/ package (one level up, in MASKING/) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MambaSSL_JEPA_Model import HARMambaConfig, MambaDownstreamClassifier
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from data.PAMAP2_data import load_PAM_loco_data, data_split_PAM, make_loaders_PAM
from data.REALWORLD_data import load_REALWORLD_loco_data, data_split_REALWORLD, make_loaders_RW
from data.REALDISP_data import load_REALDISP_data, data_split_Realdisp, make_loaders_REALDISP
from test_mamba import test_model, save_json, CLASS_NAMES
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
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

print(f"Mamba version: {mamba_ssm.__version__}")
print(f"Path: {mm.__file__}")

parser = argparse.ArgumentParser(description = "supervised_mamba")
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--fold", type = int, required = True)
parser.add_argument("--lam", type = float, required = True)
parser.add_argument("--w_per_class", type = str, default = "all")
args = parser.parse_args()
if args.w_per_class != "all" and not (args.w_per_class.isdigit() and int(args.w_per_class) > 0):
    parser.error(f"--w_per_class must be all or a positive integer, got {args.w_per_class!r}")
ENCODER_SEEDS = [42, 58, 7, 128, 92]      # pretraining seeds = which checkpoint is loaded
PROBE_SEEDS   = [11, 22, 33]      # linear-head seeds: init + batch order. Every encoder x every probe seed
DATA_SEED     = 2026                      # label subsample: identical for every encoder / objective / depth
RUN = f"{args.dataset}_lam{args.lam:g}"                                     # pretrained-encoder folder (dataset-prefixed, shared by all label fractions)
OUT = RUN if args.w_per_class == "all" else f"{RUN}_n{int(args.w_per_class)}"   # output tag: results/logs/probe .pt
os.makedirs(f"JEPA_models_pt/{OUT}", exist_ok = True)
os.makedirs("logs", exist_ok = True)
if args.dataset == "OPP":
    if args.fold in [1, 2, 3, 4]:
        training_files, validation_files, test_files = data_split_OPP(args.fold)
    else:
        raise ValueError(f"Fold must be 1, 2, 3 or 4. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, verbose = True, drill = False)
    make_loaders = make_loaders_OPP
    NUM_SENSOR_FEATURES = 45
    d_model = 384
    d_intermediate = 512
    n_layer = 8
    ssm_cfg = {"expand":4, "layer":"Mamba2", "headdim":8, "d_ssm":768, "dt_min":0.001, "dt_max":0.1}
    num_heads = 6
elif args.dataset == "PAM":
    if args.fold in range(1, 9):
        training_files, validation_files, test_files = data_split_PAM(args.fold)
    else:
        raise ValueError(f"PAM fold must be 1..8. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_PAM_loco_data(training_files, validation_files, test_files, verbose = True)
    make_loaders = make_loaders_PAM
    NUM_SENSOR_FEATURES = 27
    #d_model = 160
    #d_intermediate = 320
    #n_layer = 8
    #ssm_cfg = {"expand":2, "layer":"Mamba2", "headdim":8, "d_ssm":192, "dt_min":0.001, "dt_max":0.1}
    #num_heads = 8    
elif args.dataset == "RW":
    # taken from PRETRAINING: the encoder's input conv has in_channels = 9 * NUM_SENSORS (same for every seed of the fold)
    _ckpt = f"JEPA_models_pt/{RUN}/JEPA_model_{args.dataset}_fold{args.fold}_seed42.pt"
    _w = torch.load(_ckpt, map_location = "cpu", weights_only = True)["context_encoder.convolutional_input.weight"]
    NUM_SENSORS = int(_w.shape[1]) // 9
    assert NUM_SENSORS in (1, 2) and _w.shape[1] == 9 * NUM_SENSORS, f"unexpected encoder in_channels {_w.shape[1]} in {_ckpt}"
    print(f"RW: NUM_SENSORS = {NUM_SENSORS} (read from pretrained encoder {_ckpt})")
    if args.fold in range(1, 16):
        training_files, validation_files, test_files = data_split_REALWORLD(args.fold, num_sensors = NUM_SENSORS)
    else:
        raise ValueError(f"RW fold must be 1..15. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_REALWORLD_loco_data(training_files, validation_files, test_files, verbose = True, num_sensors = NUM_SENSORS)
    make_loaders = make_loaders_RW
    NUM_SENSOR_FEATURES = 9 * NUM_SENSORS

elif args.dataset == "RD":
    # taken from PRETRAINING: the encoder's input conv has in_channels = 9 * NUM_SENSORS (same for every seed of the fold)
    _ckpt = f"JEPA_models_pt/{RUN}/JEPA_model_{args.dataset}_fold{args.fold}_seed42.pt"
    _w = torch.load(_ckpt, map_location = "cpu", weights_only = True)["context_encoder.convolutional_input.weight"]
    NUM_SENSORS = int(_w.shape[1]) // 9
    assert NUM_SENSORS in (6, 5, 3) and _w.shape[1] == 9 * NUM_SENSORS, f"unexpected encoder in_channels {_w.shape[1]} in {_ckpt}"
    print(f"RD: NUM_SENSORS = {NUM_SENSORS} (read from pretrained encoder {_ckpt})")
    if args.fold in range(1, 18):
        training_files, validation_files, test_files = data_split_Realdisp(args.fold, scenarios = ["self", "ideal"])
    else:
        raise ValueError(f"RD fold must be 1..17. Got {args.fold}")
    X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_REALDISP_data(training_files, validation_files, test_files, NUM_SENSORS, verbose = True)
    make_loaders = make_loaders_REALDISP
    NUM_SENSOR_FEATURES = 9 * NUM_SENSORS

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
class_names = CLASS_NAMES[args.dataset]

def load_pretrained_encoder(model, device, fold, seed):
    checkpoint_path = f"JEPA_models_pt/{RUN}/JEPA_model_{args.dataset}_fold{fold}_seed{seed}.pt"

    state = torch.load(checkpoint_path, map_location = device, weights_only=True)
    encoder_state = {k.replace("context_encoder.", "", 1): v for k,v in state.items() if k.startswith("context_encoder.")}
    model.encoder.load_state_dict(encoder_state, strict = True)
    if "pe" in state:                                                            # context encoder was trained with pe (pe_target is the EMA copy used by the target encoder)
        model.pe = state["pe"].to(device)
    elif "pe_target" in state:                                                   # fallback for old checkpoints that only carry the EMA copy
        model.pe = state["pe_target"].to(device)
    print("JEPA weights loaded with PE:", model.pe is not None)
#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model(model, val_loader, device, criterion):
    '''
    Validation: avg loss per window and accuracy per window.
    '''
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)
        logits_ = model(x_batch)                                               # forward Pass
        loss = criterion(logits_, y_batch)                                     # computes mean batch loss

        bsize = y_batch.size(0)
        total_loss += loss.item() * bsize                                      # Total loss contribution of this batch

        predictions = logits_.argmax(dim = -1)                                 # gets the predicted class for each window
        total_correct += (predictions == y_batch).sum().item()                 # counts correct predictions
        total_samples += bsize

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    report = classification_report(all_labels, all_preds, labels = list(range(len(class_names))), target_names = class_names, zero_division = 0)
    return total_loss / total_samples, total_correct / total_samples, report
# --------------------------------
@torch.no_grad()
def embed_and_rank(model, loader, device, tag):
    model.eval()

    Z = []
    # --------------------------------------------------
    # 1. Extract one embedding per IMU window
    # --------------------------------------------------
    for x_batch, _ in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        # Raw IMU -> tokens
        t = model.encoder.tokenize(x_batch)

        # Add positional embedding if used
        if model.pe is not None:
            if model.pe.ndim == 3:
                pe = model.pe[:, :t.shape[1]]
            else:
                pe = model.pe[:t.shape[1]]

            t = t + pe.to(device=t.device, dtype=t.dtype)

        # Mamba encoder
        h = model.encoder.encode_tokens(t)

        # Mean pool over sequence/time dimension
        # [B, T, D] -> [B, D]
        z = h.mean(dim=1)

        Z.append(z.float().cpu())
    # --------------------------------------------------
    # 2. Basic checks
    # --------------------------------------------------
    if not Z:
        raise ValueError("Loader is empty.")

    Z = torch.cat(Z, dim=0)

    if len(Z) < 2:
        raise ValueError("At least two windows are required.")

    # --------------------------------------------------
    # 3. Center embeddings
    # --------------------------------------------------
    Zc = Z - Z.mean(dim=0, keepdim=True)
    # --------------------------------------------------
    # 4. Singular values / variance spectrum
    # --------------------------------------------------
    sv = torch.linalg.svdvals(Zc)

    # Variance associated with each singular direction
    energy = sv.square()
    total_energy = energy.sum()

    if total_energy.item() <= 1e-12:
        # All embeddings are effectively identical
        eff = 0.0
        d95 = 0
        d99 = 0

    else:
        # Fraction of representation variance
        # explained by each singular direction
        p = energy / total_energy

        # Effective rank
        q = sv / (sv.sum() + 1e-12)
        entropy = -(q * q.clamp_min(1e-12).log()).sum()
        eff = torch.exp(entropy).item()

        # Cumulative explained variance
        ev = torch.cumsum(p, dim=0)

        # Number of dimensions required for
        # 95% and 99% of variance
        d95 = int((ev < 0.95).sum().item()) + 1
        d99 = int((ev < 0.99).sum().item()) + 1

    # --------------------------------------------------
    # 5. Mean cosine similarity between different windows
    # --------------------------------------------------
    Zn = F.normalize(Z, dim=1, eps=1e-12)

    n = len(Zn)

    # Equivalent to computing all pairwise cosine
    # similarities, but WITHOUT creating an NxN matrix.
    sum_vec = Zn.sum(dim=0)

    all_dot_products = sum_vec.square().sum()
    self_dot_products = Zn.square().sum()

    mcos = (
        (all_dot_products - self_dot_products)
        / (n * (n - 1))
    ).item()

    # --------------------------------------------------
    # 6. Print diagnostics
    # --------------------------------------------------
    print(
        f"[RANK:{tag}] "
        f"N={Z.shape[0]} "
        f"D={Z.shape[1]} "
        f"eff_rank={eff:.2f} "
        f"d95={d95} "
        f"d99={d99} "
        f"mean_cos={mcos:.4f}"
    )
    return eff
#  ----------------------------------------------------------------------------------------------------------------------
#  ---------------------------------------------------- LOSO TRAINING ---------------------------------------------------
def subsample_per_class(X_windows, y_windows, w_per_class, seed):
    rng = np.random.default_rng(seed)   # own RNG: depends only on seed -> same windows for every encoder
    keep = []
    for class_ in np.unique(y_windows):
        idx = np.where(y_windows == class_)[0]
        idx = rng.permutation(idx) # take the indices of each class and shuffle them
        keep.append(idx[:int(w_per_class)]) # keep only the first w_per_class indices of each class
    keep = np.sort(np.concatenate(keep, axis = 0))
    return X_windows[keep], y_windows[keep]

def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
for seed in ENCODER_SEEDS:
  for probe_seed in PROBE_SEEDS:
    tag = f"{seed}_p{probe_seed}"                                                # output name: encoder seed + probe seed, e.g. seed42_p11
    g = set_seed(probe_seed)                                                     # head init + batch order (generator g)
    if args.w_per_class == "all":
        X_tr, y_tr = X_windows, y_windows
    else:
        X_tr, y_tr = subsample_per_class(X_windows, y_windows, args.w_per_class, DATA_SEED)
        print(f"[low-label] w_per_class = {args.w_per_class} | train windows: {len(y_tr)} | class counts: {dict(Counter(y_tr.tolist()))}")
    train_loader, val_loader, test_loader, label_encoder = make_loaders(X_tr, y_tr, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)

    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(len(label_encoder.classes_))
    assert num_classes == len(class_names), f"{num_classes} classes in training labels but {len(class_names)} class names for {args.dataset}"
    config = HARMambaConfig(num_sensor_features = NUM_SENSOR_FEATURES)
    torch.manual_seed(seed)                                                      # random-init encoder depends on the ENCODER seed only (one distinct random encoder per outer seed)
    model = MambaDownstreamClassifier(config, num_classes)
    torch.manual_seed(probe_seed)                                                # re-seed AFTER building the model: encoder construction consumed RNG (more for deeper encoders)
    model.classifier.apply(reset_parameters)                                     # so the head init depends on probe_seed only, not on encoder depth
    model.to(device, non_blocking = True)
    num_epochs = 120 #60 if args.w_per_class == "all" else 500   
    patience   = 15 #10 if args.w_per_class == "all" else 20    # low-label: val loss is noisier per epoch
    lr = 2e-4 
    
    # LOAD PRETRAINED ENCODER
    run_rank_diag = probe_seed == PROBE_SEEDS[0]                                 # rank diagnostics are encoder-level: run once per encoder seed, not per probe seed
    if run_rank_diag:
        embed_and_rank(model, test_loader, device, f"randinit_f{args.fold}_s{seed}")
    load_pretrained_encoder(model, device, args.fold, seed)
    # FREEZE encoder 
    for param in model.encoder.parameters():
        param.requires_grad = False
    if run_rank_diag:
        embed_and_rank(model, test_loader, device, f"JEPA_f{args.fold}_s{seed}")
        
    #WCE
    #y_train_encoded = label_encoder.transform(np.asarray(y_windows))
    #counts = np.bincount(y_train_encoded, minlength = num_classes)
    #weights = torch.tensor(1.0 / counts, dtype=torch.float32, device = device)
    #criterion = nn.CrossEntropyLoss(weight = weights)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},{"params": model.classifier.parameters(), "lr": 3e-4}], weight_decay = 1e-4)
    optimizer = torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr = lr, weight_decay = 0.0)
 #  ----------- TRAINING -----------
    model_name = f"JEPA_models_pt/{OUT}/model_JEPA_CLA_{args.dataset}_fold{args.fold}_seed{tag}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = None
    best_state = None
    bad_epochs = 0
    prof_out = None
    with open(f"logs/model_JEPA_CLA_{OUT}_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | ENCODER SEED: {seed} | PROBE SEED: {probe_seed} | DATA SEED: {DATA_SEED}\n")
        log_file.flush()
        for epoch in range(num_epochs):
            epoch_start = time.time() # START EPOCH TIME
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
    
            loop = tqdm(train_loader, desc= f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (x_batch, y_batch) in enumerate(loop):
                x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)    
                optimizer.zero_grad()                                             # clear previous gradients
                logits_ = model(x_batch)                                          # forward pass
                loss = criterion(logits_, y_batch)                                # computes mean batch loss
                loss.backward()                                                   # backward pass: compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                optimizer.step()                                                  # update model weights
                bsize = y_batch.size(0)
                total_loss += loss.item() * bsize                                 # Total loss contribution of this batch
                
                predictions = logits_.argmax(dim = -1)                            # gets the predicted class for each window: picks the highest score in the last dimension (one highest score per window among the 4 classes)
                total_correct += (predictions == y_batch).sum().item()            # number of correct predictions
                total_samples += bsize
                loop.set_postfix(loss= f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.4f}")
    
            train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
            val_loss, val_acc, report = validate_model(model, val_loader, device, criterion)

            epoch_time = time.time() - epoch_start # END EPOCH TIME
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "tr_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc)                
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s") 
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s\n")
            log_file.flush()
                
            if val_loss < best_val_loss - 1e-3:
                best_val_loss = float(val_loss)
                best_val_acc = float(val_acc)
                best_epoch = epoch + 1                   
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
                    
        # Load Best Model !            
        if best_state is not None:
            model.load_state_dict(best_state)
            head_state = {k: v for k, v in best_state.items() if k.startswith("classifier.")}   # save ONLY the probe head (LayerNorm + Linear, ~1 MB); the frozen encoder is already in JEPA_model_*.pt
            assert len(head_state) > 0, "no classifier.* keys found in best_state"
            torch.save(head_state, model_name)
            # Run val on best model to generate report
            _, _, report = validate_model(model, val_loader, device, criterion) 
            print(report)
            log_file.write(f"\nBest Model Validation Report: \n{str(report)}")        
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")                           
        
        #  ----------- TEST -----------         
        acc, report, f1, conf_matrix = test_model(model, test_loader, device, class_names = class_names)
        seed_result = {
            "seed": int(seed),
            "probe_seed": int(probe_seed),
            "data_seed": int(DATA_SEED),
            "fold": int(args.fold),
            "history": epoch_history,
            "summary": {
                "lam": float(args.lam),
                "w_per_class": args.w_per_class,
                "n_train_windows": int(len(y_tr)),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "test_accuracy": float(acc),
                "test_report": report,
                "test_f1": float(f1),
                "test_conf_matrix": conf_matrix.tolist()               
            }
        }
        save_json(args.dataset, args.fold, tag, seed_result, out_dir = f"results_json/{OUT}")
        print(f"{'-'*90}")
        print(f"Test Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
        log_file.write(f"\nTest Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
#  ----------------------------------------------------------------------------------------------------------------------
    

