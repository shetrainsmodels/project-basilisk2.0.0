from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from MambaSSL_JEPA_Model import MambaJEPA, MambaEncoderModel, PredictorTransformer
import numpy as np
import torch
import json
import os

# Class names in LabelEncoder order (sorted raw label ids) for each dataset:
#   OPP locomotion ids 1, 2, 4, 5 -> STAND, WALK, SIT, LIE
#   PAM activity ids   1, 2, 3, 4 -> LIE, SIT, STAND, WALK
#   RW  activity ids   1, 2, 3, 4 -> LIE, SIT, STAND, WALK
CLASS_NAMES = {
    "OPP": ["STAND", "WALK", "SIT", "LIE"],
    "PAM": ["LIE", "SIT", "STAND", "WALK"],
    "RW":  ["LIE", "SIT", "STAND", "WALK"],
    "RD":  [f"L{k}" for k in range(1, 34)],   # REALDISP L1..L33 in LabelEncoder (numeric) order
}

@torch.no_grad()
def test_model(model, test_loader, device, class_names = None):
    '''
    Runs inference on the test set and computes evaluation metrics.
    class_names: per-class names in LabelEncoder order (default: OPP names, unchanged behaviour).
    '''
    if class_names is None:
        class_names = CLASS_NAMES["OPP"]
    model.eval()
    all_preds = []
    all_labels = []
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)
        
        logits_ = model(x_batch)
        predictions = logits_.argmax(dim = -1)

        all_preds.extend(predictions.numpy(force = True)) # move to CPU
        all_labels.extend(y_batch.numpy(force = True)) # move to CPU

    acc = accuracy_score(all_labels, all_preds)    
    labels = list(range(len(class_names)))   # fixed label set: a test subject may lack some classes (e.g. REALDISP subject 6)
    report = classification_report(all_labels, all_preds, labels = labels, target_names = class_names, output_dict = True, zero_division = 0)
    present = sorted(set(int(y) for y in all_labels))   # classes the TEST subject actually performed (REALDISP subject 6 has 21/33; OPP/PAM/RW always all)
    f1 = f1_score(all_labels, all_preds, labels = present, average = "macro", zero_division = 0) # macro over PRESENT classes only: an activity the subject never did has no test windows and would count as F1 = 0
    conf_matrix = confusion_matrix(all_labels, all_preds, labels = labels)
    return acc, report, f1, conf_matrix

def save_json(dataset, fold, seed, seed_result, out_dir = "results_json") -> None:
    '''
    Saves the results of a single seed run to a JSON file for the given fold.
    Results are stored under "runs" keyed by seed for easy lookup.
    '''
    os.makedirs(out_dir, exist_ok = True)
    json_path = os.path.join(out_dir, f"{dataset}_fold{fold}_results.json")
    # check if json file exists.
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_info = json.load(f)
    else:
        json_info = {
            "fold": int(fold),
            "runs": {}
        }
    # add seed results into the dict
    json_info["runs"][f"seed_{seed}"] = seed_result
    # atomic save: write a temp file in the same folder, then rename over the old one. If the job is killed mid-write
    # (timeout, preemption) the previous file stays intact instead of being truncated and losing every saved probe.
    tmp_path = f"{json_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(json_info, f, indent = 4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, json_path)
    
            
        
    