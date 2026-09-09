from .data_pipeline import (load_pamap2, filter_loco_pam, interpolation_pam, remove_zero_label_rows, incomplete_labeled_rows,
                            divide_features_labels, acc_data_scaling_pam, mag_data_norm_pam, resample_pam, mag_data_rotation_pam,
                            sliding_window_wrapper_group)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from .preprocessing import fit_labelencoder, Dataset_HAR
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# PAMAP2 locomotion activity ids: 1 = lying, 2 = sitting, 3 = standing, 4 = walking.
# LabelEncoder sorts them, so encoded 0..3 = LIE, SIT, STAND, WALK.
PAM_CLASS_NAMES = ["LIE", "SIT", "STAND", "WALK"]
PAM_SUBJECTS = ["101", "102", "103", "104", "105", "106", "107", "108"]   # 109 has ~no Protocol data -> excluded
PAM_BASE_PATH = '../DATASETS/PAMAP2_Dataset/Protocol/'
PAM_WINDOW, PAM_STRIDE = 90, 30     # 3.0 s / 1.0 s at 30 Hz -> 18 tokens with conv_stride = 5 (same as OPP)
PAM_VAL_FRACTION = 0.20             # last contiguous 20% of each TRAINING subject's stream -> validation

def print_post_standardize_stats(df: pd.DataFrame, name: str, acc_gyro_cols: list[int]):
    x = df.iloc[:, acc_gyro_cols].to_numpy(dtype=np.float64)
    ch_mean = x.mean(axis=0)
    ch_std = x.std(axis=0)

    print(f"\n{'-'*30} {name} ACC/GYRO after StandardScaler {'-'*30}")
    print(f"global mean: {x.mean():.6f}")
    print(f"global std : {x.std():.6f}")
    print(f"max |channel mean| : {np.abs(ch_mean).max():.6f}")
    print(f"median |channel mean| : {np.median(np.abs(ch_mean)):.6f}")
    print(f"channel std range : [{ch_std.min():.6f}, {ch_std.max():.6f}]")

def print_mag_rotation_stats(Xw: np.ndarray, name: str):
    mag_cols = [axis + offset for offset in range(6, Xw.shape[2], 9) for axis in range(3)]   # any number of 9-channel devices (PAMAP2 27, RealWorld 18/27)
    M = Xw[:, :, mag_cols]                     # [W, L, 9]
    per_window_mean = M.mean(axis=1)           # [W, 9]
    flat = M.reshape(-1, M.shape[-1])          # [W*L, 9]

    print(f"\n{'-'*30} {name} MAG after window demeaning {'-'*30}")
    print(f"max |window mean|  : {np.abs(per_window_mean).max():.6e}")
    print(f"mean |window mean| : {np.abs(per_window_mean).mean():.6e}")
    print(f"channel global mean range : [{flat.mean(axis=0).min():.6e}, {flat.mean(axis=0).max():.6e}]")
    print(f"channel std range         : [{flat.std(axis=0).min():.6f}, {flat.std(axis=0).max():.6f}]")

def print_window_label_distribution(yw: np.ndarray, name: str):
    labels, counts = np.unique(yw, return_counts=True)
    total = len(yw)
    print(f"\n{name} window label distribution")
    for lb, cnt in zip(labels, counts):
        print(f"label {lb}: {cnt} ({cnt/total:.4f})")


def data_split_PAM(fold_id: int = 1) -> tuple[list, list, list]:
    '''
    Leave-One-Subject-Out split for PAMAP2 (Protocol files only).
    fold_id k (1..8) holds out subject 10k as TEST; the remaining 7 subjects provide training.
    Validation is carved from the TRAINING subjects inside load_PAM_loco_data (last 20% of each subject's stream),
    so validation_files is returned empty here (kept for signature compatibility with data_split_OPP).
    '''
    if fold_id not in range(1, len(PAM_SUBJECTS) + 1):
        raise ValueError(f"PAM fold must be in 1..{len(PAM_SUBJECTS)}. Got {fold_id}")
    test_subject = PAM_SUBJECTS[fold_id - 1]
    train_subjects = [s for s in PAM_SUBJECTS if s != test_subject]

    training_files = [PAM_BASE_PATH + 'subject' + s + '.dat' for s in train_subjects]
    validation_files = []
    test_files = [PAM_BASE_PATH + 'subject' + test_subject + '.dat']
    return training_files, validation_files, test_files


def _split_train_val_per_subject(df: pd.DataFrame, val_fraction: float, period: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    df layout: 0..26 | 'ts' | 27 (Label) | 'group_id' (subject). Rows of each subject are in time order.
    PAMAP2 activities are performed as contiguous blocks in a FIXED order (lie -> sit -> stand -> ... -> walk), so taking the
    tail of a subject's whole stream would give a single-class validation set. Instead, for every contiguous activity run
    (same label, no timestamp gap) the last val_fraction of its rows -> validation, the rest -> training.
    Validation stays contiguous in time (no window crosses the train/val cut: the cut becomes a timestamp gap) and contains
    every class of every training subject.
    '''
    train_parts, val_parts = [], []
    for subject, gdf in df.groupby("group_id", sort=False):
        gdf = gdf.sort_values("ts", kind="stable").reset_index(drop=True)
        labels = gdf[27].to_numpy()
        ts = gdf["ts"].to_numpy(dtype=np.float64)
        new_run = np.zeros(len(gdf), dtype=bool)
        new_run[1:] = (labels[1:] != labels[:-1]) | (np.diff(ts) > 1.5 * period) | (np.diff(ts) <= 0)
        run_id = np.cumsum(new_run)
        for _, rdf in gdf.groupby(run_id, sort=False):
            n_val = int(round(len(rdf) * val_fraction))
            train_parts.append(rdf.iloc[:len(rdf) - n_val])
            val_parts.append(rdf.iloc[len(rdf) - n_val:])
    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True)
    return train_df, val_df


def load_PAM_loco_data(training_files, validation_files, test_files, verbose = False, drill = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load PAMAP2 (Protocol) and process it for the model, mirroring load_OPP_loco_data (strict LOSO):
      filter locomotion (1-4) + 27 sensor columns -> interpolate short NaN gaps -> physics scaling (acc -> g, mag unit vector)
      -> per-training-subject contiguous train/val split -> resample 100 Hz -> 30 Hz (per contiguous segment)
      -> StandardScaler (acc/gyro) fitted on TRAIN only -> sliding windows 90/30 per segment -> per-window mag demeaning.
    TEST = the held-out subject only (never enters the train/val pool).
    validation_files is ignored (must be empty): validation comes from the training subjects.
    '''
    assert len(validation_files) == 0, "PAM: validation is carved from the training subjects, validation_files must be empty"

    training_data = load_pamap2(training_files, add_group_id = True)
    test_data = load_pamap2(test_files, add_group_id = True)
    if verbose:
        print(f"{'-'*90}")
        print(f"Raw training set shape: {training_data.shape}\nRaw test set shape: {test_data.shape}")

    # ----- Selection of the columns / locomotion labels (timestamp kept until resampling) -----
    training_data_selected = filter_loco_pam(training_data, keep_timestamp = True)
    test_data_selected = filter_loco_pam(test_data, keep_timestamp = True)

    # ----- Clean label 0 (no-op after the 1-4 filter, kept for symmetry with OPP) -----
    training_data_nan = remove_zero_label_rows(training_data_selected)
    test_data_nan = remove_zero_label_rows(test_data_selected)
    if verbose:
        print(f"{'-'*90}")
        print(f"Sensor subset training shape: {training_data_nan.shape}\nSensor subset test shape: {test_data_nan.shape}")

    training_nan = incomplete_labeled_rows(training_data_nan)
    test_nan = incomplete_labeled_rows(test_data_nan)
    if verbose:
        print(f"{'-'*70}")
        print(f"Rows containing NaNs - training: {training_nan}\nRows containing NaNs - test: {test_nan}")

    training_data_nan = training_data_nan.reset_index(drop=True)
    test_data_nan = test_data_nan.reset_index(drop=True)

    # ----- Interpolation of short NaN gaps (<= 30 samples = 0.3 s); longer gaps removed -----
    training_data_cleaned = interpolation_pam(training_data_nan, max_gap=30)
    test_data_cleaned = interpolation_pam(test_data_nan, max_gap=30)
    if verbose:
        print(f"{'-'*90}")
        print(f"After interpolation. Training shape: {training_data_cleaned.shape}\nAfter interpolation. Test shape: {test_data_cleaned.shape}")
        print("\nDescriptive statistics for ACC channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 0:3].describe())
        print("\nDescriptive statistics for GYRO channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 3:6].describe())
        print("\nDescriptive statistics for MAG channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 6:9].describe())

# --------------------------------------------------
# 1) physics-based scaling (acc m/s^2 -> g; gyro already rad/s; mag -> unit vector)
# --------------------------------------------------
    training_data_scaled = acc_data_scaling_pam(training_data_cleaned)
    test_data_scaled = acc_data_scaling_pam(test_data_cleaned)

    training_data_scaled = mag_data_norm_pam(training_data_scaled)
    test_data_scaled = mag_data_norm_pam(test_data_scaled)

# --------------------------------------------------
# 2) TRAIN / VAL split (strict LOSO): last 20% of each TRAINING subject's stream -> validation
# --------------------------------------------------
    new_training_data_scaled, new_validation_data_scaled = _split_train_val_per_subject(training_data_scaled, PAM_VAL_FRACTION)

    print(f"{'-'*20}TRAIN/VAL split per training subject (test = held-out subject){'-'*20}")
    print("Training SUBJECTS (rows)")
    print(new_training_data_scaled["group_id"].value_counts().sort_index())
    print("Validation SUBJECTS (rows)")
    print(new_validation_data_scaled["group_id"].value_counts().sort_index())
    print("Test SUBJECT (rows)")
    print(test_data_scaled["group_id"].value_counts().sort_index(), "\n")
    train_subj = set(new_training_data_scaled["group_id"]); val_subj = set(new_validation_data_scaled["group_id"]); test_subj = set(test_data_scaled["group_id"])
    assert train_subj.isdisjoint(test_subj) and val_subj.isdisjoint(test_subj), "LEAK: test subject appears in train/val"
    assert len(test_subj) == 1, f"PAM LOSO expects exactly one test subject, got {test_subj}"

# --------------------------------------------------
# 3) Resample 100 Hz -> 30 Hz per contiguous segment (drops 'ts', group_id becomes '<subject>-s<k>')
# --------------------------------------------------
    new_training_data_scaled = resample_pam(new_training_data_scaled, verbose = verbose)
    new_validation_data_scaled = resample_pam(new_validation_data_scaled, verbose = verbose)
    test_data_scaled = resample_pam(test_data_scaled, verbose = verbose)

# --------------------------------------------------
# 4) Standardize ACC/GYRO: statistics fitted on the TRAINING split only (after resampling = what the model sees)
# --------------------------------------------------
    acc_gyro_cols = [axis + offset for offset in range(0, 27, 9) for axis in range(6)]
    scaler = StandardScaler()
    new_training_data_scaled.iloc[:, acc_gyro_cols] = scaler.fit_transform(new_training_data_scaled.iloc[:, acc_gyro_cols].values)
    new_validation_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(new_validation_data_scaled.iloc[:, acc_gyro_cols].values)
    test_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(test_data_scaled.iloc[:, acc_gyro_cols].values)

    print("\nTraining Split Label Proportion")
    print(new_training_data_scaled.iloc[:, -2].value_counts(normalize=True).sort_index())
    print("Validation Split Label Proportion")
    print(new_validation_data_scaled.iloc[:, -2].value_counts(normalize=True).sort_index())
    print("Test Split Label Proportion")
    print(test_data_scaled.iloc[:, -2].value_counts(normalize=True).sort_index())

    if verbose:
        print_post_standardize_stats(new_training_data_scaled, "TRAIN", acc_gyro_cols)
        print_post_standardize_stats(new_validation_data_scaled, "VAL", acc_gyro_cols)
        print_post_standardize_stats(test_data_scaled, "TEST", acc_gyro_cols)

    # ----- Features & Labels (group_id kept inside features for per-segment windowing) -----
    X_features, y_labels = divide_features_labels(new_training_data_scaled)
    X_val_features, y_val_labels = divide_features_labels(new_validation_data_scaled)
    X_test_features, y_test_labels = divide_features_labels(test_data_scaled)
    if verbose:
        print(f"{'-'*90}")
        print(f"Training features: {X_features.shape} | Training labels: {y_labels.shape}")
        print(f"Validation features: {X_val_features.shape} | Validation labels: {y_val_labels.shape}")
        print(f"Test features: {X_test_features.shape}, Test labels: {y_test_labels.shape}")

    # ----- Sliding windows per contiguous segment (no window crosses a segment/subject boundary) -----
    X_windows, y_windows = sliding_window_wrapper_group(X_features, y_labels, window_size=PAM_WINDOW, stride=PAM_STRIDE)
    X_validation_windows, y_validation_windows = sliding_window_wrapper_group(X_val_features, y_val_labels, window_size=PAM_WINDOW, stride=PAM_STRIDE)
    X_test_windows, y_test_windows = sliding_window_wrapper_group(X_test_features, y_test_labels, window_size=PAM_WINDOW, stride=PAM_STRIDE)

    # ----- Only mag: per window demeaning -----
    X_windows = mag_data_rotation_pam(X_windows)
    X_validation_windows = mag_data_rotation_pam(X_validation_windows)
    X_test_windows = mag_data_rotation_pam(X_test_windows)

    assert X_windows.shape[1:] == (PAM_WINDOW, 27), f"unexpected window shape {X_windows.shape}"
    assert set(np.unique(y_validation_windows)) == set(np.unique(y_windows)) == {1, 2, 3, 4}, \
        f"train/val must contain all 4 locomotion classes: train {np.unique(y_windows)}, val {np.unique(y_validation_windows)}"

    if verbose:
        print_mag_rotation_stats(X_windows, "TRAIN")
        print_mag_rotation_stats(X_validation_windows, "VAL")
        print_mag_rotation_stats(X_test_windows, "TEST")

        print_window_label_distribution(y_windows, "TRAIN")
        print_window_label_distribution(y_validation_windows, "VAL")
        print_window_label_distribution(y_test_windows, "TEST")

        print(f"{'-'*90}")
        print(f"Training (windows): {X_windows.shape}. Training Labels: {y_windows.shape}")
        print(f"Validation (windows): {X_validation_windows.shape}. Validation Labels: {y_validation_windows.shape}")
        print(f"Test (windows): {X_test_windows.shape}. Test Labels: {y_test_windows.shape}")

    if drill:
        train_subjects = sorted({Path(f).stem[-3:] for f in training_files})
        optional_subjects = [s for s in ["101","105","106","108"] if s in train_subjects] + ["109"]
        drill_files = ['../DATASETS/PAMAP2_Dataset/Optional/subject' + subject + '.dat' for subject in optional_subjects]
        print(f"OPTIONAL pretraining files: {drill_files}")
        drill_data = load_pamap2(drill_files, add_group_id = True)
        drill_sel = filter_loco_pam(drill_data, keep_timestamp = True, keep_labels = None)
        drill_sel = remove_zero_label_rows(drill_sel)
        drill_nan = incomplete_labeled_rows(drill_sel)
        print(f"Rows containing NaNs - OPTIONAL: {drill_nan}")
        drill_sel = drill_sel.reset_index(drop=True)
        drill_sel = interpolation_pam(drill_sel, max_gap=30)
        drill_sel = acc_data_scaling_pam(drill_sel)
        drill_sel = mag_data_norm_pam(drill_sel)
        drill_sel = resample_pam(drill_sel, verbose = verbose)
        drill_sel.iloc[:, acc_gyro_cols] = scaler.transform(drill_sel.iloc[:, acc_gyro_cols].values)
        drill_X, drill_y = divide_features_labels(drill_sel)
        drill_X_windows, drill_y_windows = sliding_window_wrapper_group(drill_X, drill_y, window_size=PAM_WINDOW, stride=PAM_STRIDE)
        drill_X_windows = mag_data_rotation_pam(drill_X_windows)
        print(f"Drill windows: {drill_X_windows.shape}. Drill Labels: {drill_y_windows.shape}")
        X_windows = np.concatenate([X_windows, drill_X_windows], axis=0)
        y_windows = np.concatenate([y_windows, drill_y_windows], axis=0)

    return X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows

def make_loaders_PAM(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator, verbose = False) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    '''
    Creates DataLoaders for training, validation and test sets (same settings as make_loaders_OPP).
    Called once per seed — generator ensures reproducible shuffling.
    '''
    label_encoder = fit_labelencoder(X_windows, y_windows)
    training_dataset = Dataset_HAR(X_windows, y_windows, label_encoder = label_encoder)
    validation_dataset = Dataset_HAR(X_validation_windows, y_validation_windows, label_encoder = label_encoder)
    test_dataset = Dataset_HAR(X_test_windows, y_test_windows, label_encoder = label_encoder)

    train_loader = DataLoader(training_dataset, batch_size = 128, shuffle = True, generator = generator, num_workers = 2, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(validation_dataset, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)

    if verbose:
        # encoded index -> raw id via the fitted encoder; Optional activities (pretraining only) get OPT_<id> names
        name_of_raw = {1: "LIE", 2: "SIT", 3: "STAND", 4: "WALK"}
        label_to_name = {i: name_of_raw.get(int(raw), f"OPT_{int(raw)}") for i, raw in enumerate(label_encoder.classes_)}
        class_counts = {name: 0 for name in label_to_name.values()}
        for _, y_batch in train_loader:
            for label in y_batch:
                class_counts[label_to_name[label.item()]] += 1
        print(f"{'-'*90}")
        print("Training set class distribution:")
        print(class_counts)
        print(f"{'-'*90}")
    return train_loader, val_loader, test_loader, label_encoder
