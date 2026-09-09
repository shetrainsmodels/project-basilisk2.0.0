from .data_pipeline import (load_realworld, filter_loco_rw, remove_zero_label_rows, incomplete_labeled_rows,
                              divide_features_labels, acc_data_scaling_rw, mag_data_norm_rw, resample_rw,
                              mag_data_rotation_rw, sliding_window_wrapper_group)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from .preprocessing import fit_labelencoder, Dataset_HAR
from .PAMAP2_data import print_post_standardize_stats, print_mag_rotation_stats, print_window_label_distribution
import pandas as pd, numpy as np, torch
from pathlib import Path

REALWORLD_CLASS_NAMES = ["LIE", "SIT", "STAND", "WALK"]
REALWORLD_SUBJECTS = [f"{i}" for i in range(1, 16)]  # subject1 to subject15
REALWORLD_BASE_PATH = '../DATASETS/REALWORLD_Dataset/'
RW_VAL_FRACTION = 0.2  # 20% of each training subject's stream for validation

def data_split_REALWORLD(fold_id: int = 1, num_sensors: int = 2) -> tuple[list, list, list]:
    '''
    Leave-One-Subject-Out split for REALWORLD (Protocol files only).
    fold_id k (1..15) holds out subject k as TEST; the remaining 14 subjects provide training.
    Validation is carved from the TRAINING subjects inside load_REALWORLD_loco_data (last 20% of each subject's stream),
    so validation_files is returned empty here (kept for signature compatibility with data_split_OPP).
    '''
    if fold_id not in range(1, len(REALWORLD_SUBJECTS) + 1):
        raise ValueError(f"REALWORLD fold must be in 1..{len(REALWORLD_SUBJECTS)}. Got {fold_id}")
    test_subject = REALWORLD_SUBJECTS[fold_id - 1]
    train_subjects = [s for s in REALWORLD_SUBJECTS if s != test_subject]

    if num_sensors == 1:
        RW_PATH = REALWORLD_BASE_PATH + 'forearm/'          # 9 channels: forearm acc/gyr/mag
    elif num_sensors == 2:
        RW_PATH = REALWORLD_BASE_PATH + 'chest_forearm/'    # 18 channels: chest + forearm
    else:
        raise ValueError(f"num_sensors must be 1 (forearm) or 2 (chest+forearm); got {num_sensors}")
    training_files = [RW_PATH + 'subject' + s + '_realworld.dat' for s in train_subjects]
    validation_files = []
    test_files = [RW_PATH + 'subject' + test_subject + '_realworld.dat']
    return training_files, validation_files, test_files

def _split_train_val_per_subject_rw(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    df layout: 0..n-1 sensor features | 'ts' | n (Label) | 'group_id' (subject). Rows of each subject are in time order
    as written in the .dat files (do NOT sort by 'ts': RealWorld timestamps restart at 0 for every recording).
    RealWorld records each activity as one contiguous session, so for every recording (same label, timestamp not
    restarting) the last val_fraction of its rows -> validation, the rest -> training. Removed dropouts inside a
    recording (timestamp jumps) are NOT run boundaries here; resample_pam splits on them afterwards.
    Validation stays contiguous in time and contains every class of every training subject.
    '''
    label_col = [c for c in df.columns if c not in ("ts", "group_id")][-1]   # 27 for 3 devices, 18 for chest_forearm (devices)
    train_parts, val_parts = [], []
    for subject, gdf in df.groupby("group_id", sort=False):
        gdf = gdf.reset_index(drop=True)                                     # keep file order
        labels = gdf[label_col].to_numpy()
        ts = gdf["ts"].to_numpy(dtype=np.float64)
        new_run = np.zeros(len(gdf), dtype=bool)
        new_run[1:] = (labels[1:] != labels[:-1]) | (np.diff(ts) <= 0)      # label change or timestamp restart
        run_id = np.cumsum(new_run)
        for _, rdf in gdf.groupby(run_id, sort=False):
            n_val = int(round(len(rdf) * val_fraction))
            train_parts.append(rdf.iloc[:len(rdf) - n_val])
            val_parts.append(rdf.iloc[len(rdf) - n_val:])
    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True)
    return train_df, val_df

def load_REALWORLD_loco_data(training_files, validation_files, test_files, verbose = False, num_sensors: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    num_sensors: 1 = forearm (9 sensor columns), 2 = chest + forearm (18). Passed from the script (hard-coded in pretraining).
    '''
    n_features = 9 * num_sensors
    assert len(validation_files) == 0, "REALWORLD: validation is carved from the training subjects, validation_files must be empty"

    training_data = load_realworld(training_files, add_group_id = True)
    test_data = load_realworld(test_files, add_group_id = True)
    if verbose:
        print(f"{'-'*90}")
        print(f"Raw training set shape: {training_data.shape}\nRaw test set shape: {test_data.shape}")

    # ----- Selection of the columns / locomotion labels (timestamp kept until resampling) -----
    training_data_selected = filter_loco_rw(training_data, keep_timestamp = True)
    test_data_selected = filter_loco_rw(test_data, keep_timestamp = True)

    # ----- Clean label 0 -----
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

    # --------------------------------------------------
    # 1) physics-based scaling (acc m/s^2 -> g; gyro already rad/s; mag -> unit vector)
    # --------------------------------------------------
    training_data_scaled = acc_data_scaling_rw(training_data_nan, n_features)   # acc m/s^2 -> g (same units as PAMAP2)
    test_data_scaled = acc_data_scaling_rw(test_data_nan, n_features)

    training_data_scaled = mag_data_norm_rw(training_data_scaled, n_features)     # mag -> unit vector
    test_data_scaled = mag_data_norm_rw(test_data_scaled, n_features)


    # --------------------------------------------------
    # 2) TRAIN / VAL split (strict LOSO): last 20% of each TRAINING subject's stream -> validation
    # --------------------------------------------------
    new_training_data_scaled, new_validation_data_scaled = _split_train_val_per_subject_rw(training_data_scaled, RW_VAL_FRACTION)

    print(f"{'-'*20}TRAIN/VAL split per training subject (test = held-out subject){'-'*20}")
    print("Training SUBJECTS (rows)")
    print(new_training_data_scaled["group_id"].value_counts().sort_index())
    print("Validation SUBJECTS (rows)")
    print(new_validation_data_scaled["group_id"].value_counts().sort_index())
    print("Test SUBJECT (rows)")
    print(test_data_scaled["group_id"].value_counts().sort_index(), "\n")
    train_subj = set(new_training_data_scaled["group_id"]); val_subj = set(new_validation_data_scaled["group_id"]); test_subj = set(test_data_scaled["group_id"])
    assert train_subj.isdisjoint(test_subj) and val_subj.isdisjoint(test_subj), "LEAK: test subject appears in train/val"
    assert len(test_subj) == 1, f"REALWORLD LOSO expects exactly one test subject, got {test_subj}"

    # --------------------------------------------------
    # 3) Resample 50 Hz -> 30 Hz per contiguous segment
    # --------------------------------------------------
    new_training_data_scaled = resample_rw(new_training_data_scaled, n_features, verbose = verbose) 
    new_validation_data_scaled = resample_rw(new_validation_data_scaled, n_features, verbose = verbose) 
    test_data_scaled = resample_rw(test_data_scaled, n_features, verbose = verbose) 

    # --------------------------------------------------
    # 4) Standardize ACC/GYRO: statistics fitted on the TRAINING split only (after resampling = what the model sees)
    # --------------------------------------------------
    acc_gyro_cols = [axis + offset for offset in range(0, n_features, 9) for axis in range(6)]
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
    X_windows, y_windows = sliding_window_wrapper_group(X_features, y_labels, window_size=90, stride=30)
    X_validation_windows, y_validation_windows = sliding_window_wrapper_group(X_val_features, y_val_labels, window_size=90, stride=30)
    X_test_windows, y_test_windows = sliding_window_wrapper_group(X_test_features, y_test_labels, window_size=90, stride=30)

    # ----- Only mag: per window demeaning -----
    X_windows = mag_data_rotation_rw(X_windows) 
    X_validation_windows = mag_data_rotation_rw(X_validation_windows)
    X_test_windows = mag_data_rotation_rw(X_test_windows)

    assert X_windows.shape[1:] == (90, n_features), f"unexpected window shape {X_windows.shape}"
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

    # NOTE: no 'drill' / optional pretraining data for RealWorld (the PAMAP2 Optional block does not apply here).
    #       The 'drill' argument is kept only for signature compatibility with load_OPP_loco_data / load_PAM_loco_data.

    return X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows

def make_loaders_RW(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator, verbose = False) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
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
