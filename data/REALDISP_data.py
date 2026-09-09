from .data_pipeline import (load_realdisp, filter_realdisp, remove_zero_label_rows, incomplete_labeled_rows, interpolation_pam,
                              acc_data_scaling_rw, mag_data_norm_rw, mag_data_rotation_rw, resample_rw,
                              divide_features_labels, sliding_window_wrapper_group)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from .preprocessing import fit_labelencoder, Dataset_HAR
from .PAMAP2_data import print_post_standardize_stats, print_mag_rotation_stats, print_window_label_distribution
import pandas as pd, numpy as np, torch
from pathlib import Path
import glob 


Realdisp_SUBJECTS = [f"{i}" for i in range(1, 18)]  # subject1 to subject17
Realdisp_BASE_PATH = '../DATASETS/REALDISP_Dataset/'
RD_VAL_FRACTION = 0.2  # 20% of each training subject's stream for validation
RD_WINDOW, RD_STRIDE = 90, 30  # sliding window / stride in samples at 30 Hz after resampling (3 s / 1 s), same as OPP, PAM, RW
RD_VAL_MIN_ROWS = 150  # raw rows (3 s at 50 Hz) = minimum a validation slice needs to yield ONE 90-sample window after 30 Hz resampling

# REALDISP activity set (dataset manual, Table 2). Label 0 = no activity (removed by remove_zero_label_rows).
REALDISP_CLASS_NAMES = {
    1:  "Walking",
    2:  "Jogging",
    3:  "Running",
    4:  "Jump up",
    5:  "Jump front & back",
    6:  "Jump sideways",
    7:  "Jump leg/arms open/closed",
    8:  "Jump rope",
    9:  "Trunk twist (arms outstretched)",
    10: "Trunk twist (elbows bended)",
    11: "Waist bends forward",
    12: "Waist rotation",
    13: "Waist bends (reach foot with opposite hand)",
    14: "Reach heels backwards",
    15: "Lateral bend (10x to the left + 10x to the right)",
    16: "Lateral bend arm up (10x to the left + 10x to the right)",
    17: "Repetitive forward stretching",
    18: "Upper trunk and lower body opposite twist",
    19: "Arms lateral elevation",
    20: "Arms frontal elevation",
    21: "Frontal hand claps",
    22: "Arms frontal crossing",
    23: "Shoulders high amplitude rotation",
    24: "Shoulders low amplitude rotation",
    25: "Arms inner rotation",
    26: "Knees (alternatively) to the breast",
    27: "Heels (alternatively) to the backside",
    28: "Knees bending (crouching)",
    29: "Knees (alternatively) bend forward",
    30: "Rotation on the knees",
    31: "Rowing",
    32: "Elliptic bike",
    33: "Cycling",
}

def data_split_Realdisp(fold_id: int = 1, scenarios: list =["self", "ideal"]) -> tuple[list, list, list]:
    '''
    Leave-One-Subject-Out split for REALDISP (all scenario files of each subject: subject<k>_<scenario>.log).
    fold_id k (1..17) holds out subject k as TEST; the remaining 16 subjects provide training.
    Validation is carved from the TRAINING subjects inside load_REALDISP_data (last 20% of every activity run),
    so validation_files is returned empty here.
    '''
    if fold_id not in range(1, len(Realdisp_SUBJECTS) + 1):
        raise ValueError(f"Realdisp fold must be in 1..{len(Realdisp_SUBJECTS)}. Got {fold_id}")
    test_subject = Realdisp_SUBJECTS[fold_id - 1]
    train_subjects = [s for s in Realdisp_SUBJECTS if s != test_subject]

    training_files = []
    for s in train_subjects:
        for sc in scenarios:
            training_files += sorted(glob.glob(Realdisp_BASE_PATH + f'subject{s}_{sc}*.log'))   # subject{s}_ : the '_' avoids subject1 matching subject10..17
    validation_files = []
    test_files = []
    for sc in scenarios:
        test_files += sorted(glob.glob(Realdisp_BASE_PATH + f'subject{test_subject}_{sc}*.log'))

    assert len(training_files) > 0 and len(test_files) > 0, f"No REALDISP .log files found under {Realdisp_BASE_PATH}"
    return training_files, validation_files, test_files

def _split_train_val_per_subject_RD(df: pd.DataFrame, val_fraction: float, period: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    df layout: 0..n-1 sensor features | 'ts' | n (Label) | 'group_id' (subject). Rows of each subject are in file order
    (do NOT sort by 'ts': REALDISP timestamps restart at 0 at every recording restart inside a file).
    REALDISP activities are performed as contiguous blocks in a FIXED order (L1 -> L2 -> ... -> L33), so taking the
    tail of a subject's whole stream would give a single-class validation set. Instead, for every contiguous activity run
    (same label, no timestamp gap at the raw 50 Hz period = 0.02 s) the last val_fraction of its rows -> validation, the rest -> training,
    with a minimum of RD_VAL_MIN_ROWS rows per validation slice so that every class of every training subject gets validation windows.
    Validation stays contiguous in time (no window crosses the train/val cut: the cut becomes a timestamp gap) and contains
    every class of every training subject.
    '''
    label_col = [c for c in df.columns if c not in ("ts", "group_id")][-1]   # 27 (pam), 45 (opp) or 81 (all)
    train_parts, val_parts = [], []
    for subject, gdf in df.groupby("group_id", sort=False):
        gdf = gdf.reset_index(drop=True)                                     # keep file order
        labels = gdf[label_col].to_numpy()
        ts = gdf["ts"].to_numpy(dtype=np.float64)
        new_run = np.zeros(len(gdf), dtype=bool)
        new_run[1:] = (labels[1:] != labels[:-1]) | (np.diff(ts) > 1.5 * period) | (np.diff(ts) <= 0)
        run_id = np.cumsum(new_run)
        for _, rdf in gdf.groupby(run_id, sort=False):
            n_val = int(round(len(rdf) * val_fraction))
            if len(rdf) >= 2 * RD_VAL_MIN_ROWS:                 # short runs (L26/L27 ~8 s, L4/L8 ~11 s): 20% < 150 rows would never form
                n_val = max(n_val, RD_VAL_MIN_ROWS)             # a window -> guarantee one validation window, keep >= 150 rows for train
            train_parts.append(rdf.iloc[:len(rdf) - n_val])
            val_parts.append(rdf.iloc[len(rdf) - n_val:])
    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True)
    return train_df, val_df


def load_REALDISP_data(training_files, validation_files, test_files, NUM_SENSORS, verbose = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load REALDISP (subject<k>_<scenario>.log files) and process it for the model, mirroring load_REALWORLD_loco_data (strict LOSO):
      -> sensor selection (NUM_SENSORS: 6 = OPP + RC / 5 = OPP / 3 = PAM; 9 = all is NOT supported by filter_realdisp; n_features = 9 channels per sensor)
      -> drop label 0 -> interpolate short NaN gaps -> physical scaling (acc -> g, mag -> unit vector)
      -> per-training-subject train/val split (last 20% of every activity run) -> resample 50 Hz -> 30 Hz (per contiguous segment)
      -> StandardScaler (acc/gyro) fitted on TRAIN only -> sliding windows 90/30 per segment -> per-window mag demeaning.
    TEST = the held-out subject only (never enters the train/val pool).
    validation_files is ignored (must be empty): validation comes from the training subjects.
    '''

    assert len(validation_files) == 0, "REALDISP: validation is carved from the training subjects, validation_files must be empty"

    training_data = load_realdisp(training_files, add_group_id = True)
    test_data = load_realdisp(test_files, add_group_id = True)
    if verbose:
        print(f"{'-'*90}")
        print(f"Raw training set shape: {training_data.shape}\nRaw test set shape: {test_data.shape}")

    # ----- Selection of the columns for OPP or PAM configuration (timestamp kept until resampling) -----
    training_data_selected = filter_realdisp(training_data, keep_timestamp = True, keep_sensors = NUM_SENSORS)
    test_data_selected = filter_realdisp(test_data, keep_timestamp = True, keep_sensors = NUM_SENSORS)
    n_features = training_data_selected.shape[1] - 3   # sensor channels only ('ts', label, 'group_id' excluded): 27 (pam), 45 (opp) or 81 (all)

    # ----- Remove label 0 rows (no activity: ~72% of REALDISP rows; also removes the single NaN row of subject15_mutual7) -----
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
    training_data_cleaned = interpolation_pam(training_data_nan, max_gap=30) # SAME FUNCTION AS PAM
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
    training_data_scaled = acc_data_scaling_rw(training_data_cleaned, n_features)   # acc m/s^2 -> g (same units as PAMAP2/RealWorld)
    test_data_scaled = acc_data_scaling_rw(test_data_cleaned, n_features)

    training_data_scaled = mag_data_norm_rw(training_data_scaled, n_features)     # mag -> unit vector (Xsens field strength is arbitrary)
    test_data_scaled = mag_data_norm_rw(test_data_scaled, n_features)

# --------------------------------------------------
# 2) TRAIN / VAL split (strict LOSO): last 20% of every activity run of each TRAINING subject -> validation
# --------------------------------------------------
    new_training_data_scaled, new_validation_data_scaled = _split_train_val_per_subject_RD(training_data_scaled, RD_VAL_FRACTION)

    print(f"{'-'*20}TRAIN/VAL split per training subject (test = held-out subject){'-'*20}")
    print("Training SUBJECTS (rows)")
    print(new_training_data_scaled["group_id"].value_counts().sort_index())
    print("Validation SUBJECTS (rows)")
    print(new_validation_data_scaled["group_id"].value_counts().sort_index())
    print("Test SUBJECT (rows)")
    print(test_data_scaled["group_id"].value_counts().sort_index(), "\n")
    train_subj = set(new_training_data_scaled["group_id"]); val_subj = set(new_validation_data_scaled["group_id"]); test_subj = set(test_data_scaled["group_id"])
    assert train_subj.isdisjoint(test_subj) and val_subj.isdisjoint(test_subj), "LEAK: test subject appears in train/val"
    assert len(test_subj) == 1, f"REALDISP LOSO expects exactly one test subject, got {test_subj}"

# --------------------------------------------------
# 3) Resample 50 Hz -> 30 Hz per contiguous segment (drops 'ts', group_id becomes '<subject>-s<k>')
# --------------------------------------------------
    new_training_data_scaled = resample_rw(new_training_data_scaled, n_features, verbose = verbose)   # same 50 Hz -> 30 Hz resampler as RealWorld
    new_validation_data_scaled = resample_rw(new_validation_data_scaled, n_features, verbose = verbose)
    test_data_scaled = resample_rw(test_data_scaled, n_features, verbose = verbose)

# --------------------------------------------------
# 4) Standardize ACC/GYRO: statistics fitted on the TRAINING split only (after resampling = what the model sees)
# --------------------------------------------------
    acc_gyro_cols = [axis + offset for offset in range(0, n_features, 9) for axis in range(6)]   # acc xyz + gyro xyz of EVERY sensor (mag excluded)
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
    X_windows, y_windows = sliding_window_wrapper_group(X_features, y_labels, window_size=RD_WINDOW, stride=RD_STRIDE)
    X_validation_windows, y_validation_windows = sliding_window_wrapper_group(X_val_features, y_val_labels, window_size=RD_WINDOW, stride=RD_STRIDE)
    X_test_windows, y_test_windows = sliding_window_wrapper_group(X_test_features, y_test_labels, window_size=RD_WINDOW, stride=RD_STRIDE)

    # ----- Only mag: per window demeaning -----
    X_windows = mag_data_rotation_rw(X_windows)
    X_validation_windows = mag_data_rotation_rw(X_validation_windows)
    X_test_windows = mag_data_rotation_rw(X_test_windows)

    assert X_features.shape[1] - 1 == n_features, f"feature count changed along the pipeline: {X_features.shape[1] - 1} vs {n_features}"
    assert X_windows.shape[1:] == (RD_WINDOW, n_features), f"unexpected window shape {X_windows.shape}, expected (*, {RD_WINDOW}, {n_features})"
    # REALDISP has 33 activity classes (L1..L33, see REALDISP_CLASS_NAMES). The LabelEncoder is fitted on TRAIN only,
    # so every class seen in VAL or TEST must also exist in TRAIN (otherwise label_encoder.transform fails later).
    train_classes = set(np.unique(y_windows)); val_classes = set(np.unique(y_validation_windows)); test_classes = set(np.unique(y_test_windows))
    assert train_classes.issubset(set(REALDISP_CLASS_NAMES)), f"unexpected labels in train: {train_classes - set(REALDISP_CLASS_NAMES)}"
    assert val_classes.issubset(train_classes), f"VAL contains classes missing from TRAIN: {sorted(val_classes - train_classes)}"
    assert test_classes.issubset(train_classes), f"TEST contains classes missing from TRAIN: {sorted(test_classes - train_classes)}"
    if train_classes != val_classes:
        print(f"WARNING: VAL is missing train classes {sorted(train_classes - val_classes)} (short runs dropped by resampling/windowing)")

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

    return X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows

def make_loaders_REALDISP(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator, verbose = False) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
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
        # encoded index -> raw id via the fitted encoder -> REALDISP activity name (L1..L33)
        label_to_name = {i: f"L{int(raw)}: {REALDISP_CLASS_NAMES[int(raw)]}" for i, raw in enumerate(label_encoder.classes_)}
        class_counts = {name: 0 for name in label_to_name.values()}
        for _, y_batch in train_loader:
            for label in y_batch:
                class_counts[label_to_name[label.item()]] += 1
        print(f"{'-'*90}")
        print("Training set class distribution:")
        print(class_counts)
        print(f"{'-'*90}")
    return train_loader, val_loader, test_loader, label_encoder
