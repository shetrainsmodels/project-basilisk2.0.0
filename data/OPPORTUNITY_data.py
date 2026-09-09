from .data_pipeline import (load_opportunity, remove_zero_label_rows, incomplete_labeled_rows, remove_all_nan_rows, sliding_window, divide_features_labels, acc_data_scaling, gyro_data_scaling, mag_data_norm, mag_data_rotation, sliding_window_wrapper_group, add_segment_id_opp, opp_file_id)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .preprocessing import fit_labelencoder, Dataset_HAR
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import torch
from pathlib import Path

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
    mag_cols = [axis + offset for offset in range(6, 45, 9) for axis in range(3)]
    M = Xw[:, :, mag_cols]                     # [W, L, 15]
    per_window_mean = M.mean(axis=1)           # [W, 15]
    flat = M.reshape(-1, M.shape[-1])          # [W*L, 15]

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

        
def data_split_OPP (fold_id: int = 1) -> tuple[list, list, list]:
    '''
    Leave-One-Subject-Out split for Opportunity dataset.
    For a given fold_id, the corresponding subject is held out as test.
    The remaining 3 subjects provide training (ADL1-4) and validation (ADL5).
    '''
    subjects = ["S1", "S2", "S3", "S4"]
    test_subject = subjects[fold_id-1]
    train_subjects = [ subject for subject in subjects if subject != test_subject]

    training_files = ['../DATASETS/OpportunityUCIDataset/dataset/' + subject + '-ADL' + str(ADL_session) +'.dat' for subject in train_subjects for ADL_session in range(1, 5)]
    test_files = ['../DATASETS/OpportunityUCIDataset/dataset/' + test_subject + '-ADL' + str(ADL_session) +'.dat' for ADL_session in range(1, 6)]
    validation_files = ['../DATASETS/OpportunityUCIDataset/dataset/' + subject + '-ADL5.dat' for subject in train_subjects]
    return training_files, validation_files, test_files
    
def load_OPP_loco_data(training_files, validation_files, test_files, verbose = False, drill = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load OPPORTUNIY dataset and process all the data for the model:
    Clean Data
    Scaling: ACC/GYRO
    Normalization: MAG
    '''
    training_data = load_opportunity(training_files, add_group_id = True) #last mod: previously false now no overlapping among ADL files
    validation_data = load_opportunity(validation_files, add_group_id = True)
    test_data = load_opportunity(test_files, add_group_id = True)
    
    if verbose:
        print(f"{'-'*90}")
        print(f"Raw training set shape: {training_data.shape}\nValidation training set shape: {validation_data.shape}\nRaw test set shape: {test_data.shape}")
        
    # ----- Selection of the Columns -----
    # 5 Devices, 3 sensors each device = 15 sensors total
    imu_columns = list(range(37,46))+ list(range(50,59)) + list(range(63,72)) + list(range(76,85)) + list(range(89,98))
    imu_columns.append(243) #labels
    training_data_selected = training_data.iloc[:, imu_columns].copy()
    training_data_selected["group_id"] = training_data["group_id"].values
    validation_data_selected = validation_data.iloc[:, imu_columns].copy()
    validation_data_selected["group_id"] = validation_data["group_id"].values
    test_data_selected = test_data.iloc[:, imu_columns].copy()
    test_data_selected["group_id"] = test_data["group_id"].values
    
    # ----- Clean label ------ #last mod: REMOVE ALL ROWS WITH LABEL 0. 
    training_data_nan = remove_zero_label_rows(training_data_selected)
    validation_data_nan = remove_zero_label_rows(validation_data_selected)
    test_data_nan = remove_zero_label_rows(test_data_selected)
    if verbose:
        print(f"{'-'*90}")
        print(f"Sensor subset training shape: {training_data_nan.shape}\nSensor Validation training shape: {validation_data_nan.shape}\nSensor Test training shape: {test_data_nan.shape}")
    
    # ----- For verification if there are rows with any NaN value  ----- 
    training_nan = incomplete_labeled_rows(training_data_nan)
    validation_nan = incomplete_labeled_rows(validation_data_nan)
    test_nan = incomplete_labeled_rows(test_data_nan)
    if verbose:
        print(f"{'-'*70}")
        print(f"Rows containing NaNs - training: {training_nan}\nRows containing NaNs - validation: {validation_nan}\nRows containing NaNs - test: {test_nan}")

    # ----- Saving data NaNs ----- 
    #training_data_nan.to_csv("OPP_training_nan.csv", index = False)
    #validation_data_nan.to_csv("OPP_validation_nan.csv", index = False)
    #test_data_nan.to_csv("OPP_test_nan.csv", index = False)

    # ----- Clean NaNs ----- 
    training_data_cleaned = remove_all_nan_rows(training_data_nan)
    validation_data_cleaned = remove_all_nan_rows(validation_data_nan)
    test_data_cleaned = remove_all_nan_rows(test_data_nan)
    if verbose:
        print(f"{'-'*90}")
        print(f"Training shape after NaN removal: {training_data_cleaned.shape}\nValidation shape after NaN removal: {validation_data_cleaned.shape}\nTest shape after NaN removal: {test_data_cleaned.shape}")

    # ----- Contiguous segments (temporal-splicing fix): must run BEFORE reset_index, while the index is still the raw row position -----
    training_data_cleaned = add_segment_id_opp(training_data_cleaned)
    validation_data_cleaned = add_segment_id_opp(validation_data_cleaned)
    test_data_cleaned = add_segment_id_opp(test_data_cleaned)
    if verbose:
        print(f"{'-'*90}")
        print(f"Contiguous segments - training: {training_data_cleaned['group_id'].nunique()} | validation: {validation_data_cleaned['group_id'].nunique()} | test: {test_data_cleaned['group_id'].nunique()}")

    # ----- Reset index ----- 
    training_data_cleaned.columns = list(range(training_data_cleaned.shape[1] - 1)) + [training_data_cleaned.columns[-1]]
    training_data_cleaned = training_data_cleaned.reset_index(drop=True)
    
    validation_data_cleaned.columns = list(range(validation_data_cleaned.shape[1] - 1)) + [validation_data_cleaned.columns[-1]]
    validation_data_cleaned = validation_data_cleaned.reset_index(drop=True)
    
    test_data_cleaned.columns = list(range(test_data_cleaned.shape[1] - 1)) + [test_data_cleaned.columns[-1]]
    test_data_cleaned = test_data_cleaned.reset_index(drop=True)
    if verbose:
        print(f"{'-'*90}")
        print(f"Reindexed training shape: {training_data_cleaned.shape}\nReindexed validation shape: {validation_data_cleaned.shape}\nReindexed test shape: {test_data_cleaned.shape}")

# --------------------------------------------------
# 1) physics-based scaling
# --------------------------------------------------
    # Physics-based scaling: considering what was observed in the OPP_plots, sensor values has different relative magnitude and distribution. 
    # By normalizing them, part of the distributional infor could be removed, even though it contains rich patterns that define cross-correlations among sensors. Therefore, physics-based scaling has been performed on the data.
    # compute statistics
    if verbose: 
        print("\n")
        print("Descriptive statistics for ACC channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 0:3].describe())
        print("\nDescriptive statistics for GYRO channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 3:6].describe())
        print("\nDescriptive statistics for MAG channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 6:9].describe())
        print(f"{'-'*90}")
    
    # SCALING ACC
    training_data_scaled = acc_data_scaling(training_data_cleaned)
    validation_data_scaled = acc_data_scaling(validation_data_cleaned)
    test_data_scaled = acc_data_scaling(test_data_cleaned)

    # SCALING GYRO                
    training_data_scaled = gyro_data_scaling(training_data_scaled)
    validation_data_scaled = gyro_data_scaling(validation_data_scaled)
    test_data_scaled = gyro_data_scaling(test_data_scaled)

    # NORM MAG               
    training_data_scaled = mag_data_norm(training_data_scaled)
    validation_data_scaled = mag_data_norm(validation_data_scaled)
    test_data_scaled = mag_data_norm(test_data_scaled)
    if verbose:
        print("\n")
        print("Descriptive statistics for ACC channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 0:3].describe())
        print("\nDescriptive statistics for scaled GYRO channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 3:6].describe())
        print("\nDescriptive statistics for normalized MAG channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 6:9].describe())
    
    # ----- Saving data scaled/cleaned ----
    #training_data_scaled.to_csv("OPP_training_scaled.csv", index = False)
    #validation_data_scaled.to_csv("OPP_validation_scaled.csv", index = False)
    #test_data_scaled.to_csv("OPP_test_scaled.csv", index = False)
    
# --------------------------------------------------
# 2) columns to standardize FIRST TRY ACC AND GYRO
# --------------------------------------------------
    acc_gyro_cols = [axis + offset for offset in range(0, 45, 9) for axis in range(6)]

    # ----- Stratify TRAIN / VAL (strict LOSO) -----
    # Pool = all files of the TRAINING subjects (ADL1-4 + ADL5). Split at file level (group_id) into
    # train (4/5 of the files) and a class-balanced validation set (1/5 of the files).
    # TEST = all sessions (ADL1-5) of the held-out subject, never enters this pool.
    combined_pool = pd.concat([validation_data_scaled, training_data_scaled], axis = 0, ignore_index = True)
    y = combined_pool.iloc[:, -2].values       # Labels
    groups = opp_file_id(combined_pool["group_id"]).values  # FILE id (S2-ADL3): the split stays at file level, segments of one file never straddle train/val

    sgkf = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 42)
    train_idx, val_idx = next(sgkf.split(combined_pool, y, groups))   # sklearn yields (train_index, test_index)

    new_training_data_scaled = combined_pool.iloc[train_idx].copy().reset_index(drop=True)
    new_validation_data_scaled = combined_pool.iloc[val_idx].copy().reset_index(drop=True)

    # ----- Standardize ACC/GYRO: statistics fitted on the (new) TRAINING split only -----
    scaler = StandardScaler()
    new_training_data_scaled.iloc[:, acc_gyro_cols] = scaler.fit_transform(new_training_data_scaled.iloc[:, acc_gyro_cols].values)
    new_validation_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(new_validation_data_scaled.iloc[:, acc_gyro_cols].values)
    test_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(test_data_scaled.iloc[:, acc_gyro_cols].values)

    print(f"{'-'*20}STRATIFY TRAIN/VAL (test = held-out subject){'-'*20}")
    print("Training GROUPS (rows per file | contiguous segments)")
    print(opp_file_id(new_training_data_scaled["group_id"]).value_counts().sort_index(), f"| segments: {new_training_data_scaled['group_id'].nunique()}")
    print("Validation GROUPS (rows per file | contiguous segments)")
    print(opp_file_id(new_validation_data_scaled["group_id"]).value_counts().sort_index(), f"| segments: {new_validation_data_scaled['group_id'].nunique()}")
    print("Test GROUPS (held-out subject)")
    print(opp_file_id(test_data_scaled["group_id"]).value_counts().sort_index(), f"| segments: {test_data_scaled['group_id'].nunique()}", "\n")

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

    # ----- Features&Labels -----
    X_features, y_labels = divide_features_labels(new_training_data_scaled)
    X_val_features, y_val_labels = divide_features_labels(new_validation_data_scaled)
    X_test_features, y_test_labels = divide_features_labels(test_data_scaled)
    if verbose:
        print(f"{'-'*90}")
        print(f"Training features: {X_features.shape} | Training labels: {y_labels.shape}")
        print(f"Validation features: {X_val_features.shape} | Validation labels: {y_val_labels.shape}")
        print(f"Test features: {X_test_features.shape}, Test labels: {y_test_labels.shape}")

    # ----- Sliding Windows -----

    X_windows, y_windows = sliding_window_wrapper_group(X_features, y_labels, window_size=90, stride=30)
    X_validation_windows, y_validation_windows = sliding_window_wrapper_group(X_val_features, y_val_labels, window_size=90, stride=30)
    X_test_windows, y_test_windows = sliding_window_wrapper_group(X_test_features, y_test_labels, window_size=90, stride=30)
    
    # ----- Only mag: per window demeaning -----
    X_windows = mag_data_rotation(X_windows)
    X_validation_windows = mag_data_rotation(X_validation_windows)
    X_test_windows = mag_data_rotation(X_test_windows)
    
    if verbose:
        print_mag_rotation_stats(X_windows, "TRAIN")
        print_mag_rotation_stats(X_validation_windows, "VAL")
        print_mag_rotation_stats(X_test_windows, "TEST")
    
        print_window_label_distribution(y_windows, "TRAIN")
        print_window_label_distribution(y_validation_windows, "VAL")
        print_window_label_distribution(y_test_windows, "TEST")
    
    if verbose:
        print(f"{'-'*90}")
        print(f"Training (windows): {X_windows.shape}. Training Labels: {y_windows.shape}")
        print(f"Validation (windows): {X_validation_windows.shape}. Validation Labels: {y_validation_windows.shape}")
        print(f"Test (windows): {X_test_windows.shape}. Test Labels: {y_test_windows.shape}")            


    if drill:
        train_subjects = sorted({Path(f).stem.split('-')[0] for f in training_files})
        drill_files = ['../DATASETS/OpportunityUCIDataset/dataset/' + s + '-Drill.dat' for s in train_subjects] 
        print(f"DRILL pretraining files: {drill_files}")
        drill_data = load_opportunity(drill_files, add_group_id = True)
        drill_sel = drill_data.iloc[:, imu_columns].copy()
        drill_sel["group_id"] = drill_data["group_id"].values

        drill_sel = remove_zero_label_rows(drill_sel)           # same cleaning as ADL training
        drill_sel = remove_all_nan_rows(drill_sel)
        drill_sel = add_segment_id_opp(drill_sel)               # temporal-splicing fix, BEFORE reset_index
        drill_sel.columns = list(range(drill_sel.shape[1] - 1)) + [drill_sel.columns[-1]]
        drill_sel = drill_sel.reset_index(drop=True)
        drill_sel = acc_data_scaling(drill_sel)
        drill_sel = gyro_data_scaling(drill_sel)
        drill_sel = mag_data_norm(drill_sel)
        drill_sel.iloc[:, acc_gyro_cols] = scaler.transform(drill_sel.iloc[:, acc_gyro_cols].values)   # scaler fitted on ADL train, NOT refit
        X_drill_features, y_drill_labels = divide_features_labels(drill_sel)
        X_drill_windows, y_drill_windows = sliding_window_wrapper_group(X_drill_features, y_drill_labels, window_size=90, stride=30)
        X_drill_windows = mag_data_rotation(X_drill_windows)
        print(f"DRILL windows added to pretraining: {X_drill_windows.shape} | ADL training windows: {X_windows.shape}")
        X_windows = np.concatenate([X_windows, X_drill_windows], axis=0)
        y_windows = np.concatenate([y_windows, y_drill_windows], axis=0)

    return X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows
    
def make_loaders_OPP(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator, verbose = False) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    '''
    Creates DataLoaders for training, validation and test sets.
    Called once per seed — generator ensures reproducible shuffling.

    Args:
    X_windows, y_windows: training windows and labels
    X_validation_windows, y_validation_windows: validation windows and labels
    X_test_windows, y_test_windows: test windows and labels
    generator: seeded for reproducibility
    verbose: print class counts and batch shapes
    '''
    # ----- LabelEncoder, Transform, DataLoader -----
    label_encoder = fit_labelencoder(X_windows, y_windows)
    training_dataset = Dataset_HAR(X_windows, y_windows, label_encoder = label_encoder)
    validation_dataset = Dataset_HAR(X_validation_windows, y_validation_windows, label_encoder = label_encoder)
    test_dataset = Dataset_HAR(X_test_windows, y_test_windows, label_encoder = label_encoder)

    train_loader = DataLoader(training_dataset, batch_size = 128, shuffle = True, generator = generator, num_workers = 2, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(validation_dataset, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, num_workers = 0, pin_memory = True)
    # ----- Samples per laber checking and batch size ----- 
    if verbose:
        label_to_name = {0: "STAND", 1: "WALK", 2: "SIT", 3:"LIE"}
        class_counts = {name: 0 for name in label_to_name.values()}
        for _, y_batch in train_loader:
            for label in y_batch:
                label_idx = label.item()
                class_name = label_to_name[label_idx]
                class_counts[class_name] += 1
        print(f"{'-'*90}")
        print("Training set class distribution:")
        print(class_counts)
        print(f"{'-'*90}")
    return train_loader, val_loader, test_loader, label_encoder