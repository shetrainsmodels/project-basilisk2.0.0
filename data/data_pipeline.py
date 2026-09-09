import pandas as pd
from pathlib import Path 
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.signal import resample_poly

# ======================================================================================================================
#                                              OPPORTUNITY FUNCTIONS
# ======================================================================================================================

def load_opportunity(files_path, add_group_id = False) -> pd.DataFrame:
    '''
    Read multiple .dat files from the Opportunity dataset and vertically concatenate
    them into a single DataFrame.

    If add_group_id=True, append a last column called 'group_id'
    with the session/file name, e.g. S1-ADL5.
    '''
    dataframes = []
    row_counts = 0

    for file in files_path:
        df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')

        if add_group_id:
            session_name = Path(file).stem   # e.g. S1-ADL5
            df["group_id"] = session_name    # added as LAST column

        dataframes.append(df)
        row_counts += len(df)

    complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    assert len(complete_dataframe) == row_counts, (
        f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}")
    return complete_dataframe
    
def remove_zero_label_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove rows where the label is 0.
    Works for:
      - train: label in last column
      - train/val/test: label in second-to-last column if last column is 'group_id'
    '''
    label_col_idx = -2 if dataframe.columns[-1] == "group_id" else -1
    zero_label = dataframe.iloc[:, label_col_idx] == 0
    return dataframe[~(zero_label)]

def remove_all_nan_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove any row with any NaN value. 
    (No interpolation done in the dataset later)
    '''
    mask_nans = dataframe.isna().any(axis = 1)
    return dataframe.loc[~mask_nans].copy()

def add_segment_id_opp(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    OPP temporal-splicing fix. Rows were deleted (label 0, NaN) but windows were still taken over consecutive REMAINING rows
    of a file, so a window could span a deleted interval. Call this right AFTER the row removals and BEFORE reset_index:
    at that point the pandas index is still the raw row position. A new contiguous segment starts where the file (group_id)
    changes or the raw row position jumps by more than 1. group_id becomes '<file>-s<k>' (e.g. S1-ADL3-s7), so
    sliding_window_wrapper_group windows every segment on its own and no window crosses a deleted interval.
    Segments shorter than the window (90 rows) yield no windows and are dropped by the windowing.
    '''
    df = dataframe.copy()
    gid = df["group_id"].to_numpy()
    pos = df.index.to_numpy()
    new_seg = np.ones(len(df), dtype=bool)
    new_seg[1:] = (gid[1:] != gid[:-1]) | (np.diff(pos) != 1)
    new_file = np.ones(len(df), dtype=bool)
    new_file[1:] = gid[1:] != gid[:-1]
    seg_global = np.cumsum(new_seg)                                             # 1, 2, 3 ... over the whole frame
    seg_at_file_start = np.maximum.accumulate(np.where(new_file, seg_global, 0))
    k = seg_global - seg_at_file_start                                          # segment counter restarts at 0 in every file
    df["group_id"] = [f"{g}-s{i}" for g, i in zip(gid, k)]
    return df

def opp_file_id(group_id: pd.Series) -> pd.Series:
    '''Strip the '-s<k>' segment suffix added by add_segment_id_opp -> file id (S1-ADL3).'''
    return group_id.astype(str).str.replace(r"-s\d+$", "", regex=True)
    
def incomplete_labeled_rows(dataframe: pd.DataFrame) -> int:
    '''
    Count rows with at least one NaN in sensor columns only.
    Excludes:
      - label column in train
      - label + group_id in val/test/train
    '''
    sensor_data = dataframe.iloc[:, :-2] if dataframe.columns[-1] == "group_id" else dataframe.iloc[:, :-1]
    incomplete_rows = sensor_data.isna().any(axis=1)
    return int(incomplete_rows.sum())

def divide_features_labels(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # last mod for the windows per session to avoid creating art. windows change this in SP as well. what a mess
    if dataframe.columns[-1] == "group_id":
        sensor_ft = dataframe.iloc[:, :-2]
        group_id = dataframe[["group_id"]] #pd not series
        features = pd.concat([sensor_ft, group_id], axis = 1)
        labels = dataframe.iloc[:, -2]
    else:
        features = dataframe.iloc[:, :-1]
        labels = dataframe.iloc[:, -1]
    return features, labels

def acc_data_scaling (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Relative magnitude differ across sensor types.
    For ACC data which is in mili g, this is divided by 1000 to obtain g, bringing values to a smaller range.
    '''
    dataframe_ = dataframe.copy()
    acc_cols = [axi + offset for offset in range(0, 45, 9) for axi in range(3)]
    dataframe_.iloc[:, acc_cols] /= 1000
    return dataframe_ 

def gyro_data_scaling (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    GYRO data is milli rad/s, bringing values to a smaller range: divide by 1000 to obtain rad/s.
    '''
    dataframe_ = dataframe.copy()
    gyro_cols = [axi + offset for offset in range(3, 45, 9) for axi in range(3)]
    dataframe_.iloc[:, gyro_cols] /= 1000
    return dataframe_

def mag_data_norm (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    For MAG the magnitude equation is used to remove the strenght of the signal and mantain the direction, which 
    tells us about the body orientation. 
    '''
    dataframe_ = dataframe.copy()
    for offset in range (6, 45, 9):
        mag_x = dataframe_.iloc[:, offset]
        mag_y = dataframe_.iloc[:, offset + 1]
        mag_z = dataframe_.iloc[:, offset + 2]
        magnitude = np.sqrt((mag_x ** 2) + (mag_y ** 2) + (mag_z ** 2))
        for axi in range(3):
            new_axi = dataframe_.iloc[:, axi + offset] / magnitude
            dataframe_.iloc[:, axi + offset] = new_axi
    return dataframe_

def sliding_window(X, y, window_size: int, stride: int, return_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Row-wise sliding windows and assign a label policy: majority vote.
    Returns: features_windows: [#windows, window_size, #features]
             labels_windows: [#windows]
    '''

    
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y, (pd.Series, pd.DataFrame))
    X_values = X.values if X_is_df else np.asarray(X)
    y_values = y.values.ravel() if y_is_series else np.asarray(y).ravel()

    num_rows, num_feats = X_values.shape
    X_windows = []
    y_windows = []
    window_count = 0
    # Slide Window
    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        Xw = X_values[start:end, :]
        yw = y_values[start:end]

        win_label = Counter(yw).most_common(1)[0][0] # counts times each label appears. Take the tuple. Take the label (first position)
        X_windows.append(Xw)
        y_windows.append(int(win_label))
        window_count += 1

    # Stack Outputs
    if len(X_windows) == 0:
        if return_numpy:
            return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)
        else:
            return [], []

    if return_numpy:
        X_windows = np.stack(X_windows).astype(np.float32)  # [W,L,F]
        y_windows = np.asarray(y_windows, dtype=np.int64)   # [W]
    
    return X_windows, y_windows

#new function for group_id
def sliding_window_wrapper_group(X, y, window_size: int, stride: int)-> Tuple[np.ndarray, np.ndarray]:
    
    '''Wrapper to create sliding windows per group_id to avoid creating artificial windows among sessions.'''
    
    X_windows_all = []
    y_windows_all = []
    
    for _, X_group in X.groupby("group_id"):
        idx_group = X_group.index
        y_group = y.loc[idx_group] #filter labels
        X_group = X_group.drop(columns=["group_id"]) #data with no group

        X_windows, y_windows = sliding_window(X_group, y_group, window_size=window_size, stride=stride)
        if len(X_windows) > 0:
            X_windows_all.append(X_windows)
            y_windows_all.append(y_windows)

    if len(X_windows_all) == 0: #same check as in sliding_window
        num_feats = X.drop(columns=["group_id"]).shape[1]
        return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_windows_all = np.concatenate(X_windows_all, axis = 0)
    y_windows_all = np.concatenate(y_windows_all, axis = 0)            
    return X_windows_all, y_windows_all 
    
def mag_data_rotation(X_windows) -> np.array:
    '''
    Demeaning
    '''
    X_w = X_windows.copy()
    mag_cols = [axis + offset for offset in range(6, 45, 9) for axis in range(3)]
    for channel in mag_cols:
        mean_val = X_w[:, :, channel].mean(axis = 1, keepdims = True)  # [W, 1] for windows W compute mean in that column.
        X_w[:, :, channel] -= mean_val  # remove orientation, keep rotation dynamics
    return X_w


# ======================================================================================================================
#                                              PAMAP2 FUNCTIONS
# Layout convention (same as OPP): sensor features first, then Label, then 'group_id' as the LAST column.
# PAMAP2: 3 IMUs hand, chest, ankle = 27 sensor columns.
# ======================================================================================================================

def load_pamap2(files_path, add_group_id = False) -> pd.DataFrame:
    '''
    Read multiple .dat files from the PAMAP2 dataset and vertically concatenate them into a single DataFrame.
    If add_group_id=True, append a last column called 'group_id' with the subject id: e.g. 101, 102, ..., 109.
    '''
    dataframes = []
    row_counts = 0

    for file in files_path:
        df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')
        df = df.astype(np.float32)
        if add_group_id:
            subject_id = Path(file).stem[-3:]   # subject101 -> 101
            df["group_id"] = subject_id         # added as LAST column

        dataframes.append(df)
        row_counts += len(df)

    complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    assert len(complete_dataframe) == row_counts, (
        f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}")
    return complete_dataframe


def filter_loco_pam(dataframe: pd.DataFrame, keep_timestamp: bool = False, keep_labels = (1, 2, 3, 4)) -> pd.DataFrame:
    '''
    Filter PAMAP2 data for the locomotion setup used in this project.
    - take activityID (col 1) as label
    - keep only rows whose activityID is in keep_labels (default locomotion 1 lie, 2 sit, 3 stand, 4 walk);
      keep_labels=None keeps ALL rows (used for the Optional pretraining data; label 0 is removed later by remove_zero_label_rows)
    - drop timestamp, activityID, heart rate
    - remove temperature, second accelerometer (6g), and orientation
    - keep group_id untouched as the LAST column if it exists

    Output columns: 0..26 sensor features, [ 'ts' if keep_timestamp ], 27 (Label), [ 'group_id' ]
    '''
    dataset = dataframe.copy()

    group_id = None
    if "group_id" in dataset.columns:
        group_id = dataset["group_id"].copy()
        dataset = dataset.drop(columns=["group_id"])

    if keep_labels is not None:
        keep_rows = dataset.iloc[:, 1].isin(list(keep_labels))
        dataset = dataset.loc[keep_rows].copy()

        if group_id is not None:
            group_id = group_id.loc[keep_rows].copy()

    label = dataset.iloc[:, 1].copy()
    timestamp = dataset.iloc[:, 0].copy()
    dataset = dataset.drop(columns=[0, 1, 2])  # drop timestamp, label, heart rate

    imu_chunk = 17
    temp_cols = [col for col in range(3, dataset.shape[1], imu_chunk)]  # remove temp
    acc_second_cols = [col + axis for col in range(7, dataset.shape[1], imu_chunk) for axis in range(3)]  # remove second acc
    ori_cols = [col + axis for col in range(16, dataset.shape[1], imu_chunk) for axis in range(4)]  # remove orientation

    dataset = dataset.drop(columns=temp_cols + acc_second_cols + ori_cols)
    assert dataset.shape[1] == 27, f"PAMAP2 sensor subset must have 27 columns, got {dataset.shape[1]}"
    dataset.columns = range(dataset.shape[1])         # reset feature cols -> 0..26
    if keep_timestamp:
        dataset["ts"] = timestamp.values              # timestamp (seconds) right before the label
    dataset[27] = label.values.astype(np.int64)        # label column, named 27 (int) as in the OPP layout

    if group_id is not None:
        dataset["group_id"] = group_id.values         # group_id added to LAST col
    return dataset


def interpolation_pam(dataframe: pd.DataFrame, max_gap: int = 30) -> pd.DataFrame:
    '''
    Interpolate short NaN chunks in the sensor columns only.
    Expected layout: 0..n-1 sensor features (int column names), [ 'ts' ], n (label, int name), [ 'group_id' ].
    A NaN chunk of length <= max_gap is linearly interpolated ONLY if the valid rows right before and after it belong
    to the same recording: same label, same group_id (subject) and no timestamp jump. After the label filter,
    unrelated activities become adjacent rows, so a chunk sitting on such a boundary is NOT interpolated
    (it would blend two different activities / subjects). Longer chunks and skipped chunks are removed at the end.
    '''
    df = dataframe.copy()
    int_cols = [c for c in df.columns if isinstance(c, (int, np.integer))]
    label_col = max(int_cols)
    sensor_cols = [c for c in int_cols if c != label_col]

    nan_mask = df.loc[:, sensor_cols].isna().any(axis=1).to_numpy()
    nan_positions = np.where(nan_mask)[0]
    if len(nan_positions) == 0:
        return df

    labels = df[label_col].to_numpy()
    groups = df["group_id"].to_numpy() if "group_id" in df.columns else None
    ts = df["ts"].to_numpy(dtype=np.float64) if "ts" in df.columns else None
    period = None
    if ts is not None:
        dts = np.diff(ts)
        period = float(np.median(dts[dts > 0])) if np.any(dts > 0) else None   # sampling period (0.01 s PAMAP2, 0.02 s RealDisp)

    gaps = np.diff(nan_positions)
    seg_starts = np.insert(np.where(gaps > 1)[0] + 1, 0, 0) # start of segment nan
    seg_ends = np.append(np.where(gaps > 1)[0], len(nan_positions) - 1) # end of segment nan

    n_rows = len(df)
    for s, e in zip(seg_starts, seg_ends):
        start = nan_positions[s]
        end = nan_positions[e]
        seg_len = end - start + 1
        if seg_len > max_gap:
            continue
        lo, hi = start - 1, end + 1
        if lo < 0 or hi >= n_rows:                       # chunk touches the frame edge: no valid neighbour on one side
            continue
        if labels[lo] != labels[hi]:                     # label boundary
            continue
        if groups is not None and groups[lo] != groups[hi]:   # subject boundary
            continue
        if period is not None:
            dt = ts[hi] - ts[lo]
            if dt <= 0 or dt > (seg_len + 1 + 5) * period:    # timestamp jump / restart: different recording
                continue
        rows = df.index[lo: hi + 1]
        interp = df.loc[rows, sensor_cols].interpolate(method="linear", axis=0, limit_area="inside")
        df.loc[rows, sensor_cols] = interp

    keep_rows = ~df.loc[:, sensor_cols].isna().any(axis=1)
    return df.loc[keep_rows].reset_index(drop=True)

def acc_data_scaling_pam (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    PAMAP2 and RealWorld acceleration is in m/s^2. Divide by 9.81 to convert to g (OPP acc is converted to g as well).
    Assumes 9 sensor columns per device (acc xyz, gyro xyz, mag xyz), acc first, sensor columns starting at 0.
    '''
    dataframe_ = dataframe.copy()
    acc_cols = [axi + offset for offset in range(0, 27, 9) for axi in range(3)]
    dataframe_.iloc[:, acc_cols] /= 9.81
    return dataframe_

def mag_data_norm_pam(dataframe: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    '''
    Normalize each magnetometer 3D vector by its magnitude: removes absolute field strength, keeps direction.
    '''
    dataframe_ = dataframe.copy()
    for offset in range(6, 27, 9):
        mag_x = dataframe_.iloc[:, offset]
        mag_y = dataframe_.iloc[:, offset + 1]
        mag_z = dataframe_.iloc[:, offset + 2]

        magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        magnitude = magnitude.clip(lower = eps)

        for axi in range(3):
            dataframe_.iloc[:, axi + offset] = dataframe_.iloc[:, axi + offset] / magnitude
    return dataframe_


def resample_pam(dataframe: pd.DataFrame, up: int = 3, down: int = 10, period: float = 0.01, min_raw_len: int = 300, verbose: bool = False) -> pd.DataFrame:
    '''
    Resample PAMAP2 from 100 Hz to exactly 30 Hz with scipy.signal.resample_poly (FIR anti-aliasing) BEFORE sliding windows.

    Input layout : 0..26 sensor features | 'ts' (seconds) | 27 (Label) | 'group_id' (subject)
    Output layout: 0..26 sensor features | 27 (Label) | group_id

    - Resampling is done per contiguous segment (rows whose timestamps are consecutive at the raw period): the filter never
      blends across a gap created by NaN-row removal or by non-locomotion activities.
    - Segments shorter than min_raw_len raw samples (default 300 = 3 s = one window after resampling) are dropped.
    '''
    assert "ts" in dataframe.columns and "group_id" in dataframe.columns, "resample_pam expects 'ts' and 'group_id' columns"
    feature_cols = [c for c in dataframe.columns if c not in ("ts", 27, "group_id")]
    assert len(feature_cols) == 27, f"expected 27 sensor columns, got {len(feature_cols)}"

    out_frames = []
    for subject, subject_df in dataframe.groupby("group_id", sort=False):
        subject_df = subject_df.reset_index(drop=True)
        ts = subject_df["ts"].to_numpy(dtype=np.float64)
        X = subject_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
        y = subject_df[27].to_numpy()

        # contiguous segments: break where the timestamp jumps by more than 1.5 raw periods (or goes backwards)
        dt = np.diff(ts)
        breaks = np.where((dt > 1.5 * period) | (dt <= 0))[0] + 1
        bounds = np.concatenate([[0], breaks, [len(subject_df)]])

        n_raw_kept, n_res, n_seg_kept, n_seg_dropped = 0, 0, 0, 0
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            seg_len = e - s
            if seg_len < min_raw_len:
                n_seg_dropped += 1
                continue
            n_blocks = seg_len // down
            s_e = s + n_blocks * down                        # truncate to a multiple of 'down'
            Xr = resample_poly(X[s:s_e], up, down, axis=0)  # [n_blocks*up, 27]
            assert Xr.shape[0] == n_blocks * up, f"resample_poly length mismatch {Xr.shape[0]} vs {n_blocks * up}"
            y_blocks = y[s:s_e].reshape(n_blocks, down)
            y_major = np.array([Counter(b.tolist()).most_common(1)[0][0] for b in y_blocks], dtype=np.int64)
            yr = np.repeat(y_major, up)

            seg_df = pd.DataFrame(Xr.astype(np.float32), columns=feature_cols)
            seg_df[27] = yr
            seg_df["group_id"] = f"{subject}-s{n_seg_kept}"
            out_frames.append(seg_df)
            n_seg_kept += 1
            n_raw_kept += (s_e - s)
            n_res += Xr.shape[0]

        if verbose:
            ratio = n_res / n_raw_kept if n_raw_kept else float("nan")
            print(f"[resample_pam] subject {subject}: raw rows {len(subject_df)} | kept raw {n_raw_kept} in {n_seg_kept} segments "
                  f"(dropped {n_seg_dropped} short segments) | resampled rows {n_res} | ratio {ratio:.4f} (target {up/down:.4f})")

    if len(out_frames) == 0:
        raise ValueError("resample_pam produced no data")
    return pd.concat(out_frames, axis=0, ignore_index=True)


def mag_data_rotation_pam(X_windows) -> np.ndarray:
    '''
    Per-window demeaning of the magnetometer channels (3 IMUs).
    '''
    X_w = X_windows.copy()
    mag_cols = [axis + offset for offset in range(6, 27, 9) for axis in range(3)]
    for channel in mag_cols:
        mean_val = X_w[:, :, channel].mean(axis=1, keepdims=True)
        X_w[:, :, channel] -= mean_val
    return X_w


# ======================================================================================================================
#                                              REALWORLD FUNCTIONS
# Layout convention (same as OPP): sensor features first, then Label, then 'group_id' as the LAST column.
# Realworld: 3 IMUs chest, forearm, upper arm = 27 sensor columns or 2 IMUs chest, forearm = 18 sensor columns.
# ======================================================================================================================

def load_realworld(files_path, add_group_id = False) -> pd.DataFrame:
      '''
      Read multiple .dat files from the RealWorld dataset and vertically concatenate them into a single DataFrame.
      If add_group_id=True, append a last column called 'group_id' with the subject id: e.g. 1, 2, ..., 15.
      '''
      dataframes = []
      row_counts = 0
  
      for file in files_path:                                                                                                                                                                                                
          df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')
          df = df.astype(np.float32)
          if add_group_id:
              subject_id = int(Path(file).stem.split('_')[0].removeprefix('subject'))   # subject3_realworld -> 3
              df["group_id"] = subject_id         # added as LAST column

          dataframes.append(df)
          row_counts += len(df)

      complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

      assert len(complete_dataframe) == row_counts, (
          f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}")
      return complete_dataframe

def filter_loco_rw(dataframe: pd.DataFrame, keep_timestamp: bool = False, keep_labels = (1, 2, 3, 4)) -> pd.DataFrame:
    '''
    RealWorld: reorder the raw .dat layout (timestamp | activityID | sensors | group_id) into the pipeline layout
    (0..n-1 sensors | 'ts' | n = label | 'group_id'). Label filtering is a no-op on these files (labels are 1..4 only).
    '''
    df = dataframe.copy()
    group_id = df.pop("group_id") if "group_id" in df.columns else None
    if keep_labels is not None:
        keep = df.iloc[:, 1].isin(list(keep_labels))                                                                                                                                                                       
        df = df.loc[keep]
        if group_id is not None:
            group_id = group_id.loc[keep]
    ts, label = df.iloc[:, 0].copy(), df.iloc[:, 1].copy()
    df = df.drop(columns=[0, 1])
    n = df.shape[1]                                                                                                                                                                                                        
    df.columns = list(range(n))
    if keep_timestamp:
        df["ts"] = ts.values
    df[n] = label.values.astype(int)
    if group_id is not None:                                                                                                                                                                                               
        df["group_id"] = group_id.values
    return df

def resample_rw(dataframe: pd.DataFrame, n_features: int, up: int = 3, down: int = 5, period: float = 0.02, min_raw_len: int = 150, verbose: bool = False) -> pd.DataFrame:
    '''
    Resample RealWorld from 50 Hz to exactly 30 Hz with scipy.signal.resample_poly (FIR anti-aliasing) BEFORE sliding windows.
    n_features = 9 * num_sensors (18 = chest+forearm, 27 = chest+forearm+upperarm); the label column is named n_features.

    Input layout : 0..n_features-1 sensor features | 'ts' (seconds) | n_features (Label) | 'group_id' (subject)
    Output layout: 0..n_features-1 sensor features | n_features (Label) | group_id

    - Resampling is done per contiguous segment (rows whose timestamps are consecutive at the raw period 0.02 s): the filter
      never blends across a removed dropout (timestamp jump) or a recording restart (timestamp back to 0).
    - Segments shorter than min_raw_len raw samples (default 150 = 3 s = one window after resampling) are dropped.
    '''
    assert "ts" in dataframe.columns and "group_id" in dataframe.columns, "resample_rw expects 'ts' and 'group_id' columns"
    label_col = n_features
    feature_cols = list(range(n_features))
    assert all(c in dataframe.columns for c in feature_cols) and label_col in dataframe.columns, \
        f"resample_rw expects sensor columns 0..{n_features-1} and label column {n_features}, got {list(dataframe.columns)[:5]}..."
    assert len([c for c in dataframe.columns if c not in ("ts", "group_id")]) == n_features + 1, \
        f"expected {n_features} sensor columns + 1 label, got {len(dataframe.columns) - 2}"

    out_frames = []
    for subject, subject_df in dataframe.groupby("group_id", sort=False):
        subject_df = subject_df.reset_index(drop=True)
        ts = subject_df["ts"].to_numpy(dtype=np.float64)
        X = subject_df.loc[:, feature_cols].to_numpy(dtype=np.float64)
        y = subject_df[label_col].to_numpy()

        # contiguous segments: break where the timestamp jumps by more than 1.5 raw periods (or goes backwards)
        dt = np.diff(ts)
        breaks = np.where((dt > 1.5 * period) | (dt <= 0))[0] + 1
        bounds = np.concatenate([[0], breaks, [len(subject_df)]])

        n_raw_kept, n_res, n_seg_kept, n_seg_dropped = 0, 0, 0, 0
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            seg_len = e - s
            if seg_len < min_raw_len:
                n_seg_dropped += 1
                continue
            n_blocks = seg_len // down
            s_e = s + n_blocks * down                        # truncate to a multiple of 'down'
            Xr = resample_poly(X[s:s_e], up, down, axis=0)  # [n_blocks*up, n_features]
            assert Xr.shape[0] == n_blocks * up, f"resample_poly length mismatch {Xr.shape[0]} vs {n_blocks * up}"
            y_blocks = y[s:s_e].reshape(n_blocks, down)
            y_major = np.array([Counter(b.tolist()).most_common(1)[0][0] for b in y_blocks], dtype=np.int64)
            yr = np.repeat(y_major, up)

            seg_df = pd.DataFrame(Xr.astype(np.float32), columns=feature_cols)
            seg_df[label_col] = yr
            seg_df["group_id"] = f"{subject}-s{n_seg_kept}"
            out_frames.append(seg_df)
            n_seg_kept += 1
            n_raw_kept += (s_e - s)
            n_res += Xr.shape[0]

        if verbose:
            ratio = n_res / n_raw_kept if n_raw_kept else float("nan")
            print(f"[resample_rw] subject {subject}: raw rows {len(subject_df)} | kept raw {n_raw_kept} in {n_seg_kept} segments "
                  f"(dropped {n_seg_dropped} short segments) | resampled rows {n_res} | ratio {ratio:.4f} (target {up/down:.4f})")

    if len(out_frames) == 0:
        raise ValueError("resample_rw produced no data")
    return pd.concat(out_frames, axis=0, ignore_index=True)

def acc_data_scaling_rw (dataframe: pd.DataFrame, n_features: int) -> pd.DataFrame:
    '''
    RealWorld acceleration is in m/s^2. Divide by 9.81 to convert to g (OPP acc is converted to g as well).
    Assumes 9 sensor columns per device (acc xyz, gyro xyz, mag xyz), acc first, sensor columns starting at 0.
    '''
    dataframe_ = dataframe.copy()
    acc_cols = [axi + offset for offset in range(0, n_features, 9) for axi in range(3)]   # n_features = 9 * num_sensors: sensor columns only
    dataframe_.iloc[:, acc_cols] /= 9.81
    return dataframe_

def mag_data_norm_rw(dataframe: pd.DataFrame, n_features: int, eps: float = 1e-8) -> pd.DataFrame:
    '''
    Normalize each magnetometer 3D vector by its magnitude: removes absolute field strength, keeps direction.
    '''
    dataframe_ = dataframe.copy()
    for offset in range(6, n_features, 9):
        mag_x = dataframe_.iloc[:, offset]
        mag_y = dataframe_.iloc[:, offset + 1]
        mag_z = dataframe_.iloc[:, offset + 2]

        magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        magnitude = magnitude.clip(lower = eps)

        for axi in range(3):
            dataframe_.iloc[:, axi + offset] = dataframe_.iloc[:, axi + offset] / magnitude
    return dataframe_

def mag_data_rotation_rw(X_windows) -> np.ndarray:
    '''
    Per-window demeaning of the magnetometer channels (3 IMUs).
    '''
    X_w = X_windows.copy()
    mag_cols = [axis + offset for offset in range(6, X_windows.shape[2], 9) for axis in range(3)]
    for channel in mag_cols:
        mean_val = X_w[:, :, channel].mean(axis=1, keepdims=True)
        X_w[:, :, channel] -= mean_val
    return X_w

# ======================================================================================================================
#                                              REALDISP FUNCTIONS
# Layout convention (it can have OPP or PAM configuration, 5 and 3 sensors respectively): sensor features first, then Label, then 'group_id' as the LAST column.
# Realworld: 5 IMUs RLA, RUA, BACK, LUA, LLA = 45 sensor columns or 3 IMUs CHEST(BACK PAM), RLA(HAND PAM), ANKLE(RC) = 18 sensor columns.
# ======================================================================================================================


def load_realdisp(files_path, add_group_id = False) -> pd.DataFrame:
      '''
      Read multiple .log files from the RealDisp dataset and vertically concatenate them into a single DataFrame.
      If add_group_id=True, append a last column called 'group_id' with the subject id: e.g. 1, 2, ..., 17.
      '''
      dataframes = []
      row_counts = 0
  
      for file in files_path:                                                                                                                                                                                                
          df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')
          df = df.astype(np.float32)
          if add_group_id:
              subject_id = int(Path(file).stem.split('_')[0].removeprefix('subject'))   # subject3_realworld -> 3
              df["group_id"] = subject_id         # added as LAST column

          dataframes.append(df)
          row_counts += len(df)

      complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

      assert len(complete_dataframe) == row_counts, (
          f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}")
      return complete_dataframe

def filter_realdisp(dataframe: pd.DataFrame, keep_timestamp: bool = False, keep_sensors: int = 6 ) -> pd.DataFrame:
    '''
    Filter RealDisp data for OPP/PAM setup.
    Raw layout: 0 = timestamp (s), 1 = timestamp (us), 2..118 = 9 sensors x 13 cols (acc, gyro, mag, quat), 119 = label
    Sensor block starts: RLA 2, RUA 15, BACK 28, LUA 41, LLA 54, RC 67, RT 80, LT 93, LC 106 (acc/gyro/mag = first 9 cols of each block)
    - keep_sensors (int) must be 6, 5 or 3 (9 = all sensors is NOT implemented; ValueError otherwise). 6 keeps BACK, RUA, RLA, LUA, LLA, RC (54 cols),
      5 keeps the OPP layout BACK, RUA, RLA, LUA, LLA (45 cols), 3 keeps the PAM layout RLA (hand), BACK (chest), RC (ankle) (27 cols)
    Output columns: 0..n-1 sensor features, [ 'ts' if keep_timestamp ], n (Label), [ 'group_id' ]
    '''
    dataset = dataframe.copy()

    group_id = None
    if "group_id" in dataset.columns:
        group_id = dataset["group_id"].copy()
        dataset = dataset.drop(columns=["group_id"])

    label = dataset.iloc[:, 119].copy()                                      # label column (taken BEFORE slicing sensors)
    timestamp = dataset.iloc[:, 0].copy() + dataset.iloc[:, 1].copy() / 1e6  # timestamp in seconds (taken BEFORE slicing sensors)

    if keep_sensors == 6: # keep 6 sensors: BACK, RUA, RLA, LUA, LLA, RC (OPP layout + right calf = union of OPP and PAM)
        dataset = pd.concat([dataset.iloc[:, 28:37], dataset.iloc[:, 15:24], dataset.iloc[:, 2:11], dataset.iloc[:, 41:50], dataset.iloc[:, 54:63], dataset.iloc[:, 67:76]], axis=1)
        expected_cols = 54
    elif keep_sensors == 5: # keep 5 sensors: BACK, RUA, RLA, LUA, LLA
        dataset = pd.concat([dataset.iloc[:, 28:37], dataset.iloc[:, 15:24], dataset.iloc[:, 2:11], dataset.iloc[:, 41:50], dataset.iloc[:, 54:63]], axis=1)
        expected_cols = 45
    elif keep_sensors == 3: # keep 3 sensors: RLA (hand), BACK (chest), RC (ankle)
        dataset = pd.concat([dataset.iloc[:, 2:11], dataset.iloc[:, 28:37], dataset.iloc[:, 67:76]], axis=1)
        expected_cols = 27
    else:
        raise ValueError(f"keep_sensors must be 6, 5 or 3; got {keep_sensors}")

    assert dataset.shape[1] == expected_cols, f"RealDisp {keep_sensors} subset must have {expected_cols} columns, got {dataset.shape[1]}"

    n = dataset.shape[1]
    dataset.columns = range(n)                        # reset feature cols -> 0..n-1
    if keep_timestamp:
        dataset["ts"] = timestamp.values              # timestamp (seconds) right before the label
    dataset[n] = label.values.astype(np.int64)        # label column
    if group_id is not None:
        dataset["group_id"] = group_id.values         # group_id added to LAST col
    return dataset

