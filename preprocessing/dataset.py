import pandas as pd
import numpy as np
import torch
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, Subset

from utils import set_seed

set_seed(42)

class SlidingWindowDataset(Dataset):
    """
    Builds sliding windows for time series prediction.
    Supports configurable lags and roll-forward updates.
    """

    def __init__(self, df: pd.DataFrame, window_size: int, horizon: int,
                 features: List[str], target: str, lags: List[int]):
        df = df.sort_values("day").reset_index(drop=True)
        self.df_orig = df.copy()   # keep original for DOY and holidays
        self.df = df.copy()
        self.features = features
        self.target = target
        self.window_size = window_size
        self.horizon = horizon
        self.lags = sorted(lags)

        # keep buffer of observed + predicted target values
        self.target_buffer = list(self.df[target].values)

        self._build_windows()

    def _build_windows(self):
        vals = self.df[self.features].to_numpy().astype(np.float32)
        tgt  = self.df[[self.target]].to_numpy().reshape(-1).astype(np.float32)
        days = self.df["day"].to_numpy()
        n = len(self.df)

        X_list, y_list, tstart = [], [], []
        for i in range(n - self.window_size - self.horizon + 1):
            X_list.append(vals[i:i + self.window_size])
            y_list.append(tgt[i + self.window_size: i + self.window_size + self.horizon])
            tstart.append(days[i + self.window_size])

        self.X = torch.tensor(np.asarray(X_list), dtype=torch.float32)   # (N, T, F)
        self.y = torch.tensor(np.asarray(y_list), dtype=torch.float32)   # (N, H)
        self.t_start = np.asarray(tstart)                                # (N,)

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

    def roll_forward(self, preds: np.ndarray, shift: int = 1):
        """
        Roll forward by `shift` steps (days).
        Update target buffer, lag features, and DOY features accordingly.
        preds: model predictions (can be longer than shift, but at least shift long).
        """
        if preds.ndim > 1:
            preds = preds.reshape(-1)

        if len(preds) < shift:
            raise ValueError(f"Need at least {shift} predictions, got {len(preds)}")

        preds = preds[:shift]  # only use first `shift` predictions

        # extend buffer
        self.target_buffer.extend(preds.tolist())

        # update df with new target + lag features
        last_idx = len(self.df) - 1
        for step in range(shift):
            new_idx = last_idx + step + 1
            if new_idx >= len(self.df_orig):
                break  # stop if beyond true_y range

            row = self.df_orig.iloc[new_idx].copy()
            row[self.target] = self.target_buffer[new_idx]

            # update lag features from buffer
            for lag in self.lags:
                if new_idx - lag >= 0:
                    row[f"lag_{lag}"] = self.target_buffer[new_idx - lag]
                else:
                    row[f"lag_{lag}"] = np.nan

            self.df = pd.concat([self.df, row.to_frame().T], ignore_index=True)

        # trim past rows so dataset advances in time
        self.df = self.df.iloc[shift:].reset_index(drop=True)

        # rebuild windows
        self._build_windows()


##############################################
# preproess features

def preprocess_df(
    df_group: pd.DataFrame,
    target: str,
    lags: List[int],
    features: List[str],
    val_ratio: float,
    window_size: int,
    horizon: int,
    scaler_choice: str = "Std",
):
    """
    Preprocessing pipeline for training.
    """

    # Sort and reset index
    dfb = df_group.sort_values("day").reset_index(drop=True)

    # --- Inline build_threshold logic ---
    n = len(dfb)
    split_row = int(n * (1 - val_ratio))
    split_row = min(max(split_row, 0), n - 1)
    threshold = dfb.sort_values("day")["day"].iloc[split_row]

    # Fit scalers + get final feature set
    tscaler = fit_scalers_on_training_data(
        dfb, target, threshold, scaler_choice
    )

    # Apply transformations
    dfb_scaled, final_features = apply_transformations(
        dfb, target, lags, features, tscaler
    )

    # Build dataset
    full_ds = SlidingWindowDataset(dfb_scaled, window_size, horizon, final_features, target, lags)

    # Split train/val
    train_idx = [i for i, t in enumerate(full_ds.t_start) if t < threshold]
    val_idx   = [i for i, t in enumerate(full_ds.t_start) if t >= threshold]
    train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds, val_idx)

    return train_ds, val_ds, tscaler, final_features

############################################################

def fit_scalers_on_training_data(
    df_group: pd.DataFrame,
    target: str,
    threshold: pd.Timestamp,
    scaler_choice: str = "Std",
):
    """
    Fit feature and target scalers on training partition only.
    Returns fitted scalers and final feature list.
    """
    # restrict to training rows for fitting scalers
    df_train = df_group[df_group["day"] < threshold].sort_values("day").copy()

    # Fit target scaler based on choice
    if scaler_choice == "Std":
        tgt_scaler = StandardScaler()
    elif scaler_choice == "MinMax":
        tgt_scaler = MinMaxScaler()
    else:  # None
        tgt_scaler = None

    if tgt_scaler is not None:
        tgt_scaler.fit(df_train[[target]])

    return tgt_scaler

############################################################
# engineering the features

def apply_transformations(
    df_group: pd.DataFrame,
    target: str,
    lags: List[int],
    features: List[str],
    tscaler,
):
    """
    Apply lag features, time features and scaling using already fitted scalers.
    Returns transformed dataframe and final feature list.
    """
    
    dfb = df_group.sort_values("day").copy()

    # ---First, Apply scaling if scaler is provided ---
    if tscaler is not None:
        dfb[target] = tscaler.transform(dfb[[target]])

    # --- Add lag features directly ---
    for lag in lags:
        dfb[f"lag_{lag}"] = dfb[target].shift(lag)
    dfb = dfb.dropna().reset_index(drop=True)

    # Create features list that includes lag features
    lag_features = [f"lag_{lag}" for lag in lags]
    all_features = features + lag_features

    # --- Add time-based features directly ---
    new_features = all_features.copy()

    if "DOY" in all_features:
        dfb["DOY_sin"] = np.sin(2 * np.pi * dfb["DOY"] / 365)
        dfb["DOY_cos"] = np.cos(2 * np.pi * dfb["DOY"] / 365)
        new_features.remove("DOY")
        new_features.extend(["DOY_sin", "DOY_cos"])

    if "day_of_week" in all_features:
        dummies = pd.get_dummies(dfb["day_of_week"], prefix="dow", dtype=np.uint8)
        dfb = pd.concat([dfb, dummies], axis=1)
        new_features.remove("day_of_week")
        new_features.extend(dummies.columns.tolist())

    return dfb, new_features


