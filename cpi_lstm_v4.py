

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor


# config stuff
CSV_PATH = "/Users/nikhilneelagaru/Desktop/fred-md.csv"
TARGET = "CPIAUCSL"

HORIZON = 1       # how many months ahead to predict
SEQ_LEN = 24      # input window in months
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15   # rest goes to test

# feature selection
VARIANCE_EPS = 1e-8
TOP_K_CORR = 60
TOP_K_RF = 30

# model
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20

# training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def load_and_transform(path, target):
    """
    loads the csv, first-differences the features for stationarity, 
    and returns everything aligned
    
    Returns
    -------
    feat_diff    : dataframe of first differenced features (no target col)
    target_level : raw CPIAUCSL levels
    dates        : dates aligned to feat_diff
    """
    
    raw = pd.read_csv(path)
    raw["sasdate"] = pd.to_datetime(raw["sasdate"])
    raw = raw.sort_values("sasdate").reset_index(drop=True)
    
    dates = raw["sasdate"]
    target_level = raw[target].astype(float)
    
    features = raw.drop(columns=["sasdate", target])
    features = features.apply(pd.to_numeric, errors="coerce")
    
    # drop all-nan cols then fill gaps
    features = features.dropna(axis=1, how="all")
    features = features.ffill().bfill()
    
    # first difference
    feat_diff = features.diff()
    
    valid_idx = feat_diff.dropna().index
    feat_diff = feat_diff.loc[valid_idx].reset_index(drop=True)
    target_level = target_level.loc[valid_idx].reset_index(drop=True)
    dates = dates.loc[valid_idx].reset_index(drop=True)
    
    return feat_diff, target_level, dates


def select_features(X_train, y_train_delta, top_k_corr=TOP_K_CORR, top_k_rf=TOP_K_RF):
    """
    3-stage feature selection, all fit on training data only so nothing leaks
    
    stage 1: drop near-constant cols by variance
    stage 2: keep top K by |pearson corr| with target delta
    stage 3: refine to top K by random forest importance
    
    Returns
    -------
    list of column names to keep
    """
    
    # stage 1
    vt = VarianceThreshold(threshold=VARIANCE_EPS)
    vt.fit(X_train.values)
    cols_stage1 = X_train.columns[vt.get_support()].tolist()
    X1 = X_train[cols_stage1]
    
    # stage 2
    corrs = X1.apply(lambda c: np.corrcoef(c.values, y_train_delta)[0, 1])
    corrs = corrs.fillna(0.0).abs().sort_values(ascending=False)
    cols_stage2 = corrs.head(min(top_k_corr, len(corrs))).index.tolist()
    X2 = X1[cols_stage2]
    
    # stage 3
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=SEED)
    rf.fit(X2.values, y_train_delta)
    importances = pd.Series(rf.feature_importances_, index=X2.columns)
    importances = importances.sort_values(ascending=False)
    cols_final = importances.head(min(top_k_rf, len(importances))).index.tolist()
    
    print(f"feature selection: {X_train.shape[1]} -> {len(cols_stage1)} (var) -> {len(cols_stage2)} (corr) -> {len(cols_final)} (RF)")
    
    return cols_final


def build_sequences(features_arr, level_arr, seq_len, horizon):
    """
    builds (X, y) pairs where X is a window ending at time t
    and y is the change in CPIAUCSL from t to t+horizon.
    features at t+horizon are never touched.
    
    Returns
    -------
    X            : (N, seq_len, F)
    y_delta      : (N,)
    anchor_levels: (N,) -- the level at t, used to recover the forecast level
    anchors      : (N,) -- the index t
    """
    
    n = len(features_arr)
    X, y, base, anchors = [], [], [], []
    
    for t in range(seq_len - 1, n - horizon):
        X.append(features_arr[t - seq_len + 1 : t + 1])
        y.append(level_arr[t + horizon] - level_arr[t])
        base.append(level_arr[t])
        anchors.append(t)
    
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(base, dtype=np.float32),
        np.asarray(anchors, dtype=np.int64),
    )


class LSTMForecaster(nn.Module):
    """
    simple LSTM with a small MLP head on the last time step
    """
    
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def evaluate(model, loader, mse, mae):
    """
    runs model in eval mode over a dataloader and returns avg mse and mae
    """
    
    model.eval()
    s_mse, s_mae, n = 0.0, 0.0, 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            bsz = yb.size(0)
            s_mse += mse(pred, yb).item() * bsz
            s_mae += mae(pred, yb).item() * bsz
            n += bsz
    
    n = max(n, 1)
    
    return s_mse / n, s_mae / n


def train_model(model, train_loader, val_loader,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """
    trains with early stopping on val MSE. restores best weights at the end.
    
    Returns
    -------
    history dict with train/val mse, rmse, mae per epoch
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    
    history = {
        "train_mse": [], "train_rmse": [], "train_mae": [],
        "val_mse": [],   "val_rmse": [],   "val_mae": [],
    }
    
    best_val = float("inf")
    best_state = None
    no_improve = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # recompute in eval mode for clean numbers
        t_mse, t_mae = evaluate(model, train_loader, mse, mae)
        v_mse, v_mae = evaluate(model, val_loader, mse, mae)
        t_rmse = float(np.sqrt(t_mse))
        v_rmse = float(np.sqrt(v_mse))
        
        history["train_mse"].append(t_mse)
        history["train_rmse"].append(t_rmse)
        history["train_mae"].append(t_mae)
        history["val_mse"].append(v_mse)
        history["val_rmse"].append(v_rmse)
        history["val_mae"].append(v_mae)
        
        improved = v_mse < best_val - 1e-6
        
        if improved:
            best_val = v_mse
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        
        flag = " *" if improved else ""
        print(f"epoch {epoch:03d} | train_mse {t_mse:.5f} | val_mse {v_mse:.5f} | val_rmse {v_rmse:.5f} | val_mae {v_mae:.5f}{flag}")
        
        if no_improve >= patience:
            print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def predict_array(model, X):
    """
    runs inference on a numpy array and returns predictions as numpy
    """
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return model(x).cpu().numpy()


def report_metrics(y_true, y_pred, label):
    """
    prints rmse / mae / mape for a set of predictions
    """
    
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100)
    
    print(f"{label:18s} RMSE {rmse:.4f} | MAE {mae:.4f} | MAPE {mape:.3f}%")
    
    return rmse, mae, mape


def plot_losses(history, path="loss_curves.png"):
    """
    saves a 2x2 grid of train/val loss curves (mse, mse log, rmse, mae)
    """
    
    epochs = np.arange(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    axes[0, 0].plot(epochs, history["train_mse"], label="train MSE")
    axes[0, 0].plot(epochs, history["val_mse"], label="val MSE")
    axes[0, 0].set_title("MSE per epoch")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MSE (standardized delta)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].semilogy(epochs, history["train_mse"], label="train MSE")
    axes[0, 1].semilogy(epochs, history["val_mse"], label="val MSE")
    axes[0, 1].set_title("MSE per epoch (log scale)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, which="both")
    
    axes[1, 0].plot(epochs, history["train_rmse"], label="train RMSE")
    axes[1, 0].plot(epochs, history["val_rmse"], label="val RMSE")
    axes[1, 0].set_title("RMSE per epoch")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(epochs, history["train_mae"], label="train MAE")
    axes[1, 1].plot(epochs, history["val_mae"], label="val MAE")
    axes[1, 1].set_title("MAE per epoch")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def plot_pred_vs_actual(dates_test, y_true_delta, y_pred_delta,
                        y_true_level, y_pred_level,
                        path="pred_vs_actual.png"):
    """
    saves a 2x2 grid: level plot, delta plot, scatter, residuals
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    
    axes[0, 0].plot(dates_test, y_true_level, label="actual CPIAUCSL")
    axes[0, 0].plot(dates_test, y_pred_level, label="predicted CPIAUCSL", alpha=0.85)
    axes[0, 0].set_title("Test set: CPIAUCSL level (actual vs predicted)")
    axes[0, 0].set_ylabel("CPIAUCSL")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(dates_test, y_true_delta, label="actual delta")
    axes[0, 1].plot(dates_test, y_pred_delta, label="predicted delta", alpha=0.85)
    axes[0, 1].set_title("Test set: h step change in CPIAUCSL")
    axes[0, 1].set_ylabel("delta CPIAUCSL")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].scatter(y_true_level, y_pred_level, s=12, alpha=0.6)
    lo = float(min(y_true_level.min(), y_pred_level.min()))
    hi = float(max(y_true_level.max(), y_pred_level.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1, 0].set_title("Predicted vs actual scatter (level)")
    axes[1, 0].set_xlabel("actual")
    axes[1, 0].set_ylabel("predicted")
    axes[1, 0].grid(alpha=0.3)
    
    residuals = y_pred_level - y_true_level
    axes[1, 1].plot(dates_test, residuals)
    axes[1, 1].axhline(0, color="k", linewidth=0.8)
    axes[1, 1].set_title("Residuals (predicted minus actual, level)")
    axes[1, 1].set_ylabel("residual")
    axes[1, 1].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    print(f"device: {DEVICE}")
    print(f"horizon: {HORIZON} month(s) ahead")
    
    # load and difference
    feat_diff, target_level, dates = load_and_transform(CSV_PATH, TARGET)
    print(f"after differencing: {len(feat_diff)} rows, {feat_diff.shape[1]} candidate features")
    
    # split sizes
    n = len(feat_diff)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    
    # build the supervised target deltas
    level_arr = target_level.values.astype(np.float32)
    y_delta_full = np.full(n, np.nan, dtype=np.float32)
    y_delta_full[: n - HORIZON] = level_arr[HORIZON:] - level_arr[: n - HORIZON]
    
    # fit feature selector only on rows whose target time is in train
    fs_mask = np.zeros(n, dtype=bool)
    fs_mask[: max(n_train - HORIZON, 0)] = True
    fs_mask &= ~np.isnan(y_delta_full)
    
    selected_cols = select_features(feat_diff.loc[fs_mask], y_delta_full[fs_mask])
    feat_diff = feat_diff[selected_cols]
    
    # standardize features using training stats only
    feat_train = feat_diff.iloc[:n_train].values
    f_mean = feat_train.mean(axis=0)
    f_std = feat_train.std(axis=0) + 1e-8
    feat_std = (feat_diff.values - f_mean) / f_std
    
    # build sequences
    X_all, y_all_raw, base_all, anchors_all = build_sequences(feat_std, level_arr, SEQ_LEN, HORIZON)
    
    # assign samples to splits based on where the target time lands
    target_times = anchors_all + HORIZON
    train_sel = target_times < n_train
    val_sel = (target_times >= n_train) & (target_times < n_train + n_val)
    test_sel = target_times >= n_train + n_val
    
    # standardize target using train samples only
    y_mean = float(y_all_raw[train_sel].mean())
    y_std = float(y_all_raw[train_sel].std() + 1e-8)
    y_all = (y_all_raw - y_mean) / y_std
    
    X_train, y_train = X_all[train_sel], y_all[train_sel]
    X_val, y_val     = X_all[val_sel],   y_all[val_sel]
    X_test, y_test   = X_all[test_sel],  y_all[test_sel]
    
    base_test = base_all[test_sel]
    anchors_test = anchors_all[test_sel]
    
    print(f"sequences: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    
    # build model
    model = LSTMForecaster(input_size=X_train.shape[2]).to(DEVICE)
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # train
    history = train_model(model, train_loader, val_loader)
    
    # predict on test set and invert standardization
    pred_std = predict_array(model, X_test)
    y_pred_delta = pred_std * y_std + y_mean
    y_true_delta = y_test  * y_std + y_mean
    
    # recover levels using the anchor (base_test = level[t], no leakage)
    y_pred_level = base_test + y_pred_delta
    y_true_level = base_test + y_true_delta
    
    # metrics
    print()
    print("test set metrics")
    report_metrics(y_true_delta, y_pred_delta, "delta CPIAUCSL")
    report_metrics(y_true_level, y_pred_level, "level CPIAUCSL")
    
    # plots (dates correspond to t + HORIZON)
    test_target_times = anchors_test + HORIZON
    dates_test = pd.to_datetime(dates.iloc[test_target_times].values)
    
    plot_losses(history, "loss_curves.png")
    plot_pred_vs_actual(dates_test, y_true_delta, y_pred_delta,
                        y_true_level, y_pred_level, "pred_vs_actual.png")
    
    plt.show()


if __name__ == "__main__":
    main()