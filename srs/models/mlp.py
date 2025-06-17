import locale
import os
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import calendar
import torch
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# constants that were globals in the tutor's notebook
WD_DEFAULT          = [1,2,3,4,5,6,7]
PRICE_LAGS_DEFAULT  = [1,2,7]
DA_LAG_DEFAULT      = [0]
FUEL_LAGS_DEFAULT   = [2]

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP model                                               
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

# Extended MLP
class DeepMLP(nn.Module):
    """
    Same spirit as SimpleMLP but with
      • n_hidden hidden layers (≥1)
      • Dropout after every activation
    """
    def __init__(
        self,
        input_dim : int,
        hidden_dim: int,
        output_dim: int,
        n_hidden  : int = 2,      
        dropout_p : float = 0.10,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]

        for _ in range(n_hidden - 1):
            layers += [
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ]

        layers += [nn.Dropout(dropout_p),
                   nn.Linear(hidden_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Train and evaluate the MLP for ONE forecast date              
def train_mlp(
    tensors: Dict[str, torch.Tensor],
    hidden_dim: int    = 50,
    lr: float          = 1e-3,
    weight_decay: float= 1e-3,
    batch_size: int    = 32,
    epochs: int        = 60,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits the SimpleMLP on (X_train, y_train) and returns:
      unstandardized forecast, unstandardized true target
    """
    device = device or tensors["X_train"].device
    input_dim  = tensors["X_train"].shape[1]
    output_dim = tensors["y_train"].shape[1]

    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)

    loader = DataLoader(TensorDataset(tensors["X_train"], tensors["y_train"]),
                        batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

    # predict one step
    model.eval()
    with torch.no_grad():
        pred = model(tensors["X_test"]).squeeze(0)

    # unstandardise
    unstd_pred = pred * tensors["std_y"] + tensors["mean_y"]
    unstd_true = tensors["y_test"].squeeze(0) * tensors["std_y"] + tensors["mean_y"]
    return unstd_pred.cpu(), unstd_true.cpu()

# Rolling-window MLP
def build_mlp_rolling_forecasts(
    regmat_df   : pd.DataFrame,
    dep_indices : List[int],
    window      : int,
    horizon     : int,
    start_row   : int,                       
    hidden_dim  : int,
    lr          : float,
    weight_decay: float,
    batch_size  : int,
    epochs      : int,
    device      : torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    """
    Returns (pred_tensor, true_tensor) of shape (horizon, S)
    where S = 24 hourly series.
    """
    device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regmat   = torch.tensor(regmat_df.values, dtype=torch.float32, device=device)
    dep_var  = regmat[:, dep_indices]
    regmat[:, dep_indices] = 0        

    S = dep_var.shape[1]
    preds = torch.empty((horizon, S), device=device)
    trues = torch.empty((horizon, S), device=device)

    train_rmses = [] # added Bekzod to collect train rmses
    test_rmses  = [] # added Bekzod to collect test rmses
    
    for n in range(horizon):
        idx = start_row + n

        mean_x = regmat[idx-window:idx].mean(0, keepdim=True)
        std_x  = regmat[idx-window:idx].std(0, keepdim=True)
        std_x[std_x == 0] = 1
        X_train = (regmat[idx-window:idx] - mean_x) / std_x
        X_test  = ((regmat[idx] - mean_x) / std_x).unsqueeze(0)

        mean_y = dep_var[idx-window:idx].mean(0, keepdim=True)
        std_y  = dep_var[idx-window:idx].std(0, keepdim=True)
        std_y[std_y == 0] = 1
        y_train = (dep_var[idx-window:idx] - mean_y) / std_y
        y_true  = dep_var[idx].unsqueeze(0)

        loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0
                            )

        # model = SimpleMLP(X_train.shape[1], hidden_dim, S).to(device) #<- tutor's one
        model = DeepMLP(
            input_dim  = X_train.shape[1],
            hidden_dim = hidden_dim,
            output_dim = y_train.shape[1],
            n_hidden   = 2,       
            dropout_p  = 0.10,
        ).to(device)
        loss_fn  = nn.MSELoss()
        opt      = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_std = model(X_test).squeeze(0)
            pred_train_std = model(X_train)
            train_pred = pred_train_std * std_y + mean_y # <- added Bekzod
            train_rmse_day = torch.sqrt(((train_pred - (dep_var[idx-window:idx]))**2).mean()) # <- added Bekzod
            train_rmses.append(train_rmse_day.item())
            
        pred = pred_std * std_y + mean_y         

        test_rmse_day = torch.sqrt(((pred - y_true.squeeze(0))**2).mean()) # <- added Bekzod
        test_rmses.append(test_rmse_day.item()) # <- added Bekzod
        
        preds[n] = pred
        trues[n] = y_true.squeeze(0)
        
    return preds.cpu(), trues.cpu(), train_rmses, test_rmses # <- added train_rmses and test_rmses


def build_mlp_rolling_forecasts_weighted_loss(
        regmat_df   : pd.DataFrame,
        dep_indices : List[int],
        window      : int,
        horizon     : int,
        start_row   : int,
        hidden_dim  : int,
        lr          : float,
        weight_decay: float,
        batch_size  : int,
        epochs      : int,
        device      : torch.device | None = None,
        alpha       : float = 0.002,          # time-decay weight
):
    """
    Rolling one-day-ahead MLP with EXPONENTIAL time-decay applied
    **only in the MSE loss**.

    Returns: preds, trues, train_rmse_series, test_rmse_series
    Shapes:  (horizon, 24), same for preds/trues
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    regmat = torch.tensor(regmat_df.values, dtype=torch.float32, device=device)
    dep_var = regmat[:, dep_indices]
    regmat[:, dep_indices] = 0       

    S      = dep_var.shape[1]
    preds  = torch.empty((horizon, S), device=device)
    trues  = torch.empty((horizon, S), device=device)
    train_rmses, test_rmses = [], []

    # pre-compute weight vector (oldest -> newest)
    w_full = torch.exp(-alpha * torch.arange(window-1, -1, -1, device=device))

    for n in range(horizon):
        idx = start_row + n

        # standardise features & targets on this window
        mean_x = regmat[idx-window:idx].mean(0, keepdim=True)
        std_x  = regmat[idx-window:idx].std(0, keepdim=True).clamp_min_(1e-9)
        X_train = (regmat[idx-window:idx] - mean_x) / std_x
        X_test  = ((regmat[idx] - mean_x) / std_x).unsqueeze(0)

        mean_y = dep_var[idx-window:idx].mean(0, keepdim=True)
        std_y  = dep_var[idx-window:idx].std(0, keepdim=True).clamp_min_(1e-9)
        y_train = (dep_var[idx-window:idx] - mean_y) / std_y
        y_true  = dep_var[idx].unsqueeze(0)

        loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=batch_size, shuffle=False, num_workers=0)

        model = DeepMLP(input_dim=X_train.shape[1],
                        hidden_dim=hidden_dim,
                        output_dim=y_train.shape[1],
                        n_hidden=2,
                        dropout_p=0.10).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # training with per-sample weighting
        for _ in range(epochs):
            model.train()
            for batch_idx, (xb, yb) in enumerate(loader):
                opt.zero_grad()

                y_pred = model(xb)

                # slice of w_full to this mini-batch
                offset   = batch_idx * xb.size(0)
                w_batch  = w_full[offset: offset + xb.size(0)].unsqueeze(1)

                loss = ((y_pred - yb) ** 2 * w_batch).mean()
                loss.backward()
                opt.step()

        # forecasts and evaluation
        model.eval()
        with torch.no_grad():
            # train RMSE
            pred_train_std = model(X_train)
            resid_train    = pred_train_std - y_train 
            train_rmse_day = torch.sqrt((resid_train**2).mean()) * std_y.mean()
            train_rmses.append(train_rmse_day.item())

            # test (next-day)
            pred_std = model(X_test).squeeze(0)
            pred     = pred_std * std_y + mean_y
            test_rmse_day = torch.sqrt(((pred - y_true.squeeze(0))**2).mean())
            test_rmses.append(test_rmse_day.item())

        preds[n] = pred
        trues[n] = y_true.squeeze(0)

    return preds.cpu(), trues.cpu(), train_rmses, test_rmses





# Optuna tuning on evaluation block                            
def tune_mlp_hyperparameters(
    regmat_df   : pd.DataFrame,
    dep_indices : List[int],
    eval_window : Tuple[int,int],     
    n_trials    : int = 50,
    device      : torch.device | None = None,
) -> Tuple[Dict[str,any], optuna.Study]:
    """
    Searches {learning_rate, window (D), hidden_dim, weight_decay}
    Returns (best_param_dict, study)
    """
    start_row, horizon = eval_window
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        lr          = trial.suggest_float("lr",      2**-16, 1e-2, log=True)
        window      = trial.suggest_int  ("D",       28, 730)
        hidden_dim  = trial.suggest_int  ("hidden",  4, 256)
        weight_decay= trial.suggest_float("wd",      2**-16, 1e-2, log=True)

        preds, trues = build_mlp_rolling_forecasts(
            regmat_df, dep_indices,
            window     = window,
            horizon    = horizon,
            start_row  = start_row,
            hidden_dim = hidden_dim,
            lr         = lr,
            weight_decay = weight_decay,
            device     = device,
            epochs     = 60,
            batch_size = 32,
        )
        rmse = torch.sqrt(((preds - trues) ** 2).mean()).item()
        return rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=7200)
    return study.best_params, study

