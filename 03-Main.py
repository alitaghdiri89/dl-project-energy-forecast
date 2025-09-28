import torch
import optuna
from optuna.trial import Trial
import numpy as np

from models import LSTMModel, GRUModel, TCNModel, MLPModel
from preprocessing import load_group, preprocess_df
from training import train_model, evaluate_model

from utils import set_seed

import config as cfg

set_seed(42)

#------------------------------

EPOCHS = 100
OPTUNA_TRIALS = 100

MODEL_CLASS = MLPModel     #TCNModel #GRUModel  #LSTMModel #MLPModel

# MODE = "block" or "house"
MODE = "house"       # change here
TRAINED_KEYS = ['MAC000002', 'MAC000033', 'MAC000092', 'MAC000156', 'MAC000246', 'MAC004500', 'MAC001074', 'MAC003223']   # list of block_numbers or LCLids depending on MODE

#MODE = "block"       # change here
#TRAINED_KEYS = [0, 25, 36, 51, 61, 90, 108]   # list of block_numbers or LCLids depending on MODE

cfg.OUTMODELS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Model save
# -----------------------------
 
def save_model(obj, mode: str, key):
    model_name = MODEL_CLASS.__name__

    if mode == "block":
        model_path = cfg.OUTMODELS_DIR / f"{model_name}_block_{key}.pt"
    else:  # house
        model_path = cfg.OUTMODELS_DIR / f"{model_name}_house_{key}.pt"

    torch.save(obj, model_path)
    print(f"Saved {len(obj)} horizons for {mode} {key} → {model_path}")

def objective_factory(df_group, window_size, horizon, val_ratio):
    def objective(trial: Trial):
        
        # ---- collect trial params into one dict ----
        general_config = {
            "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "clip_norm":    trial.suggest_float("clip_norm", 0.5, 5.0, step=0.5),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
            "scaler":       trial.suggest_categorical("scaler", ["Std", "MinMax", "None"]),
        }

        model_config = MODEL_CLASS.suggest_config(trial)

        config = {**model_config, **general_config}

        train_ds, val_ds, tscaler, new_features = preprocess_df(
            df_group=df_group,
            target=cfg.TARGET,
            lags=cfg.LAGS,
            features=cfg.FEATURES,
            val_ratio=val_ratio,
            window_size=window_size,
            horizon=horizon,
            scaler_choice=config["scaler"],
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            return float("inf")

        # ---- model training ----
        model, train_losses, val_losses = train_model(
            train_ds, val_ds,
            input_size=len(new_features),
            horizon=horizon,
            config=config,
            epochs=EPOCHS,
            patience=8,
            model_class=MODEL_CLASS
        )

        if model is None:
            return float("inf")

        preds, targets = evaluate_model(model, val_ds)

        mse = torch.mean((preds - targets) ** 2).item()
        rmse = np.sqrt(mse)

        # ---- save useful objects in trial ----
        trial.set_user_attr("best_model", model)
        trial.set_user_attr("tgt_scaler", tscaler)
        trial.set_user_attr("config", config)
        trial.set_user_attr("train_loss", train_losses)
        trial.set_user_attr("val_loss", val_losses)

        return rmse
    return objective

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
 
    for key in TRAINED_KEYS:
        df_group = load_group(MODE, key, cfg.PICKLE_BLOCK, cfg.PICKLE_HOUSE)
        print(f"\n=== Optimizing {MODE.capitalize()} {key} ===")

        group_models = {}  # collect all horizons for this group

        for window_size, horizon, val_ratio in cfg.SETUPS:
            print(f"→ Window Size: {window_size}, Horizon: {horizon}, Val Ratio: {val_ratio}")

            study = optuna.create_study(direction="minimize")
            study.optimize(objective_factory(df_group, window_size, horizon, val_ratio),
                           n_trials=OPTUNA_TRIALS, n_jobs=1)

            best_trial = study.best_trial
            model = best_trial.user_attrs["best_model"]
            tscaler = best_trial.user_attrs["tgt_scaler"]
            train_loss = best_trial.user_attrs["train_loss"]
            val_loss = best_trial.user_attrs["val_loss"]

            group_models[f"horizon_{horizon}"] = {
                "model_state_dict": model.state_dict(),
                "scalers": tscaler,
                "features": cfg.FEATURES,
                "config": best_trial.params,
                "rmse": study.best_value,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            print(f"{MODE.capitalize()} {key} | Horizon {horizon} | Best RMSE {study.best_value:.4f}")
            print(f"Params: {best_trial.params}")

        save_model(group_models, MODE, key)


                