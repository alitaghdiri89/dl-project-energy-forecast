import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch

from utils import set_seed

set_seed(42)

def train_model(train_ds, val_ds, input_size, horizon, config, epochs, patience, model_class):

    # Handle edge case of empty datasetsss
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"Not enough data — train={len(train_ds)}, val={len(val_ds)}")
        return None, [], []

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=min(config['batch_size'], len(train_ds)), shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(config['batch_size'], len(val_ds)),   shuffle=False, drop_last=True)

    # Model, optimizer, criterion, scheduler
    model = model_class(
        input_size=input_size,
        horizon=horizon,
        **{k: config[k] for k in model_class.REQUIRED_KEYS}
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_norm'])
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))
        train_losses.append(tr_loss)

        # Validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                va_loss += criterion(pred, y).item()
        va_loss /= max(1, len(val_loader))
        val_losses.append(va_loss)

        # Step scheduler
        scheduler.step(va_loss)

        print(f"Epoch {epoch+1:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        # Early stopping
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

# Evaluation function to get predictions and targets
def evaluate_model(model: nn.Module, ds: Dataset):
    loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=True)
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            p = model(X).cpu()
            preds.append(p)
            targets.append(y.cpu())
    preds = torch.vstack(preds)
    targets = torch.vstack(targets)

    return preds, targets
