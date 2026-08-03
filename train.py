from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra

from tqdm import tqdm

from data import (
    SpeechDatasetDual,
    collate_fn_dual,
    load_metadata,
    split_signals_by_prompt,
)
from model import TransformerRegressorDual


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_rmse(y_true, y_pred):
    y_true_100 = y_true * 100
    y_pred_100 = y_pred * 100
    mse = torch.mean((y_true_100 - y_pred_100) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def batch_pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    vx = preds - preds.mean()
    vy = targets - targets.mean()
    eps = 1e-8
    corr = (vx * vy).sum() / (
        torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + eps
    )
    return corr


def build_model(cfg):
    return TransformerRegressorDual(
        input_dim=len(cfg.data.feature_cols),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        n_hearing=cfg.model.n_hearing,
    )


def build_dataloaders(cfg):
    if isinstance(cfg.datasets, list):
        raise NotImplementedError("Currently only accepting on a single dataset")

    lpath = cfg.data.csv_path.format(dataset=cfg.datasets, subset="train", side="l")
    rpath = cfg.data.csv_path.format(dataset=cfg.datasets, subset="train", side="r")

    df_l, df_r = load_metadata(lpath, rpath)
    train_signals, val_signals = split_signals_by_prompt(
        df_l, cfg.train.val_split, cfg.seed
    )

    train_dataset = SpeechDatasetDual(
        df_l, df_r, train_signals, feature_cols=cfg.data.feature_cols, train=True
    )
    val_dataset = SpeechDatasetDual(
        df_l, df_r, val_signals, feature_cols=cfg.data.feature_cols, train=True
    )

    if cfg.debug:
        train_dataset.signals = train_dataset.signals[:10]
        val_dataset.signals = val_dataset.signals[:10]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn_dual,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_dual,
    )
    return train_loader, val_loader


def train_one_epoch(
    model, dataloader, optimizer, device, corr_lambda: float, use_tqdm: bool
):
    model.train()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0

    if use_tqdm:
        dataloader = tqdm(dataloader, desc="Training")

    for (
        feats_l,
        mask_l,
        feats_r,
        mask_r,
        hearing_labels,
        targets,
        _signal_ids,
    ) in dataloader:
        feats_l, mask_l = feats_l.to(device), mask_l.to(device)
        feats_r, mask_r = feats_r.to(device), mask_r.to(device)
        hearing_labels, targets = hearing_labels.to(device), targets.to(device)

        preds = model(feats_l, mask_l, feats_r, mask_r, hearing_labels)
        mse = mse_loss_fn(preds, targets)
        corr = batch_pearson_corr(preds, targets)
        loss = mse + corr_lambda * (1.0 - corr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        n += targets.size(0)

    return total_loss / n


def evaluate(model, dataloader, data_subset, device, use_tqdm):
    model.eval()
    y_true_all, y_pred_all = [], []

    if use_tqdm:
        dataloader = tqdm(dataloader, desc=f"Evaluating {data_subset}")

    with torch.no_grad():
        for (
            feats_l,
            mask_l,
            feats_r,
            mask_r,
            hearing_labels,
            targets,
            _signal_ids,
        ) in dataloader:
            feats_l, mask_l = feats_l.to(device), mask_l.to(device)
            feats_r, mask_r = feats_r.to(device), mask_r.to(device)
            hearing_labels, targets = hearing_labels.to(device), targets.to(device)
            preds = model(feats_l, mask_l, feats_r, mask_r, hearing_labels)
            y_true_all.append(targets.cpu())
            y_pred_all.append(preds.cpu())
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    return compute_rmse(y_true_all, y_pred_all)


def save_train_log(log, path: Path):
    df = pd.DataFrame(log, columns=["epoch", "train_loss", "train_rmse", "val_rmse"])
    df.to_csv(path, index=False)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)

    train_loader, val_loader = build_dataloaders(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    best_val_rmse = float("inf")
    train_log = []
    best_model_path = save_dir / "model.pt"
    train_log_path = save_dir / "train_log.csv"

    use_tqdm = cfg.debug or cfg.device == "cpu"

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.train.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg.train.corr_lambda, use_tqdm
        )
        train_rmse = evaluate(model, train_loader, "train", device, use_tqdm)
        val_rmse = evaluate(model, val_loader, "valid", device, use_tqdm)
        print(f"Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.4f}")
        print(f"Val RMSE:   {val_rmse:.4f}")

        train_log.append([epoch, train_loss, train_rmse, val_rmse])

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            print(
                f"--> Saved new best model at epoch {epoch}, Val RMSE: {val_rmse:.4f}"
            )

        save_train_log(train_log, train_log_path)


if __name__ == "__main__":
    main()
