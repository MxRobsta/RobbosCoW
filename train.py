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
from sam import SAM
from WhiSQI.models.whisper_ni_predictors import cpcTransformer


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

    if cfg.model.name == "scow":
        return TransformerRegressorDual(
            input_dim=len(cfg.data.feature_cols),
            d_model=cfg.model.d_model,
            nhead=cfg.model.nhead,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            n_hearing=cfg.model.n_hearing,
        )
    elif cfg.model.name == "whisqi":
        return cpcTransformer(model_type="multi")
    else:
        raise NotImplementedError(f"Couldn't recognise model {cfg.name}")


def build_singledataset(
    dataset_name,
    val_split,
    csv_path,
    features,
    requires_audio,
    audio_dir,
    audio_ftype,
    seed,
    debug,
):

    lpath = csv_path.format(dataset=dataset_name, subset="train", side="left")
    rpath = csv_path.format(dataset=dataset_name, subset="train", side="right")

    df_l, df_r = load_metadata(lpath, rpath)

    if dataset_name == "cpc3":
        df_l["correctness"] /= 100
        df_r["correctness"] /= 100

    train_signals, val_signals = split_signals_by_prompt(df_l, val_split, seed)

    train_dataset = SpeechDatasetDual(
        dataset_name,
        df_l,
        df_r,
        train_signals,
        features,
        requires_audio,
        audio_dir,
        audio_ftype,
        train=True,
    )
    val_dataset = SpeechDatasetDual(
        dataset_name,
        df_l,
        df_r,
        val_signals,
        features,
        requires_audio,
        audio_dir,
        audio_ftype,
        train=True,
    )

    if debug:
        train_dataset.signals = train_dataset.signals[:10]
        val_dataset.signals = val_dataset.signals[:10]

    return train_dataset, val_dataset


def build_multiloader(
    dataset_name,
    val_split,
    csv_path,
    features,
    requires_audio,
    audio_dir,
    audio_ftype,
    batch_size,
    seed,
    debug,
):
    datasets = dataset_name.split("+")

    train, val = {}, {}
    for ds in datasets:

        if isinstance(audio_dir, str):
            # Just one dataset so a single audio dir is provided
            this_audio_dir = audio_dir
            ftype = audio_ftype
        else:
            # Multiple datasets so we have a dictionary of audio dirs for each ds
            this_audio_dir = audio_dir[ds]
            ftype = audio_ftype[ds]

        this_audio_dir = this_audio_dir.format(subset="train")

        t, v = build_singledataset(
            ds,
            val_split,
            csv_path,
            features,
            requires_audio,
            this_audio_dir,
            ftype,
            seed,
            debug,
        )
        train[ds] = t
        val[ds] = v

    if len(datasets) == 1:
        # Single dataset, just return it with a data loader
        train_loader = DataLoader(
            train[datasets[0]],
            batch_size,
            shuffle=True,
            collate_fn=collate_fn_dual,
        )
        val[datasets[0]] = DataLoader(
            val[datasets[0]],
            batch_size,
            shuffle=False,
            collate_fn=collate_fn_dual,
        )
        return train_loader, val

    # Multiple datasets - creating train
    core = train[datasets[0]]  # type: SpeechDatasetDual
    core.dataset_name = dataset_name
    core.set_hl_levels()
    for ds in datasets[1:]:
        core.df_l = pd.concat([core.df_l, train[ds].df_l])
        core.df_r = pd.concat([core.df_r, train[ds].df_r])
        core.signals += train[ds].signals

    train_loader = DataLoader(
        core, batch_size, shuffle=True, collate_fn=collate_fn_dual
    )

    for ds in datasets:
        val[ds] = DataLoader(
            val[ds],
            batch_size,
            shuffle=False,
            collate_fn=collate_fn_dual,
        )

    return train_loader, val


def get_optimizer(
    optim_name: str, model: torch.nn.Module, lr: float
) -> torch.optim.Optimizer:
    if "sam" in optim_name:
        _, base_name = optim_name.split("-")
        if base_name.lower() == "sgd":
            base = torch.optim.SGD
        else:
            raise NotImplementedError(
                f"Can't use {base_name} as base optimiser, add code here"
            )
        optimiser = SAM(model.parameters(), base, lr, momentum=0.9)
        one_step = False
    elif optim_name.lower() == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
        one_step = True

    return optimiser, one_step


def train_one_epoch(
    model,
    requires_audio,
    dataloader,
    optimiser,
    one_step,
    device,
    corr_lambda: float,
    use_tqdm: bool,
):
    model.train()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0

    y_true_all, y_pred_all = [], []

    if use_tqdm:
        dataloader = tqdm(dataloader, desc="Training")

    for (
        audio,
        feats_l,
        mask_l,
        feats_r,
        mask_r,
        hearing_labels,
        targets,
        _signal_ids,
    ) in dataloader:
        if requires_audio:
            preds = model(audio[:, 0, :].to(device))
        else:
            feats_l, mask_l = feats_l.to(device), mask_l.to(device)
            feats_r, mask_r = feats_r.to(device), mask_r.to(device)
            hearing_labels = hearing_labels.to(device)

            preds = model(feats_l, mask_l, feats_r, mask_r, hearing_labels)

        targets = targets.to(device)

        mse = mse_loss_fn(preds, targets)
        corr = batch_pearson_corr(preds, targets)
        loss = mse + corr_lambda * (1.0 - corr)

        if one_step:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        else:
            loss.backward()
            optimiser.first_step(zero_grad=True)

            if requires_audio:
                preds = model(audio[:, 0, :])
            else:
                preds = model(feats_l, mask_l, feats_r, mask_r, hearing_labels)

            mse = mse_loss_fn(preds, targets)
            corr = batch_pearson_corr(preds, targets)
            loss = mse + corr_lambda * (1.0 - corr)

            optimiser.second_step(zero_grad=True)

        total_loss += loss.item() * targets.size(0)
        n += targets.size(0)

        y_true_all.append(targets.cpu())
        y_pred_all.append(preds.cpu())

    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)

    return total_loss / n, compute_rmse(y_true_all, y_pred_all)


def evaluate(model, requires_audio, dataloader, data_subset, device, use_tqdm):
    model.eval()
    y_true_all, y_pred_all = [], []

    if use_tqdm:
        dataloader = tqdm(dataloader, desc=f"Evaluating {data_subset}")

    with torch.no_grad():
        for (
            audio,
            feats_l,
            mask_l,
            feats_r,
            mask_r,
            hearing_labels,
            targets,
            _signal_ids,
        ) in dataloader:
            if requires_audio:
                preds = model(audio[:, 0, :].to(device))
            else:
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
    requires_audio = cfg.model.requires_audio

    train_loader, val_loader = build_multiloader(
        cfg.dataset.name,
        cfg.train.val_split,
        cfg.data.csv_path,
        cfg.data.feature_cols,
        cfg.model.requires_audio,
        cfg.dataset.signal_dir,
        cfg.dataset.ftype,
        cfg.train.batch_size,
        cfg.seed,
        cfg.debug,
    )
    optimiser, one_step = get_optimizer(cfg.train.optim, model, cfg.train.lr)

    best_val_rmse = {v: float("inf") for v in val_loader.keys()}
    train_log = []
    train_log_path = save_dir / "train_log.csv"

    use_tqdm = cfg.debug or cfg.device == "cpu" or cfg.progress

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.train.epochs}")
        train_loss, train_rmse = train_one_epoch(
            model,
            requires_audio,
            train_loader,
            optimiser,
            one_step,
            device,
            cfg.train.corr_lambda,
            use_tqdm,
        )

        this_log = {"epoch": epoch, "train_loss": train_loss, "train_rmse": train_rmse}
        for ds, loader in val_loader.items():
            val_rmse = evaluate(
                model, requires_audio, loader, "valid", device, use_tqdm
            )
            this_log[f"{ds}.val_rmse"] = val_rmse

            if val_rmse < best_val_rmse[ds]:
                best_val_rmse[ds] = val_rmse
                best_model_path = save_dir / f"{ds}.model.pt"
                torch.save(model.state_dict(), best_model_path)

        train_log.append(this_log)
        pd.DataFrame(train_log).to_csv(train_log_path)

        fstring = "{:<16}: {:>6.2f} | {:<16}: {:>6.2f}"
        print(
            fstring.format(
                "Train RMSE",
                this_log["train_rmse"],
                "Train Loss",
                this_log["train_loss"],
            )
        )
        fstring = fstring.split("|")[0]
        for ds in val_loader.keys():
            print(fstring.format(f"Val {ds} RMSE", this_log[f"{ds}.val_rmse"]))


if __name__ == "__main__":
    main()
