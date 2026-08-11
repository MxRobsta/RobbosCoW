from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import hydra

from data import (
    SpeechDatasetDual,
    collate_fn_dual_infer,
    collect_signals_by_prompt,
    load_metadata,
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


def build_infer_loader(dataset_name, df_l, df_r, feature_cols, batch_size):
    signals = collect_signals_by_prompt(df_l)
    dataset = SpeechDatasetDual(
        dataset_name,
        df_l=df_l,
        df_r=df_r,
        signals=signals,
        feature_cols=feature_cols,
        train=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_dual_infer
    )
    return loader


def get_split_paths(cfg, split: str):

    if split in ["train", "eval"] or (split == "valid" and cfg.dataset.name == "clip1"):
        left = cfg.data.csv_path.format(subset=split, side="left")
        right = cfg.data.csv_path.format(subset=split, side="right")
    else:
        # CPC3 data calls it "dev" not "valid"
        left = cfg.data.csv_path.format(subset="dev", side="left")
        right = cfg.data.csv_path.format(subset="dev", side="right")
    return left, right


def run_inference(model, loader, device, clamp=True, scale_to_percentage=True):
    model.eval()
    preds_scaled_list = []
    preds_out_list = []
    signal_ids_list = []
    targets_list = []

    with torch.no_grad():
        for (
            feats_l,
            mask_l,
            feats_r,
            mask_r,
            hearing_labels,
            targets,
            signal_ids,
        ) in loader:
            feats_l, mask_l = feats_l.to(device), mask_l.to(device)
            feats_r, mask_r = feats_r.to(device), mask_r.to(device)
            hearing_labels = hearing_labels.to(device)
            preds = model(feats_l, mask_l, feats_r, mask_r, hearing_labels)
            if clamp:
                preds = preds.clamp(0, 1)
            preds_scaled = preds.detach().cpu()
            preds_out = preds_scaled * 100 if scale_to_percentage else preds_scaled
            preds_scaled_list.extend(preds_scaled.tolist())
            preds_out_list.extend(preds_out.tolist())
            signal_ids_list.extend(signal_ids)
            targets_list.extend(targets.tolist())

    return preds_scaled_list, preds_out_list, signal_ids_list, targets_list


def write_predictions(path: Path, signal_ids, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"signal_ID": signal_ids, "intelligibility_score": preds}).to_csv(
        path, index=False, header=False
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )

    save_dir = Path(cfg.save_dir)
    # save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    checkpoint_path = save_dir / "model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    splits = cfg.infer.get("splits", ["valid", "eval"])
    for split in splits:
        print(f"\nRunning inference for split: {split}")
        csv_l, csv_r = get_split_paths(cfg, split)
        df_l, df_r = load_metadata(csv_l, csv_r)
        has_targets = "correctness" in df_l.columns
        loader = build_infer_loader(
            cfg.dataset.name, df_l, df_r, cfg.data.feature_cols, cfg.train.batch_size
        )

        preds_scaled, preds_out, signal_ids, targets = run_inference(
            model,
            loader,
            device,
            clamp=cfg.infer.clamp_scores,
            scale_to_percentage=cfg.infer.scale_to_percentage,
        )

        if has_targets:
            y_true = torch.tensor(targets, dtype=torch.float32)
            y_pred = torch.tensor(preds_scaled, dtype=torch.float32)
            rmse = compute_rmse(y_true, y_pred)
            corr = batch_pearson_corr(y_pred, y_true).item()
            print(f"{split} RMSE: {rmse:.4f}")
            print(f"{split} Pearson: {corr:.4f}")
        else:
            print(f"No targets available for split '{split}', skipping metrics.")

        output_name = cfg.infer.output_name.format(split=split)
        submission_path = save_dir / output_name
        write_predictions(submission_path, signal_ids, preds_out)
        print(f"Inference CSV saved: {submission_path}")


if __name__ == "__main__":
    main()
