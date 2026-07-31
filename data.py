import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SpeechDatasetDual(Dataset):
    def __init__(self, df_l, df_r, signals, feature_cols, train=True):
        self.df_l = df_l
        self.df_r = df_r
        self.signals = signals
        self.feature_cols = feature_cols
        self.hearing_loss_map = {"No Loss": 0, "Mild": 1, "Moderate": 2}
        self.train = train

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal_id = self.signals[idx]
        df_signal_l = self.df_l[self.df_l["signal"] == signal_id]
        df_signal_r = self.df_r[self.df_r["signal"] == signal_id]

        def process_df(df_signal):
            sentence_row = df_signal[df_signal["word"] == "SENTENCE_SCORE"]
            word_rows = df_signal[df_signal["word"] != "SENTENCE_SCORE"]
            df_sorted = pd.concat([sentence_row, word_rows], axis=0)
            feats = torch.tensor(
                df_sorted[self.feature_cols].values, dtype=torch.float32
            )
            cls_token = torch.zeros((1, feats.shape[1]))
            hearing_token = torch.zeros((1, feats.shape[1]))
            feats = torch.cat([cls_token, hearing_token, feats], dim=0)
            return feats

        feats_l = process_df(df_signal_l)
        feats_r = process_df(df_signal_r)
        hearing_label = self.hearing_loss_map[df_signal_l["hearing_loss"].iloc[0]]

        if "correctness" in df_signal_l.columns:
            target_val = df_signal_l["correctness"].iloc[0]
        else:
            target_val = 0.0
        target = torch.tensor(target_val, dtype=torch.float32)

        return feats_l, feats_r, hearing_label, target, signal_id


def collate_fn_dual(batch):
    feats_l, feats_r, hearing_labels, targets, signal_ids = zip(*batch)
    max_len_l = max(f.shape[0] for f in feats_l)
    max_len_r = max(f.shape[0] for f in feats_r)
    dim = feats_l[0].shape[1]

    padded_l, mask_l = [], []
    padded_r, mask_r = [], []

    for f in feats_l:
        L = f.shape[0]
        pad_len = max_len_l - L
        if pad_len > 0:
            f = torch.cat([f, torch.zeros((pad_len, dim))], dim=0)
        padded_l.append(f)
        mask_l.append(
            torch.cat(
                [
                    torch.ones(L, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )

    for f in feats_r:
        L = f.shape[0]
        pad_len = max_len_r - L
        if pad_len > 0:
            f = torch.cat([f, torch.zeros((pad_len, dim))], dim=0)
        padded_r.append(f)
        mask_r.append(
            torch.cat(
                [
                    torch.ones(L, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )

    feats_l = torch.stack(padded_l)
    feats_r = torch.stack(padded_r)
    mask_l = torch.stack(mask_l)
    mask_r = torch.stack(mask_r)
    hearing_labels = torch.tensor(hearing_labels, dtype=torch.long)
    targets = torch.stack(targets)

    return feats_l, mask_l, feats_r, mask_r, hearing_labels, targets, list(signal_ids)


def collate_fn_dual_infer(batch):
    feats_l, feats_r, hearing_labels, targets, signal_ids = zip(*batch)
    max_len_l = max(f.shape[0] for f in feats_l)
    max_len_r = max(f.shape[0] for f in feats_r)
    dim = feats_l[0].shape[1]

    padded_l, mask_l = [], []
    padded_r, mask_r = [], []

    for f in feats_l:
        L = f.shape[0]
        pad_len = max_len_l - L
        if pad_len > 0:
            f = torch.cat([f, torch.zeros((pad_len, dim))], dim=0)
        padded_l.append(f)
        mask_l.append(
            torch.cat(
                [
                    torch.ones(L, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )

    for f in feats_r:
        L = f.shape[0]
        pad_len = max_len_r - L
        if pad_len > 0:
            f = torch.cat([f, torch.zeros((pad_len, dim))], dim=0)
        padded_r.append(f)
        mask_r.append(
            torch.cat(
                [
                    torch.ones(L, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )

    feats_l = torch.stack(padded_l)
    feats_r = torch.stack(padded_r)
    mask_l = torch.stack(mask_l)
    mask_r = torch.stack(mask_r)
    hearing_labels = torch.tensor(hearing_labels, dtype=torch.long)
    targets = torch.stack(targets)

    return feats_l, mask_l, feats_r, mask_r, hearing_labels, targets, list(signal_ids)


def load_metadata(csv_l: str, csv_r: str):
    df_l = pd.read_csv(csv_l).fillna(0.0)
    df_r = pd.read_csv(csv_r).fillna(0.0)
    return df_l, df_r


def split_signals_by_prompt(df_l, val_size: float, seed: int):
    prompt_to_signals = df_l.groupby("prompt")["signal"].unique().to_dict()
    all_prompts = list(prompt_to_signals.keys())
    train_prompts, val_prompts = train_test_split(
        all_prompts, test_size=val_size, random_state=seed
    )

    train_signals = []
    for prompt in train_prompts:
        train_signals.extend(prompt_to_signals[prompt])

    val_signals = []
    for prompt in val_prompts:
        val_signals.extend(prompt_to_signals[prompt])

    return train_signals, val_signals


def collect_signals_by_prompt(df):
    prompt_to_signals = df.groupby("prompt")["signal"].unique().to_dict()
    signals = []
    for prompt_signals in prompt_to_signals.values():
        signals.extend(prompt_signals)
    return signals
