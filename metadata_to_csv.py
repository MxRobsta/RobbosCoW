import csv
import hydra
from itertools import product
import json
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Word handling rules (currently disabled; keep empty to mirror notebook behavior)
DOUBLE_WORDS: List[str] = []
EXCEPTION_WORDS: Dict[str, int] = {}
SPLIT_WORDS: Dict[str, List[str]] = {}


def process_prompt(prompt: str) -> Tuple[str, List[Tuple[str, int]], int]:
    """
    Return cleaned prompt, list of (word, count) pairs, and total count.
    """
    prompt_no_special = prompt.replace("*", "").replace(",", "")
    words: List[Tuple[str, int]] = []
    total_count = 0
    prompt_words_for_csv: List[str] = []

    for token in prompt_no_special.split():
        token_lower = token.lower()

        if token_lower in EXCEPTION_WORDS:
            cnt = EXCEPTION_WORDS[token_lower]
            norm = token.replace("\u2019", "'").lower()
            total_count += cnt
            words.append((norm, cnt))
            prompt_words_for_csv.append(norm)
            continue

        norm = token.replace("\u2019", "'").lower()
        if norm in SPLIT_WORDS:
            for sub in SPLIT_WORDS[norm]:
                words.append((sub, 1))
                total_count += 1
                prompt_words_for_csv.append(sub)
            continue

        if token_lower in DOUBLE_WORDS:
            words.append((token, 2))
            total_count += 2
            prompt_words_for_csv.append(token)
        else:
            words.append((token, 1))
            total_count += 1
            prompt_words_for_csv.append(token)

    prompt_clean = " ".join(prompt_words_for_csv)
    return prompt_clean, words, total_count


def get_severity(lid: str, listeners: dict, thresholds: dict):
    left = np.mean(listeners[lid]["audiogram_levels_l"])
    right = np.mean(listeners[lid]["audiogram_levels_r"])
    best = min(left, right)

    for th in thresholds:
        if best >= th["low"] and best < th["high"]:
            return th["name"]
    return "Severe"


def clip1(
    data: Iterable[dict],
    csv_path: Path,
    sig_key: str = "signal",
    listeners: dict = None,
    thresholds: dict = None,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "signal",
            "prompt",
            "response",
            "n_words",
            "words_correct",
            "correctness",
            "hearing_loss",
            "word",
            "word_count",
        ]
        writer.writerow(header)

        for entry in data:
            prompt_clean, words, _ = process_prompt(entry["prompt"])

            if "hearing_loss" not in entry and listeners is not None:
                _, _, _, listener = entry[sig_key].split("_")
                hearing_loss = get_severity(listener, listeners, thresholds)
            else:
                hearing_loss = entry.get("hearing_loss", "")

            for word, word_count in words:
                word_clean = word.replace("\u2019", "'")
                writer.writerow(
                    [
                        entry[sig_key],
                        prompt_clean,
                        entry.get("response", ""),
                        entry["n_words"],
                        entry.get("words_correct", ""),
                        entry.get("correctness", ""),
                        hearing_loss,
                        word_clean,
                        word_count,
                    ]
                )


def check_word_counts(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    word_count_sum = df.groupby("signal")["word_count"].sum().reset_index()
    word_count_sum = word_count_sum.rename(columns={"word_count": "word_count_sum"})
    n_words_df = df.drop_duplicates("signal")[["signal", "n_words"]]
    check_df = pd.merge(word_count_sum, n_words_df, on="signal")
    check_df["match"] = check_df["word_count_sum"] == check_df["n_words"]
    mismatch = check_df[~check_df["match"]]
    # (Optional) inspect mismatch if needed:
    if len(mismatch):
        print(mismatch.head(20))


@hydra.main(version_base=None, config_path="config", config_name="json_to_csv")
def main(cfg: DictConfig):
    # Get the splits to work on
    subsets = cfg.subsets
    if isinstance(subsets, str):
        subsets = [subsets]

    sides = cfg.sides
    if isinstance(sides, str):
        sides = [sides]

    # Iterate over all possibilities
    for subset, side in product(subsets, sides):

        # Derive base paths for left/right outputs
        csv_fpath = Path(cfg.output_fpath.format(subset=subset, side=side))
        csv_fpath.parent.mkdir(exist_ok=True, parents=True)

        if cfg.dataset.name == "clip1":
            json_path = Path(cfg.input_fpath.format(subset=subset))

            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            csv_fpath.parent.mkdir(parents=True, exist_ok=True)
            clip1(data, csv_fpath)
        elif cfg.dataset.name == "cpc3":
            if subset == "valid":
                subset = "dev"
            if subset in ["dev", "eval"]:
                json_path = Path(cfg.input_fpath.format(subset=subset + "_full"))

                with open(json_path, "r") as f:
                    data = json.load(f)

                clip1(data, csv_fpath, "signal_encoded")
            else:
                json_path = Path(cfg.input_fpath.format(subset=subset))

                with open(json_path, "r") as f:
                    data = json.load(f)

                with open(cfg.dataset.listeners_fpath, "r") as f:
                    listeners = json.load(f)

                clip1(data, csv_fpath, "signal", listeners, cfg.dataset.hl_thresholds)

        print(f"CSV saved to {csv_fpath}")

        # check_word_counts(csv_fpath)


if __name__ == "__main__":
    main()
