import argparse
import csv
import json
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


def save_to_csv(data: Iterable[dict], csv_path: Path) -> None:
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
            for word, word_count in words:
                word_clean = word.replace("\u2019", "'")
                writer.writerow(
                    [
                        entry["signal"],
                        prompt_clean,
                        entry.get("response", ""),
                        entry["n_words"],
                        entry.get("words_correct", ""),
                        entry.get("correctness", ""),
                        entry.get("hearing_loss", ""),
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata JSON to left/right CSVs."
    )
    parser.add_argument("--json", required=True, help="Path to input metadata JSON.")
    args = parser.parse_args()

    json_path = Path(args.json)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Derive base paths for left/right outputs
    base = json_path.with_suffix("")
    csv_left = base.with_name(base.name + "_l.csv")
    csv_right = base.with_name(base.name + "_r.csv")

    csv_left.parent.mkdir(parents=True, exist_ok=True)
    csv_right.parent.mkdir(parents=True, exist_ok=True)

    save_to_csv(data, csv_left)
    save_to_csv(data, csv_right)
    print(f"CSV saved to {csv_left} and {csv_right}")

    check_word_counts(csv_left)
    check_word_counts(csv_right)


if __name__ == "__main__":
    main()
