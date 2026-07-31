import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
from tqdm import tqdm
import warnings
from transcription_scorer import SentenceScorer
import yaml
from pathlib import Path
import soundfile as sf

warnings.filterwarnings("ignore")

# -----------------------------
# Contractions
THIS_DIR = Path(__file__).resolve().parent
CONTRACTIONS_PATH = THIS_DIR / "contractions.csv"
contractions_df = pd.read_csv(
    CONTRACTIONS_PATH, sep=",", header=None, names=["short", "long"]
)
# Create a dictionary: "can't" -> ["can", "not"]
CONTRACTIONS = {
    row["short"].lower(): row["long"].lower().split()
    for _, row in contractions_df.iterrows()
}
scorer = SentenceScorer(CONTRACTIONS_PATH)


# -----------------------------
def load_audio(audio_path, target_sr=16000, channel="both"):
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        data, sr = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(data.T).float()
    else:
        waveform = waveform.float()
    if channel == "both":
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    elif channel == "left":
        waveform = waveform[0:1, :]
    elif channel == "right":
        if waveform.shape[0] > 1:
            waveform = waveform[1:2, :]
        else:
            waveform = waveform[0:1, :]
    else:
        raise ValueError(f"Unknown channel option: {channel}")

    waveform = waveform.squeeze(0)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform, target_sr


def clean_token(token):
    token = token.replace("Ġ", "")
    token = re.sub(r"[^\w']", "", token)
    return token.strip().lower()


@torch.no_grad()
def predict_word_probs(
    model, processor, audio_path, channel="both", is_english_only=False, device="cpu"
):
    audio, sr = load_audio(audio_path, channel=channel)
    inputs = processor(
        audio.numpy(), sampling_rate=sr, return_tensors="pt", return_attention_mask=True
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if is_english_only:
        generated = model.generate(
            input_features,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
        )
    else:
        generated = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="en",
            task="transcribe",
            return_dict_in_generate=True,
        )

    token_ids = generated.sequences[0]
    decoder_input_ids = token_ids[:-1].unsqueeze(0)
    labels = token_ids[1:].unsqueeze(0)

    outputs = model(
        encoder_outputs=model.get_encoder()(input_features),
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    token_probs = probs[0, torch.arange(labels.shape[1]), labels[0]].tolist()
    raw_tokens = processor.tokenizer.convert_ids_to_tokens(labels[0])

    words = []
    word_probs = []
    current_tokens = []
    current_probs = []

    for tok, p in zip(raw_tokens, token_probs):
        if tok.startswith("Ġ") and current_tokens:
            word_str = "".join(clean_token(t) for t in current_tokens)
            avg_prob = sum(current_probs) / len(current_probs)
            if word_str in CONTRACTIONS:
                for w in CONTRACTIONS[word_str]:
                    words.append(w)
                    word_probs.append(avg_prob)
            else:
                words.append(word_str)
                word_probs.append(avg_prob)
            current_tokens = []
            current_probs = []

        current_tokens.append(tok)
        current_probs.append(p)

    if current_tokens:
        word_str = "".join(clean_token(t) for t in current_tokens[:-1])
        avg_prob = sum(current_probs) / len(current_probs)
        if word_str in CONTRACTIONS:
            for w in CONTRACTIONS[word_str]:
                words.append(w)
                word_probs.append(avg_prob)
        else:
            words.append(word_str)
            word_probs.append(avg_prob)

    hypothesis = " ".join(words[1:])
    return dict(zip(words[1:], word_probs[1:])), hypothesis


def process_csv_with_model(
    csv_path, base_dir, model_name, channel="both", is_train=True
):
    print(f"\nProcessing {csv_path} with model {model_name} (channel={channel}) ...")
    df = pd.read_csv(csv_path)
    column_name = f"{model_name.split('/')[-1][8:]}"
    if column_name not in df.columns:
        df[column_name] = 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    is_english_only = model_name.endswith(".en")

    for signal, group in tqdm(df.groupby("signal"), desc=f"{model_name} signals"):
        audio_path = f"{base_dir}/{signal}.flac"
        reference = str(group["prompt"].values[0])
        word_prob_dict, hypothesis = predict_word_probs(
            model,
            processor,
            audio_path,
            channel=channel,
            is_english_only=is_english_only,
            device=device,
        )

        # update word-level probabilities
        for idx in group.index:
            word = str(df.at[idx, "word"]).lower()
            if word != "SENTENCE_SCORE":
                df.at[idx, column_name] = word_prob_dict.get(word, 0.0)

        # sentence-level score
        results = scorer.score([reference], [hypothesis])
        total_words = results.substitutions + results.deletions + results.hits
        sentence_score = results.hits / total_words if total_words > 0 else 0.0

        # handle SENTENCE_SCORE row
        score_row_idx = group.index[group["word"] == "SENTENCE_SCORE"].tolist()
        if score_row_idx:
            # update existing row
            df.at[score_row_idx[0], column_name] = sentence_score
        else:
            # create new row
            new_row = {
                "signal": signal,
                "prompt": reference,
                "hearing_loss": group["hearing_loss"].values[0],
                "word": "SENTENCE_SCORE",
                "word_count": 0,
                column_name: sentence_score,
            }
            if is_train:
                new_row.update(
                    {
                        "response": group["response"].values[0],
                        "n_words": group["n_words"].values[0],
                        "words_correct": group["words_correct"].values[0],
                        "correctness": group["correctness"].values[0],
                    }
                )
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"✅ Updated CSV with column '{column_name}'")


# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Whisper inference over metadata CSVs."
    )
    parser.add_argument(
        "--config",
        default="feature_config.yaml",
        help="Path to YAML config (whisper section).",
    )
    parser.add_argument(
        "--splits",
        help="Comma-separated splits to run (train,valid,eval). Overrides config.splits.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to run (defaults to all datasets whose split is selected).",
    )
    parser.add_argument(
        "--channel",
        choices=["left", "right"],
        help="Filter datasets by channel (left/right). If omitted, uses dataset channel.",
    )
    args = parser.parse_args()

    cfg_raw = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg = cfg_raw.get("whisper", {})
    splits = cfg.get("splits", ["eval"])
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    models = cfg["models"]
    datasets = cfg_raw.get("datasets", [])
    channel_filter = args.channel
    selected = []
    for entry in datasets:
        if entry.get("split") not in splits:
            continue
        entry_channel = entry.get("channel", "right")
        if channel_filter and entry_channel != channel_filter:
            continue
        selected.append(entry)
    if args.datasets:
        allowed = set(args.datasets)
        selected = [e for e in selected if e.get("name") in allowed]

    for entry in selected:
        csv_path = entry["csv_path"]
        signal_dir = entry["signal_dir"]
        is_train = entry.get("is_train", entry.get("split") == "train")
        channel = entry.get("channel", "right")
        for model_name in models:
            process_csv_with_model(
                csv_path, signal_dir, model_name, channel=channel, is_train=is_train
            )


if __name__ == "__main__":
    main()
