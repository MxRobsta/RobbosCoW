"""
Compute STOI/eSTOI/PESQ and DNSMOS metrics for metadata CSVs, using a single YAML config.

Usage:
  python compute_metrics.py --config metrics_config.yaml
  # optional filters:
  python compute_metrics.py --config metrics_config.yaml --datasets signal_eval_left signal_eval_right --quality --dns
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample, resample_poly
from tqdm import tqdm
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from pystoi import stoi as stoi_metric
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'pystoi'. Install it with `pip install pystoi`."
    ) from exc

try:
    from pesq import pesq as pesq_metric
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'pesq'. Install it with `pip install pesq`."
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'onnxruntime'. Install it with `pip install onnxruntime`."
    ) from exc

TARGET_SAMPLE_RATE = 16_000
MIN_REQUIRED_SAMPLES = TARGET_SAMPLE_RATE // 4
QUALITY_COLUMNS = ["stoi", "estoi", "pesq"]
DNS_COLUMNS = ["dns_sig", "dns_bak", "dns_ovrl"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    degraded_dir: Optional[Path]
    reference_dir: Optional[Path]
    audio_dir: Optional[Path]
    channel: str  # "left" or "right"


def load_audio_poly(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path.as_posix(), always_2d=True)
    audio = audio.T  # (channels, samples)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down, axis=1)
    return audio.astype(np.float32)


def load_audio_resample(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path.as_posix())
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    if sr != target_sr:
        n_samples = int(audio.shape[1] * (target_sr / sr))
        audio = resample(audio, n_samples, axis=1)
    return audio.astype(np.float32)


def _safe_stoi(
    clean: np.ndarray, degraded: Optional[np.ndarray], sr: int, extended: bool = False
) -> Optional[float]:
    if degraded is None:
        return None
    try:
        return float(stoi_metric(clean, degraded, sr, extended=extended))
    except Exception:
        return None


def _safe_pesq(
    clean: np.ndarray, degraded: Optional[np.ndarray], sr: int
) -> Optional[float]:
    if degraded is None:
        return None
    mode = "wb" if sr >= 16_000 else "nb"
    try:
        return float(pesq_metric(sr, clean, degraded, mode))
    except Exception:
        return None


def compute_quality(
    signal_id: str, degraded_dir: Path, reference_dir: Path, channel: str
) -> Dict[str, Optional[float]]:
    metrics = {column: None for column in QUALITY_COLUMNS}
    degraded_path = degraded_dir / f"{signal_id}.flac"
    reference_path = reference_dir / f"{signal_id}_unproc.flac"

    if not degraded_path.exists():
        alt = degraded_dir / f"{signal_id}_unproc.flac"
        if alt.exists():
            degraded_path = alt
        else:
            return metrics

    if not reference_path.exists():
        alt_ref = reference_dir / f"{signal_id}.flac"
        if alt_ref.exists():
            reference_path = alt_ref
        else:
            return metrics

    try:
        degraded_wave = load_audio_poly(degraded_path)
        reference_wave = load_audio_poly(reference_path)
    except Exception:
        return metrics

    min_len = min(degraded_wave.shape[-1], reference_wave.shape[-1])
    if min_len < MIN_REQUIRED_SAMPLES:
        return metrics

    degraded_wave = degraded_wave[:, :min_len]
    reference_wave = reference_wave[:, :min_len]
    reference_mono = reference_wave.mean(axis=0, keepdims=True)

    clean_np = np.ascontiguousarray(reference_mono.squeeze(0), dtype=np.float32)
    channel_idx = 0 if channel == "left" else 1
    if channel_idx >= degraded_wave.shape[0]:
        return metrics

    degraded_np = np.ascontiguousarray(degraded_wave[channel_idx], dtype=np.float32)

    metrics["stoi"] = _safe_stoi(clean_np, degraded_np, TARGET_SAMPLE_RATE)
    metrics["estoi"] = _safe_stoi(
        clean_np, degraded_np, TARGET_SAMPLE_RATE, extended=True
    )
    metrics["pesq"] = _safe_pesq(clean_np, degraded_np, TARGET_SAMPLE_RATE)
    return metrics


class DNSMOS:
    def __init__(self, onnx_path: Path):
        if not onnx_path.exists():
            raise FileNotFoundError(f"DNSMOS ONNX model not found: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path.as_posix())
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.expected_len: Optional[int] = None
        if len(input_shape) == 2 and isinstance(input_shape[1], int):
            self.expected_len = input_shape[1]

    def __call__(self, audio_16k: np.ndarray) -> np.ndarray:
        batch = self._prepare_batches(audio_16k)
        mos = self.session.run(None, {self.input_name: batch})[0]
        if mos.ndim == 2:
            return mos.mean(axis=0)
        return mos[0]

    def _prepare_batches(self, audio_16k: np.ndarray) -> np.ndarray:
        if self.expected_len is None or self.expected_len <= 0:
            return audio_16k[np.newaxis, :]
        seg_len = self.expected_len
        total = audio_16k.shape[0]
        if total <= seg_len:
            padded = np.zeros(seg_len, dtype=np.float32)
            padded[:total] = audio_16k
            return padded[np.newaxis, :]
        num_segments = int(np.ceil(total / seg_len))
        segments = []
        for idx in range(num_segments):
            start = idx * seg_len
            end = min(start + seg_len, total)
            chunk = np.zeros(seg_len, dtype=np.float32)
            chunk[: end - start] = audio_16k[start:end]
            segments.append(chunk)
        return np.stack(segments, axis=0)


def resolve_audio_path(base_dir: Path, signal_id: str) -> Optional[Path]:
    primary = base_dir / f"{signal_id}.flac"
    if primary.exists():
        return primary
    alt = base_dir / f"{signal_id}_unproc.flac"
    if alt.exists():
        return alt
    return None


def compute_dnsmos(
    signal_id: str, audio_dir: Path, channel: str, model: DNSMOS
) -> Dict[str, Optional[float]]:
    metrics = {col: None for col in DNS_COLUMNS}
    audio_path = resolve_audio_path(audio_dir, signal_id)
    if audio_path is None:
        return metrics

    try:
        audio = load_audio_resample(audio_path)
    except Exception:
        return metrics

    channel_idx = 0 if channel == "left" else 1
    if channel_idx >= audio.shape[0]:
        channel_idx = 0
    waveform = audio[channel_idx]
    if waveform.shape[-1] < MIN_REQUIRED_SAMPLES:
        return metrics

    try:
        sig, bak, ovrl = model(waveform)
    except Exception:
        return metrics

    metrics["dns_sig"] = float(sig)
    metrics["dns_bak"] = float(bak)
    metrics["dns_ovrl"] = float(ovrl)
    return metrics


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def update_quality(config: DatasetConfig) -> None:
    if config.degraded_dir is None or config.reference_dir is None:
        print(
            f"[SKIP] {config.name}: missing degraded/reference paths for quality metrics"
        )
        return
    if not config.csv_path.exists():
        print(f"[SKIP] {config.name}: missing CSV {config.csv_path}")
        return
    if not config.degraded_dir.exists() or not config.reference_dir.exists():
        print(f"[SKIP] {config.name}: missing audio dir(s) for quality metrics")
        return

    df = pd.read_csv(config.csv_path)
    if "signal" not in df.columns:
        print(f"[SKIP] {config.name}: 'signal' column not found")
        return
    ensure_columns(df, QUALITY_COLUMNS)

    signal_ids = df["signal"].dropna().unique()
    if len(signal_ids) == 0:
        print(f"[SKIP] {config.name}: no signals")
        return

    print(
        f"[INFO] {config.name}: computing STOI/eSTOI/PESQ for {len(signal_ids)} signals"
    )
    for signal_id in tqdm(signal_ids, desc=f"{config.name} quality"):
        metrics = compute_quality(
            signal_id, config.degraded_dir, config.reference_dir, config.channel
        )
        mask = df["signal"] == signal_id
        for column, value in metrics.items():
            if value is not None:
                df.loc[mask, column] = value

    df.to_csv(config.csv_path, index=False)
    print(f"[DONE] {config.name}: quality metrics saved to {config.csv_path}")


def update_dns(config: DatasetConfig, model: DNSMOS) -> None:
    if config.audio_dir is None:
        print(f"[SKIP] {config.name}: missing audio_dir for DNSMOS")
        return
    if not config.csv_path.exists():
        print(f"[SKIP] {config.name}: missing CSV {config.csv_path}")
        return
    if not config.audio_dir.exists():
        print(f"[SKIP] {config.name}: missing audio dir {config.audio_dir}")
        return

    df = pd.read_csv(config.csv_path)
    if "signal" not in df.columns:
        print(f"[SKIP] {config.name}: 'signal' column not found")
        return
    ensure_columns(df, DNS_COLUMNS)

    signal_ids = df["signal"].dropna().unique()
    if len(signal_ids) == 0:
        print(f"[SKIP] {config.name}: no signals")
        return

    print(f"[INFO] {config.name}: computing DNSMOS for {len(signal_ids)} signals")
    for signal_id in tqdm(signal_ids, desc=f"{config.name} dns"):
        metrics = compute_dnsmos(signal_id, config.audio_dir, config.channel, model)
        mask = df["signal"] == signal_id
        for column, value in metrics.items():
            if value is not None:
                df.loc[mask, column] = value

    df.to_csv(config.csv_path, index=False)
    print(f"[DONE] {config.name}: DNSMOS saved to {config.csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute STOI/eSTOI/PESQ and DNSMOS and append to CSVs."
    )
    parser.add_argument(
        "--config",
        default="feature_config.yaml",
        help="Path to YAML config (metrics section).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to process (default: all from config).",
    )
    parser.add_argument(
        "--splits", help="Comma-separated splits to process (train,valid,eval)."
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Compute quality metrics (STOI/eSTOI/PESQ).",
    )
    parser.add_argument("--dns", action="store_true", help="Compute DNSMOS metrics.")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset_configs(raw_cfg: Dict) -> Dict[str, DatasetConfig]:
    datasets = {}
    for entry in raw_cfg.get("datasets", []):
        cfg = DatasetConfig(
            name=entry["name"],
            csv_path=Path(entry["csv_path"]),
            degraded_dir=(
                Path(entry["degraded_dir"]) if entry.get("degraded_dir") else None
            ),
            reference_dir=(
                Path(entry["reference_dir"]) if entry.get("reference_dir") else None
            ),
            audio_dir=Path(entry["audio_dir"]) if entry.get("audio_dir") else None,
            channel=entry.get("channel", "right"),
        )
        datasets[cfg.name] = cfg
    return datasets


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    metrics_cfg = cfg.get("metrics", cfg)
    datasets_cfg = build_dataset_configs(cfg)

    selected = (
        args.datasets
        if args.datasets
        else metrics_cfg.get("default_datasets", list(datasets_cfg.keys()))
    )
    if args.splits:
        allowed_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
        filtered = []
        for name in selected:
            ds = datasets_cfg.get(name)
            if ds is None:
                continue
            # infer split from name if not provided in config
            split = None
            if "train" in ds.name:
                split = "train"
            elif "valid" in ds.name:
                split = "valid"
            elif "eval" in ds.name:
                split = "eval"
            if split is None or split in allowed_splits:
                filtered.append(name)
        selected = filtered
    cfg_quality = metrics_cfg.get("quality", True)
    cfg_dns = metrics_cfg.get("dns", True)
    do_quality = args.quality or (not args.quality and not args.dns and cfg_quality)
    do_dns = args.dns or (not args.quality and not args.dns and cfg_dns)

    if do_dns:
        dnsmos_model_path = Path(
            metrics_cfg.get("dnsmos_model", "code/sig_bak_ovr.onnx")
        )
        dnsmos_model = DNSMOS(dnsmos_model_path)
    else:
        dnsmos_model = None

    for name in selected:
        if name not in datasets_cfg:
            raise ValueError(f"Dataset '{name}' not found in config.")
        ds = datasets_cfg[name]
        if do_quality:
            update_quality(ds)
        if do_dns and dnsmos_model is not None:
            update_dns(ds, dnsmos_model)


if __name__ == "__main__":
    main()
