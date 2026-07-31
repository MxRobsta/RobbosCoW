# 🏆 Stereo Chorus of Whispers

> **Winner Solution** of the
>[Cadenza Lyric Intelligibility Prediction (CLIP) Challenge](https://cadenzachallenge.org/docs/clip1/intro)
>at **ICASSP 2026**
>
> 🥇 **Rank: 1st Place** | [Official Results (T045)](https://cadenzachallenge.org/docs/clip1/results)

![System Architecture](figure.png)

## 🛠️ Setup

### 1. Environment Installation

```bash
git clone https://github.com/jinlongbin/Stereo-Chorus-of-Whispers.git
cd Stereo-Chorus-of-Whispers

conda create -n cadenza python=3.10
conda activate cadenza

# Install PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Install PESQ (Windows users need conda-forge)
conda install -c conda-forge pesq
```

## 🚀 Quick Start (Inference Only)

You can run inference immediately using pre-trained checkpoint
`checkpoints/final/model.pt`for validation and evaluation.

```bash
python infer.py
```

Outputs:

- `checkpoints/final/inference.csv`
- `checkpoints/final/eval_inference.csv`

## 1. Data Preparation (JSON -> CSV)

- Download the train/valid/eval data packages from
[Zenodo](https://zenodo.org/records/17789818).
- Extract them into the `data/` directory.
- **Important**: Replace the original metadata JSONs with the updated versions
from `cadenza_clip1_data.metadata.v1.2.tar.gz`.

**Expected Directory Structure:**

```[file-tree]
data/
  ├── cadenza_data_train/
  │   ├── metadata/train_metadata.json  # from cadenza_clip1_data.metadata.v1.2.tar.gz
  │   └── train/
  │       ├── signals/*.flac
  │       └── unprocessed/*.flac
  ├── cadenza_data_valid/
  │   ├── metadata/valid_metadata.json  # from cadenza_clip1_data.metadata.v1.2.tar.gz
  │   └── valid/
  │       ├── signals/*.flac
  │       └── unprocessed/*.flac
  └── cadenza_data_eval/
      ├── metadata/eval_metadata.json   # from cadenza_clip1_data.metadata.v1.2.tar.gz
      └── eval/
          ├── signals/*.flac
          └── unprocessed/*.flac
```

**Generate CSV Metadata:**

```[bash]
python metadata_to_csv.py --json data/cadenza_data_train/metadata/train_metadata.json
python metadata_to_csv.py --json data/cadenza_data_valid/metadata/valid_metadata.json
python metadata_to_csv.py --json data/cadenza_data_eval/metadata/eval_metadata.json
```

Outputs:

- `data/cadenza_data_train/metadata/train_metadata_l.csv`,
`data/cadenza_data_train/metadata/train_metadata_r.csv`
- `data/cadenza_data_valid/metadata/valid_metadata_l.csv`,
`data/cadenza_data_valid/metadata/valid_metadata_r.csv`
- `data/cadenza_data_eval/metadata/eval_metadata_l.csv`,
`data/cadenza_data_eval/metadata/eval_metadata_r.csv`

**Note**: The CSVs are already generated under
`data/cadenza_data_*/metadata/*.csv`, regenerate from JSON if needed.

## 2. Feature extraction (Whisper scoring)

Extract stereo-aware Whisper features. Configuration is defined in
`feature_config.yaml`.

```[bash]
# Run for all default splits
python whisper_inference.py --config feature_config.yaml

# [Optional] Run specific splits or channels
python whisper_inference.py --config feature_config.yaml --splits=train,valid,eval
python whisper_inference.py --config feature_config.yaml --channel=right
```

**Note**: If you are using the provided CSVs, features might already be present.
Run these steps to refresh or re-compute.

## 3. Metric Computation (STOI / eSTOI / PESQ / DNSMOS)

Compute audio quality. Configuration is defined in `feature_config.yaml`.

```[bash]
# Compute all metrics
python compute_metrics.py --config feature_config.yaml

# [Optional] Compute specific metrics or datasets
python compute_metrics.py --config feature_config.yaml --quality --datasets \
signal_eval_left signal_eval_right
python compute_metrics.py --config feature_config.yaml --dns
```

**Note**: If you are using the provided CSVs, features might already be present.
Run these steps to refresh or re-compute.

## 4. Model training & inference

Configuration file: `checkpoints/final/config.yaml`

### Training

To train the model from scratch:

```[bash]
python train.py
```

- Artifacts: Saved to checkpoints/final/ (includes model.pt, train_log.csv).

### Inference

To run inference using the trained model (defaults to Valid + Eval splits):

```[bash]
python infer.py

# [Optional] Run inference on evaluation set only
python infer.py infer.splits=[eval]
```

Outputs

- `checkpoints/final/valid_inference.csv`
- `checkpoints/final/eval_inference.csv`

## Citation

```[bibtex]
@inproceedings{scow2026,
  title        = {Stereo Chorus of Whispers: Perceptually-Augmented
  Ear-Specific Intelligibility Prediction},
  author       = {Longbin Jin and Heungsoo Kim and Youngmin Kim},
  booktitle    = {Proc. IEEE ICASSP},
  year         = {2026},
  note         = {To appear}
}
```
