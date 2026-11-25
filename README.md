# OTTO – Multi-Objective Recommender System

Modular, production-ready 2-stage recommender system for the Kaggle **[OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system)** competition:

1. **Data Processing & Validation (`Polars`)**: Fast, memory-efficient lazy execution, deterministic session sampling, and leak-free temporal validation splitting.
2. **Stage 1 – Candidate Generation (`Co-visitation Matrices`)**: Multi-objective item-to-item co-visitation retrieval (`clicks`, `carts`, `orders`) combined with session history and popularity fallbacks.
3. **Stage 2 – Reranking (`LightGBM`)**: Objective-specific Gradient Boosted Decision Tree rankers (`lambdarank`) trained on session, item, and interaction features to predict the top-20 items per objective.

---

## Zero-Local-Storage Design (Google Colab + Google Drive)

This repository contains **lightweight source code, configuration, and notebooks only**. All heavy files live exclusively on Google Drive under `/content/drive/MyDrive/OTTO/`:

```text
/content/drive/MyDrive/OTTO/
├── kaggle_token.txt       # Kaggle API token (KGAT_..., never committed to Git)
├── data/
│   ├── raw/               # Optimized Parquet dataset (radek1/otto-full-optimized-memory-footprint)
│   ├── sample/            # 5% session subsample for rapid iteration
│   └── splits/            # Temporal train / validation splits & test labels
├── artifacts/
│   ├── candidates/        # Top-K candidate tables & co-visitation matrices
│   ├── features/          # Engineered session-item feature matrices
│   └── models/            # Trained LightGBM models (clicks, carts, orders)
└── submissions/           # Generated Kaggle submission CSVs
```

---

## Repository Structure

```text
otto-recommender-system/
├── configs/
│   └── config.yaml
├── notebooks/
│   └── run_colab.ipynb
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_sample.py
│   │   └── split.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── recall.py
│   ├── candidates/
│   │   ├── __init__.py
│   │   └── covisitation.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   └── models/
│       ├── __init__.py
│       ├── train.py
│       └── predict.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart (Google Colab)

1. Save your Kaggle API token (`KGAT_...`) to Google Drive at `/content/drive/MyDrive/OTTO/kaggle_token.txt`.
2. Push this repository to GitHub and open `notebooks/run_colab.ipynb` in Google Colab.
3. Update `REPO_URL` in the notebook and run all cells to mount Drive, install dependencies, download the Parquet dataset directly to Drive, and execute the pipeline.

---

## CLI Usage

Run the full end-to-end pipeline on the fast 5% sample:

```bash
python main.py --mode sample --stage all
```

Run an individual stage (`sample`, `split`, `candidates`, `features`, `train`, `predict`):

```bash
python main.py --mode sample --stage candidates
```

Run the full 100% dataset pipeline:

```bash
python main.py --mode full --stage all
```
