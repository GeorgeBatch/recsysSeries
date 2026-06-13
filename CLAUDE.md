# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Tutorial series implementing recommendation systems from scratch, accompanying the [mlwhiz.com RecSys Series](https://www.mlwhiz.com). Notebooks are designed for Kaggle kernels or GPU cloud instances (AWS EC2 with NVIDIA A10G) but can also run locally on a normal laptop or use Apple Silicon via MPS.

## Local Development Setup

The repo has a uv-managed Python 3.11 virtual environment at `.venv/`, registered as a Jupyter kernel:

```bash
# Create and populate (already done — only needed on a fresh clone)
uv venv .venv --python 3.11
uv pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm transformers Pillow faiss-cpu jupyter ipykernel ipywidgets kaggle
source .venv/bin/activate
python -m ipykernel install --user --name recsys --display-name "RecSys (Python 3.11)"

# Launch
source .venv/bin/activate
jupyter notebook
```

Select the **RecSys (Python 3.11)** kernel. Use `faiss-cpu` locally (not `faiss-gpu`); MPS is available on Apple Silicon.

## Device Selection

All notebooks should use this pattern (some older cells only have the cuda/cpu fallback and need updating for local use):

```python
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
```

## Data Sources & Local Paths

Datasets live in `kaggle/input/` at the repo root (gitignored), mirroring the Kaggle `/kaggle/input/` convention. Jupyter sets cwd to each notebook's directory, so notebooks reference data as `../kaggle/input/`.

Download with the Kaggle CLI (installed in `.venv`):

```bash
source .venv/bin/activate

# Section 2 — MovieLens-1M
kaggle datasets download -d odedgolden/movielens-1m-dataset \
  -p kaggle/input/movielens-1m-dataset --unzip
kaggle datasets download -d mohamedelmallah1/movielens-1m-with-posters-and-metadata \
  -p kaggle/input/movielens-1m-with-posters-and-metadata --unzip

# Section 3 & 4 — Steam Games metadata
kaggle datasets download -d fronkongames/steam-games-dataset \
  -p kaggle/input/steam-games-dataset --unzip
# Steam interaction data downloads at runtime from mcauleylab.ucsd.edu
```

The Section 2 notebook (`2. Two Tower/`) has two path strings that reference the original Kaggle absolute paths and need updating for local use:
- `path = '/kaggle/input/movielens-1m-dataset/'` → `'../kaggle/input/movielens-1m-dataset/'`
- `pd.read_csv("/kaggle/input/movielens-1m-with-posters-and-metadata/movies.csv")` → `"../kaggle/input/movielens-1m-with-posters-and-metadata/movies.csv"`

**Section 1** uses bundled data in `1. Recsys Fundamentals/ml-100k/` — no download needed.

## Architecture Overview

The series progresses through four recommendation paradigms:

**1. Fundamentals** — Content-based and collaborative filtering on MovieLens-100K. Classic non-neural baselines.

**2. Two Tower** — Dual-encoder retrieval model on MovieLens-1M. Key techniques: in-batch negatives (diagonal labels on `B×B` logit matrix), temperature scaling, and logQ correction (subtracting `log(item_popularity)` from logits to debias popular items). Item tower can be precomputed for ANN serving.

**3. Sequential Recommenders** — GRU4Rec (RNN) and SASRec (self-attention transformer) trained on Steam Games with 5-core filtering. Both use next-item prediction with in-batch negatives. Evaluation: Recall@K and NDCG@K against all items.

**4. Semantic IDs** — Four-notebook pipeline on Steam Games:
- **Notebook 1 (RQ-VAE):** Embeds game descriptions with Qwen3-Embedding-4B (MRL-truncated to 1024D), trains an RQVAE to produce 3-level discrete codes (256 codes each), adds a 4th collision-resolution token. Saves checkpoints.
- **Notebook 2 (SASRec baseline):** Standard SASRec with integer item IDs.
- **Notebook 3 (Semantic SASRec):** SASRec trained on 4-token semantic ID sequences instead of single item IDs.
- **Notebook 4 (TIGER):** Autoregressive generation of the full semantic ID tuple for the next item.

## Checkpoint System (Section 4)

Notebook 1 saves two checkpoints to `checkpoints/` (gitignored) that all subsequent notebooks depend on:

| Checkpoint | File(s) | Contents |
|---|---|---|
| 1 | `item_embeddings.pt`, `data_checkpoint1.pkl` | Qwen3 embeddings, games_df, user_sequences |
| 2 | `data_checkpoint2.pkl` | semantic_ids_3/4, semantic_token_ids, app_id_to_tokens |

Each downstream notebook has load-checkpoint cells to skip the expensive embedding/RQVAE steps.

## Code Conventions

- **ID mappings:** always use contiguous integer indices (`item2idx`, `user2idx`) for embedding tables; original IDs are preserved in reverse maps (`idx2item`).
- **Evaluation:** temporal train/val/test splits — never random splits for sequential models.
- **In-batch negatives:** all retrieval models use cross-entropy over the `B×B` similarity matrix; batch size is tuned large (1024+) to provide sufficient negatives.
- **Gradient flow through VQ:** the RQVAE uses the rotation trick (Fifty et al. 2025) rather than straight-through estimator.
