# Recommendation Systems Series

A comprehensive tutorial series on building recommendation systems from scratch, covering fundamental algorithms to modern deep learning approaches.

## Overview

This repository contains hands-on implementations accompanying the [mlwhiz.com RecSys Series](https://www.mlwhiz.com). Each notebook provides a complete implementation with practical code, making it ideal for ML engineers who want to go beyond theory and build real systems.

## Repository Structure

```
recsysSeries/
├── 1. Recsys Fundamentals/
│   ├── 1. ContentBased.ipynb
│   └── 2. Collaborative Filtering.ipynb
├── 2. Two Tower/
│   └── two-tower-network-with-temperature-and-logq.ipynb
├── 3. Sequential Recommenders/
│   ├── gru4rec/
│   │   └── gru4rec-steam-dataset.ipynb
│   └── sasrec/
│       └── sasrec-on-steam-games.ipynb
└── 4. Semantic Ids/
    ├── semantic-ids-steam-1-rqvae.ipynb
    ├── semantic-ids-steam-2-sasrec.ipynb
    ├── semantic-ids-steam-3-semantic-sasrec.ipynb
    └── semantic-ids-steam-4-tiger.ipynb
```

## Tutorials

### 1. RecSys Fundamentals

**Notebooks:** `1. Recsys Fundamentals/`

**Blog posts:**
- [RecSys Fundamentals: The Art and Science of Digital Matchmaking](https://www.mlwhiz.com/p/the-recommenders-playbook-algorithms) — collaborative filtering, content-based filtering, and hybrid approaches
- [How Recommendation Systems Learned to Think](https://www.mlwhiz.com/p/the-algorithmic-journey-of-recommender) — from Tapestry (1992) to modern neural approaches

Implementations of the two foundational recommendation paradigms: content-based filtering (recommending based on item features) and collaborative filtering (recommending based on user behavior patterns).

---

### 2. Two Tower Networks

**Notebook:** `2. Two Tower/two-tower-network-with-temperature-and-logq.ipynb`

**Blog posts:**
- [How YouTube Finds Your Next Video in Milliseconds](https://www.mlwhiz.com/p/building-youtube-scale-recommendation) — deep dive on Two-Tower architecture, in-batch negatives, temperature scaling, and logQ correction
- [The 3-Stage Funnel Behind Every Modern Recommender System](https://www.mlwhiz.com/p/the-recommendation-engine-under-the) — where Two-Tower fits in the retrieval → ranking → re-ranking pipeline

Two-Tower models are the dominant architecture for large-scale candidate retrieval. By keeping user and item towers independent, item embeddings can be precomputed offline and served via approximate nearest neighbor search for sub-millisecond latency at billion-item scale.

---

### 3. Sequential Recommenders

**Notebooks:** `3. Sequential Recommenders/`

**Blog post:**
- [From RNNs to Transformers: Building Sequential Recommenders (Part 1)](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders) — GRU4Rec, SASRec, production deployment with FAISS

Implementations of sequential models that treat user history as a sequence to predict the next item:

- **GRU4Rec** (`gru4rec/`) — RNN-based approach using GRUs, the first deep learning model applied to session-based recommendation
- **SASRec** (`sasrec/`) — self-attention based model that captures long-range dependencies, ~17x faster than GRU4Rec+ while achieving better accuracy

Both are trained and evaluated on the Steam Games dataset.

---

### 4. Semantic IDs & Generative Retrieval

**Notebooks:** `4. Semantic Ids/`

**Blog post:**
- [From RNNs to Transformers: Building Sequential Recommenders (Part 2)](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders) — Semantic IDs, TIGER, HSTU, and generative recommenders in production

A four-part implementation series on generative recommendation using semantic IDs on the Steam Games dataset:

1. **RQ-VAE** (`semantic-ids-steam-1-rqvae.ipynb`) — train a Residual Quantized VAE to encode item metadata into discrete semantic ID tuples
2. **SASRec baseline** (`semantic-ids-steam-2-sasrec.ipynb`) — standard SASRec as the baseline
3. **Semantic SASRec** (`semantic-ids-steam-3-semantic-sasrec.ipynb`) — SASRec trained on semantic IDs instead of arbitrary item IDs
4. **TIGER** (`semantic-ids-steam-4-tiger.ipynb`) — Transformer Index for GEnerative Recommenders: autoregressively generate the semantic IDs of the next item

Semantic IDs replace arbitrary interaction-dependent item IDs with content-aware discrete representations, enabling better generalization and cold-start performance.

---

## Blog Series

All notebooks accompany the **mlwhiz.com RecSys Series**:

| # | Post | Topic |
|---|------|-------|
| 1 | [RecSys Fundamentals](https://www.mlwhiz.com/p/the-recommenders-playbook-algorithms) | Content-based & collaborative filtering |
| 2 | [How RecSys Learned to Think](https://www.mlwhiz.com/p/the-algorithmic-journey-of-recommender) | Algorithmic history & matrix factorization |
| 3 | [The 3-Stage Funnel](https://www.mlwhiz.com/p/the-recommendation-engine-under-the) | Retrieval → ranking → re-ranking pipeline |
| 4 | [YouTube-Scale Two-Tower](https://www.mlwhiz.com/p/building-youtube-scale-recommendation) | Two-Tower models, in-batch negatives, ANN |
| 5 | [Sequential Recommenders Part 1](https://www.mlwhiz.com/p/rnns-to-transformers-sequential-recommenders) | GRU4Rec, SASRec on Steam Games |
| 6 | Sequential Recommenders Part 2 *(coming soon)* | Semantic IDs, TIGER, HSTU |
