# RankLab

**A responsible news recommendation platform with fairness and bias analysis.**

RankLab implements and compares five recommendation approaches on the
[EB-NeRD](https://recsys.eb.dk/) benchmark (RecSys 2024 Challenge dataset),
with a focus on reducing popularity bias and improving long-tail content
exposure. It ships with a live A/B testing dashboard tracking both
recommendation quality and fairness metrics.

## Why

News recommenders that optimise pure engagement systematically over-expose
already-popular articles, starving the long tail and narrowing what readers
see. RankLab makes that trade-off measurable: every model is evaluated not
only on accuracy (Recall/NDCG/MRR) but on **diversity, freshness, long-tail
exposure, and popularity bias** — with side-by-side comparison of graph
collaborative filtering (LightGCN) and contrastive debiasing (XSimGCL).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 docker compose                │
                 │                                              │
 EB-NeRD  ───►   │  ┌────────────┐    ┌──────────────────────┐  │
 (recsys.eb.dk)  │  │  data/      │    │  backend (FastAPI)   │  │
                 │  │  download   │───►│                      │  │
                 │  │  preprocess │    │  /recommend          │  │
                 │  └────────────┘    │  /metrics/{model}     │  │
                 │        │           │  /ab-test/*           │  │
                 │        ▼           │  /models  /health     │  │
                 │  ┌────────────┐    └─────┬──────────┬─────┘  │
                 │  │  models/    │          │          │        │
                 │  │  popularity │◄─ train ─┘          ▼        │
                 │  │  recency    │              ┌────────────┐  │
                 │  │  item_cf    │              │ db          │  │
                 │  │  lightgcn   │              │ PostgreSQL  │  │
                 │  │  xsimgcl    │              │ (A/B obs.)  │  │
                 │  └────────────┘              └────────────┘  │
                 │        │                                      │
                 │        ▼                                      │
                 │  ┌──────────────────────────────────────┐    │
                 │  │  frontend (React + Vite + Recharts)   │    │
                 │  │  Model Comparison · Fairness ·        │    │
                 │  │  A/B Testing · Recommendations        │    │
                 │  └──────────────────────────────────────┘    │
                 └──────────────────────────────────────────────┘
```

## Dataset: EB-NeRD

EB-NeRD (Ekstra Bladet News Recommendation Dataset) was released for the
RecSys 2024 Challenge. It contains user click histories, impression logs
(clicked articles within in-view lists), and rich article metadata
(category, publish time, entities, sentiment) from the Danish newspaper
Ekstra Bladet.

- `behaviors.parquet` — impression logs with clicked articles
- `history.parquet` — per-user click history
- `articles.parquet` — article metadata

**Getting the data.** The dataset is gated behind a license agreement at
<https://recsys.eb.dk/>. Extract the small subset and point RankLab at it:

```bash
# After downloading from recsys.eb.dk, set in .env:
EBNERD_RAW_PATH=./data/raw/ebnerd_small   # folder with train/, validation/, articles.parquet
```

Alternatively set `EBNERD_VARIANT=small` to download directly, or
`EBNERD_VARIANT=demo` for a synthetic fallback with the same schema so the
stack runs offline.

## Models

| Model | Idea | Reference |
|---|---|---|
| `popularity` | Time-decayed global click counts | classic baseline |
| `recency` | Exponential decay over article age + popularity tiebreak | classic baseline |
| `item_cf` | Item-item CF: cosine similarity on TF-IDF article text, weighted by click history | Sarwar et al., WWW 2001 |
| `lightgcn` | Light graph convolution over the user-item bipartite graph; BPR loss | He et al., *LightGCN*, SIGIR 2020 |
| `xsimgcl` | LightGCN + noise-augmented cross-layer contrastive learning (InfoNCE, τ=0.2, λ=0.1) | Yu et al., *XSimGCL*, 2023 |

All models are implemented **from scratch** in PyTorch/NumPy — no
recommendation libraries. LightGCN follows the paper exactly: embeddings
only, no feature transformations, no nonlinearities, mean layer combination.
XSimGCL perturbs every propagation layer with sign-aligned noise and
contrasts the final embedding against an intermediate layer from the same
forward pass, pushing item representations toward uniformity on the
hypersphere — which is precisely what reduces popularity bias.

All models handle cold start: unknown users receive a popularity prior
instead of crashing.

## Results (EB-NeRD small, test split)

Evaluated on **18,826 users · 8,916 items · 4.38M interactions** (500 test
users, k=10). Metrics are written to `models/checkpoints/metrics.json` after
training.

| Model | Recall@10 | NDCG@10 | MRR | Long-tail % | Pop. bias |
|---|---|---|---|---|---|
| Popularity | 0.0056 | 0.0086 | 0.0304 | 0.0% | +0.297 |
| Recency | 0.0064 | 0.0095 | 0.0314 | 0.0% | +0.295 |
| Item-CF | **0.0136** | **0.0125** | 0.0298 | **57.9%** | +0.092 |
| LightGCN | 0.0102 | 0.0129 | **0.0386** | 11.4% | +0.297 |
| XSimGCL | 0.0102 | 0.0128 | 0.0384 | 11.6% | +0.297 |

Item-CF achieves the strongest recall; LightGCN leads on MRR. At the tested
GNN settings (embed dim 32, early stopping), XSimGCL did not materially shift
popularity bias relative to LightGCN on this split — the fairness tooling
surfaces that honestly.

## Metrics (all from scratch)

**Quality** (`evaluation/metrics.py`)
- **Recall@10** — share of held-out clicks recovered in the top 10
- **NDCG@10** — rank-discounted gain, normalised by the ideal ranking
- **MRR** — reciprocal rank of the first hit
- **p95 latency** — 95th percentile inference time per request

**Fairness** (`evaluation/fairness.py`)
- **Diversity** — fraction of item pairs in a list from different categories
- **Freshness** — mean age (hours) of recommended articles
- **Long-tail exposure** — share of recommendations outside the top-20% most
  popular items
- **Popularity bias** — Pearson correlation between item popularity and
  recommendation frequency (lower = fairer)

**A/B testing** (`evaluation/ab_testing.py`)
- Deterministic 50/50 user hashing into arms, observation logging into
  PostgreSQL, and Welch's t-test for significance.

## Quick start

### Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This boots PostgreSQL, downloads/generates data, preprocesses it, trains all
five models, and serves:

- API: <http://localhost:8000> (interactive docs at `/docs`)
- Dashboard: <http://localhost:5173>

### Local development

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m data.download                # EBNERD_VARIANT=small for real data
python -m data.preprocess
python -m api.train                    # add --epochs 5 for a quick run
uvicorn api.main:app --reload          # API on :8000

cd dashboard && npm install && npm run dev   # dashboard on :5173
```

Without `DATABASE_URL` set, the A/B framework falls back to a local SQLite
file, so PostgreSQL is optional outside Docker.

### Running a single model

```bash
python -m api.train --models lightgcn --epochs 50
python -m api.train --models xsimgcl popularity
```

### Tests

```bash
pytest tests/ -v
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Service + data status |
| `GET /models` | Available models with trained/loaded status |
| `POST /recommend` | `{user_id, model_name, k}` → ranked articles with scores + explanations |
| `GET /metrics/{model_name}` | Cached quality + fairness metrics |
| `GET /scatter/{model_name}` | Popularity-vs-exposure scatter data |
| `GET /ab-test/results` | Current A/B test results with significance |
| `POST /ab-test/start` | Start a new A/B test between two models |

## Dashboard

1. **Model Comparison** — Recall@10 / NDCG@10 / MRR bar charts + full metric
   table with the best model highlighted.
2. **Fairness Analysis** — diversity / freshness / long-tail bars, plus a
   popularity-vs-exposure scatter contrasting LightGCN and XSimGCL.
3. **A/B Testing** — live arm comparison, Welch's t-test significance badge,
   p95 latency per arm.
4. **Recommendations** — query any user with any model; every item shows its
   metadata and a human-readable explanation.

## References

- He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
  **LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation.** SIGIR 2020.
- Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023).
  **XSimGCL: Towards Extremely Simple Graph Contrastive Learning for
  Recommendation.** IEEE TKDE 2023 (extends SimGCL, SIGIR 2022).
- Kruse, J., et al. (2024). **EB-NeRD: A Large-Scale Dataset for News
  Recommendation.** RecSys 2024 Challenge.
- Rendle, S., et al. (2009). **BPR: Bayesian Personalized Ranking from
  Implicit Feedback.** UAI 2009.

## Resume bullets

> - Built **RankLab**, a responsible news-recommendation platform on the
>   EB-NeRD benchmark (RecSys 2024): five models implemented from scratch in
>   PyTorch — popularity/recency baselines, item-item CF, **LightGCN** (SIGIR
>   2020) and **XSimGCL** (TKDE 2023) — served via FastAPI + PostgreSQL +
>   React, fully containerised with Docker Compose.
> - Implemented every evaluation metric from scratch (Recall@10, NDCG@10,
>   MRR, p95 latency) plus fairness instrumentation (intra-list diversity,
>   freshness, long-tail exposure, popularity-bias correlation) on EB-NeRD
>   small (18K users, 8.9K items, 4.4M interactions).
> - Designed an A/B testing framework with deterministic user bucketing,
>   PostgreSQL-backed observation logging, and Welch's t-test significance
>   reporting, surfaced in a live Recharts dashboard.
