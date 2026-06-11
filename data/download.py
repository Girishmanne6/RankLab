"""Download the EB-NeRD dataset (RecSys 2024 benchmark).

EB-NeRD is published by Ekstra Bladet at https://recsys.eb.dk/. The small
subset is distributed as a zip archive containing parquet files:

    behaviors.parquet  - impression logs (clicks within inview lists)
    history.parquet    - per-user click history (list columns)
    articles.parquet   - article metadata (category, publish time, entities)

Usage:
    python -m data.download             # uses EBNERD_VARIANT from env
    python -m data.download --variant small
    python -m data.download --variant demo   # synthetic offline fallback

If the real download is unavailable (no network / license gate), the script
falls back to generating a realistic synthetic dataset with the exact same
schema so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

# Official distribution bucket referenced from https://recsys.eb.dk/
EBNERD_URLS = {
    "demo_real": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
}

CATEGORIES = [
    "nyheder", "sport", "underholdning", "krimi", "politik",
    "musik", "biler", "forbrug", "sex_og_samliv", "ferie",
    "penge", "vejret", "teknologi", "sundhed", "madopskrifter",
]


def download_file(url: str, dest: Path) -> bool:
    """Stream a file to disk with a progress bar. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
        print(f"Download failed ({exc}).")
        if dest.exists():
            dest.unlink()
        return False


def download_ebnerd(variant: str, raw_dir: Path) -> bool:
    """Download and extract a real EB-NeRD archive."""
    url = EBNERD_URLS.get("small" if variant == "small" else "demo_real")
    archive = raw_dir / f"ebnerd_{variant}.zip"
    print(f"Downloading EB-NeRD '{variant}' from {url}")
    if not download_file(url, archive):
        return False
    print("Extracting...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()
    return True


# ---------------------------------------------------------------------------
# Synthetic fallback (identical schema to EB-NeRD)
# ---------------------------------------------------------------------------

def generate_synthetic(
    raw_dir: Path,
    n_users: int = 2_000,
    n_articles: int = 1_200,
    n_impressions: int = 60_000,
    seed: int = config.SEED,
) -> None:
    """Generate a synthetic news dataset with EB-NeRD's schema.

    Key realism properties:
      * power-law article popularity (a few hits, a long tail)
      * per-user category affinity (users mostly click 2-3 categories)
      * recency effect (recent articles get more clicks)
      * timestamps span three weeks of activity
    """
    rng = np.random.default_rng(seed)
    print(f"Generating synthetic EB-NeRD-style data: "
          f"{n_users} users, {n_articles} articles, {n_impressions} impressions")

    end_ts = pd.Timestamp("2023-05-25 00:00:00")
    start_ts = end_ts - pd.Timedelta(days=21)

    # --- articles.parquet -------------------------------------------------
    article_ids = np.arange(100_000, 100_000 + n_articles)
    categories = rng.choice(len(CATEGORIES), size=n_articles)
    published = start_ts.value + (
        rng.random(n_articles) * (end_ts - start_ts).value
    ).astype("int64")
    # Power-law base popularity, exponent chosen to give a heavy tail.
    base_pop = rng.zipf(1.6, size=n_articles).astype(np.float64)
    base_pop = np.clip(base_pop, 1, 5_000)

    articles = pd.DataFrame(
        {
            "article_id": article_ids,
            "title": [
                f"{CATEGORIES[c].title()} story {i}"
                for i, c in enumerate(categories)
            ],
            "subtitle": [
                f"In-depth coverage of {CATEGORIES[c]} topic number {i}"
                for i, c in enumerate(categories)
            ],
            "body": [
                f"Article about {CATEGORIES[c]} with details on subject {i}. "
                + " ".join(
                    rng.choice(
                        ["analysis", "report", "exclusive", "interview",
                         "breaking", "update", "feature", CATEGORIES[c]],
                        size=20,
                    )
                )
                for i, c in enumerate(categories)
            ],
            "category": categories,
            "category_str": [CATEGORIES[c] for c in categories],
            "published_time": pd.to_datetime(published),
            "article_type": "article_default",
            "premium": rng.random(n_articles) < 0.1,
            "total_pageviews": base_pop,
            "sentiment_score": rng.uniform(0, 1, n_articles).round(4),
            "sentiment_label": rng.choice(
                ["Positive", "Neutral", "Negative"], size=n_articles
            ),
            "topics": [
                list(rng.choice(CATEGORIES, size=2, replace=False))
                for _ in range(n_articles)
            ],
        }
    )

    # --- user click behaviour ---------------------------------------------
    user_ids = np.arange(10_000, 10_000 + n_users)
    # Each user prefers 2-3 categories.
    user_pref_cats = [
        rng.choice(len(CATEGORIES), size=rng.integers(2, 4), replace=False)
        for _ in range(n_users)
    ]

    pub_ts = articles["published_time"].astype("int64").to_numpy()

    def article_scores(now_ns: int, prefs: np.ndarray) -> np.ndarray:
        """Unnormalised click propensity for every article at a moment."""
        age_h = np.maximum((now_ns - pub_ts) / 3.6e12, 0.0)
        recency = np.exp(-age_h / 48.0)            # 48h half-life-ish decay
        cat_boost = np.where(np.isin(categories, prefs), 4.0, 1.0)
        not_published = pub_ts > now_ns
        score = base_pop * recency * cat_boost
        score[not_published] = 0.0
        return score

    rows_user, rows_item, rows_ts = [], [], []
    impressions_per_user = rng.poisson(
        n_impressions / n_users, size=n_users
    ).clip(5, 200)

    for u_idx in tqdm(range(n_users), desc="users"):
        prefs = user_pref_cats[u_idx]
        n_imp = impressions_per_user[u_idx]
        times = np.sort(
            start_ts.value
            + (rng.random(n_imp) * (end_ts - start_ts).value).astype("int64")
        )
        for t in times:
            scores = article_scores(int(t), prefs)
            total = scores.sum()
            if total <= 0:
                continue
            item = rng.choice(n_articles, p=scores / total)
            rows_user.append(user_ids[u_idx])
            rows_item.append(article_ids[item])
            rows_ts.append(t)

    behaviors = pd.DataFrame(
        {
            "impression_id": np.arange(len(rows_user)),
            "user_id": rows_user,
            "article_ids_clicked": [[a] for a in rows_item],
            "impression_time": pd.to_datetime(np.array(rows_ts)),
            "device_type": rng.integers(1, 4, size=len(rows_user)),
            "is_subscriber": rng.random(len(rows_user)) < 0.2,
        }
    )

    # --- history.parquet (per-user list columns, like EB-NeRD) -------------
    clicks = behaviors.explode("article_ids_clicked").rename(
        columns={"article_ids_clicked": "article_id"}
    )
    history = (
        clicks.sort_values("impression_time")
        .groupby("user_id")
        .agg(
            article_id_fixed=("article_id", list),
            impression_time_fixed=("impression_time", list),
        )
        .reset_index()
    )

    out = raw_dir / "ebnerd_demo"
    out.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(out / "articles.parquet", index=False)
    behaviors.to_parquet(out / "behaviors.parquet", index=False)
    history.to_parquet(out / "history.parquet", index=False)
    print(f"Synthetic dataset written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EB-NeRD dataset")
    parser.add_argument(
        "--variant",
        default=config.EBNERD_VARIANT,
        choices=["demo", "small"],
        help="'small' = real EB-NeRD small subset, 'demo' = synthetic fallback",
    )
    args = parser.parse_args()

    raw_dir = config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    marker = raw_dir / "ebnerd_demo" / "behaviors.parquet"
    small_marker = raw_dir / "ebnerd_small" / "train" / "behaviors.parquet"
    if marker.exists() or small_marker.exists():
        print("Raw data already present, skipping download.")
        return

    if args.variant == "small":
        if download_ebnerd("small", raw_dir):
            return
        print("Falling back to synthetic demo dataset.")
    generate_synthetic(raw_dir)


if __name__ == "__main__":
    main()
