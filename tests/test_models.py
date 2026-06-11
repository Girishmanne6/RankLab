"""Model behaviour tests: valid rankings, cold start, checkpointing."""

from __future__ import annotations

import numpy as np
import pytest

from models import (
    ItemCFModel,
    LightGCNModel,
    PopularityModel,
    RecencyModel,
    XSimGCLModel,
)


def fit(model, ds, **kwargs):
    return model.fit(
        ds["train"], ds["articles"], ds["n_users"], ds["n_items"], **kwargs
    )


@pytest.fixture(scope="module")
def trained(dataset):
    """Train every model once for the whole module (GNNs use 3 epochs)."""
    return {
        "popularity": fit(PopularityModel(), dataset),
        "recency": fit(RecencyModel(), dataset),
        "item_cf": fit(ItemCFModel(top_k_neighbors=10), dataset),
        "lightgcn": fit(
            LightGCNModel(epochs=3, patience=2, batch_size=128),
            dataset,
            val=dataset["val"],
        ),
        "xsimgcl": fit(
            XSimGCLModel(epochs=3, patience=2, batch_size=128),
            dataset,
            val=dataset["val"],
        ),
    }


ALL_MODELS = ["popularity", "recency", "item_cf", "lightgcn", "xsimgcl"]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_returns_k_items(trained, name):
    recs = trained[name].recommend(0, k=10)
    assert len(recs) == 10


@pytest.mark.parametrize("name", ALL_MODELS)
def test_items_in_valid_range(trained, dataset, name):
    recs = trained[name].recommend(1, k=10)
    for item, _ in recs:
        assert 0 <= item < dataset["n_items"]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_no_duplicate_items(trained, name):
    recs = trained[name].recommend(2, k=10)
    items = [i for i, _ in recs]
    assert len(items) == len(set(items))


@pytest.mark.parametrize("name", ALL_MODELS)
def test_scores_descending(trained, name):
    recs = trained[name].recommend(3, k=10)
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_excludes_seen_items(trained, name):
    model = trained[name]
    seen = model.seen(4)
    recs = model.recommend(4, k=10, exclude_seen=True)
    assert not seen & {i for i, _ in recs}


@pytest.mark.parametrize("name", ALL_MODELS)
def test_cold_start_user(trained, dataset, name):
    """Unknown users must still get a full valid ranking (no crash)."""
    unknown = dataset["n_users"] + 999
    recs = trained[name].recommend(unknown, k=10)
    assert len(recs) == 10
    for item, _ in recs:
        assert 0 <= item < dataset["n_items"]


def test_popularity_ranks_by_count(dataset):
    model = PopularityModel(half_life_hours=None)
    fit(model, dataset)
    counts = dataset["train"]["item_idx"].value_counts()
    top_item = counts.index[0]
    scores = model.score_all(0)
    assert int(np.argmax(scores)) == top_item


def test_popularity_time_decay_changes_scores(dataset):
    plain = fit(PopularityModel(half_life_hours=None), dataset).score_all(0)
    decayed = fit(PopularityModel(half_life_hours=24.0), dataset).score_all(0)
    assert not np.allclose(plain, decayed)


def test_recency_prefers_newer_articles(dataset):
    model = fit(RecencyModel(alpha=0.0), dataset)
    scores = model.score_all(0)
    published = dataset["articles"].set_index("item_idx")["published_ts"]
    newest = int(published.idxmax())
    oldest = int(published.idxmin())
    assert scores[newest] > scores[oldest]


def test_item_cf_similarity_matrix_shape(trained, dataset):
    sim = trained["item_cf"].sim_
    assert sim.shape == (dataset["n_items"], dataset["n_items"])
    assert (sim.diagonal() == 0).all()  # no self-similarity


def test_item_cf_personalizes(trained):
    """Two users with different histories should get different scores."""
    s0 = trained["item_cf"].score_all(0)
    s1 = trained["item_cf"].score_all(1)
    assert not np.allclose(s0, s1)


def test_lightgcn_embedding_shapes(trained, dataset):
    model = trained["lightgcn"]
    assert model.user_emb_.shape == (dataset["n_users"], model.embedding_dim)
    assert model.item_emb_.shape == (dataset["n_items"], model.embedding_dim)


def test_lightgcn_training_reduces_loss(trained):
    history = trained["lightgcn"].history_
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]


def test_lightgcn_save_load_roundtrip(trained, dataset, tmp_path):
    model = trained["lightgcn"]
    path = tmp_path / "lightgcn.pkl"
    model.save(path)
    loaded = LightGCNModel.load(path)
    np.testing.assert_allclose(loaded.score_all(5), model.score_all(5))
    assert loaded.recommend(5, k=10) == model.recommend(5, k=10)


def test_xsimgcl_save_load_roundtrip(trained, tmp_path):
    model = trained["xsimgcl"]
    path = tmp_path / "xsimgcl.pkl"
    model.save(path)
    loaded = XSimGCLModel.load(path)
    np.testing.assert_allclose(loaded.score_all(7), model.score_all(7))


def test_xsimgcl_differs_from_lightgcn(trained):
    """Contrastive regularisation must change the learned embeddings."""
    assert not np.allclose(
        trained["xsimgcl"].item_emb_, trained["lightgcn"].item_emb_
    )


def test_simple_model_save_load_roundtrip(trained, tmp_path):
    model = trained["popularity"]
    path = tmp_path / "pop.pkl"
    model.save(path)
    loaded = PopularityModel.load(path)
    np.testing.assert_allclose(loaded.score_all(0), model.score_all(0))


@pytest.mark.parametrize("name", ALL_MODELS)
def test_explanations_are_strings(trained, name):
    model = trained[name]
    recs = model.recommend(0, k=3)
    for item, _ in recs:
        text = model.explain(0, item)
        assert isinstance(text, str) and len(text) > 0
