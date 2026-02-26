import numpy as np

from terrabit.metrics import (
    cosine_similarity_correlation,
    effective_rank,
    evaluate_compression,
    isotropy_score,
    knn_recall,
    reconstruction_cosine,
    reconstruction_mse,
)


def _rand_embeddings(n=200, d=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class TestKnnRecall:
    def test_identity(self):
        x = _rand_embeddings()
        recall = knn_recall(x, x, k=10)
        assert recall == 1.0

    def test_noisy(self):
        x = _rand_embeddings()
        noise = np.random.default_rng(99).standard_normal(x.shape).astype(np.float32) * 0.01
        recall = knn_recall(x, x + noise, k=10)
        assert recall > 0.5

    def test_identity_cosine(self):
        x = _rand_embeddings()
        recall = knn_recall(x, x, k=10, metric="cosine")
        assert recall == 1.0

    def test_identity_euclidean(self):
        x = _rand_embeddings()
        recall = knn_recall(x, x, k=10, metric="euclidean")
        assert recall == 1.0


class TestCosineSimilarityCorrelation:
    def test_identity(self):
        x = _rand_embeddings()
        corr = cosine_similarity_correlation(x, x, seed=0)
        assert corr > 0.99

    def test_scaled(self):
        x = _rand_embeddings()
        corr = cosine_similarity_correlation(x, x * 2.0, seed=0)
        assert corr > 0.99


class TestReconstruction:
    def test_mse_identity(self):
        x = _rand_embeddings()
        assert reconstruction_mse(x, x) == 0.0

    def test_cosine_identity(self):
        x = _rand_embeddings()
        assert abs(reconstruction_cosine(x, x) - 1.0) < 1e-5


class TestIsotropy:
    def test_isotropic(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 10)).astype(np.float32)
        score = isotropy_score(x)
        assert score > 0.3  # random gaussian should be fairly isotropic

    def test_anisotropic(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 10)).astype(np.float32)
        x[:, 0] *= 100  # one dominant dimension
        score = isotropy_score(x)
        assert score < 0.05


class TestEffectiveRank:
    def test_full_rank(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 10)).astype(np.float32)
        er = effective_rank(x)
        assert er > 5  # should be close to 10

    def test_low_rank(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((500, 2)).astype(np.float32)
        proj = rng.standard_normal((2, 20)).astype(np.float32)
        x = base @ proj
        er = effective_rank(x)
        assert er < 5  # inherently 2D


class TestEvaluateCompression:
    def test_full_eval(self):
        x = _rand_embeddings(n=100, d=32)
        x_comp = x[:, :16]
        results = evaluate_compression(x, x_comp, x_recon=None, k=5, seed=0)
        assert "knn_recall" in results
        assert "cosine_sim_correlation" in results
        assert "effective_rank_orig" in results
        assert "effective_rank_comp" in results
        assert "isotropy_orig" in results
        assert "isotropy_comp" in results
        assert "reconstruction_mse" not in results

    def test_with_reconstruction(self):
        x = _rand_embeddings(n=100, d=32)
        results = evaluate_compression(x, x, x_recon=x, k=5, seed=0)
        assert results["reconstruction_mse"] == 0.0
        assert results["reconstruction_cosine"] > 0.99

    def test_with_cosine_knn(self):
        x = _rand_embeddings(n=100, d=32)
        results = evaluate_compression(x, x, x_recon=x, k=5, knn_metric="cosine", seed=0)
        assert results["knn_recall"] == 1.0
