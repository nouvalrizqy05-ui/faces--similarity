"""
tests/test_pca_engine.py
========================
Unit tests untuk modul pca_engine.py
Jalankan: python -m pytest tests/ -v
"""

import numpy as np
import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pca_engine import (
    FacePCA,
    mse, psnr, ssim,
    cosine_similarity, euclidean_distance,
    compression_ratio,
    IMG_SIZE,
)


# ─── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Data sintetik 50 sampel, 200 fitur (bukan gambar asli)."""
    np.random.seed(42)
    X = np.random.rand(50, 200).astype(np.float64)
    return X


@pytest.fixture
def fitted_pca(synthetic_data):
    """FacePCA yang sudah di-fit pada data sintetik."""
    pca = FacePCA(n_components=10)
    pca.fit(synthetic_data)
    return pca


# ─── Test FacePCA ──────────────────────────────────────────────────────────────

class TestFacePCA:

    def test_fit_sets_attributes(self, synthetic_data):
        pca = FacePCA(n_components=10)
        pca.fit(synthetic_data)
        assert pca.is_fitted
        assert pca.mean_face_ is not None
        assert pca.components_ is not None
        assert pca.eigenvalues_ is not None
        assert pca.explained_variance_ratio_ is not None

    def test_mean_face_shape(self, synthetic_data):
        pca = FacePCA(n_components=5)
        pca.fit(synthetic_data)
        assert pca.mean_face_.shape == (synthetic_data.shape[1],)

    def test_components_shape(self, synthetic_data):
        k = 8
        pca = FacePCA(n_components=k)
        pca.fit(synthetic_data)
        assert pca.components_.shape == (k, synthetic_data.shape[1])

    def test_transform_shape(self, synthetic_data, fitted_pca):
        Z = fitted_pca.transform(synthetic_data)
        assert Z.shape == (synthetic_data.shape[0], fitted_pca.n_components)

    def test_fit_transform_equals_fit_then_transform(self, synthetic_data):
        pca1 = FacePCA(n_components=10)
        Z1   = pca1.fit_transform(synthetic_data)

        pca2 = FacePCA(n_components=10)
        pca2.fit(synthetic_data)
        Z2   = pca2.transform(synthetic_data)

        np.testing.assert_allclose(np.abs(Z1), np.abs(Z2), atol=1e-8)

    def test_reconstruct_shape(self, synthetic_data, fitted_pca):
        Z    = fitted_pca.transform(synthetic_data)
        Xhat = fitted_pca.reconstruct(Z)
        assert Xhat.shape == synthetic_data.shape

    def test_reconstruct_improves_with_more_components(self, synthetic_data):
        mse_vals = []
        for k in [2, 5, 10, 20]:
            k = min(k, synthetic_data.shape[0] - 1, synthetic_data.shape[1])
            pca  = FacePCA(n_components=k)
            Z    = pca.fit_transform(synthetic_data)
            Xhat = pca.reconstruct(Z)
            mse_vals.append(mse(synthetic_data, Xhat))
        # MSE harus menurun seiring bertambahnya k
        assert mse_vals[0] >= mse_vals[-1], "MSE seharusnya menurun dengan k lebih besar"

    def test_explained_variance_ratio_sums_leq_1(self, fitted_pca):
        total = np.sum(fitted_pca.explained_variance_ratio_)
        assert 0 < total <= 1.001  # toleransi floating point

    def test_explained_variance_ratio_nonnegative(self, fitted_pca):
        assert np.all(fitted_pca.explained_variance_ratio_ >= 0)

    def test_n_components_for_variance_in_range(self, fitted_pca):
        k = fitted_pca.n_components_for_variance(0.80)
        assert 1 <= k <= fitted_pca.n_components

    def test_cumulative_explained_variance_monotone(self, fitted_pca):
        cumvar = fitted_pca.cumulative_explained_variance()
        assert np.all(np.diff(cumvar) >= -1e-10), "Cumulative EVR harus monoton naik"

    def test_unfitted_transform_raises(self):
        pca = FacePCA(n_components=5)
        with pytest.raises(RuntimeError):
            pca.transform(np.random.rand(10, 200))

    def test_centering_makes_mean_zero(self, synthetic_data):
        pca = FacePCA(n_components=5)
        pca.fit(synthetic_data)
        Xc  = synthetic_data - pca.mean_face_
        assert abs(Xc.mean()) < 1e-10

    def test_save_and_load(self, synthetic_data, fitted_pca):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name
        try:
            fitted_pca.save(tmp_path)
            pca2 = FacePCA()
            pca2.load(tmp_path)
            np.testing.assert_allclose(fitted_pca.mean_face_, pca2.mean_face_)
            np.testing.assert_allclose(fitted_pca.components_, pca2.components_)
        finally:
            os.unlink(tmp_path)

    def test_single_sample_transform(self, synthetic_data, fitted_pca):
        single = synthetic_data[0:1]
        Z = fitted_pca.transform(single)
        assert Z.shape == (1, fitted_pca.n_components)

    def test_eigenvalues_decreasing(self, synthetic_data):
        pca = FacePCA(n_components=15)
        pca.fit(synthetic_data)
        diffs = np.diff(pca.eigenvalues_)
        # Eigenvalue harus menurun (atau tetap)
        assert np.all(diffs <= 1e-8), "Eigenvalue harus terurut menurun"


# ─── Test Metrik ───────────────────────────────────────────────────────────────

class TestMetrics:

    def test_mse_identical_arrays(self):
        x = np.random.rand(100)
        assert mse(x, x) == pytest.approx(0.0)

    def test_mse_nonnegative(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        assert mse(x, y) >= 0

    def test_psnr_identical_arrays(self):
        x = np.random.rand(100)
        result = psnr(x, x)
        assert result == float('inf')

    def test_psnr_decreases_with_error(self):
        x = np.ones(100) * 0.5
        y1 = x + 0.01 * np.ones(100)
        y2 = x + 0.1  * np.ones(100)
        assert psnr(x, y1) > psnr(x, y2)

    def test_ssim_identical_arrays(self):
        x = np.random.rand(200)
        result = ssim(x, x)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_ssim_range(self):
        x = np.random.rand(200)
        y = np.random.rand(200)
        result = ssim(x, y)
        assert -1.0 <= result <= 1.0 + 1e-6

    def test_compression_ratio_positive(self):
        cr = compression_ratio(n_pixels=10000, k=50, m_samples=100)
        assert cr > 1.0

    def test_compression_ratio_increases_with_smaller_k(self):
        cr_small = compression_ratio(10000, 10, 100)
        cr_large = compression_ratio(10000, 80, 100)
        assert cr_small > cr_large


# ─── Test Similarity Functions ─────────────────────────────────────────────────

class TestSimilarity:

    def test_cosine_identical_vectors(self):
        z = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(z, z) == pytest.approx(1.0)

    def test_cosine_opposite_vectors(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([-1.0, 0.0])
        assert cosine_similarity(z1, z2) == pytest.approx(-1.0)

    def test_cosine_orthogonal_vectors(self):
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 1.0])
        assert cosine_similarity(z1, z2) == pytest.approx(0.0)

    def test_cosine_zero_vector(self):
        z1 = np.zeros(5)
        z2 = np.random.rand(5)
        result = cosine_similarity(z1, z2)
        assert result == 0.0

    def test_cosine_range(self):
        for _ in range(20):
            z1 = np.random.rand(50) - 0.5
            z2 = np.random.rand(50) - 0.5
            sim = cosine_similarity(z1, z2)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6

    def test_cosine_symmetric(self):
        z1 = np.random.rand(50)
        z2 = np.random.rand(50)
        assert cosine_similarity(z1, z2) == pytest.approx(cosine_similarity(z2, z1))

    def test_euclidean_identical_vectors(self):
        z = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(z, z) == pytest.approx(0.0)

    def test_euclidean_nonnegative(self):
        z1 = np.random.rand(50)
        z2 = np.random.rand(50)
        assert euclidean_distance(z1, z2) >= 0.0

    def test_euclidean_triangle_inequality(self):
        z1 = np.random.rand(50)
        z2 = np.random.rand(50)
        z3 = np.random.rand(50)
        d12 = euclidean_distance(z1, z2)
        d23 = euclidean_distance(z2, z3)
        d13 = euclidean_distance(z1, z3)
        assert d13 <= d12 + d23 + 1e-10


# ─── Test Integrasi Ringan ──────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self):
        """Test alur lengkap: fit → transform → reconstruct → similarity."""
        np.random.seed(0)
        X = np.random.rand(30, 150).astype(np.float64)

        pca = FacePCA(n_components=10)
        Z   = pca.fit_transform(X)
        Xhat = pca.reconstruct(Z)

        # MSE harus kecil tapi > 0 (lossy compression)
        err = mse(X, Xhat)
        assert 0 < err < 1.0

        # Dua vektor dari array yang sama harus similarity tinggi
        z1, z2 = Z[0], Z[0].copy()
        assert cosine_similarity(z1, z2) == pytest.approx(1.0)

    def test_same_person_higher_similarity_than_different(self):
        """
        Wajah dari orang yang sama (sedikit berbeda) seharusnya
        lebih mirip daripada wajah dari orang yang berbeda.
        """
        np.random.seed(1)
        base   = np.random.rand(200)
        same   = base + np.random.rand(200) * 0.05  # sedikit noise
        diff   = np.random.rand(200)                 # benar-benar berbeda

        pca = FacePCA(n_components=20)
        pca.fit(np.vstack([base.reshape(1,-1)] * 20))

        z_base = pca.transform(base.reshape(1,-1))[0]
        z_same = pca.transform(same.reshape(1,-1))[0]
        z_diff = pca.transform(diff.reshape(1,-1))[0]

        sim_same = cosine_similarity(z_base, z_same)
        sim_diff = cosine_similarity(z_base, z_diff)

        assert sim_same > sim_diff, \
            f"Wajah sama ({sim_same:.4f}) seharusnya > wajah berbeda ({sim_diff:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
