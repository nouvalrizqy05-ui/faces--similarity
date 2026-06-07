"""
pca_engine.py
=============
Implementasi PCA dari scratch menggunakan NumPy (tanpa sklearn.decomposition.PCA)
untuk tujuan akademik — menunjukkan eigenvalue & eigenvector secara eksplisit.

Referensi: Modul Aljabar Linear — Eigenvector, Eigenvalue, dan PCA
"""

import numpy as np
import os
import cv2
import pickle
from pathlib import Path


# ─────────────────────────────────────────────
# Konstanta default
# ─────────────────────────────────────────────
IMG_SIZE     = (100, 100)   # Ukuran standar setiap gambar
N_COMPONENTS = 50           # Jumlah principal component default


# ─────────────────────────────────────────────
# 1. Preprocessing Citra
# ─────────────────────────────────────────────

def load_face_image(path: str, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Membaca gambar wajah, mengubah ke grayscale, resize, normalisasi,
    lalu flatten menjadi vektor 1D.

    Parameters
    ----------
    path     : str   — path ke file gambar
    img_size : tuple — ukuran target (width, height)

    Returns
    -------
    np.ndarray shape (img_size[0] * img_size[1],) dengan nilai [0, 1]
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized   = cv2.resize(gray, img_size)
    normalized = resized.astype(np.float64) / 255.0
    return normalized.flatten()


def load_dataset(dataset_path: str, img_size: tuple = IMG_SIZE):
    """
    Membaca seluruh gambar dari struktur folder:
        dataset/
          ├── 001/
          │     ├── 001A02.JPG
          │     └── ...
          └── 002/
                └── ...

    Returns
    -------
    X      : np.ndarray (m, n)   — matriks data wajah
    labels : np.ndarray (m,)     — label ID orang per gambar
    paths  : list[str]           — path file per gambar
    """
    X, labels, paths = [], [], []

    for person_id in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_id)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(person_dir, fname)
                try:
                    vec = load_face_image(fpath, img_size)
                    X.append(vec)
                    labels.append(person_id)
                    paths.append(fpath)
                except Exception:
                    pass  # lewati file rusak

    return np.array(X), np.array(labels), paths


# ─────────────────────────────────────────────
# 2. Implementasi PCA dari Scratch (via SVD)
# ─────────────────────────────────────────────

class FacePCA:
    """
    PCA untuk data wajah, diimplementasikan dari scratch menggunakan SVD.

    Alur sesuai panduan (Aljabar Linear — Tahap 1–12):
        1. Load data  → matriks X ∈ ℝ^(m×n)
        2. Centering  → Xc = X − μ
        3. SVD        → Xc = U Σ V^T
        4. Pilih k komponen (V_k = principal components = eigenfaces)
        5. Proyeksi   → Z = Xc · V_k
        6. Rekonstruksi → X̂ = Z · V_k^T + μ
    """

    def __init__(self, n_components: int = N_COMPONENTS):
        self.n_components   = n_components
        self.mean_face_     = None   # μ  — wajah rata-rata
        self.components_    = None   # V_k — eigenvectors (eigenfaces)
        self.singular_values_ = None # Σ diagonal
        self.eigenvalues_   = None   # λ_i = σ_i² / (m-1)
        self.explained_variance_ratio_ = None
        self._n_samples     = None
        self.is_fitted      = False

    # ── 2a. Fit ─────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray):
        """
        Latih model PCA.

        Parameters
        ----------
        X : np.ndarray (m, n) — data wajah (baris = sampel, kolom = piksel)
        """
        m, n = X.shape
        self._n_samples = m

        # Langkah 4: Centering — Xc = X − μ
        self.mean_face_ = np.mean(X, axis=0)        # μ ∈ ℝ^n
        Xc = X - self.mean_face_                    # Xc ∈ ℝ^(m×n)

        # Langkah 6: SVD — Xc = U Σ V^T
        # full_matrices=False → Reduced SVD, lebih efisien
        U, sigma, VT = np.linalg.svd(Xc, full_matrices=False)

        # Pilih k komponen terbesar
        k = min(self.n_components, len(sigma))
        self.singular_values_ = sigma[:k]
        self.components_      = VT[:k]              # V_k ∈ ℝ^(k×n)

        # Eigenvalue dari singular value: λ_i = σ_i² / (m-1)
        self.eigenvalues_ = (sigma ** 2) / (m - 1)
        ev_total          = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_[:k] / ev_total

        self.is_fitted = True
        return self

    # ── 2b. Transform ────────────────────────────────────────────────────
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyeksikan data ke ruang PCA.
        Z = Xc · V_k^T

        Returns
        -------
        Z : np.ndarray (m, k)
        """
        self._check_fitted()
        Xc = X - self.mean_face_
        return Xc @ self.components_.T             # Z ∈ ℝ^(m×k)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── 2c. Reconstruct ──────────────────────────────────────────────────
    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """
        Rekonstruksi citra dari ruang PCA.
        X̂ = Z · V_k + μ

        Parameters
        ----------
        Z : np.ndarray (m, k) — data di ruang PCA

        Returns
        -------
        X̂ : np.ndarray (m, n) — citra rekonstruksi (nilai bisa di luar [0,1])
        """
        self._check_fitted()
        return Z @ self.components_ + self.mean_face_

    # ── 2d. Utility ──────────────────────────────────────────────────────
    def cumulative_explained_variance(self) -> np.ndarray:
        """Cumulative explained variance ratio per komponen."""
        return np.cumsum(self.explained_variance_ratio_)

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Jumlah komponen minimum untuk mencapai threshold explained variance."""
        cumvar = self.cumulative_explained_variance()
        indices = np.where(cumvar >= threshold)[0]
        return int(indices[0] + 1) if len(indices) > 0 else self.n_components

    def get_eigenfaces(self, img_size: tuple = IMG_SIZE) -> np.ndarray:
        """
        Kembalikan eigenfaces sebagai array citra 2D.

        Returns
        -------
        np.ndarray (k, h, w)
        """
        self._check_fitted()
        faces = []
        for comp in self.components_:
            face = comp.reshape(img_size[1], img_size[0])
            # Normalisasi ke [0,1] untuk visualisasi
            face = (face - face.min()) / (face.max() - face.min() + 1e-8)
            faces.append(face)
        return np.array(faces)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit. Panggil fit() terlebih dahulu.")

    # ── 2e. Save / Load ──────────────────────────────────────────────────
    def save(self, path: str):
        """Simpan model ke file pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        """Muat model dari file pickle."""
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        return self


# ─────────────────────────────────────────────
# 3. Metrik Evaluasi
# ─────────────────────────────────────────────

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error antara citra asli dan rekonstruksi."""
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray,
         max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Semakin tinggi → kualitas rekonstruksi semakin baik.
    > 40 dB = sangat baik
    """
    error = mse(original, reconstructed)
    if error == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / error))


def ssim(x: np.ndarray, y: np.ndarray,
         k1: float = 0.01, k2: float = 0.03,
         data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM).
    Mendekati 1 → sangat mirip.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x, mu_y   = np.mean(x), np.mean(y)
    sigma_x2     = np.var(x)
    sigma_y2     = np.var(y)
    sigma_xy     = np.mean((x - mu_x) * (y - mu_y))
    numerator    = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator  = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(numerator / denominator)


def compression_ratio(n_pixels: int, k: int, m_samples: int) -> float:
    """
    Rasio kompresi PCA.
    Data asli  : m × n piksel
    Data simpan: n×k (eigenvectors) + m×k (proyeksi) + n (mean)
    """
    original_size    = m_samples * n_pixels
    compressed_size  = n_pixels * k + m_samples * k + n_pixels
    return original_size / compressed_size


# ─────────────────────────────────────────────
# 4. Fungsi Kemiripan Wajah
# ─────────────────────────────────────────────

def euclidean_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """Jarak Euclidean antara dua vektor PCA."""
    return float(np.linalg.norm(z1 - z2))


def cosine_similarity(z1: np.ndarray, z2: np.ndarray) -> float:
    """
    Cosine similarity antara dua vektor PCA.
    Range [-1, 1]; mendekati 1 = sangat mirip.
    """
    norm1 = np.linalg.norm(z1)
    norm2 = np.linalg.norm(z2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(z1, z2) / (norm1 * norm2))


def compare_faces(img_path_1: str, img_path_2: str,
                  pca_model: FacePCA,
                  threshold_cosine: float = 0.80,
                  img_size: tuple = IMG_SIZE) -> dict:
    """
    Bandingkan dua gambar wajah menggunakan PCA.

    Returns
    -------
    dict dengan kunci:
        cosine_similarity, euclidean_distance, is_similar,
        face1_pca, face2_pca
    """
    v1 = load_face_image(img_path_1, img_size)
    v2 = load_face_image(img_path_2, img_size)

    z1 = pca_model.transform(v1.reshape(1, -1))[0]
    z2 = pca_model.transform(v2.reshape(1, -1))[0]

    cos_sim  = cosine_similarity(z1, z2)
    euc_dist = euclidean_distance(z1, z2)
    is_sim   = cos_sim >= threshold_cosine

    return {
        "cosine_similarity"  : cos_sim,
        "euclidean_distance" : euc_dist,
        "is_similar"         : is_sim,
        "face1_pca"          : z1,
        "face2_pca"          : z2,
    }


def recognize_face(query_path: str,
                   pca_model: FacePCA,
                   X_pca: np.ndarray,
                   labels: np.ndarray,
                   threshold: float = 0.80,
                   img_size: tuple = IMG_SIZE) -> tuple:
    """
    Cari wajah paling mirip dari database.

    Returns
    -------
    (best_label, best_similarity, all_similarities)
    """
    vec   = load_face_image(query_path, img_size)
    z_q   = pca_model.transform(vec.reshape(1, -1))[0]

    sims  = np.array([cosine_similarity(z_q, z) for z in X_pca])
    best_idx   = int(np.argmax(sims))
    best_sim   = float(sims[best_idx])
    best_label = labels[best_idx] if best_sim >= threshold else "Tidak dikenal"

    return best_label, best_sim, sims
