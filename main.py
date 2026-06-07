"""
train.py
========
Script untuk melatih model PCA dari dataset FG-NET.

Cara pakai:
    python src/train.py --dataset dataset/ --n_components 50 --output models/
"""

import argparse
import os
import sys
import time
import numpy as np
import pickle
from pathlib import Path

# Tambahkan src/ ke path
sys.path.insert(0, os.path.dirname(__file__))
from pca_engine import (
    FacePCA, load_dataset, mse, psnr, ssim, compression_ratio,
    IMG_SIZE, N_COMPONENTS
)


def evaluate_reconstruction(pca: FacePCA, X: np.ndarray) -> dict:
    """Evaluasi kualitas rekonstruksi untuk berbagai nilai k."""
    Z  = pca.transform(X)
    X_hat = pca.reconstruct(Z)

    mse_val  = mse(X, X_hat)
    psnr_val = psnr(X, X_hat, max_val=1.0)
    ssim_val = np.mean([ssim(X[i], X_hat[i]) for i in range(min(50, len(X)))])
    cr_val   = compression_ratio(X.shape[1], pca.n_components, X.shape[0])

    return {
        "mse"               : mse_val,
        "psnr_db"           : psnr_val,
        "ssim"              : ssim_val,
        "compression_ratio" : cr_val,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def main(args):
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  FACE SIMILARITY PCA — TRAINING")
    print("=" * 60)

    # ── 1. Load Dataset ──────────────────────────────────────────
    print(f"\n[1/5] Loading dataset dari: {args.dataset}")
    t0 = time.time()
    X, labels, paths = load_dataset(args.dataset, IMG_SIZE)
    print(f"      Loaded {X.shape[0]} gambar dari {len(np.unique(labels))} orang")
    print(f"      Shape matriks X: {X.shape}  ({time.time()-t0:.1f}s)")

    # ── 2. Fit PCA ───────────────────────────────────────────────
    print(f"\n[2/5] Menjalankan PCA (k={args.n_components}) via SVD...")
    t0 = time.time()
    pca = FacePCA(n_components=args.n_components)
    Z   = pca.fit_transform(X)
    print(f"      Selesai  ({time.time()-t0:.1f}s)")
    print(f"      Mean face shape  : {pca.mean_face_.shape}")
    print(f"      Eigenfaces shape : {pca.components_.shape}")
    print(f"      Top-5 eigenvalues: {pca.eigenvalues_[:5].round(2)}")

    # ── 3. Explained Variance Report ─────────────────────────────
    print(f"\n[3/5] Explained Variance Report:")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    for t in thresholds:
        k_needed = pca.n_components_for_variance(t)
        print(f"      {int(t*100)}% variance → {k_needed} komponen dibutuhkan")
    total_ev = float(np.sum(pca.explained_variance_ratio_))
    print(f"      Total dengan k={args.n_components}: {total_ev*100:.2f}%")

    # ── 4. Evaluasi Rekonstruksi ──────────────────────────────────
    print(f"\n[4/5] Evaluasi Rekonstruksi:")
    metrics = evaluate_reconstruction(pca, X)
    print(f"      MSE              : {metrics['mse']:.6f}")
    print(f"      PSNR             : {metrics['psnr_db']:.2f} dB")
    print(f"      SSIM             : {metrics['ssim']:.4f}")
    print(f"      Compression Ratio: {metrics['compression_ratio']:.2f}x")

    # ── 5. Simpan Model ───────────────────────────────────────────
    print(f"\n[5/5] Menyimpan model...")

    model_path = os.path.join(args.output, "pca_model.pkl")
    pca.save(model_path)
    print(f"      Model PCA    → {model_path}")

    # Simpan juga data proyeksi + label untuk recognize_face()
    db_path = os.path.join(args.output, "face_database.pkl")
    with open(db_path, 'wb') as f:
        pickle.dump({"Z": Z, "labels": labels, "paths": paths,
                     "metrics": metrics, "img_size": IMG_SIZE}, f)
    print(f"      Face database → {db_path}")

    print("\n" + "=" * 60)
    print("  TRAINING SELESAI")
    print("=" * 60)
    return pca, Z, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PCA model untuk face similarity FG-NET")
    parser.add_argument("--dataset",      default="dataset/",
                        help="Path ke folder dataset terorganisir")
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS,
                        help="Jumlah principal components (default: 50)")
    parser.add_argument("--output",       default="models/",
                        help="Folder output untuk model")
    args = parser.parse_args()
    main(args)
