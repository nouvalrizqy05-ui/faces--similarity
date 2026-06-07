"""
compare.py
==========
CLI tool untuk membandingkan dua gambar wajah menggunakan model PCA terlatih.

Cara pakai:
    python src/compare.py wajah_1.jpg wajah_2.jpg
    python src/compare.py wajah_1.jpg wajah_2.jpg --threshold 0.75
    python src/compare.py wajah_1.jpg wajah_2.jpg --model models/pca_model.pkl
"""

import argparse
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pca_engine import FacePCA, compare_faces, recognize_face, IMG_SIZE


def load_model(model_path: str) -> FacePCA:
    pca = FacePCA()
    pca.load(model_path)
    return pca


def load_database(db_path: str) -> dict:
    with open(db_path, 'rb') as f:
        return pickle.load(f)


def print_separator():
    print("─" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Bandingkan dua wajah menggunakan PCA/SVD")
    parser.add_argument("image1",     help="Path gambar wajah pertama")
    parser.add_argument("image2",     nargs="?", default=None,
                        help="Path gambar wajah kedua (opsional)")
    parser.add_argument("--model",    default="models/pca_model.pkl",
                        help="Path model PCA")
    parser.add_argument("--db",       default="models/face_database.pkl",
                        help="Path face database")
    parser.add_argument("--threshold",type=float, default=0.80,
                        help="Threshold cosine similarity (default: 0.80)")
    parser.add_argument("--mode",     choices=["compare", "recognize"],
                        default="compare",
                        help="Mode: compare (2 gambar) atau recognize (cari dari DB)")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"ERROR: Model tidak ditemukan di {args.model}")
        print("       Jalankan dulu: python src/train.py")
        sys.exit(1)

    print_separator()
    print("  FACE SIMILARITY PCA/SVD")
    print_separator()

    pca = load_model(args.model)
    print(f"  Model loaded   : {args.model}")
    print(f"  n_components   : {pca.n_components}")
    print(f"  Threshold      : {args.threshold}")

    # ── Mode: Compare ─────────────────────────────────────────────
    if args.mode == "compare":
        if args.image2 is None:
            print("ERROR: Mode compare membutuhkan dua gambar.")
            sys.exit(1)

        print_separator()
        print(f"  Gambar 1 : {args.image1}")
        print(f"  Gambar 2 : {args.image2}")
        print_separator()

        result = compare_faces(
            args.image1, args.image2, pca,
            threshold_cosine=args.threshold
        )

        cos  = result["cosine_similarity"]
        euc  = result["euclidean_distance"]
        sim  = result["is_similar"]

        print(f"  Cosine Similarity   : {cos:.4f}")
        print(f"  Euclidean Distance  : {euc:.4f}")
        print_separator()

        if sim:
            print(f"  ✓ MIRIP  (similarity {cos:.4f} ≥ threshold {args.threshold})")
        else:
            print(f"  ✗ TIDAK MIRIP  (similarity {cos:.4f} < threshold {args.threshold})")

        print_separator()

    # ── Mode: Recognize ──────────────────────────────────────────
    elif args.mode == "recognize":
        if not os.path.exists(args.db):
            print(f"ERROR: Database tidak ditemukan di {args.db}")
            sys.exit(1)

        db = load_database(args.db)
        print_separator()
        print(f"  Query    : {args.image1}")
        print(f"  Database : {len(db['labels'])} gambar")
        print_separator()

        best_label, best_sim, all_sims = recognize_face(
            args.image1, pca, db["Z"], db["labels"],
            threshold=args.threshold
        )

        print(f"  Hasil Identifikasi  : {best_label}")
        print(f"  Similarity terbaik  : {best_sim:.4f}")

        # Top-5
        top5_idx = np.argsort(all_sims)[::-1][:5]
        print(f"\n  Top-5 kandidat:")
        for rank, idx in enumerate(top5_idx, 1):
            print(f"    {rank}. Person {db['labels'][idx]}  "
                  f"— similarity {all_sims[idx]:.4f}  "
                  f"({os.path.basename(db['paths'][idx])})")
        print_separator()


if __name__ == "__main__":
    main()
