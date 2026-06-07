"""
reorganize_dataset.py
=====================
Reorganisasi dataset FG-NET dari format flat (images/XXXAYYY.JPG)
ke format per-orang (dataset/XXX/XXXAYYY.JPG) sesuai panduan.

Cara pakai:
    python src/reorganize_dataset.py --input images_raw/images/ --output dataset/
"""

import os
import re
import shutil
import argparse


FGNET_PATTERN = re.compile(r'^(\d{3})A\d+[a-z]?\.JPG$', re.IGNORECASE)


def reorganize(input_dir: str, output_dir: str, dry_run: bool = False) -> dict:
    """
    Reorganisasi gambar FG-NET ke struktur per-orang.

    Returns
    -------
    dict: {'moved': int, 'skipped': int, 'persons': set}
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory tidak ditemukan: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    files   = os.listdir(input_dir)
    moved   = 0
    skipped = 0
    persons = set()

    for fname in sorted(files):
        m = FGNET_PATTERN.match(fname)
        if not m:
            print(f"  [SKIP] {fname} — tidak cocok pola FG-NET")
            skipped += 1
            continue

        person_id  = m.group(1)
        person_dir = os.path.join(output_dir, person_id)
        src_path   = os.path.join(input_dir, fname)
        dst_path   = os.path.join(person_dir, fname)

        persons.add(person_id)

        if not dry_run:
            os.makedirs(person_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        else:
            print(f"  [DRY] {fname} → {dst_path}")

        moved += 1

    return {"moved": moved, "skipped": skipped, "persons": persons}


def main():
    parser = argparse.ArgumentParser(
        description="Reorganisasi dataset FG-NET ke struktur per-orang")
    parser.add_argument("--input",   default="images_raw/images/",
                        help="Folder flat FG-NET (berisi file XXXAYYY.JPG)")
    parser.add_argument("--output",  default="dataset/",
                        help="Folder output per-orang")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tampilkan rencana tanpa benar-benar menyalin")
    args = parser.parse_args()

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Dry run: {args.dry_run}\n")

    result = reorganize(args.input, args.output, dry_run=args.dry_run)

    print(f"\nSelesai!")
    print(f"  File dipindah : {result['moved']}")
    print(f"  File dilewati : {result['skipped']}")
    print(f"  Total orang   : {len(result['persons'])}")


if __name__ == "__main__":
    main()
