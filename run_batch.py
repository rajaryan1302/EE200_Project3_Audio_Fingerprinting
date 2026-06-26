"""
run_batch.py  –  Command-Line Batch Identification
====================================================
Identify a folder of query clips against the fingerprint database and
write results.csv.

Usage:
    python run_batch.py --query_dir query_clips/ --db db/fingerprint_db.pkl

Output:
    results.csv   (two columns: filename, prediction)
"""

import os
import csv
import argparse
import librosa

from audio_processor import (
    SAMPLE_RATE,
    load_database,
    match_audio,
)

SUPPORTED = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def run_batch(query_dir: str, db_path: str, output_csv: str = "results.csv"):
    db = load_database(db_path)

    query_files = sorted(
        f for f in os.listdir(query_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    )

    if not query_files:
        print(f"No supported audio files found in '{query_dir}'.")
        return

    rows = []
    for filename in query_files:
        filepath = os.path.join(query_dir, filename)
        print(f"  {filename} …", end=" ", flush=True)
        try:
            y, sr  = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            best, _, _ = match_audio(y, sr, db, use_pairs=True)
            prediction = best if best else "no_match"
        except Exception as exc:
            prediction = f"error"
            print(f"ERROR: {exc}", end=" ")

        print(f"→ {prediction}")
        rows.append({"filename": filename, "prediction": prediction})

    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  results.csv written ({len(rows)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch audio identification")
    parser.add_argument("--query_dir", default="query_clips",
                        help="Directory of query audio clips")
    parser.add_argument("--db", default="db/fingerprint_db.pkl",
                        help="Path to fingerprint database")
    parser.add_argument("--output", default="results.csv",
                        help="Output CSV filename")
    args = parser.parse_args()
    run_batch(args.query_dir, args.db, args.output)
