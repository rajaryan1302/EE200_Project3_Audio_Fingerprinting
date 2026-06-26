"""
build_db.py  –  Pre-build the fingerprint database
====================================================
Run ONCE before deploying to Streamlit Cloud (or any server) to
create  db/fingerprint_db.pkl  from the songs/ directory.

The resulting .pkl file should be committed to the repository so the
deployed app loads instantly without re-indexing at startup.

Usage:
    python build_db.py [--songs songs/] [--db db/fingerprint_db.pkl]
"""

import os
import argparse
from audio_processor import build_database

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fingerprint database")
    parser.add_argument("--songs", default="songs",
                        help="Directory containing reference audio files")
    parser.add_argument("--db", default="db/fingerprint_db.pkl",
                        help="Output path for the database pickle file")
    args = parser.parse_args()

    if not os.path.isdir(args.songs):
        print(f"ERROR: songs directory not found: '{args.songs}'")
        raise SystemExit(1)

    db = build_database(args.songs, db_path=args.db)
    print(f"\n✅  Database built: {len(db) - 1} unique hashes across "
          f"{len(db.get('__song_list__', []))} songs.")
    print(f"   Saved to: {args.db}")
