"""
generate_report_plots.py  –  Q3A Experiment Script
====================================================
Run this script to reproduce ALL plots required by the assignment report.
Outputs are saved to the report_plots/ directory as high-resolution PNGs.

Usage:
    python generate_report_plots.py [--song PATH] [--query PATH]

If no song/query are supplied, a synthetic test tone is generated so the
script can run end-to-end without real audio files.  Replace with real
songs for the actual submission.

Plots produced
--------------
  01_dft_baseline.png          – Full-song DFT (timing lost)
  02_spectrogram.png           – Standard STFT spectrogram
  03_window_comparison.png     – Short vs long window comparison
  04_constellation.png         – Sparse peak constellation
  05_offset_histogram.png      – Offset histograms for all songs
  06_single_vs_pairs.png       – Single-peak vs paired-hash experiment
  07_noise_robustness.png      – Match confidence vs added noise SNR
  08_pitch_shift.png           – Match confidence vs pitch shift (semitones)
"""

import os
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_processor import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT,
    compute_spectrogram, find_peaks_2d, generate_hashes,
    plot_dft_baseline, plot_spectrogram, plot_window_comparison,
    plot_constellation, match_audio, build_database, load_database,
    experiment_single_vs_pairs, experiment_noise_robustness,
    pitch_shift_and_match,
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR  = os.path.join(BASE_DIR, "songs")
DB_PATH    = os.path.join(BASE_DIR, "db", "fingerprint_db.pkl")
OUT_DIR    = os.path.join(BASE_DIR, "report_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC AUDIO GENERATOR  (fallback when no real files are present)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_song(
    duration: float = 30.0,
    sr: int = SAMPLE_RATE,
    freqs: list = None
) -> np.ndarray:
    """
    Create a multi-tone synthetic 'song' for testing without real audio.

    The signal is a sum of sinusoids at the given frequencies, with random
    AM envelopes so the spectrogram looks interesting.
    """
    if freqs is None:
        freqs = [220, 330, 440, 550, 660, 880, 1100, 1320]

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)

    rng = np.random.default_rng(42)
    for f in freqs:
        # Random amplitude envelope (slow AM)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t)
        y += am * np.sin(2 * np.pi * f * t)

    # Add gentle pink-ish noise
    noise = rng.normal(0, 0.05, t.shape)
    y = (y + noise).astype(np.float32)
    y /= np.max(np.abs(y) + 1e-9)   # normalise to [-1, 1]
    return y


def ensure_test_environment(song_path, query_path):
    """
    If real audio files aren't available, create synthetic ones and build a
    minimal test database so all experiments can still run.
    """
    # ── songs directory ────────────────────────────────────────────────────
    if not os.path.isdir(SONGS_DIR) or not any(
        f.endswith((".mp3", ".wav", ".flac")) for f in os.listdir(SONGS_DIR)
    ):
        print("[SETUP] No real songs found – generating synthetic test audio.")
        os.makedirs(SONGS_DIR, exist_ok=True)

        import soundfile as sf

        for name, freqs in [
            ("song_A", [220, 440, 880]),
            ("song_B", [330, 660, 990]),
            ("song_C", [415, 554, 831]),
        ]:
            wav_path = os.path.join(SONGS_DIR, f"{name}.wav")
            if not os.path.exists(wav_path):
                y_syn = make_synthetic_song(freqs=freqs)
                sf.write(wav_path, y_syn, SAMPLE_RATE)
                print(f"  Created {wav_path}")

    # ── query file ─────────────────────────────────────────────────────────
    if query_path is None or not os.path.exists(query_path):
        import soundfile as sf
        query_path = os.path.join(BASE_DIR, "query_clips", "test_query.wav")
        os.makedirs(os.path.dirname(query_path), exist_ok=True)
        if not os.path.exists(query_path):
            # Use first 10 s of song_A as a clean query
            y_syn = make_synthetic_song(duration=10.0, freqs=[220, 440, 880])
            sf.write(query_path, y_syn, SAMPLE_RATE)
            print(f"  Created test query: {query_path}")

    # ── song_path for single-file plots ───────────────────────────────────
    if song_path is None or not os.path.exists(song_path):
        files = [
            os.path.join(SONGS_DIR, f)
            for f in os.listdir(SONGS_DIR)
            if f.endswith(".wav")
        ]
        song_path = files[0] if files else None

    return song_path, query_path


def save(fig: plt.Figure, filename: str):
    """Save figure to report_plots/ and close it to free memory."""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate report plots.")
    parser.add_argument("--song",  default=None, help="Path to a reference song")
    parser.add_argument("--query", default=None, help="Path to a query clip")
    args = parser.parse_args()

    # ── Prepare environment ────────────────────────────────────────────────
    song_path, query_path = ensure_test_environment(args.song, args.query)

    # ── Load / build database ──────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        print("[DB] Loading existing database…")
        db = load_database(DB_PATH)
    else:
        print("[DB] Building database from songs/ directory…")
        db = build_database(SONGS_DIR, db_path=DB_PATH)

    # Load audio for single-song plots
    print(f"\n[AUDIO] Loading: {song_path}")
    y_song, sr = librosa.load(song_path, sr=SAMPLE_RATE, mono=True)

    print(f"[AUDIO] Loading query: {query_path}")
    y_query, _  = librosa.load(query_path, sr=SAMPLE_RATE, mono=True)

    song_label = os.path.splitext(os.path.basename(song_path))[0]

    # ── Plot 01: DFT Baseline ─────────────────────────────────────────────
    print("\n[PLOT 01] DFT baseline…")
    fig = plot_dft_baseline(y_song, sr, title=f"DFT Magnitude – '{song_label}'")
    save(fig, "01_dft_baseline.png")

    # ── Plot 02: Spectrogram ──────────────────────────────────────────────
    print("[PLOT 02] Spectrogram…")
    S_db, _, _ = compute_spectrogram(y_song, sr)
    fig = plot_spectrogram(S_db, sr, HOP_LENGTH,
                           title=f"Spectrogram – '{song_label}'")
    save(fig, "02_spectrogram.png")

    # ── Plot 03: Window comparison ────────────────────────────────────────
    print("[PLOT 03] Window-length comparison…")
    fig = plot_window_comparison(y_song, sr)
    save(fig, "03_window_comparison.png")

    # ── Plot 04: Constellation ────────────────────────────────────────────
    print("[PLOT 04] Constellation map…")
    peaks = find_peaks_2d(S_db)
    fig   = plot_constellation(S_db, peaks, sr, HOP_LENGTH,
                               title=f"Constellation – '{song_label}'  ({len(peaks)} peaks)")
    save(fig, "04_constellation.png")

    # ── Plot 05: Offset histograms ────────────────────────────────────────
    print("[PLOT 05] Offset histograms…")
    _, _, fig = match_audio(y_query, sr, db, use_pairs=True)
    save(fig, "05_offset_histogram.png")

    # ── Plot 06: Single vs Pairs ──────────────────────────────────────────
    print("[PLOT 06] Single peaks vs paired hashes…")
    fig = experiment_single_vs_pairs(
        y_query, sr, db,
        query_name=os.path.splitext(os.path.basename(query_path))[0]
    )
    save(fig, "06_single_vs_pairs.png")

    # ── Plot 07: Noise robustness ─────────────────────────────────────────
    print("[PLOT 07] Noise robustness…")
    fig = experiment_noise_robustness(
        y_query, sr, db,
        snr_levels_db=[40, 30, 20, 15, 10, 5, 0, -5]
    )
    save(fig, "07_noise_robustness.png")

    # ── Plot 08: Pitch shift ──────────────────────────────────────────────
    print("[PLOT 08] Pitch-shift robustness…")
    fig = pitch_shift_and_match(
        y_query, sr, db,
        semitones_list=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    )
    save(fig, "08_pitch_shift.png")

    print(f"\n✅  All plots saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
