"""
audio_processor.py  –  Core Audio Fingerprinting Engine
=========================================================
Implements a Shazam-style audio fingerprinting pipeline:

  1. DFT baseline          – full-song spectrum (timing lost)
  2. STFT spectrogram      – time × frequency energy map
  3. Peak / constellation  – local maxima retained as landmarks
  4. Paired hashing        – anchor + target peak pairs → compact fingerprint
  5. Database build        – hash table keyed on (f1, f2, Δt)
  6. Matching              – offset histogram → song identity
  7. Robustness tests      – noise & pitch-shift stress tests

All functions are self-contained and return (figure, data) pairs so they
can be called from both the CLI report scripts and the Streamlit front-end.
"""

import os
import hashlib
import pickle
from typing import Optional, Dict, List, Tuple

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")           # headless – no GUI required
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL ALGORITHM CONSTANTS
#  Tune these to balance speed vs. robustness.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 22050   # Hz – librosa default; matches most reference tracks
HOP_LENGTH         = 512     # STFT hop (≈23 ms at 22 kHz)
N_FFT              = 2048    # STFT window size (≈93 ms)
N_FFT_SHORT        = 512     # Short window for time-res experiment
N_FFT_LONG         = 8192    # Long window for freq-res experiment

# Constellation (peak picking) parameters
PEAK_NEIGHBOURHOOD = 15      # Maximum-filter footprint (time × freq bins)
PEAK_AMP_MIN_DB    = -50     # Only keep peaks above this dB floor
PEAKS_PER_FRAME    = 5       # Max peaks retained per time-frame

# Pairing / hashing parameters
FAN_VALUE          = 12      # Number of target peaks paired with each anchor
MIN_HASH_TIME_DELTA = 0      # Minimum Δt between anchor and target (frames)
MAX_HASH_TIME_DELTA = 300    # Maximum Δt (frames) – ~4.7 s at hop=512
FREQ_BITS          = 10      # Frequency quantisation bits
TIME_BITS          = 10      # Time-delta quantisation bits

# Matching parameters
OFFSET_HISTOGRAM_MIN_VOTES = 3   # Minimum aligned offsets to claim a match


# ─────────────────────────────────────────────────────────────────────────────
#  1 · DFT BASELINE  (full-song magnitude spectrum)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dft_baseline(
    y: np.ndarray,
    sr: int,
    title: str = "DFT Magnitude Spectrum (full song)"
):
    """
    Demonstrates why DFT alone is insufficient for song identification.
    """

    # Use only first 20 seconds
    y = y[:20 * sr]

    N = len(y)
    N_pad = int(2 ** np.ceil(np.log2(N)))

    Y = np.fft.rfft(y, n=N_pad)

    magnitude_db = 20 * np.log10(np.abs(Y) + 1e-10)
    freqs = np.fft.rfftfreq(N_pad, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        freqs / 1000,
        magnitude_db,
        linewidth=1
    )

    # Zoom where most musical energy exists
    ax.set_xlim(0, 5)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    ax.text(
        3.2,
        np.max(magnitude_db)-15,
        "Timing information lost\nEntire song compressed\ninto one spectrum",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2 · STFT SPECTROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) spectrogram.

    A short window is slid along the signal with step `hop_length`.
    At each position we take the DFT of the windowed slice, producing a
    column of complex values.  Stacking columns → 2-D time×frequency array.

    Returns
    -------
    S_db    : log-power spectrogram  (freq_bins × time_frames)
    freqs   : frequency axis (Hz)
    times   : time axis (s)
    """
    # STFT – returns complex matrix (freq_bins × time_frames)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")

    # Convert amplitude to decibels; ref=np.max gives 0 dB at the loudest bin
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)          # Hz
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length   # seconds
    )
    return S_db, freqs, times


def plot_spectrogram(
    S_db: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot a log-power spectrogram using librosa's display utilities.

    A spectrogram encodes time on the x-axis, frequency on the y-axis,
    and energy (dB) as colour.  Unlike the plain DFT, we retain timing.

    Parameters
    ----------
    S_db       : log-power spectrogram array (freq × time)
    sr         : sample rate
    hop_length : STFT hop used to produce S_db
    title      : plot title
    ax         : optional existing Axes to draw into

    Returns
    -------
    matplotlib Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if standalone:
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  3 · WINDOW-SIZE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def plot_window_comparison(
    y: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH
) -> plt.Figure:
    """
    Demonstrate the STFT time-frequency resolution tradeoff.
    """
    start = min(80 * sr, len(y) // 2)
    end   = min(start + 15 * sr, len(y))
    y = y[start:end]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5)
    )

    configs = [
        (256,   "Short Window\nExcellent Time Resolution\nPoor Frequency Resolution"),
        (16384, "Long Window\nExcellent Frequency Resolution\nPoor Time Resolution")
    ]

    for ax, (n_fft, label) in zip(axes, configs):

        S_db, _, _ = compute_spectrogram(
            y,
            sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        plot_spectrogram(
            S_db,
            sr,
            hop_length,
            title=label,
            ax=ax
        )

    fig.suptitle(
        "STFT Window-Length Tradeoff",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  4 · PEAK PICKING  →  CONSTELLATION MAP
# ─────────────────────────────────────────────────────────────────────────────

from scipy.ndimage import maximum_filter
import numpy as np

def find_peaks_2d(
    S_db: np.ndarray,
    neighbourhood: int = 15,
    percentile: float = 98.5,
    max_peaks: int = 3000
) -> np.ndarray:
    """
    Robust peak detector for Shazam-style fingerprinting.
    """

    struct = np.ones(
        (neighbourhood, neighbourhood),
        dtype=bool
    )

    local_max = (
        maximum_filter(
            S_db,
            footprint=struct,
            mode="nearest"
        ) == S_db
    )

    threshold = max(
        np.percentile(S_db, percentile),
        -55
    )

    peak_mask = local_max & (S_db >= threshold)

    freq_idxs, time_idxs = np.where(peak_mask)

    if len(freq_idxs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    strengths = S_db[freq_idxs, time_idxs]

    if len(strengths) > max_peaks:

        keep = np.argsort(strengths)[-max_peaks:]

        freq_idxs = freq_idxs[keep]
        time_idxs = time_idxs[keep]

    peaks = np.stack(
        [freq_idxs, time_idxs],
        axis=1
    )

    peaks = peaks[np.argsort(peaks[:, 1])]

    return peaks.astype(np.int32)


def plot_constellation(
    S_db,
    peaks,
    sr,
    hop_length,
    title="Constellation Map"
):
    fig, ax = plt.subplots(figsize=(14, 6))

    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        alpha=0.15,
        ax=ax
    )

    freq_bins = peaks[:, 0]
    time_frames = peaks[:, 1]

    times_sec = librosa.frames_to_time(
        time_frames,
        sr=sr,
        hop_length=hop_length
    )

    freqs_hz = librosa.fft_frequencies(
        sr=sr,
        n_fft=(S_db.shape[0]-1)*2
    )[freq_bins]

    ax.scatter(
        times_sec,
        freqs_hz,
        s=15,
        alpha=0.8
    )

    ax.set_title(
        f"{title}\nDetected Spectral Peaks = {len(peaks)}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5 · PAIRED HASHING
# ─────────────────────────────────────────────────────────────────────────────

def generate_hashes(
    peaks: np.ndarray,
    fan_value: int = FAN_VALUE,
    min_delta_t: int = MIN_HASH_TIME_DELTA,
    max_delta_t: int = MAX_HASH_TIME_DELTA,
    use_pairs: bool = True
) -> List[Tuple[int, int]]:
    """
    Convert a constellation of peaks into a list of (hash, time_offset) tuples.

    Paired-hash construction:
      For each anchor peak  (f1, t1):
        for each of the next `fan_value` peaks  (f2, t2)  with t2 > t1:
          hash_key = encode(f1, f2, Δt)   where Δt = t2 − t1
          emit (hash_key, t1)

    A 32-bit integer is used as the hash:
        bits [21:12] = f1  quantised to FREQ_BITS
        bits [11: 2] = f2  quantised to FREQ_BITS
        bits [ 1: 0] = Δt  quantised to TIME_BITS

    (In practice we build the key as a Python int; 64-bit is fine.)

    Single-peak mode (use_pairs=False):
      Emit only (f1, t1) pairs — no pairing — to demonstrate why
      pairs are dramatically more discriminative.

    Parameters
    ----------
    peaks      : (N, 2) array [freq_bin, time_frame], sorted by time
    fan_value  : how many targets to pair with each anchor
    min/max_delta_t : valid time gap range (frames)
    use_pairs  : True → paired hashes; False → single-peak hashes

    Returns
    -------
    List of (hash_int, absolute_time_frame_of_anchor) tuples
    """
    hashes = []

    # Sort by time frame so we always look forward in time
    peaks_sorted = peaks[peaks[:, 1].argsort()]

    for i, (f1, t1) in enumerate(peaks_sorted):
        if not use_pairs:
            # Single-peak mode: hash encodes only f1
            h = int(f1) << TIME_BITS
            hashes.append((h, int(t1)))
            continue

        # Paired-hash mode: find the next `fan_value` peaks ahead in time
        paired = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2, t2 = peaks_sorted[j]
            delta_t = int(t2) - int(t1)

            if delta_t < min_delta_t:
                continue
            if delta_t > max_delta_t:
                break          # peaks are time-sorted, so we can break early

            # Quantise frequencies and time-delta into a compact integer key
            FREQ_QUANT = 4

            f1_q = (int(f1) // FREQ_QUANT) & ((1 << FREQ_BITS) - 1)
            f2_q = (int(f2) // FREQ_QUANT) & ((1 << FREQ_BITS) - 1)
            dt_q = int(delta_t) & ((1 << TIME_BITS) - 1)

            h = (f1_q << (FREQ_BITS + TIME_BITS)) | (f2_q << TIME_BITS) | dt_q
            hashes.append((h, int(t1)))

            paired += 1
            if paired >= fan_value:
                break

    return hashes


# ─────────────────────────────────────────────────────────────────────────────
#  6 · DATABASE  BUILD & LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def build_database(
    audio_dir: str,
    db_path: str = "db/fingerprint_db.pkl",
    extensions: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
) -> Dict:
    """
    Index all audio files in `audio_dir` into a fingerprint database.

    Database schema:
        {
          hash_key : [(song_name, anchor_time_frame), ...],
          ...
          "__song_list__": [song_name, ...]
        }

    The song label stored is the filename WITHOUT the extension, matching
    the assignment requirement.

    Parameters
    ----------
    audio_dir  : directory containing reference songs
    db_path    : where to pickle the finished database
    extensions : supported audio formats

    Returns
    -------
    db dict (also written to db_path)
    """
    db: Dict[int, List[Tuple[str, int]]] = {}
    song_list: List[str] = []

    audio_files = [
        f for f in os.listdir(audio_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in '{audio_dir}'.")

    print(f"[DB BUILD] Found {len(audio_files)} files in '{audio_dir}'.")

    for filename in sorted(audio_files):
        song_name = os.path.splitext(filename)[0]   # ← strip extension
        filepath  = os.path.join(audio_dir, filename)

        print(f"  Indexing: {filename} → label '{song_name}'")
        try:
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            print(f"    ✗ Could not load {filename}: {exc}")
            continue

        S_db, _, _ = compute_spectrogram(y, sr)
        peaks      = find_peaks_2d(S_db)
        hashes     = generate_hashes(peaks, use_pairs=True)

        for h, t in hashes:
            db.setdefault(h, []).append((song_name, t))

        song_list.append(song_name)
        print(f"    ✓ {len(peaks)} peaks, {len(hashes)} hashes")

    db["__song_list__"] = song_list

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "wb") as fh:
        pickle.dump(db, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DB BUILD] Done. Database saved to '{db_path}'.")
    return db


def load_database(db_path: str = "db/fingerprint_db.pkl") -> Dict:
    """Load a pre-built fingerprint database from disk."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. "
            "Run build_database() first."
        )
    with open(db_path, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
#  7 · MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_audio(
    y: np.ndarray,
    sr: int,
    db: Dict,
    use_pairs: bool = True
) -> Tuple[Optional[str], Dict[str, np.ndarray], plt.Figure]:
    """
    Identify an audio clip against the fingerprint database.

    Method:
      1. Fingerprint the query clip.
      2. For each hash match in the DB, record the time offset:
             offset = db_anchor_time − query_anchor_time
         A true match produces a large spike at a single offset value.
         False matches scatter uniformly.
      3. The song with the largest spike (votes) wins.

    Parameters
    ----------
    y         : query audio waveform
    sr        : sample rate
    db        : fingerprint database dict
    use_pairs : use paired hashes (True) or single peaks (False)

    Returns
    -------
    best_match : song name (or None)
    offsets    : dict mapping song_name → array of offset values
    fig        : offset histogram figure
    """
    # ── Fingerprint the query ────────────────────────────────────────────────
    S_db, _, _ = compute_spectrogram(y, sr)
    peaks      = find_peaks_2d(S_db)
    query_hashes = generate_hashes(peaks, use_pairs=use_pairs)

    # ── Accumulate offset votes per song ────────────────────────────────────
    #   For every hash in the query that also appears in the DB,
    #   we compute offset = db_time − query_time for each DB entry.
    song_offsets: Dict[str, List[int]] = {}

    for h, q_time in query_hashes:
        if h not in db or h == "__song_list__":
            continue
        for song_name, db_time in db[h]:
            offset = db_time - q_time
            song_offsets.setdefault(song_name, []).append(offset)

    # ── Find the best-matching song ─────────────────────────────────────────
    best_song   = None
    best_votes  = 0
    offset_arrays: Dict[str, np.ndarray] = {}

    for song_name, offsets in song_offsets.items():
        arr = np.array(offsets)
        offset_arrays[song_name] = arr

        # Count the most popular offset (histogram peak)
        if len(arr) > 0:
            counts, _ = np.histogram(
                arr,
                bins=np.arange(
                    arr.min(),
                    arr.max() + 6,
                    5
                )
            )

            max_votes = counts.max()
            if max_votes > best_votes:
                best_votes = max_votes
                best_song  = song_name

    if best_votes < OFFSET_HISTOGRAM_MIN_VOTES:
        best_song = None   # not confident enough

    # ── Plot offset histograms ───────────────────────────────────────────────

    n_songs = len(offset_arrays)

    if n_songs == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No hash matches found",
            ha="center",
            va="center",
            fontsize=14
        )
        ax.axis("off")
        return best_song, offset_arrays, fig


    # Show only top 5 songs by number of matches
    top_songs = sorted(
        offset_arrays.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]

    n_plots = len(top_songs)

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(12, max(4, 3 * n_plots)),
        squeeze=False
    )

    for idx, (song_name, arr) in enumerate(top_songs):

        ax = axes[idx][0]

        peak_votes = 0

        if len(arr) > 0:

            # Adaptive bin count
            bins = min(
                100,
                max(
                    20,
                    len(np.unique(arr)) // 2
                )
            )

            counts, edges = np.histogram(
                arr,
                bins=bins
            )

            peak_idx = np.argmax(counts)
            peak_votes = counts[peak_idx]

            peak_center = (
                edges[peak_idx] +
                edges[peak_idx + 1]
            ) / 2

            # Histogram
            ax.hist(
                arr,
                bins=bins,
                alpha=0.85
            )

            # Dominant offset location
            ax.axvline(
                peak_center,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Peak votes = {peak_votes}"
            )

            ax.legend()

        is_match = (song_name == best_song)

        if is_match:
            ax.set_facecolor("#FFF5F5")

        ax.set_title(
            f"{song_name}"
            f"{'  ← MATCH ✓' if is_match else ''}"
            f"   (peak votes = {peak_votes})",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_xlabel("Time Offset (frames)")
        ax.set_ylabel("Votes")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Offset Histograms | Mode = "
        f"{'Paired Hashes' if use_pairs else 'Single Peaks'}",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    return best_song, offset_arrays, fig


# ─────────────────────────────────────────────────────────────────────────────
#  8 · SINGLE-PEAK vs PAIRED-HASH EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def experiment_single_vs_pairs(
    y_query: np.ndarray,
    sr: int,
    db: Dict,
    query_name: str = "query"
) -> plt.Figure:
    """
    Run matching twice – once with single peaks, once with pairs – and
    return a comparison figure of the offset histograms.

    Why pairs win:
      A single-peak hash encodes only one frequency value.  Many songs
      share the same prominent frequency at different times, producing
      many false offset matches spread across all offsets.

      A paired hash encodes (f1, f2, Δt) – the probability that two songs
      share the same *pair* of frequencies at the same *time gap* is tiny,
      so false matches nearly vanish and the true match spike dominates.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, use_pairs, label in [
        (axes[0], False, "Single-peak hashes  (high false-match noise)"),
        (axes[1], True,  "Paired hashes  (decisive match spike)"),
    ]:
        best, offsets, _ = match_audio(y_query, sr, db, use_pairs=use_pairs)
        if offsets:
            top_song = max(offsets, key=lambda k: len(offsets[k]))
            arr = offsets[top_song]
            if len(arr) > 0:
                bins = np.arange(arr.min(), arr.max() + 2) - 0.5
                ax.hist(arr, bins=bins, color="#FF6B35", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"{label}\n  → identified as: {best or 'no match'} (query: {query_name})",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Time offset (frames)")
        ax.set_ylabel("Votes")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Experiment: Single Peaks vs Paired Hashes",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  9 · ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def add_white_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise at the specified signal-to-noise ratio (dB).

    SNR_dB = 10 * log10( P_signal / P_noise )
    → P_noise = P_signal / 10^(SNR_dB/10)
    """
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def experiment_noise_robustness(
    y_query: np.ndarray,
    sr: int,
    db: Dict,
    snr_levels_db: List[float] = [40, 30, 20, 10, 5, 0, -5]
) -> plt.Figure:
    """
    Test identification accuracy as we add increasing amounts of white noise.

    For each SNR level:
      1. Corrupt the query with noise.
      2. Attempt matching.
      3. Record number of votes for the best match.

    Plots votes (confidence) vs SNR level so we can see the failure point.

    Returns
    -------
    matplotlib Figure
    """
    votes_list   = []
    matched_list = []

    for snr in snr_levels_db:
        y_noisy = add_white_noise(y_query, snr_db=snr)
        best, offsets, _ = match_audio(y_noisy, sr, db, use_pairs=True)

        if offsets and best and best in offsets:
            arr = offsets[best]
            if len(arr) > 0:
                counts, _ = np.histogram(
                    arr,
                    bins=np.arange(
                        arr.min(),
                        arr.max() + 6,
                        5
                    )
                )

                v = int(counts.max())
            else:
                v = 0
        else:
            v = 0
        votes_list.append(v)
        matched_list.append(best)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ECC71" if m is not None else "#E74C3C" for m in matched_list]
    bars   = ax.bar(
        [str(s) for s in snr_levels_db],
        votes_list,
        color=colors,
        edgecolor="white",
        linewidth=0.8
    )

    # Annotate each bar with the predicted label
    for bar, pred in zip(bars, matched_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            pred or "✗",
            ha="center", va="bottom", fontsize=8, rotation=30
        )

    ax.set_xlabel("SNR (dB)  –  lower = more noise", fontsize=12)
    ax.set_ylabel("Peak offset votes (confidence)", fontsize=12)
    ax.set_title(
        "Noise Robustness: Match Confidence vs Added Noise\n"
        "(green = correct match, red = failure)",
        fontsize=13, fontweight="bold"
    )
    ax.axhline(OFFSET_HISTOGRAM_MIN_VOTES, color="orange", linestyle="--",
               label=f"Min-votes threshold = {OFFSET_HISTOGRAM_MIN_VOTES}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def pitch_shift_and_match(
    y_query: np.ndarray,
    sr: int,
    db: Dict,
    semitones_list: List[float] = [-2, -1, -0.5, 0, 0.5, 1, 2]
) -> plt.Figure:
    """
    Test the effect of pitch-shifting (or time-stretching) the query.

    librosa.effects.pitch_shift changes frequency content without altering
    duration.  Because our hashes encode raw FFT bin indices, even a half-
    semitone shift moves peaks to different bins, making hashes mismatch.

    This demonstrates that the system is NOT pitch-invariant, and suggests
    using chroma features or log-frequency bins as a robustness improvement.

    Returns
    -------
    matplotlib Figure
    """
    votes_list   = []
    matched_list = []

    for n_steps in semitones_list:

        if n_steps == 0:
            y_shifted = y_query.copy()
        else:
            y_shifted = librosa.effects.pitch_shift(
                y_query.astype(np.float32),
                sr=sr,
                n_steps=n_steps
            )

        best, offsets, _ = match_audio(
            y_shifted,
            sr,
            db,
            use_pairs=True
        )

        v = 0

        if offsets and best and best in offsets:

            arr = offsets[best]

            if len(arr) > 0:

                counts, _ = np.histogram(
                    arr,
                    bins=np.arange(
                        arr.min(),
                        arr.max() + 6,
                        5
                    )
                )

                v = int(counts.max())

        votes_list.append(v)
        matched_list.append(best)

        print(
            f"Shift={n_steps:+} | "
            f"Match={best} | "
            f"Votes={v}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ECC71" if m is not None else "#E74C3C" for m in matched_list]
    ax.bar(
        [str(s) for s in semitones_list],
        votes_list,
        color=colors,
        edgecolor="white",
        linewidth=0.8
    )
    for i, (pred, v) in enumerate(zip(matched_list, votes_list)):
        ax.text(i, v + 0.5, pred or "✗", ha="center", va="bottom",
                fontsize=8, rotation=30)

    ax.set_xlabel("Pitch shift (semitones)", fontsize=12)
    ax.set_ylabel("Peak offset votes", fontsize=12)
    ax.set_title(
        "Pitch-Shift Robustness: Peak Votes vs Semitone Shift\n"
        "(green = correct match, red = failure)",
        fontsize=13, fontweight="bold"
    )
    ax.axhline(OFFSET_HISTOGRAM_MIN_VOTES, color="orange", linestyle="--",
               label=f"Min-votes threshold = {OFFSET_HISTOGRAM_MIN_VOTES}")
    ax.axvline(semitones_list.index(0), color="blue", linestyle=":",
               alpha=0.5, label="No shift")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  HIGH-LEVEL CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint_and_match(
    audio_path: str,
    db: Dict,
    use_pairs: bool = True
) -> Tuple[Optional[str], plt.Figure, plt.Figure, plt.Figure]:
    """
    One-shot function: load a file, fingerprint it, match against db.

    Returns
    -------
    best_match   : matched song name (or None)
    spec_fig     : spectrogram figure
    const_fig    : constellation figure
    hist_fig     : offset histogram figure
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    S_db, _, _ = compute_spectrogram(y, sr)
    peaks       = find_peaks_2d(S_db)

    spec_fig  = plot_spectrogram(S_db, sr, HOP_LENGTH,
                                 title=f"Spectrogram – {os.path.basename(audio_path)}")
    const_fig = plot_constellation(S_db, peaks, sr, HOP_LENGTH,
                                   title=f"Constellation – {os.path.basename(audio_path)}")

    best, offsets, hist_fig = match_audio(y, sr, db, use_pairs=use_pairs)
    return best, spec_fig, const_fig, hist_fig
