# Sonic Signatures — Audio Fingerprinting System

A Shazam-style audio fingerprinting system built with Python.
Implements the full pipeline from raw audio → spectrogram → constellation
→ paired hashes → database → matching.

## Project Structure

```
audio_fingerprint/
├── app.py                    ← Streamlit web application (entry point)
├── audio_processor.py        ← Core fingerprinting engine
├── build_db.py               ← Pre-build database from songs/ directory
├── run_batch.py              ← CLI batch identification → results.csv
├── generate_report_plots.py  ← Reproduce all report figures
├── requirements.txt          ← Python dependencies
├── REPORT.md                 ← Academic report draft
├── songs/                    ← Place reference audio files here
├── query_clips/              ← Place query clips here (for CLI batch)
├── db/
│   └── fingerprint_db.pkl    ← Pre-built database (commit this!)
└── report_plots/             ← Output directory for report figures
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your songs

Place your reference audio files in `songs/`:

```
songs/
├── Song_Name_A.mp3
├── Song_Name_B.wav
└── Song_Name_C.flac
```

The fingerprint label will be the **filename without extension**
(e.g. `Song_Name_A`).

### 3. Build the database

```bash
python build_db.py
```

This creates `db/fingerprint_db.pkl`. Commit this file to your repo.

### 4. Run the web app

```bash
streamlit run app.py
```

---

## CLI Usage

### Batch identification

```bash
python run_batch.py --query_dir query_clips/ --db db/fingerprint_db.pkl
```

Writes `results.csv` with columns `filename` and `prediction`.

### Generate all report plots

```bash
python generate_report_plots.py --song songs/MySong.mp3 --query query_clips/clip.mp3
```

Plots saved to `report_plots/`.

---

## Deployment on Streamlit Community Cloud

1. Push this entire directory (including `db/fingerprint_db.pkl`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select your repository, branch, and set **Main file path** to `app.py`.
4. Click **Deploy** — no additional secrets or environment variables needed.

---

## Algorithm Overview

```
Audio file
    ↓  librosa.load()
Waveform  y[n]
    ↓  STFT  (n_fft=2048, hop=512)
Spectrogram  S[f, t]  in dB
    ↓  2-D maximum filter + amplitude threshold
Constellation  {(f_i, t_i)}  ~30–100 peaks/s
    ↓  Fan-out pairing  (anchor → FAN_VALUE targets)
Hashes  {(f1, f2, Δt) : t_anchor}
    ↓  store in dict  (hash → [(song, t_anchor), ...])
Database

Query clip  → same pipeline → query hashes
    ↓  lookup each hash in DB
    ↓  compute offset = db_t − query_t per song
    ↓  histogram: true match = sharp spike
Matched song name
```

---

## Key Parameters (audio_processor.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 22050 | Audio sample rate (Hz) |
| `HOP_LENGTH` | 512 | STFT hop (≈23 ms) |
| `N_FFT` | 2048 | STFT window size (≈93 ms) |
| `PEAK_NEIGHBOURHOOD` | 10 | Local-max filter footprint (bins) |
| `PEAK_AMP_MIN_DB` | -60 | Noise floor threshold (dB) |
| `FAN_VALUE` | 15 | Targets paired per anchor |
| `MAX_HASH_TIME_DELTA` | 200 | Max anchor-target gap (frames, ≈4.7 s) |
| `OFFSET_HISTOGRAM_MIN_VOTES` | 3 | Minimum votes to claim a match |
