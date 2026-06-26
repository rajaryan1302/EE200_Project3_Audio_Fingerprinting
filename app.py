"""
app.py  –  Streamlit Web Application: Audio Fingerprint Identifier
===================================================================
Wraps the core fingerprinting engine (audio_processor.py) in a clean,
interactive UI with two operating modes:

  ① Single-Clip Mode
     Upload one audio clip → see spectrogram, constellation, offset
     histogram, and the matched song name.

  ② Batch Mode
     Upload many query clips at once → download results.csv with
     columns  [filename, prediction].

The indexed database (db/fingerprint_db.pkl) ships with the app and is
built once at startup from the songs/ directory if it doesn't yet exist.

Deploy on Streamlit Community Cloud:
  1.  Push this repo (with db/ included) to GitHub.
  2.  Point Streamlit Cloud at app.py – no extra configuration needed.
      requirements.txt must list: streamlit, librosa, numpy, scipy, matplotlib

Run locally:
  streamlit run app.py
"""

import os
import io
import tempfile
import pickle
import csv

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

# Local engine (must be in the same directory)
from audio_processor import (
    SAMPLE_RATE,
    HOP_LENGTH,
    compute_spectrogram,
    find_peaks_2d,
    generate_hashes,
    plot_spectrogram,
    plot_constellation,
    match_audio,
    build_database,
    load_database,
    fingerprint_and_match,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#  Paths are relative to the directory that contains app.py.
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR = os.path.join(BASE_DIR, "songs")
DB_PATH   = os.path.join(BASE_DIR, "db", "fingerprint_db.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE INITIALISATION  (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎵 Indexing song database – please wait…")
def get_database():
    """
    Load or build the fingerprint database.

    The @st.cache_resource decorator ensures this function runs exactly once
    per Streamlit server session, not on every page interaction.

    Priority:
      1. Load pre-built DB from db/fingerprint_db.pkl  (fast – ships with app)
      2. Build from songs/ directory if no pkl exists   (first run / CI)
    """
    if os.path.exists(DB_PATH):
        return load_database(DB_PATH)

    if not os.path.isdir(SONGS_DIR):
        st.error(
            f"No songs directory found at '{SONGS_DIR}' and no pre-built "
            "database at '{DB_PATH}'.  Please add reference songs and restart."
        )
        st.stop()

    return build_database(SONGS_DIR, db_path=DB_PATH)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib Figure to PNG bytes for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def load_uploaded_file(uploaded_file) -> tuple:
    """
    Save an st.UploadedFile to a temp file, load with librosa, return (y, sr).
    We need a real path because librosa reads file headers.
    """
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    os.unlink(tmp_path)
    return y, sr


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sonic Signatures – Audio Fingerprinter",
    page_icon="🎵",
    layout="wide",
)

# ── Global theme CSS (Indigo / Violet palette) ─────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --accent-1: #7C3AED;   /* violet   */
        --accent-2: #C084FC;  /* lilac    */
        --accent-3: #4F46E5;  /* indigo   */
    }
    /* Tabs underline + selected tab color */
    .stTabs [data-baseweb="tab"] {
        color: #A78BFA;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-1) !important;
        border-bottom-color: var(--accent-1) !important;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, var(--accent-3), var(--accent-2));
    }
    /* Full-page gradient background */
    .stApp {
        background: linear-gradient(160deg, #0E1117 0%, #1A1430 45%, #2D1B4E 100%);
    }
    /* Sidebar gets its own subtler gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1430 0%, #120F24 100%);
    }
    /* Gradient buttons (primary + download) */
    .stButton > button[kind="primary"],
    .stDownloadButton > button {
        background: linear-gradient(90deg, var(--accent-3), var(--accent-1)) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center;
               background:linear-gradient(90deg,#7C3AED,#C084FC);
               -webkit-background-clip:text;
               -webkit-text-fill-color:transparent;
               letter-spacing:2px;'>
    🎵  Sonic Signatures
    </h1>
    <p style='text-align:center;color:#9CA3AF;font-size:1.05em;'>
    A Shazam-style audio fingerprinting system.
    Upload a clip — we'll tell you which song it is.
    </p>
    <hr style='border-color:#7C3AED33;'/>
    """,
    unsafe_allow_html=True,
)

# ── Load / build the database ─────────────────────────────────────────────────
db = get_database()
song_list = db.get("__song_list__", [])

with st.sidebar:
    st.markdown(
        "<h3 style='color:#A78BFA;'>📚 Indexed Songs</h3>",
        unsafe_allow_html=True,
    )
    if song_list:
        for name in song_list:
            st.markdown(f"- `{name}`")
    else:
        st.info("No songs indexed yet.")

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#A78BFA;'>⚙️ Settings</h3>",
        unsafe_allow_html=True,
    )
    use_pairs = st.toggle("Use paired hashes", value=True,
                          help="Paired hashes are more discriminative. "
                               "Toggle off to see single-peak mode.")
    st.markdown(
        f"**Hash mode:** {'Paired (recommended)' if use_pairs else 'Single peaks'}"
    )

# ── Mode selector ─────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🎧  Single-Clip Mode", "📂  Batch Mode"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – SINGLE-CLIP MODE
# ─────────────────────────────────────────────────────────────────────────────

with tab_single:
    st.markdown(
        "Upload **one** audio clip.  The app will show you the intermediate "
        "signal-processing steps and reveal which song it matched."
    )

    uploaded = st.file_uploader(
        "Choose an audio clip",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        key="single_uploader",
    )

    if uploaded is not None:
        st.audio(uploaded, format="audio/" + uploaded.name.split(".")[-1])

        with st.spinner("🔍 Fingerprinting…"):
            y, sr = load_uploaded_file(uploaded)

            # ── Step 1: Spectrogram ──────────────────────────────────────────
            S_db, _, _ = compute_spectrogram(y, sr)
            spec_fig   = plot_spectrogram(
                S_db, sr, HOP_LENGTH,
                title=f"Spectrogram – {uploaded.name}"
            )

            # ── Step 2: Constellation ────────────────────────────────────────
            peaks     = find_peaks_2d(S_db)
            const_fig = plot_constellation(
                S_db, peaks, sr, HOP_LENGTH,
                title=f"Constellation – {uploaded.name}  ({len(peaks)} peaks)"
            )

            # ── Step 3: Match ────────────────────────────────────────────────
            best_match, offsets, hist_fig = match_audio(
                y, sr, db, use_pairs=use_pairs
            )
            # st.write("### Match Scores")

            # for song_name, arr in offsets.items():

            #     if len(arr) == 0:
            #         continue

            #     counts, _ = np.histogram(
            #         arr,
            #         bins=np.arange(arr.min(), arr.max() + 6, 5)
            #     )

            #     st.write(
            #         f"{song_name}: "
            #         f"offsets={len(arr)} "
            #         f"peak_votes={counts.max()}"
            #     )

        # ── Display ───────────────────────────────────────────────────────────
        if best_match:
            st.success(f"✅  **Matched:**  `{best_match}`")
        else:
            st.warning("⚠️  No confident match found in the database.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Spectrogram")
            st.caption(
                "Time × frequency energy map.  Bright regions = loud frequencies."
            )
            st.image(fig_to_bytes(spec_fig), use_container_width=True)

        with col2:
            st.markdown("#### Constellation Map")
            st.caption(
                "Sparse set of local-maximum peaks retained as fingerprint landmarks."
            )
            st.image(fig_to_bytes(const_fig), use_container_width=True)

        st.markdown("#### Offset Histogram")
        st.caption(
            "For each song in the database, this histogram shows the distribution "
            "of time offsets at which hash matches occur.  A true match produces "
            "a sharp spike at a single offset; false matches scatter uniformly."
        )
        st.image(fig_to_bytes(hist_fig), use_container_width=True)

        st.markdown("---")
        st.markdown(f"**Peaks detected:** {len(peaks)}")
        query_hashes = generate_hashes(peaks, use_pairs=use_pairs)
        st.markdown(f"**Hashes generated:** {len(query_hashes)}")
        if offsets:
            best = max(offsets, key=lambda k: len(offsets[k]))
            arr  = offsets[best]
            if len(arr) > 0:
                peak_votes = int(np.bincount(arr - arr.min()).max())
                st.markdown(f"**Peak votes for top song:** {peak_votes}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – BATCH MODE
# ─────────────────────────────────────────────────────────────────────────────

with tab_batch:
    st.markdown(
        "Upload **multiple** query clips.  The app will fingerprint each one "
        "and let you download a `results.csv` file with columns "
        "`filename` and `prediction`."
    )

    uploaded_batch = st.file_uploader(
        "Choose one or more audio clips",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_batch:
        run_batch = st.button("▶  Run Batch Identification", type="primary")

        if run_batch:
            results = []   # list of (filename, prediction) tuples
            progress = st.progress(0, text="Processing…")

            for idx, uf in enumerate(uploaded_batch):
                progress.progress(
                    (idx + 1) / len(uploaded_batch),
                    text=f"Processing {uf.name}  ({idx+1}/{len(uploaded_batch)})"
                )
                try:
                    y, sr = load_uploaded_file(uf)
                    best, _, _ = match_audio(y, sr, db, use_pairs=use_pairs)
                    prediction = best if best else "no_match"
                except Exception as exc:
                    prediction = f"error: {exc}"

                results.append((uf.name, prediction))

            progress.empty()

            # ── Build CSV in memory ──────────────────────────────────────────
            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)
            writer.writerow(["filename", "prediction"])   # required header
            writer.writerows(results)
            csv_bytes = csv_buf.getvalue().encode("utf-8")

            # ── Results table ────────────────────────────────────────────────
            st.markdown("#### Results")
            st.table(
                [{"filename": r[0], "prediction": r[1]} for r in results]
            )

            # ── Download button ──────────────────────────────────────────────
            st.download_button(
                label="⬇  Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )

            # ── Summary metrics ──────────────────────────────────────────────
            matched   = sum(1 for _, p in results if p != "no_match" and not p.startswith("error"))
            unmatched = len(results) - matched
            c1, c2, c3 = st.columns(3)
            c1.metric("Total clips",   len(results))
            c2.metric("Matched",       matched,   delta=None)
            c3.metric("Unmatched",     unmatched, delta=None)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style='border-color:#7C3AED33;'/>
    <p style='text-align:center;color:#9CA3AF;font-size:0.85em;'>
    Sonic Signatures · Audio Fingerprinting System ·
    Built with <a href='https://librosa.org' target='_blank' style='color:#A78BFA;'>librosa</a>,
    <a href='https://scipy.org' target='_blank' style='color:#A78BFA;'>SciPy</a> &amp;
    <a href='https://streamlit.io' target='_blank' style='color:#A78BFA;'>Streamlit</a>
    </p>
    """,
    unsafe_allow_html=True,
)
