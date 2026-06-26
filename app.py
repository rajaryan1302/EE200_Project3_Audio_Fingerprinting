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

    /* ── Distinct branding additions ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
    code, .stCode, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

    /* Subtle dot-grid texture layered on top of the gradient */
    .stApp {
        background-image:
            radial-gradient(circle, rgba(196,132,252,0.10) 1px, transparent 1px),
            linear-gradient(160deg, #0E1117 0%, #1A1430 45%, #2D1B4E 100%);
        background-size: 22px 22px, cover;
    }

    /* Badge pill under the title */
    .badge-row { display:flex; justify-content:center; gap:8px; margin-bottom:0.6em; flex-wrap:wrap; }
    .badge-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72em;
        padding: 3px 12px;
        border-radius: 999px;
        border: 1px solid #7C3AED66;
        background: #7C3AED1A;
        color: #C4B5FD;
        letter-spacing: 0.5px;
    }

    /* Card wrapper for images (spectrogram / constellation / histogram) */
    div[data-testid="stImage"] {
        border: 1px solid #7C3AED33;
        border-radius: 14px;
        padding: 10px;
        background: linear-gradient(145deg, #1A143055, #0E111799);
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.12);
    }

    /* Custom stat-card grid */
    .stat-grid { display:flex; gap:12px; flex-wrap:wrap; margin-top:0.5em; }
    .stat-card {
        flex:1; min-width:140px;
        border: 1px solid #7C3AED40;
        border-radius: 12px;
        padding: 14px 16px;
        background: linear-gradient(145deg, #1A143066, #2D1B4E44);
    }
    .stat-card .label { font-size:0.72em; color:#A78BFA; letter-spacing:0.5px; text-transform:uppercase; }
    .stat-card .value { font-family:'JetBrains Mono', monospace; font-size:1.6em; color:#E9D5FF; margin-top:2px; }
    .stat-card { transition: transform 0.18s ease, box-shadow 0.18s ease; }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(124,58,237,0.25); }

    /* Animated shimmer on the main title */
    @keyframes shimmer { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }
    .shimmer-title {
        background: linear-gradient(270deg, #7C3AED, #C084FC, #4F46E5, #C084FC, #7C3AED);
        background-size: 400% 100%;
        animation: shimmer 6s ease-in-out infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Recolor default Streamlit alert boxes to match the theme */
    div[data-testid="stAlertContentSuccess"], div[data-baseweb="notification"]:has(div[data-testid="stAlertContentSuccess"]) {
        background: linear-gradient(135deg, #1E3A2E, #16291F) !important;
        border: 1px solid #34D39966 !important;
    }
    div[data-testid="stAlertContentWarning"], div[data-baseweb="notification"]:has(div[data-testid="stAlertContentWarning"]) {
        background: linear-gradient(135deg, #3D2E12, #2A1F0C) !important;
        border: 1px solid #FBBF2466 !important;
    }
    div[data-testid="stAlertContentInfo"], div[data-baseweb="notification"]:has(div[data-testid="stAlertContentInfo"]) {
        background: linear-gradient(135deg, #1A1430, #120F24) !important;
        border: 1px solid #7C3AED66 !important;
    }

    /* Hover-lift on image cards */
    div[data-testid="stImage"] { transition: transform 0.2s ease, box-shadow 0.2s ease; }
    div[data-testid="stImage"]:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(124,58,237,0.22); }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #120F24; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #7C3AED, #4F46E5); border-radius: 6px; }

    /* Pipeline stepper */
    .pipeline-row { display:flex; align-items:center; justify-content:center; gap:6px; margin:0.8em 0 1.2em 0; flex-wrap:wrap; }
    .pipe-step { display:flex; align-items:center; gap:6px; }
    .pipe-bubble {
        font-family:'JetBrains Mono', monospace; font-size:0.78em;
        padding:6px 13px; border-radius:999px;
        background: linear-gradient(135deg, #1A1430, #2D1B4E);
        border: 1px solid #7C3AED55; color:#E9D5FF;
    }
    .pipe-arrow { color:#7C3AED; font-size:1em; opacity:0.7; }

    /* Custom file-uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(145deg, #1A143055, #0E111799) !important;
        border: 1.5px dashed #7C3AED77 !important;
        border-radius: 14px !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #C084FC !important;
        background: linear-gradient(145deg, #2D1B4E55, #1A143099) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* Audio player card wrapper */
    div[data-testid="stAudio"] {
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #7C3AED33;
        background: linear-gradient(145deg, #1A143055, #0E111799);
    }
    div[data-testid="stAudio"] audio { width: 100%; accent-color: #7C3AED; }

    /* Decorative animated waveform bars ("now playing" indicator) */
    @keyframes wavebounce { 0%,100% { transform: scaleY(0.3); } 50% { transform: scaleY(1); } }
    .now-playing { display:flex; align-items:center; gap:3px; height:22px; margin:8px 0 2px 2px; }
    .now-playing span {
        display:block; width:3px; border-radius:2px;
        background: linear-gradient(180deg, #C084FC, #7C3AED);
        animation: wavebounce 1.1s ease-in-out infinite;
    }
    .now-playing span:nth-child(1) { height:60%; animation-delay: 0s; }
    .now-playing span:nth-child(2) { height:100%; animation-delay: 0.15s; }
    .now-playing span:nth-child(3) { height:45%; animation-delay: 0.3s; }
    .now-playing span:nth-child(4) { height:85%; animation-delay: 0.45s; }
    .now-playing span:nth-child(5) { height:55%; animation-delay: 0.6s; }
    .now-playing span:nth-child(6) { height:95%; animation-delay: 0.75s; }
    .now-playing span:nth-child(7) { height:40%; animation-delay: 0.9s; }
    .now-playing-label { font-family:'JetBrains Mono', monospace; font-size:0.72em; color:#A78BFA; margin-left:8px; letter-spacing:0.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; margin-top:-0.5em;'>
        <svg width="46" height="46" viewBox="0 0 100 100" style="margin-bottom:-8px;">
            <g fill="none" stroke="#C084FC" stroke-width="6" stroke-linecap="round">
                <line x1="10" y1="50" x2="10" y2="50"/>
                <line x1="22" y1="35" x2="22" y2="65"/>
                <line x1="34" y1="20" x2="34" y2="80"/>
                <line x1="46" y1="10" x2="46" y2="90"/>
                <line x1="58" y1="25" x2="58" y2="75"/>
                <line x1="70" y1="38" x2="70" y2="62"/>
                <line x1="82" y1="44" x2="82" y2="56"/>
                <line x1="94" y1="48" x2="94" y2="52"/>
            </g>
        </svg>
    </div>
    <h1 style='text-align:center;
               letter-spacing:1px;
               font-weight:700;
               margin-bottom:0.15em;'
        class='shimmer-title'>
    Sonic Signatures
    </h1>
    <p style='text-align:center;color:#9CA3AF;font-size:1.0em;margin-bottom:0.6em;'>
    Frequency-domain fingerprint matching — peak constellations, paired hashes, offset voting.
    </p>
    <div class='badge-row'>
        <span class='badge-pill'>EE200 · SIGNALS &amp; SYSTEMS</span>
        <span class='badge-pill'>Raj &amp; Abhinav Bajpai</span>
        <span class='badge-pill'>IIT KANPUR</span>
    </div>
    <div class='pipeline-row'>
        <div class='pipe-step'><span class='pipe-bubble'>WAVEFORM</span></div>
        <span class='pipe-arrow'>➜</span>
        <div class='pipe-step'><span class='pipe-bubble'>STFT</span></div>
        <span class='pipe-arrow'>➜</span>
        <div class='pipe-step'><span class='pipe-bubble'>CONSTELLATION</span></div>
        <span class='pipe-arrow'>➜</span>
        <div class='pipe-step'><span class='pipe-bubble'>FAN-OUT HASHES</span></div>
        <span class='pipe-arrow'>➜</span>
        <div class='pipe-step'><span class='pipe-bubble'>OFFSET VOTE</span></div>
    </div>
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
        st.markdown(
            f"""
            <div class='now-playing'>
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
                <span class='now-playing-label'>QUERY CLIP LOADED · {uploaded.name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        query_hashes = generate_hashes(peaks, use_pairs=use_pairs)
        peak_votes_val = "—"
        if offsets:
            best = max(offsets, key=lambda k: len(offsets[k]))
            arr  = offsets[best]
            if len(arr) > 0:
                peak_votes_val = int(np.bincount(arr - arr.min()).max())

        st.markdown(
            f"""
            <div class='stat-grid'>
                <div class='stat-card'>
                    <div class='label'>Peaks Detected</div>
                    <div class='value'>{len(peaks)}</div>
                </div>
                <div class='stat-card'>
                    <div class='label'>Hashes Generated</div>
                    <div class='value'>{len(query_hashes)}</div>
                </div>
                <div class='stat-card'>
                    <div class='label'>Peak Votes (Top Song)</div>
                    <div class='value'>{peak_votes_val}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
            st.markdown(
                f"""
                <div class='stat-grid'>
                    <div class='stat-card'>
                        <div class='label'>Total Clips</div>
                        <div class='value'>{len(results)}</div>
                    </div>
                    <div class='stat-card'>
                        <div class='label'>Matched</div>
                        <div class='value'>{matched}</div>
                    </div>
                    <div class='stat-card'>
                        <div class='label'>Unmatched</div>
                        <div class='value'>{unmatched}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style='border-color:#7C3AED33;'/>
    <p style='text-align:center;color:#9CA3AF;font-size:0.85em;'>
    <span style='color:#C4B5FD;font-family:"JetBrains Mono",monospace;'>SONIC SIGNATURES</span>
    &nbsp;·&nbsp; Frequency Forensics &amp; Missing Boundaries &nbsp;·&nbsp; EE200 Project ·
    Built with <a href='https://librosa.org' target='_blank' style='color:#A78BFA;'>librosa</a>,
    <a href='https://scipy.org' target='_blank' style='color:#A78BFA;'>SciPy</a> &amp;
    <a href='https://streamlit.io' target='_blank' style='color:#A78BFA;'>Streamlit</a>
    </p>
    """,
    unsafe_allow_html=True,
)
