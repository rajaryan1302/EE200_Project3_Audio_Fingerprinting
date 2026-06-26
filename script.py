import os
import random
import tempfile

import librosa
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

SONGS_DIR = "songs"
QUERY_DIR = "query_clips"

os.makedirs(QUERY_DIR, exist_ok=True)

random.seed(42)

audio_extensions = (
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a"
)

songs = sorted([
    f for f in os.listdir(SONGS_DIR)
    if f.lower().endswith(audio_extensions)
])

mapping = []

for idx, song_file in enumerate(songs, start=1):

    song_path = os.path.join(SONGS_DIR, song_file)

    print(f"[{idx}/{len(songs)}] {song_file}")

    try:

        y, sr = librosa.load(
            song_path,
            sr=None,
            mono=True
        )

        duration = len(y) / sr

        min_len = 3
        max_len = min(30, duration * 0.30)

        clip_length = random.uniform(
            min_len,
            max_len
        )

        if duration <= clip_length + 5:
            start_time = 0
        else:
            start_time = random.uniform(
                0,
                duration - clip_length
            )

        end_time = start_time + clip_length

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        clip = y[start_sample:end_sample]

        clip_name = f"clip{idx}.mp3"
        clip_path = os.path.join(
            QUERY_DIR,
            clip_name
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_wav = temp_file.name

        sf.write(
            temp_wav,
            clip,
            sr
        )

        audio = AudioSegment.from_wav(
            temp_wav
        )

        audio.export(
            clip_path,
            format="mp3",
            bitrate="192k"
        )

        os.remove(temp_wav)

        mapping.append({
            "clip": clip_name,
            "song": song_file,
            "start_sec": round(start_time, 2),
            "end_sec": round(end_time, 2),
            "duration_sec": round(
                clip_length,
                2
            )
        })

        print(
            f"Saved {clip_name} "
            f"({start_time:.1f}s -> "
            f"{end_time:.1f}s)"
        )

    except Exception as e:

        print(
            f"Failed: {song_file}"
        )

        print(e)

mapping_df = pd.DataFrame(
    mapping
)

mapping_df.to_csv(
    os.path.join(
        QUERY_DIR,
        "clip_mapping.csv"
    ),
    index=False
)

print("\nDone!")
print(
    f"Created {len(mapping)} clips"
)