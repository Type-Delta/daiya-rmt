# Stateful diarization throwaway demo

This keeps pyannote untouched and wraps its normal `Pipeline` output with a tiny
speaker memory.

Install:

```powershell
uv venv --python 3.12
uv --native-tls pip install -e ..\pyannote scipy matplotlib
uv --native-tls pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv --native-tls pip install python-certifi-win32
.venv\Scripts\python -m ensurepip --upgrade
.venv\Scripts\python -m pip install --upgrade pip
```

Smoke test without model downloads:

```powershell
.venv\Scripts\python demo.py
```

Run real audio:

```powershell
.venv\Scripts\python demo.py
```

Save a speaker-memory clustering graph after the run:

```powershell
.venv\Scripts\python demo.py --mem-graph memory_profiles.png
```

With a `.env` file:

```text
HF_TOKEN=...
AUDIO_PATH=C:\path\to\audio.wav
DEVICE=cuda
CHUNK_SECONDS=20
STRIDE_SECONDS=12
MATCH_THRESHOLD=0.38
MIN_NEW_PROFILE_SECONDS=6.0
CANDIDATE_PROMOTE_SECONDS=3.0
CANDIDATE_PROMOTE_OBSERVATIONS=2
EMBEDDING_EXCLUDE_OVERLAP=true
```

There is also a `.env.example` you can copy/edit locally.

`AUDIO_PATH` may point at a WAV file or another FFmpeg-readable format such as
MP3. For pyannote's gated models, the Hugging Face account behind `HF_TOKEN`
must have accepted the model conditions.

The interesting bit is in `speaker_memory.py`: pyannote-local labels are mapped
to persistent labels by comparing each chunk's returned `speaker_embeddings`
against stored centroids.

Weak unmatched labels no longer become permanent speakers immediately. Labels
with no exclusive speech are reported as `OVERLAP_ONLY` unless they match an
existing profile, and short unmatched labels become `CANDIDATE_*` until they
collect enough clean exclusive speech to promote into `SPEAKER_*`.

This demo preloads audio with SciPy/FFmpeg and passes an in-memory waveform to
pyannote, so it does not depend on pyannote/torchcodec file decoding.
