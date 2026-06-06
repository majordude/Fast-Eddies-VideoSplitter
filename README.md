# Fast Eddie's Video Splitter

An AI **content-based** video splitter for Windows. It splits a video where the
*content* changes — not on fixed time intervals and not on raw scene-cut
detection. Useful for breaking long home-movie transfers (e.g. digitized 8mm /
VHS reels) into separate clips per scene.

## How it works

1. **Sample frames** with FFmpeg at a configurable rate (default 0.5 fps).
2. **Embed each frame** with a small ONNX image model (MobileNetV3-Small).
   The model is downloaded automatically on first run and cached under
   `~/.video_splitter_ai/`.
3. **Compare embeddings** (cosine distance). When the content drifts past the
   sensitivity threshold — or a segment exceeds the max length — a split point
   is proposed.
4. **Export segments** with FFmpeg into `<video>_output/split_segments/`, with a
   JSON manifest of the job. Optional per-segment brightness correction is
   applied to very dark / blown-out clips.

## Requirements

- **Python 3.11** (3.10+ should work)
- **FFmpeg** — `ffmpeg.exe` and `ffprobe.exe`. These are **not** committed to the
  repo because each is ~200 MB (over GitHub's 100 MB limit). Either:
  - drop `ffmpeg.exe` / `ffprobe.exe` next to the scripts in this folder, **or**
  - install FFmpeg and put it on your `PATH`, **or**
  - pass `--ffmpeg-path` / `--ffprobe-path` explicitly.
- Python packages (see `requirements.txt`):

  ```powershell
  python -m pip install -r requirements.txt
  ```

## Usage

### CLI

```powershell
python video_splitter_ai.py --input "C:\path\video.mp4" --sensitivity 0.35 --sample-fps 0.5 --max-segment-sec 600
```

Common options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--input` | — | Input video path (required for CLI mode) |
| `--output-root` | next to input | Where `<video>_output` is created |
| `--sensitivity` | `0.35` | Cosine-distance threshold. Lower = more cuts |
| `--sample-fps` | `0.5` | Frames per second sampled for analysis |
| `--max-segment-sec` | `600` | Force a split after this many seconds |
| `--min-segment-sec` | `3` | Merge segments shorter than this |
| `--reencode` / `--copy` | `--copy` | Re-encode (accurate starts) vs stream-copy (fast) |
| `--no-gpu` | GPU if available | Force CPU-only inference |

### GUI

```powershell
python video_splitter_ai.py          # PySide6 settings GUI
# or the simple launcher:
python fast_eddies_video_split_gui.py
```

## Files

| File | Purpose |
|------|---------|
| `video_splitter_ai.py` | Main application (CLI + PySide6 GUI + all logic) |
| `fast_eddies_video_split_gui.py` | Lightweight Tk launcher / PyInstaller entry point |
| `Banner.png` | Banner image for the launcher |
| `requirements.txt` | Python dependencies |

## Packaging (optional)

Build a standalone Windows app with PyInstaller:

```powershell
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
pyinstaller --noconfirm --clean --onedir --noconsole --name "Fast Eddies Video Split" `
  --hidden-import requests `
  --add-data "Banner.png;." `
  --add-data "video_splitter_ai.py;." `
  --add-binary "ffmpeg.exe;." `
  --add-binary "ffprobe.exe;." `
  fast_eddies_video_split_gui.py
```

## License

Personal project — all rights reserved unless a license file is added.
