# Fast Eddie's Video Splitter

An AI **content-based** video splitter for Windows. It splits a video where the
*content* changes — not on fixed time intervals and not on raw scene-cut
detection. Useful for breaking long home-movie transfers (e.g. digitized 8mm /
VHS reels) into separate clips per scene.

## Quick start (Windows)

Three steps to get running on a fresh machine:

1. **Install Python.** Download Python 3.11 from
   <https://www.python.org/downloads/> and run the installer. On the first
   screen, tick **"Add python.exe to PATH"**, then finish the install.

2. **Run `setup.bat` once.** Double-click `setup.bat` (or run it from a
   terminal). It checks that Python is installed, installs all required
   packages, and downloads the AI model. Wait for it to print
   *"Setup complete."*

3. **Run `RunSplitter.bat`.** Double-click `RunSplitter.bat` to open the app.
   Pick a video, adjust the settings if you like, and click **Start**.

> **FFmpeg note:** `ffmpeg.exe` and `ffprobe.exe` ship in this folder. If you
> cloned from GitHub they are *not* included (each is ~200 MB, over GitHub's
> limit) — drop them into this folder, or install FFmpeg on your `PATH`.
> `setup.bat` will warn you if they are missing.

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
- **FFmpeg** — `ffmpeg.exe` and `ffprobe.exe`, either bundled in this folder, on
  your `PATH`, or passed via `--ffmpeg-path` / `--ffprobe-path`. The app also
  auto-detects them when they sit next to `video_splitter_ai.py`.
- Python packages — installed for you by `setup.bat`, or manually:

  ```powershell
  python -m pip install -r requirements.txt
  ```

## FFmpeg binaries (Releases)

`ffmpeg.exe` and `ffprobe.exe` are **not** stored in this repository — each is
~200 MB, over GitHub's 100 MB file limit. To run the app you need them locally.
Get them either way:

- **Recommended:** download the bundled `ffmpeg.exe` + `ffprobe.exe` from this
  repo's [**Releases**](../../releases) page and drop them in the app folder
  next to `video_splitter_ai.py`, **or**
- install FFmpeg yourself (e.g. <https://www.gyan.dev/ffmpeg/builds/> or
  `winget install Gyan.FFmpeg`) and make sure `ffmpeg`/`ffprobe` are on your `PATH`.

> Maintainer note: attach `ffmpeg.exe` and `ffprobe.exe` as assets on a GitHub
> Release so the link above resolves. Releases don't count against the repo's
> file-size limits.

## Usage

The easiest path is the [Quick start](#quick-start-windows) above
(`setup.bat` then `RunSplitter.bat`). The app can also be driven directly:

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
| `setup.bat` | One-time setup: installs packages + downloads the model |
| `RunSplitter.bat` | Launches the app with the local Python environment |
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
