# fast_eddies_video_split_gui.py
"""
Fast Eddie's Video Split - reliable Windows launcher.

Build (PowerShell, from this folder):
  python -m pip install --upgrade pyinstaller pillow requests onnxruntime pyinstaller-hooks-contrib
  pyinstaller --noconfirm --clean --onedir --noconsole --name "Fast Eddies Video Split" `
    --hidden-import requests `
    --add-data "Banner.png;." `
    --add-data "video_splitter_ai.py;." `
    --add-binary "ffmpeg.exe;." `
    fast_eddies_video_split_gui.py
"""
from __future__ import annotations

import os
import runpy
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

APP_TITLE = "Fast Eddie's Video Split"
BANNER_FILENAME = "Banner.png"
APP_SCRIPT = "video_splitter_ai.py"
FFMPEG_EXE = "ffmpeg.exe"

MAX_BANNER_WIDTH = 900
MAX_BANNER_HEIGHT = 180


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _exe_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def _resource_path(name: str) -> Path:
    p = _exe_dir() / name
    return p if p.exists() else _bundle_root() / name


def _load_banner(path: Path) -> ImageTk.PhotoImage | None:
    if not path.exists():
        return None
    img = Image.open(path)
    w, h = img.size
    scale = min(MAX_BANNER_WIDTH / w, MAX_BANNER_HEIGHT / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return ImageTk.PhotoImage(img)


def _prepend_path(dir_path: Path) -> None:
    if dir_path.exists():
        os.environ["PATH"] = str(dir_path) + os.pathsep + os.environ.get("PATH", "")


def _ensure_runtime_paths() -> None:
    """
    Why: native DLL loaders (onnxruntime, torch, etc.) need correct DLL search paths in frozen apps.
    """
    candidates = [
        _exe_dir(),
        _bundle_root(),
        _bundle_root() / "onnxruntime",
        _bundle_root() / "onnxruntime" / "capi",
    ]
    for d in candidates:
        _prepend_path(d)
        if hasattr(os, "add_dll_directory") and d.exists():
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass


def _ensure_ffmpeg_on_path() -> None:
    ffmpeg_path = _resource_path(FFMPEG_EXE)
    if ffmpeg_path.exists():
        _prepend_path(ffmpeg_path.parent)


def _diagnose_onnxruntime() -> None:
    """
    Imports onnxruntime here so we can show the REAL error if DLL loading fails.

    Fixes common OpenMP conflicts (torch/numpy/scipy vs onnxruntime) that manifest as WinError 1114.
    """
    # Must be set BEFORE importing onnxruntime
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    _ensure_runtime_paths()
    try:
        import onnxruntime as ort  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "onnxruntime failed to import inside the EXE.\n\n"
            f"REAL ERROR:\n{e!r}\n\n"
            "This is usually a native DLL initialization failure (often OpenMP conflicts or missing DLL deps).\n"
        ) from e


def _launch_app() -> None:
    _ensure_ffmpeg_on_path()
    _diagnose_onnxruntime()

    script = _resource_path(APP_SCRIPT)
    if not script.exists():
        raise FileNotFoundError(f"Missing {APP_SCRIPT} (expected at {script})")

    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


def main() -> int:
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("640x360")
    root.minsize(520, 300)

    frame = tk.Frame(root, padx=14, pady=14)
    frame.pack(fill="both", expand=True)

    banner = _load_banner(_resource_path(BANNER_FILENAME))
    if banner:
        lbl = tk.Label(frame, image=banner)
        lbl.image = banner
        lbl.pack(fill="x")
    else:
        tk.Label(frame, text=APP_TITLE, font=("Segoe UI", 20, "bold"), pady=8).pack(fill="x")

    tk.Label(frame, text="Fast launch for Video Splitter", font=("Segoe UI", 10), pady=10).pack()

    def on_launch() -> None:
        try:
            _launch_app()
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            root.lift()

    tk.Button(frame, text="Launch", font=("Segoe UI", 12, "bold"), padx=14, pady=10, command=on_launch).pack(pady=8)
    tk.Button(frame, text="Exit", padx=10, command=root.destroy).pack(pady=4)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
