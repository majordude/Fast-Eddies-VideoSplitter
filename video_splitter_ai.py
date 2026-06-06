# -*- coding: utf-8 -*-
# file: video_splitter_ai.py
"""
AI Content-Based Video Splitter (Windows 11) - Auto model download version.

What it does
------------
- Samples frames with FFmpeg (not scene-cut detection).
- Uses an ONNX image model (MobileNetV3-small) to compute embeddings.
- Splits when content changes exceed sensitivity threshold and/or max segment length.
- Exports segments into <video>_output/split_segments (next to the input video) or under an optional output root.

Install deps
------------
pip install PySide6 onnxruntime numpy pillow requests

FFmpeg
------
ffmpeg -version
ffprobe -version

Usage (CLI)
-----------
python video_splitter_ai.py --input "C:\\path\\video.mp4" --sensitivity 0.35 --sample-fps 0.5 --max-segment-sec 600

GUI
---
python video_splitter_ai.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError("onnxruntime import failed. Install with: pip install onnxruntime") from e

try:
    from PySide6 import QtCore, QtWidgets
except Exception:
    QtCore = None
    QtWidgets = None


# Stable pinned revision (HuggingFace main no longer contains the ONNX file).
MODEL_URL_DEFAULT = (
    "https://huggingface.co/qualcomm/MobileNet-v3-Small/resolve/"
    "34576ece80d8d2e79dd1f5b8962c425f083e5787/MobileNet-v3-Small.onnx"
)
MODEL_URL_FALLBACKS: Tuple[str, ...] = (
    MODEL_URL_DEFAULT,
    "https://huggingface.co/qualcomm/MobileNet-v3-Small/resolve/"
    "47dfa7e0a0de7f239ddcd71bbbaa31dafae35224/MobileNet-v3-Small.onnx",
)
MODEL_SHA256_DEFAULT: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SplitterConfig:
    input_video: Path
    output_root: Optional[Path]
    sensitivity: float
    sample_fps: float
    max_segment_sec: float
    min_segment_sec: float
    model_path: Path
    model_url: str
    model_sha256: Optional[str]
    use_gpu_if_available: bool
    reencode: bool
    video_codec: str
    crf: int
    preset: str
    audio_codec: str
    audio_bitrate: str
    ffmpeg_path: str
    ffprobe_path: str


@dataclasses.dataclass(frozen=True)
class FrameSample:
    index: int
    timestamp_sec: float
    image_path: Path


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _run(
    cmd: Sequence[str],
    *,
    capture: bool,
    text: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=text,
        check=check,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def _ensure_ffmpeg(ffmpeg_path: str, ffprobe_path: str) -> Tuple[str, str]:
    ffmpeg = ffmpeg_path or "ffmpeg"
    ffprobe = ffprobe_path or "ffprobe"

    ffmpeg_resolved = ffmpeg if Path(ffmpeg).exists() else (_which(ffmpeg) or "")
    ffprobe_resolved = ffprobe if Path(ffprobe).exists() else (_which(ffprobe) or "")

    if not ffmpeg_resolved:
        raise FileNotFoundError(
            "FFmpeg not found. Install FFmpeg and ensure ffmpeg.exe is on PATH, "
            "or pass --ffmpeg-path."
        )
    if not ffprobe_resolved:
        raise FileNotFoundError(
            "FFprobe not found. Install FFmpeg and ensure ffprobe.exe is on PATH, "
            "or pass --ffprobe-path."
        )
    return ffmpeg_resolved, ffprobe_resolved


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model_downloaded(
    model_path: Path,
    model_url: str,
    expected_sha256: Optional[str],
    log: Callable[[str], None],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        if expected_sha256:
            got = _sha256_file(model_path)
            if got.lower() != expected_sha256.lower():
                log(f"Model hash mismatch. Re-downloading. got={got}")
                model_path.unlink(missing_ok=True)
            else:
                return
        else:
            return

    urls: List[str] = []
    if model_url:
        urls.append(model_url.strip())
    urls.extend([u for u in MODEL_URL_FALLBACKS if u not in urls])

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            log(f"Downloading model: {url}")
            tmp = model_path.with_suffix(model_path.suffix + ".download")
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100.0 * done / total
                            log(f"  ... {pct:.1f}%")
            tmp.replace(model_path)

            if expected_sha256:
                got = _sha256_file(model_path)
                if got.lower() != expected_sha256.lower():
                    model_path.unlink(missing_ok=True)
                    raise ValueError(f"Model SHA256 mismatch after download: got={got}")

            log(f"Model ready: {model_path}")
            return
        except Exception as e:
            last_err = e
            log(f"Model download failed: {e}")

    raise RuntimeError(f"All model URLs failed. Last error: {last_err}")


def probe_duration_sec(ffprobe: str, input_video: Path) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_video),
    ]
    cp = _run(cmd, capture=True, text=True, check=True)
    data = json.loads(cp.stdout or "{}")
    return float(data["format"]["duration"])


_SHOWINFO_RE = re.compile(r"showinfo.*pts_time:([0-9.]+)")


def sample_frames_with_ffmpeg(
    ffmpeg: str,
    input_video: Path,
    temp_dir: Path,
    sample_fps: float,
    log: Callable[[str], None],
    cancel: Callable[[], bool],
) -> List[FrameSample]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(temp_dir / "frame_%06d.jpg")

    vf = (
        f"fps={sample_fps},"
        "scale=224:224:force_original_aspect_ratio=decrease,"
        "pad=224:224:(ow-iw)/2:(oh-ih)/2,"
        "showinfo"
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        out_pattern,
    ]

    log("Sampling frames with FFmpeg...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    timestamps: List[float] = []
    stderr_lines: List[str] = []

    try:
        assert proc.stderr is not None
        while True:
            if cancel():
                proc.kill()
                raise RuntimeError("Cancelled")
            line = proc.stderr.readline()
            if not line:
                break
            stderr_lines.append(line)
            m = _SHOWINFO_RE.search(line)
            if m:
                timestamps.append(float(m.group(1)))
    finally:
        rc = proc.wait()
        if rc != 0:
            tail = "".join(stderr_lines[-50:])
            raise RuntimeError(f"FFmpeg frame sampling failed (rc={rc}). Tail:\n{tail}")

    image_paths = sorted(temp_dir.glob("frame_*.jpg"))
    if not image_paths:
        raise RuntimeError("No frames were sampled. Check FFmpeg and input file.")

    n = min(len(image_paths), len(timestamps))
    samples: List[FrameSample] = []
    for i in range(n):
        samples.append(FrameSample(index=i, timestamp_sec=timestamps[i], image_path=image_paths[i]))

    log(f"Sampled {len(samples)} frames.")
    return samples


def _get_ort_providers(use_gpu: bool) -> List[str]:
    providers = ort.get_available_providers()
    if not use_gpu:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclasses.dataclass(frozen=True)
class OrtxModelIO:
    input_name: str
    input_shape: Tuple[int, ...]
    input_is_nchw: bool
    output_name: str
    fixed_batch: Optional[int]


def _infer_model_io(session: ort.InferenceSession) -> OrtxModelIO:
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]

    shape_raw = inp.shape
    shape: Tuple[int, ...] = tuple(int(x) if isinstance(x, (int, np.integer)) else -1 for x in shape_raw)

    fixed_batch: Optional[int] = None
    if len(shape) >= 1 and isinstance(shape_raw[0], (int, np.integer)):
        fixed_batch = int(shape_raw[0])

    input_is_nchw = True
    if len(shape) == 4:
        if shape[1] in (-1, 1, 3) and shape[2] in (-1, 224) and shape[3] in (-1, 224):
            input_is_nchw = True
        elif shape[3] in (-1, 1, 3) and shape[1] in (-1, 224) and shape[2] in (-1, 224):
            input_is_nchw = False

    return OrtxModelIO(
        input_name=inp.name,
        input_shape=shape,
        input_is_nchw=input_is_nchw,
        output_name=out.name,
        fixed_batch=fixed_batch,
    )


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image_224(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr


def compute_embeddings(
    session: ort.InferenceSession,
    io: OrtxModelIO,
    samples: Sequence[FrameSample],
    log: Callable[[str], None],
    cancel: Callable[[], bool],
) -> np.ndarray:
    log("Computing embeddings (ONNXRuntime)...")

    # Fix: some models are exported with fixed batch=1.
    # If fixed, we must feed exactly that batch size.
    batch_size = 16
    if io.fixed_batch is not None and io.fixed_batch > 0:
        batch_size = io.fixed_batch

    embs: List[np.ndarray] = []

    for start in range(0, len(samples), batch_size):
        if cancel():
            raise RuntimeError("Cancelled")
        chunk = samples[start : start + batch_size]

        # If model expects fixed batch (e.g. 1), enforce it strictly.
        if io.fixed_batch is not None and len(chunk) != io.fixed_batch:
            # For fixed batch=1 this just skips the tail (rare).
            # Better behavior is to just process the remainder with batch=1.
            if io.fixed_batch == 1:
                for s in chunk:
                    imgs1 = np.expand_dims(_load_image_224(s.image_path), axis=0)  # 1x224x224x3
                    if io.input_is_nchw:
                        imgs1 = np.transpose(imgs1, (0, 3, 1, 2))
                    outputs = session.run([io.output_name], {io.input_name: imgs1.astype(np.float32, copy=False)})
                    vec = outputs[0].astype(np.float32)
                    if vec.ndim > 2:
                        vec = vec.reshape(vec.shape[0], -1)
                    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
                    embs.append(vec)
                log(f"  embeddings: {min(start + batch_size, len(samples))}/{len(samples)}")
                continue

        imgs = np.stack([_load_image_224(s.image_path) for s in chunk], axis=0)  # NHWC
        if io.input_is_nchw:
            imgs = np.transpose(imgs, (0, 3, 1, 2))  # NCHW
        imgs = imgs.astype(np.float32, copy=False)

        outputs = session.run([io.output_name], {io.input_name: imgs})
        vec = outputs[0].astype(np.float32)
        if vec.ndim > 2:
            vec = vec.reshape(vec.shape[0], -1)

        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        embs.append(vec)

        log(f"  embeddings: {min(start + batch_size, len(samples))}/{len(samples)}")

    return np.vstack(embs)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - float(np.dot(a, b)))


def propose_split_points(
    timestamps: Sequence[float],
    embeddings: np.ndarray,
    sensitivity: float,
    max_segment_sec: float,
    min_segment_sec: float,
    log: Callable[[str], None],
) -> List[float]:
    if len(timestamps) != embeddings.shape[0]:
        raise ValueError("timestamps and embeddings length mismatch")

    split_times: List[float] = []
    last_cut_time = 0.0
    last_anchor_idx = 0

    for i in range(1, len(timestamps)):
        t = float(timestamps[i])

        if (t - last_cut_time) >= max_segment_sec:
            split_times.append(t)
            last_cut_time = t
            last_anchor_idx = i
            continue

        d = cosine_distance(embeddings[last_anchor_idx], embeddings[i])
        if d >= sensitivity:
            if (t - last_cut_time) < min_segment_sec:
                continue
            split_times.append(t)
            last_cut_time = t
            last_anchor_idx = i

    if split_times:
        merged: List[float] = []
        prev = 0.0
        for t in split_times:
            if (t - prev) < min_segment_sec and merged:
                merged[-1] = t
            else:
                merged.append(t)
            prev = merged[-1]
        split_times = merged

    log(f"Proposed {len(split_times)} split points.")
    return split_times


def _safe_stem_name(p: Path) -> str:
    """Windows-safe stem for folders."""
    bad = '\/:*?\"<>|'
    s = p.stem
    s = "".join("_" if c in bad else c for c in s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_. ")
    return s or "video"


@dataclasses.dataclass(frozen=True)
class OutputDirs:
    parent: Path
    split_dir: Path
    joined_dir: Path
    manifest_path: Path


def build_output_dirs(input_video: Path, output_root: Optional[Path]) -> OutputDirs:
    """Create <video>_output/{split_segments,joined_segments} and return paths."""
    base = _safe_stem_name(input_video)
    parent = (output_root / f"{base}_output") if output_root else (input_video.parent / f"{base}_output")
    split_dir = parent / "split_segments"
    joined_dir = parent / "joined_segments"
    split_dir.mkdir(parents=True, exist_ok=True)
    joined_dir.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        parent=parent,
        split_dir=split_dir,
        joined_dir=joined_dir,
        manifest_path=parent / "_last_split_job.json",
    )


def write_last_split_manifest(
    dirs: OutputDirs,
    config: SplitterConfig,
    duration_sec: float,
    split_times: Sequence[float],
    segment_paths: Sequence[Path],
) -> None:
    """Write a portable manifest (relative to parent output dir)."""
    parent = dirs.parent.resolve()

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(parent))
        except Exception:
            return str(p.resolve())

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_video_path": str(config.input_video.resolve()),
        "parent_output_dir": str(parent),
        "split_segments_dir": rel(dirs.split_dir),
        "joined_segments_dir": rel(dirs.joined_dir),
        "duration_sec": float(duration_sec),
        "split_times_sec": [float(x) for x in split_times],
        "segments": [rel(p) for p in segment_paths],
        "settings": {
            "sensitivity": float(config.sensitivity),
            "sample_fps": float(config.sample_fps),
            "max_segment_sec": float(config.max_segment_sec),
            "min_segment_sec": float(config.min_segment_sec),
            "use_gpu_if_available": bool(config.use_gpu_if_available),
            "reencode": bool(config.reencode),
            "video_codec": str(config.video_codec),
            "crf": int(config.crf),
            "preset": str(config.preset),
            "audio_codec": str(config.audio_codec),
            "audio_bitrate": str(config.audio_bitrate),
            "model_path": str(config.model_path),
            "model_url": str(config.model_url),
            "ffmpeg_path": str(config.ffmpeg_path),
            "ffprobe_path": str(config.ffprobe_path),
        },
    }

    dirs.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def export_segments_ffmpeg(
    ffmpeg: str,
    input_video: Path,
    out_dir: Path,
    split_times: Sequence[float],
    duration_sec: float,
    reencode: bool,
    video_codec: str,
    crf: int,
    preset: str,
    audio_codec: str,
    audio_bitrate: str,
    log: Callable[[str], None],
    cancel: Callable[[], bool],
) -> List[Path]:

    def analyze_brightness(sample_path: Path) -> float:
        try:
            img = Image.open(sample_path).convert("L").resize((64, 64))
            arr = np.asarray(img, dtype=np.float32)
            return float(arr.mean())
        except Exception:
            return 128.0

    def pick_filter(brightness: float) -> str:
        # Keep this mild. White/blown segments are left alone for manual review/deletion.
        if brightness > 235:
            return "none"
        if brightness > 180:
            return "eq=brightness=-0.05:gamma=0.95"
        if brightness < 60:
            return "eq=brightness=0.05:gamma=1.10"
        return "none"

    times = [0.0] + list(split_times) + [float(duration_sec)]
    outputs: List[Path] = []

    for i in range(len(times) - 1):
        if cancel():
            raise RuntimeError("Cancelled")

        start = times[i]
        end = times[i + 1]
        if end <= start + 0.01:
            continue

        out_path = out_dir / f"segment_{i+1:03d}_{start:.2f}s_{end:.2f}s.mp4"
        out_tmp = out_path.with_suffix(".tmp.mp4")

        brightness = 128.0
        tmp_frame = out_dir / f"_probe_{i+1:03d}.jpg"
        probe_cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.2f}",
            "-i",
            str(input_video),
            "-frames:v",
            "1",
            str(tmp_frame),
        ]
        try:
            _run(probe_cmd, capture=True, text=True, check=True)
            brightness = analyze_brightness(tmp_frame)
        except Exception:
            brightness = 128.0
        finally:
            tmp_frame.unlink(missing_ok=True)

        filter_str = pick_filter(brightness)
        log(f"Segment {i+1}: brightness={brightness:.1f} filter={filter_str}")

        base = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(input_video),
        ]

        if reencode:
            cmd = base.copy()
            if filter_str != "none":
                cmd += ["-vf", filter_str]
            cmd += [
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                audio_codec,
                "-b:a",
                audio_bitrate,
                "-movflags",
                "+faststart",
                str(out_tmp),
            ]
        else:
            cmd = base + [
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(out_tmp),
            ]

        log(f"Exporting segment {i+1}/{len(times)-1}: {start:.2f}s -> {end:.2f}s")
        try:
            _run(cmd, capture=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "")[-4000:]
            raise RuntimeError(f"FFmpeg export failed for segment {i+1}. Tail:\n{stderr}") from e

        out_tmp.replace(out_path)
        outputs.append(out_path)

    log(f"Exported {len(outputs)} segments to: {out_dir}")
    return outputs


def split_video(

    config: SplitterConfig,
    log: Callable[[str], None],
    cancel: Callable[[], bool],
) -> List[Path]:
    ffmpeg, ffprobe = _ensure_ffmpeg(config.ffmpeg_path, config.ffprobe_path)

    ensure_model_downloaded(config.model_path, config.model_url, config.model_sha256, log)

    duration_sec = probe_duration_sec(ffprobe, config.input_video)
    log(f"Video duration: {duration_sec:.2f}s")

    providers = _get_ort_providers(config.use_gpu_if_available)
    log(f"ONNXRuntime providers: {providers}")
    sess = ort.InferenceSession(str(config.model_path), providers=providers)
    io = _infer_model_io(sess)
    log(
        f"Model IO: input={io.input_name} shape={io.input_shape} "
        f"layout={'NCHW' if io.input_is_nchw else 'NHWC'} output={io.output_name} "
        f"fixed_batch={io.fixed_batch}"
    )

    with tempfile.TemporaryDirectory(prefix="video_splitter_ai_") as td:
        temp_dir = Path(td)
        samples = sample_frames_with_ffmpeg(
            ffmpeg=ffmpeg,
            input_video=config.input_video,
            temp_dir=temp_dir,
            sample_fps=config.sample_fps,
            log=log,
            cancel=cancel,
        )
        embeddings = compute_embeddings(sess, io, samples, log=log, cancel=cancel)

        timestamps = [s.timestamp_sec for s in samples]
        split_times = propose_split_points(
            timestamps=timestamps,
            embeddings=embeddings,
            sensitivity=config.sensitivity,
            max_segment_sec=config.max_segment_sec,
            min_segment_sec=config.min_segment_sec,
            log=log,
        )

    dirs = build_output_dirs(config.input_video, config.output_root)
    outputs = export_segments_ffmpeg(
        ffmpeg=ffmpeg,
        input_video=config.input_video,
        out_dir=dirs.split_dir,
        split_times=split_times,
        duration_sec=duration_sec,
        reencode=config.reencode,
        video_codec=config.video_codec,
        crf=config.crf,
        preset=config.preset,
        audio_codec=config.audio_codec,
        audio_bitrate=config.audio_bitrate,
        log=log,
        cancel=cancel,
    )

    try:
        write_last_split_manifest(
            dirs=dirs,
            config=config,
            duration_sec=duration_sec,
            split_times=split_times,
            segment_paths=outputs,
        )
        log(f"Manifest written: {dirs.manifest_path}")
    except Exception as e:
        log(f"WARNING: failed to write manifest: {e}")

    return outputs



# -------------------------
# GUI
# -------------------------

if QtCore is not None and QtWidgets is not None:

    class _WorkerSignals(QtCore.QObject):  # type: ignore[misc]
        log = QtCore.Signal(str)  # type: ignore[attr-defined]
        done = QtCore.Signal(object)  # type: ignore[attr-defined]
        failed = QtCore.Signal(str)  # type: ignore[attr-defined]

    class _SplitterWorker(QtCore.QThread):  # type: ignore[misc]
        def __init__(self, config: SplitterConfig):
            super().__init__()
            self.config = config
            self.signals = _WorkerSignals()
            self._cancel = threading.Event()

        def cancel(self) -> None:
            self._cancel.set()

        def run(self) -> None:
            try:
                outputs = split_video(
                    self.config,
                    log=lambda s: self.signals.log.emit(s),
                    cancel=lambda: self._cancel.is_set(),
                )
                self.signals.done.emit(outputs)
            except Exception as e:
                self.signals.failed.emit(str(e))

    class VideoSplitterWindow(QtWidgets.QMainWindow):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("AI Content-Based Video Splitter")
            self.setMinimumWidth(860)

            self._worker: Optional[_SplitterWorker] = None

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            form = QtWidgets.QFormLayout()
            layout.addLayout(form)

            self.in_path = QtWidgets.QLineEdit()
            btn_in = QtWidgets.QPushButton("Browse...")
            btn_in.clicked.connect(self._browse_input)
            in_row = QtWidgets.QHBoxLayout()
            in_row.addWidget(self.in_path, 1)
            in_row.addWidget(btn_in)
            form.addRow("Input video", in_row)

            self.out_root = QtWidgets.QLineEdit()
            self.out_root.setPlaceholderText("(optional) e.g. D:\\exports  (creates <video>_output under this)")
            btn_out = QtWidgets.QPushButton("Browse...")
            btn_out.clicked.connect(self._browse_output)
            out_row = QtWidgets.QHBoxLayout()
            out_row.addWidget(self.out_root, 1)
            out_row.addWidget(btn_out)
            form.addRow("Output root", out_row)

            self.sens = QtWidgets.QDoubleSpinBox()
            self.sens.setRange(0.05, 1.0)
            self.sens.setSingleStep(0.01)
            self.sens.setValue(0.35)
            self.sens.setToolTip("Cosine distance threshold. Lower = more cuts, Higher = fewer cuts.")
            form.addRow("Sensitivity", self.sens)

            self.sample_fps = QtWidgets.QDoubleSpinBox()
            self.sample_fps.setRange(0.05, 10.0)
            self.sample_fps.setSingleStep(0.05)
            self.sample_fps.setValue(0.5)
            self.sample_fps.setToolTip("Frames per second to sample for content change.")
            form.addRow("Sample FPS", self.sample_fps)

            self.max_seg = QtWidgets.QSpinBox()
            self.max_seg.setRange(5, 24 * 3600)
            self.max_seg.setValue(600)
            form.addRow("Max segment (sec)", self.max_seg)

            self.min_seg = QtWidgets.QSpinBox()
            self.min_seg.setRange(0, 3600)
            self.min_seg.setValue(3)
            form.addRow("Min segment (sec)", self.min_seg)

            self.reencode = QtWidgets.QCheckBox("Re-encode (more accurate starts, slower)")
            self.reencode.setChecked(False)
            form.addRow("Export mode", self.reencode)

            self.model_path = QtWidgets.QLineEdit(
                str(Path.home() / ".video_splitter_ai" / "MobileNet-v3-Small.onnx")
            )
            form.addRow("Model path", self.model_path)

            self.model_url = QtWidgets.QLineEdit(MODEL_URL_DEFAULT)
            self.model_url.setToolTip("Download URL for the ONNX model.")
            form.addRow("Model URL", self.model_url)

            self.use_gpu = QtWidgets.QCheckBox("Use GPU if available (CUDAExecutionProvider)")
            self.use_gpu.setChecked(True)
            form.addRow("Inference", self.use_gpu)

            btn_row = QtWidgets.QHBoxLayout()
            self.btn_start = QtWidgets.QPushButton("Start")
            self.btn_cancel = QtWidgets.QPushButton("Cancel")
            self.btn_cancel.setEnabled(False)
            self.btn_start.clicked.connect(self._start)
            self.btn_cancel.clicked.connect(self._cancel)
            btn_row.addWidget(self.btn_start)
            btn_row.addWidget(self.btn_cancel)
            btn_row.addStretch(1)
            layout.addLayout(btn_row)

            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            layout.addWidget(self.log_view, 1)

            self.status = QtWidgets.QLabel("Ready.")
            layout.addWidget(self.status)

        def _append_log(self, s: str) -> None:
            self.log_view.appendPlainText(s)
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

        def _browse_input(self) -> None:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select video",
                "",
                "Videos (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*.*)",
            )
            if p:
                self.in_path.setText(p)

        def _browse_output(self) -> None:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output root")
            if p:
                self.out_root.setText(p)

        def _set_running(self, running: bool) -> None:
            self.btn_start.setEnabled(not running)
            self.btn_cancel.setEnabled(running)

        def _start(self) -> None:
            in_p = Path(self.in_path.text().strip('" ').strip())
            if not in_p.exists():
                QtWidgets.QMessageBox.critical(self, "Error", "Input video not found.")
                return

            out_root_text = self.out_root.text().strip('" ').strip()
            out_root = Path(out_root_text) if out_root_text else None

            cfg = SplitterConfig(
                input_video=in_p,
                output_root=out_root,
                sensitivity=float(self.sens.value()),
                sample_fps=float(self.sample_fps.value()),
                max_segment_sec=float(self.max_seg.value()),
                min_segment_sec=float(self.min_seg.value()),
                model_path=Path(self.model_path.text().strip()),
                model_url=self.model_url.text().strip() or MODEL_URL_DEFAULT,
                model_sha256=MODEL_SHA256_DEFAULT,
                use_gpu_if_available=bool(self.use_gpu.isChecked()),
                reencode=bool(self.reencode.isChecked()),
                video_codec="libx264",
                crf=18,
                preset="veryfast",
                audio_codec="aac",
                audio_bitrate="192k",
                ffmpeg_path="ffmpeg",
                ffprobe_path="ffprobe",
            )

            self.log_view.clear()
            self._append_log("Starting...")
            self.status.setText("Running...")
            self._set_running(True)

            self._worker = _SplitterWorker(cfg)
            self._worker.signals.log.connect(self._append_log)
            self._worker.signals.done.connect(self._on_done)
            self._worker.signals.failed.connect(self._on_failed)
            self._worker.start()

        def _cancel(self) -> None:
            if self._worker:
                self._append_log("Cancelling...")
                self._worker.cancel()

        def _on_done(self, outputs: object) -> None:
            self._set_running(False)
            self.status.setText("Done.")
            outs = outputs if isinstance(outputs, list) else []
            self._append_log(f"Done. Exported {len(outs)} segments.")
            QtWidgets.QMessageBox.information(self, "Done", f"Exported {len(outs)} segments.")

        def _on_failed(self, msg: str) -> None:
            self._set_running(False)
            self.status.setText("Failed.")
            self._append_log(f"ERROR: {msg}")
            QtWidgets.QMessageBox.critical(self, "Failed", msg)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Content-Based Video Splitter (ONNX embeddings).")
    p.add_argument("--input", type=str, help="Input video path.")
    p.add_argument("--output-root", type=str, default="", help="Optional output root directory.")
    p.add_argument("--sensitivity", type=float, default=0.35, help="Cosine distance threshold. Lower=more cuts.")
    p.add_argument("--sample-fps", type=float, default=0.5, help="Frame sampling rate (frames per second).")
    p.add_argument("--max-segment-sec", type=float, default=600, help="Max segment duration before forced split.")
    p.add_argument("--min-segment-sec", type=float, default=3, help="Minimum segment duration (merges tiny segments).")
    p.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / ".video_splitter_ai" / "MobileNet-v3-Small.onnx"),
        help="Path to ONNX model (auto-downloaded if missing).",
    )
    p.add_argument("--model-url", type=str, default=MODEL_URL_DEFAULT, help="Model download URL.")
    p.add_argument("--use-gpu", action="store_true", help="Use GPU if available (CUDAExecutionProvider).")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU only.")
    p.add_argument("--reencode", action="store_true", help="Re-encode segments (accurate starts).")
    p.add_argument("--copy", action="store_true", help="Stream copy segments (fast, may start off-keyframe).")
    p.add_argument("--video-codec", type=str, default="libx264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", type=str, default="veryfast")
    p.add_argument("--audio-codec", type=str, default="aac")
    p.add_argument("--audio-bitrate", type=str, default="192k")
    p.add_argument("--ffmpeg-path", type=str, default="ffmpeg")
    p.add_argument("--ffprobe-path", type=str, default="ffprobe")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.input:
        if QtWidgets is None or QtCore is None:
            print("PySide6 not installed. Install with: pip install PySide6", file=sys.stderr)
            return 2
        app = QtWidgets.QApplication(sys.argv)
        w = VideoSplitterWindow()  # type: ignore[name-defined]
        w.show()
        return int(app.exec())

    input_video = Path(args.input.strip('" '))
    output_root = Path(args.output_root.strip('" ')) if args.output_root else None

    use_gpu = bool(args.use_gpu)
    if args.no_gpu:
        use_gpu = False

    reencode = bool(args.reencode)
    if args.copy:
        reencode = False

    cfg = SplitterConfig(
        input_video=input_video,
        output_root=output_root,
        sensitivity=float(args.sensitivity),
        sample_fps=float(args.sample_fps),
        max_segment_sec=float(args.max_segment_sec),
        min_segment_sec=float(args.min_segment_sec),
        model_path=Path(args.model_path),
        model_url=str(args.model_url),
        model_sha256=MODEL_SHA256_DEFAULT,
        use_gpu_if_available=use_gpu,
        reencode=reencode,
        video_codec=str(args.video_codec),
        crf=int(args.crf),
        preset=str(args.preset),
        audio_codec=str(args.audio_codec),
        audio_bitrate=str(args.audio_bitrate),
        ffmpeg_path=str(args.ffmpeg_path),
        ffprobe_path=str(args.ffprobe_path),
    )

    def log(s: str) -> None:
        print(s, flush=True)

    try:
        split_video(cfg, log=log, cancel=lambda: False)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
