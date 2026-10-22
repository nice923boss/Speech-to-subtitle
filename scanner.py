"""Scan for ffmpeg and Whisper models on the local machine."""

import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass

# Shared with app.py, which owns writing it.
STS_CONFIG_FILE = os.path.join(str(Path.home()), ".speech-to-subtitle", "config.json")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@dataclass
class WhisperModel:
    """Represents a discovered Whisper model."""
    name: str
    path: str
    size_label: str  # tiny, base, small, medium, large, etc.
    format_type: str  # "ctranslate2" or "openai"
    size_bytes: int

    def display_name(self) -> str:
        return f"{self.name} ({self.size_label}, {self._human_size()})"

    def _human_size(self) -> str:
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.0f} KB"
        if self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_bytes / (1024 * 1024):.0f} MB"
        return f"{self.size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Whisper model size keywords for identification
SIZE_KEYWORDS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "distil"]


def _detect_size_label(name: str) -> str:
    """Extract model size label from model name/path."""
    name_lower = name.lower()
    # Check from most specific to least specific
    for kw in reversed(SIZE_KEYWORDS):
        if kw in name_lower:
            return kw
    return "unknown"


def _dir_total_size(path: str) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _is_ctranslate2_model(directory: str) -> bool:
    """Check if a directory contains a CTranslate2 Whisper model."""
    has_model = os.path.isfile(os.path.join(directory, "model.bin"))
    has_config = os.path.isfile(os.path.join(directory, "config.json"))
    return has_model and has_config


def _is_openai_whisper_model(filepath: str) -> bool:
    """Check if a file is an OpenAI-format Whisper .pt model."""
    return filepath.endswith(".pt") and os.path.getsize(filepath) > 50 * 1024 * 1024


def _scan_huggingface_cache(cache_dir: str) -> list[WhisperModel]:
    """Scan a HuggingFace-style cache directory for Whisper models."""
    models = []
    if not os.path.isdir(cache_dir):
        return models

    for entry in os.listdir(cache_dir):
        entry_lower = entry.lower()
        if "whisper" not in entry_lower:
            continue

        model_dir = os.path.join(cache_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        # HuggingFace cache structure: models--org--name/snapshots/<hash>/
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                snap_path = os.path.join(snapshots_dir, snapshot)
                if os.path.isdir(snap_path) and _is_ctranslate2_model(snap_path):
                    size = _dir_total_size(snap_path)
                    name = entry.replace("models--", "").replace("--", "/")
                    models.append(WhisperModel(
                        name=name,
                        path=snap_path,
                        size_label=_detect_size_label(entry),
                        format_type="ctranslate2",
                        size_bytes=size,
                    ))
        else:
            # Flat directory with model files
            if _is_ctranslate2_model(model_dir):
                models.append(WhisperModel(
                    name=entry,
                    path=model_dir,
                    size_label=_detect_size_label(entry),
                    format_type="ctranslate2",
                    size_bytes=_dir_total_size(model_dir),
                ))

    return models


def _scan_directory_for_models(directory: str) -> list[WhisperModel]:
    """Recursively scan a directory for Whisper model files."""
    models = []
    if not os.path.isdir(directory):
        return models

    for dirpath, dirnames, filenames in os.walk(directory):
        # CTranslate2 format
        if _is_ctranslate2_model(dirpath):
            name = os.path.basename(dirpath)
            # Try to get a better name from parent dirs
            parent = os.path.basename(os.path.dirname(dirpath))
            if "whisper" in parent.lower():
                name = parent
            elif "whisper" in name.lower():
                pass
            else:
                name = f"{parent}/{name}"

            models.append(WhisperModel(
                name=name,
                path=dirpath,
                size_label=_detect_size_label(dirpath),
                format_type="ctranslate2",
                size_bytes=_dir_total_size(dirpath),
            ))
            # Don't recurse into this model's subdirectories
            dirnames.clear()
            continue

        # OpenAI .pt format
        for f in filenames:
            if f.endswith(".pt"):
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.getsize(fp) > 50 * 1024 * 1024:
                        models.append(WhisperModel(
                            name=f.replace(".pt", ""),
                            path=fp,
                            size_label=_detect_size_label(f),
                            format_type="openai",
                            size_bytes=os.path.getsize(fp),
                        ))
                except OSError:
                    pass

    return models


def scan_whisper_models(extra_dirs: list[str] | None = None) -> list[WhisperModel]:
    """Scan common locations and extra directories for Whisper models.

    Returns a deduplicated list of discovered models.
    """
    seen_paths: set[str] = set()
    all_models: list[WhisperModel] = []

    scan_dirs: list[str] = []

    # 1. HuggingFace default cache
    hf_home = os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"))
    hf_hub_cache = os.path.join(hf_home, "hub")
    scan_dirs.append(hf_hub_cache)

    # 2. Common Whisper model locations on Windows
    home = str(Path.home())
    scan_dirs.extend([
        os.path.join(home, ".cache", "whisper"),  # openai-whisper default
        os.path.join(home, "AppData", "Local", "whisper"),
    ])

    # 3. User-specified extra directories
    if extra_dirs:
        scan_dirs.extend(extra_dirs)

    for d in scan_dirs:
        d = os.path.normpath(d)
        if not os.path.isdir(d):
            continue

        # Try HuggingFace cache style first
        found = _scan_huggingface_cache(d)
        if not found:
            found = _scan_directory_for_models(d)

        for m in found:
            norm = os.path.normpath(m.path)
            if norm not in seen_paths:
                seen_paths.add(norm)
                all_models.append(m)

    # Filter out "base" size models — not used by this project
    all_models = [m for m in all_models if m.size_label != "base"]

    return all_models


def configured_scan_dirs(config: dict | None = None) -> list[str]:
    """Extra model folders from the STS config plus the known model folder.

    The GUI passes its loaded config; the worker passes nothing and reads the
    same file, so both resolve the same default model.
    """
    if config is None:
        try:
            with open(STS_CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
    extra = list(config.get("extra_scan_dirs", []))
    known = config.get("whisper_model_dir",
                       os.environ.get("WHISPER_MODEL_DIR", DEFAULT_MODEL_DIR))
    if known not in extra:
        extra.append(known)
    return extra


def preferred_model(models: list[WhisperModel]) -> WhisperModel | None:
    """large-v3 when present, else the first model found."""
    for m in models:
        if m.size_label == "large-v3":
            return m
    return models[0] if models else None


def find_ffmpeg() -> str | None:
    """Find ffmpeg executable. Returns path or None."""
    # 1. Check PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Check common Windows locations
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]

    # Also check Downloads and common project dirs
    home = str(Path.home())
    for subdir in ["Downloads", "Desktop", "Documents"]:
        base = os.path.join(home, subdir)
        if os.path.isdir(base):
            for entry in os.listdir(base):
                if "ffmpeg" in entry.lower():
                    candidate = os.path.join(base, entry, "bin", "ffmpeg.exe")
                    common_paths.append(candidate)

    for p in common_paths:
        if os.path.isfile(p):
            return p

    return None
