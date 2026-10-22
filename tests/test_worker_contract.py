"""The worker CLI is what Video Console calls; these pin its contract."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scanner  # noqa: E402
import transcribe_worker  # noqa: E402


def _model(label: str) -> scanner.WhisperModel:
    return scanner.WhisperModel(name=label, path=f"/m/{label}", size_label=label,
                                format_type="ctranslate2", size_bytes=1)


def test_preferred_model_picks_large_v3():
    models = [_model("medium"), _model("large-v3"), _model("large-v2")]
    assert scanner.preferred_model(models).size_label == "large-v3"


def test_preferred_model_falls_back_to_first():
    assert scanner.preferred_model([_model("medium"), _model("small")]).size_label == "medium"
    assert scanner.preferred_model([]) is None


def test_configured_scan_dirs_adds_known_dir_once():
    config = {"extra_scan_dirs": ["D:/a", "D:/known"], "whisper_model_dir": "D:/known"}
    assert scanner.configured_scan_dirs(config) == ["D:/a", "D:/known"]
    assert config["extra_scan_dirs"] == ["D:/a", "D:/known"]  # caller's list untouched


def test_configured_scan_dirs_reads_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"whisper_model_dir": "D:/models"}), encoding="utf-8")
    monkeypatch.setattr(scanner, "STS_CONFIG_FILE", str(cfg))
    assert scanner.configured_scan_dirs() == ["D:/models"]


def test_default_model_path_says_how_to_fix(monkeypatch):
    monkeypatch.setattr(scanner, "scan_whisper_models", lambda extra_dirs=None: [])
    with pytest.raises(FileNotFoundError, match="model_path"):
        transcribe_worker._default_model_path()


def test_check_prints_one_json_line_from_any_cwd(tmp_path):
    pytest.importorskip("ctranslate2")
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "transcribe_worker.py"), "--check"],
        cwd=str(tmp_path), capture_output=True, timeout=180,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = proc.stdout.decode("utf-8").splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["t"] == "check"
    assert report["protocol"] == transcribe_worker.PROTOCOL_VERSION
    for key in ("cuda", "model", "ffmpeg", "mfa", "opencc"):
        assert "ok" in report[key]
    assert proc.returncode == (0 if report["ok"] else 1)
