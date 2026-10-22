"""Phase 1 stability fixes: atomic SRT writes, tree kill, visible MFA fallback."""
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mfa_align  # noqa: E402


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return f'"{pid}"' in out


def test_write_atomic_backs_up_previous_srt(tmp_path):
    transcriber = pytest.importorskip("transcriber")
    path = str(tmp_path / "a.srt")
    assert transcriber._write_atomic(path, "first") is None
    backup = transcriber._write_atomic(path, "second")
    assert backup == path + ".bak"
    assert open(backup, encoding="utf-8").read() == "first"
    assert open(path, encoding="utf-8").read() == "second"
    assert not os.path.exists(path + ".tmp")


def test_run_tree_timeout_kills_grandchild(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        mfa_align._run_tree([sys.executable, "-c", script], timeout=3)
    grandchild = int(pid_file.read_text())
    time.sleep(1)
    assert not _alive(grandchild)


def test_refine_reports_why_mfa_is_unavailable(monkeypatch):
    monkeypatch.setattr(mfa_align, "check_mfa_available",
                        lambda: (False, "MFA version check timed out"))
    messages = []
    cues = [{"start": 0.0, "end": 1.0, "text": "你好"}]
    assert mfa_align.mfa_refine_cues("x.wav", cues, on_progress=messages.append) is None
    assert any("timed out" in m for m in messages)


def test_refine_reports_alignment_failure(monkeypatch):
    monkeypatch.setattr(mfa_align, "check_mfa_available", lambda: (True, "MFA 3.3.9"))

    def boom(*a, **k):
        raise RuntimeError("MFA alignment failed")

    monkeypatch.setattr(mfa_align, "mfa_align", boom)
    messages = []
    cues = [{"start": 0.0, "end": 1.0, "text": "你好"}]
    assert mfa_align.mfa_refine_cues("x.wav", cues, on_progress=messages.append) is None
    assert any("MFA alignment failed" in m for m in messages)


def test_mfa_check_runs_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(mfa_align, "_MFA_CHECK", None)
    monkeypatch.setattr(mfa_align, "_check_mfa",
                        lambda: calls.append(1) or (True, "MFA 3.3.9"))
    mfa_align.check_mfa_available()
    mfa_align.check_mfa_available()
    assert len(calls) == 1
