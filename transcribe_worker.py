"""Single-file transcription worker, run as a child process by app.py.

Why a separate process: CTranslate2/cuDNN can raise a C++ exception that never
reaches Python (Windows Error Reporting shows 0xe06d7363 followed by abort()
0xc0000409 in ucrtbase). A `try/except` in the GUI cannot catch that, so a batch
run took the whole window down with it. Running each file in its own process
means a native abort kills only the child; the GUI sees a non-zero exit code,
logs the failure and carries on with the next file. It also gives every file a
fresh CUDA context instead of stacking model loads in one long-lived process.

This is also the interface other apps (Video Console) call. The full contract
is in docs/cli-contract.md; keep the two in step.

Protocol:
    argv[1] = path to a JSON file holding TranscribeOptions fields.
              model_path may be omitted: the same default model the GUI picks
              is used (see scanner.configured_scan_dirs / preferred_model).
    argv[1] = "--check" instead: print one {"t": "check", ...} line describing
              what a run would use, then exit 0 (ok) or 1 (cannot run).
    stdout  = one JSON object per line, UTF-8:
              {"t": "progress", "phase": ..., "percent": ..., "message": ...,
               "segment_count": ...}
              {"t": "done", "path": ..., "words": ...}
              {"t": "error", "message": ...}
    exit    = 0 on success, 1 on a Python-level error.
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Bumped when a field callers rely on changes meaning or goes away.
PROTOCOL_VERSION = 1


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _default_model_path() -> str:
    from scanner import scan_whisper_models, configured_scan_dirs, preferred_model
    model = preferred_model(scan_whisper_models(extra_dirs=configured_scan_dirs()))
    if model is None:
        raise FileNotFoundError(
            "No Whisper model found. Open Speech-to-subtitle and use 掃描資料夾, "
            "or pass model_path.")
    return model.path


def _check() -> bool:
    """Report what a run would use. CUDA and a model are required; MFA, OpenCC
    and ffmpeg only degrade the result (energy alignment, simplified text)."""
    report = {"t": "check", "protocol": PROTOCOL_VERSION}

    try:
        import cuda_setup  # noqa: F401
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        report["cuda"] = {"ok": count > 0, "devices": count}
    except Exception as e:
        report["cuda"] = {"ok": False, "message": f"{type(e).__name__}: {e}"}

    try:
        report["model"] = {"ok": True, "path": _default_model_path()}
    except Exception as e:
        report["model"] = {"ok": False, "message": str(e)}

    from scanner import find_ffmpeg
    ffmpeg = find_ffmpeg()
    report["ffmpeg"] = {"ok": ffmpeg is not None, "path": ffmpeg}

    import mfa_align
    ok, msg = mfa_align.check_mfa_available()
    report["mfa"] = {"ok": ok, "message": msg}

    import s2tw_gate
    report["opencc"] = {"ok": s2tw_gate.AVAILABLE,
                        "message": s2tw_gate.status_line()}

    report["ok"] = report["cuda"]["ok"] and report["model"]["ok"]
    _emit(report)
    return report["ok"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if sys.argv[1] == "--check":
        os._exit(0 if _check() else 1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    from transcriber import transcribe, TranscribeOptions, TranscribeProgress

    if not payload.get("model_path"):
        payload = {**payload, "model_path": _default_model_path()}
    options = TranscribeOptions(**payload)

    def on_progress(p: TranscribeProgress):
        _emit({
            "t": "progress",
            "phase": p.phase,
            "percent": p.percent,
            "message": p.message,
            "segment_count": p.segment_count,
        })

    try:
        result_path = transcribe(options, on_progress=on_progress)
    except Exception as e:
        _emit({"t": "error", "message": f"{type(e).__name__}: {e}"})
        sys.stdout.flush()
        os._exit(1)

    _emit({"t": "done", "path": result_path,
           "words": os.path.splitext(result_path)[0] + ".words.json"})
    sys.stdout.flush()
    # Skip interpreter teardown: destroying the CUDA-backed model at exit is the
    # crash path this worker exists to contain. Same guard as transcribe_cli.py.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _emit({"t": "error", "message": f"{type(e).__name__}: {e}"})
        os._exit(1)
