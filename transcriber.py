"""Whisper transcription engine with SRT output."""

import os
import sys
import json
import shutil
import time
from dataclasses import dataclass
from typing import Callable

import cuda_setup  # noqa: F401  — preload NVIDIA DLLs before CUDA imports

from faster_whisper import WhisperModel
from cue_merge import merge_short_cues
from onset_refine import build_energy, snap_starts
import loop_guard

# OpenCC simplified -> Taiwan traditional. large-v3 occasionally emits simplified
# characters; convert at the source so both the SRT and the words sidecar are
# traditional and stay character-consistent for downstream alignment. s2tw is
# character-level (no vocabulary substitution), so it does not change the words
# actually spoken. No-op for non-Chinese text. When OpenCC is missing the run
# reports it loudly instead of silently emitting simplified characters.
import s2tw_gate


# Loaded models are parked here for the lifetime of the process — see the note
# in transcribe(). This makes transcribe() a once-per-process call: a second
# call in the same process loads a second model without freeing the first.
_MODEL_KEEPALIVE = []


def _to_traditional(text: str) -> str:
    # Guarded, not unconditional: the same evidence rule as the output gate, so
    # the two passes agree instead of this one creating work for that one.
    return s2tw_gate.convert_guarded(text)


def _strip_commas(text: str) -> str:
    """Remove commas from subtitle display text (half- and full-width).

    Applied only to cue/word text, never to timecodes — the SRT timecode
    millisecond separator is also a ',' and must stay intact.
    """
    return text.replace(",", "").replace("，", "")


@dataclass
class TranscribeOptions:
    """Options for transcription."""
    model_path: str
    input_file: str
    output_srt: str
    language: str | None = None  # None = auto-detect
    device: str = "cuda"
    compute_type: str = "float16"
    initial_prompt: str | None = None  # Domain keywords to guide recognition
    use_mfa: bool = True  # prefer MFA forced alignment; fall back to energy onset


@dataclass
class TranscribeProgress:
    """Progress update during transcription."""
    phase: str  # "loading", "transcribing", "writing"
    percent: float  # 0.0 to 100.0
    message: str
    segment_count: int = 0


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_atomic(path: str, text: str, keep_backup: bool = True) -> str | None:
    """Write via a temp file and os.replace, so a crash never leaves half a file.

    An existing file at `path` is copied to `path + ".bak"` first (one
    generation) when keep_backup is set, because the SRT may be a hand-edited
    copy the user re-transcribed over. Returns the backup path, or None.
    """
    backup = None
    if keep_backup and os.path.isfile(path):
        backup = path + ".bak"
        shutil.copy2(path, backup)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return backup


def _detect_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
    """Force CUDA GPU usage. Raises RuntimeError if CUDA is unavailable."""
    if compute_type == "auto":
        compute_type = "float16"
    return "cuda", compute_type


def transcribe(
    options: TranscribeOptions,
    on_progress: Callable[[TranscribeProgress], None] | None = None,
) -> str:
    """Run transcription and write SRT file.

    Call this once per process: the Whisper model it loads is never released
    (see _MODEL_KEEPALIVE). Batch callers should run one child process per file.

    Args:
        options: Transcription settings.
        on_progress: Optional callback for progress updates.

    Returns:
        Path to the generated SRT file.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If transcription fails.
    """
    if not os.path.isfile(options.input_file):
        raise FileNotFoundError(f"Input file not found: {options.input_file}")

    def report(phase: str, percent: float, msg: str, seg_count: int = 0):
        if on_progress:
            on_progress(TranscribeProgress(phase, percent, msg, seg_count))

    # Phase 1: Load model
    report("loading", 0, "Loading Whisper model...")
    device, compute_type = _detect_device_and_compute(options.device, options.compute_type)
    report("loading", 10, f"Using device: {device}, compute: {compute_type}")

    try:
        model = WhisperModel(
            options.model_path,
            device=device,
            compute_type=compute_type,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    # Park the model so it is never destroyed. Freeing a CUDA-backed
    # CTranslate2 model after a long run raises a C++ exception that no Python
    # handler can see: the process is killed by abort() instead
    # (Windows 0xe06d7363 then 0xc0000409). Measured on 01.wav and 02.wav —
    # the SRT and sidecar were fully written, then the process died 19s and 10s
    # later while this local was being released on return. Callers are
    # short-lived processes, so process exit reclaims it.
    _MODEL_KEEPALIVE.append(model)

    report("loading", 30, "Model loaded successfully.")

    # Phase 2: Transcribe
    report("transcribing", 30, "Starting transcription...")

    try:
        transcribe_kwargs = dict(
            language=options.language,
            beam_size=8,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
            ),
        )
        if options.initial_prompt:
            transcribe_kwargs['initial_prompt'] = options.initial_prompt
            report("transcribing", 31, f"Using initial_prompt ({len(options.initial_prompt)} chars)")

        segments_gen, info = model.transcribe(
            options.input_file,
            **transcribe_kwargs,
        )
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    detected_lang = info.language
    duration = info.duration
    report("transcribing", 35, f"Detected language: {detected_lang}, duration: {duration:.1f}s")

    # Collect segments
    segments = []
    for seg in segments_gen:
        segments.append(seg)
        # Estimate progress based on segment end time vs total duration
        if duration > 0:
            pct = 35 + (seg.end / duration) * 55  # 35% to 90%
            pct = min(pct, 90)
        else:
            pct = 60
        report("transcribing", pct, f"Segment {len(segments)}: {seg.text.strip()[:50]}...", len(segments))

    report("transcribing", 90, f"Transcription complete. {len(segments)} segments found.")

    # Phase 3: Write SRT (with conservative cue merge) + word-level sidecar
    report("writing", 90, "Writing SRT file...")

    output_path = options.output_srt
    if not output_path:
        base = os.path.splitext(options.input_file)[0]
        output_path = base + ".srt"

    # Build cues + word-level timestamps from Whisper segments.
    # Convert both to traditional with the same converter so the SRT text and the
    # words sidecar stay character-consistent for Quick subtitles' alignment.
    cues = [{"start": seg.start, "end": seg.end,
             "text": _to_traditional(seg.text.strip())}
            for seg in segments]
    words = [{"word": _to_traditional(w.word), "start": w.start, "end": w.end}
             for seg in segments for w in (seg.words or [])]

    # Repair repetition loops before anything else touches the cues: a loop has
    # overwritten real speech, and aligning hallucinated text with MFA only
    # spends time on words nobody said.
    cues, words, _ = loop_guard.repair_loops(
        cues, words, model, options.input_file, detected_lang,
        on_message=lambda m: report("writing", 91, m),
    )

    # Conservative merge: keep Whisper's boundaries, absorb over-short fragments.
    cues = merge_short_cues(cues)

    # Timing refinement. Prefer MFA forced alignment (precise word-boundary
    # timing) when available; it returns a word-level sidecar built from MFA's
    # own word times. If MFA is unavailable or fails, fall back to
    # energy-onset refinement (snap starts that fall in silence forward to the
    # real RMS-energy onset). Both fix raw Whisper's "subtitle appears early after
    # a pause" tendency; MFA is more precise across all cue boundaries.
    refined = None
    if options.use_mfa:
        report("writing", 92, "MFA forced alignment (precise timing)...")
        try:
            from mfa_align import mfa_refine_cues
            refined = mfa_refine_cues(
                options.input_file, cues,
                on_progress=lambda m: report("writing", 93, m),
            )
        except Exception as e:
            report("writing", 93,
                   f"MFA 無法執行，改用能量對齊：{type(e).__name__}: {e}")
            refined = None

    if refined is not None:
        cues, words = refined
    else:
        report("writing", 92, "Refining onsets to speech start...")
        energy = build_energy(options.input_file)
        cues = snap_starts(cues, energy)
        words = snap_starts(words, energy)

    # Output gate: whatever the refinement path did to the text, nothing leaves
    # this function still holding simplified characters. Cues and the sidecar are
    # gated together so both stay character-consistent.
    if s2tw_gate.AVAILABLE:
        gated = s2tw_gate.convert_cues(cues, "text")
        gated += s2tw_gate.convert_cues(words, "word")
        # The cue text is the source of truth for characters. A word judged on
        # its own has no context, so a merge character the cue resolved with
        # context has to be carried across rather than re-decided here.
        synced = s2tw_gate.sync_words_to_cues(cues, words)
        report("writing", 94,
               f"簡轉繁閘門：修正 {gated} 處，sidecar 對齊 {synced} 字"
               if gated or synced else "簡轉繁閘門：無殘留")
    else:
        report("writing", 94, s2tw_gate.UNAVAILABLE_WARNING)

    srt_text = "".join(
        f"{i}\n{_format_srt_time(c['start'])} --> {_format_srt_time(c['end'])}\n"
        f"{_strip_commas(c['text'])}\n\n"
        for i, c in enumerate(cues, start=1))
    try:
        backup = _write_atomic(output_path, srt_text)
    except OSError as e:
        raise RuntimeError(f"Failed to write SRT file: {e}") from e
    if backup:
        report("writing", 95, f"已備份原有字幕：{backup}")

    # Word-level sidecar consumed by Quick subtitles (avoids a second Whisper pass).
    # Strip commas here too so the downstream SRT stays comma-free as well.
    words_out = [{**w, "word": _strip_commas(w["word"])} for w in words]
    words_path = os.path.splitext(output_path)[0] + ".words.json"
    try:
        _write_atomic(words_path, json.dumps(words_out, ensure_ascii=False),
                      keep_backup=False)
    except OSError as e:
        raise RuntimeError(f"Failed to write words sidecar: {e}") from e

    # Verify against the bytes that actually landed on disk, not against memory.
    if s2tw_gate.AVAILABLE:
        with open(output_path, "r", encoding="utf-8") as f:
            leftover = s2tw_gate.scan_text(f.read())
        if leftover:
            lines = ", ".join(f"L{n}" for n, _, _ in leftover[:5])
            report("writing", 99,
                   f"警告：SRT 仍有 {len(leftover)} 行簡體殘留（{lines}）")

    report("writing", 100,
           f"Done! Saved to: {output_path}｜{s2tw_gate.status_line()}")
    return output_path
