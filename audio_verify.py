"""Audio-subtitle verification tool.

Runs Voice Activity Detection (VAD) on audio and cross-references against
SRT subtitles to verify timing accuracy.

Verifies the 4 subtitle pipeline rules:
  1. Each subtitle start time aligns with actual speech onset
  2. (Text verification - handled separately by Claude Code)
  3. Gaps between different sentences only where audio has >= 2s silence
  4. Overall pass/fail summary

Usage:
    python audio_verify.py <audio_file> <srt_file> [--output report.json]

Output:
    JSON report with speech/silence segments and per-rule verification results.
"""

import sys
import os
import re
import json
import argparse
from dataclasses import dataclass, asdict

import subprocess
import tempfile

import torch
import soundfile as sf

# Ensure this script's directory is on sys.path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from srt_common import parse_srt_file, time_to_ms  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimeSpan:
    start: float
    end: float
    duration: float


@dataclass
class SrtEntry:
    index: int
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# SRT parser (delegates to srt_common)
# ---------------------------------------------------------------------------

def parse_srt(path):
    """Parse an SRT file and return a list of SrtEntry (via srt_common)."""
    raw_entries = parse_srt_file(path)
    return [
        SrtEntry(
            index=e['idx'],
            start=e['start_ms'] / 1000.0,
            end=e['end_ms'] / 1000.0,
            text=e['text'],
        )
        for e in raw_entries
    ]


# ---------------------------------------------------------------------------
# VAD analysis
# ---------------------------------------------------------------------------

def _load_audio_as_16k_mono(audio_path, sample_rate=16000):
    """Load audio file as 16kHz mono using ffmpeg + soundfile.

    Uses ffmpeg to convert any format to 16kHz mono WAV in a temp file,
    then reads with soundfile. This avoids torchaudio backend issues.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path,
             '-ar', str(sample_rate), '-ac', '1', '-f', 'wav', tmp.name],
            capture_output=True, check=True,
        )
        data, sr = sf.read(tmp.name, dtype='float32')
        return torch.from_numpy(data), sr
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def run_vad(audio_path, sample_rate=16000, vad_threshold=0.65):
    """Run Silero VAD on audio file and return speech timestamps.

    Args:
        audio_path: Path to audio/video file.
        sample_rate: Target sample rate (default 16000).
        vad_threshold: VAD confidence threshold (0-1). Higher = stricter,
            only detect clear speech. Use 0.65+ for audio with background
            music/noise (e.g. sports commentary, interviews with ambient sound).

    Returns:
        (speech_segments: list[TimeSpan], total_duration: float)
    """
    # Load silero-vad model
    model, utils = torch.hub.load(
        'snakers4/silero-vad', 'silero_vad',
        trust_repo=True,
    )
    get_speech_ts = utils[0]  # get_speech_timestamps

    # Load audio as 16kHz mono via ffmpeg
    wav, sr = _load_audio_as_16k_mono(audio_path, sample_rate)

    # Run VAD with stricter settings to distinguish speech from background noise
    speech_timestamps = get_speech_ts(
        wav, model,
        sampling_rate=sample_rate,
        threshold=vad_threshold,
        min_speech_duration_ms=300,
        min_silence_duration_ms=500,
        speech_pad_ms=50,
    )

    # Convert sample indices to seconds
    speech_segments = []
    for ts in speech_timestamps:
        s = ts['start'] / sample_rate
        e = ts['end'] / sample_rate
        speech_segments.append(TimeSpan(start=round(s, 3), end=round(e, 3),
                                        duration=round(e - s, 3)))

    return speech_segments, round(len(wav) / sample_rate, 3)


def find_silence_segments(speech_segments, total_duration, min_silence=0.5):
    """Find silence gaps between speech segments.

    Args:
        speech_segments: List of TimeSpan from VAD.
        total_duration: Total audio duration in seconds.
        min_silence: Minimum silence duration to report (seconds).

    Returns:
        List of TimeSpan for silence segments.
    """
    silences = []
    prev_end = 0.0

    for seg in speech_segments:
        gap = seg.start - prev_end
        if gap >= min_silence:
            silences.append(TimeSpan(
                start=round(prev_end, 3),
                end=round(seg.start, 3),
                duration=round(gap, 3),
            ))
        prev_end = seg.end

    # Trailing silence
    trailing = total_duration - prev_end
    if trailing >= min_silence:
        silences.append(TimeSpan(
            start=round(prev_end, 3),
            end=round(total_duration, 3),
            duration=round(trailing, 3),
        ))

    return silences


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------

def _is_during_speech(time_sec, speech_segments, tolerance=0.5):
    """Check if a given time falls within or near a speech segment."""
    for seg in speech_segments:
        # Within segment (with tolerance for early start)
        if (seg.start - tolerance) <= time_sec <= seg.end:
            return True, 0.0
    # Find nearest speech segment
    min_dist = float('inf')
    for seg in speech_segments:
        dist = min(abs(time_sec - seg.start), abs(time_sec - seg.end))
        min_dist = min(min_dist, dist)
    return False, round(min_dist, 3)


def _get_silence_at(time_start, time_end, silence_segments):
    """Find the longest silence segment that overlaps with the given time range."""
    best = None
    for sil in silence_segments:
        # Check overlap
        overlap_start = max(time_start, sil.start)
        overlap_end = min(time_end, sil.end)
        if overlap_end > overlap_start:
            if best is None or sil.duration > best.duration:
                best = sil
    return best


def verify_rule_1(srt_entries, speech_segments, tolerance_ms=500):
    """Rule 1: Each subtitle start time must align with speech onset.

    Returns dict with passed/failed counts and failure details.
    """
    tolerance = tolerance_ms / 1000
    passed = []
    failed = []

    for entry in srt_entries:
        is_speech, distance = _is_during_speech(
            entry.start, speech_segments, tolerance
        )
        result = {
            'entry': entry.index,
            'srt_start': entry.start,
            'text_preview': entry.text[:30],
        }
        if is_speech:
            result['status'] = 'pass'
            passed.append(result)
        else:
            result['status'] = 'fail'
            result['nearest_speech_distance_ms'] = round(distance * 1000)
            failed.append(result)

    return {
        'rule': 'Rule 1: Start time aligns with speech',
        'tolerance_ms': tolerance_ms,
        'total': len(srt_entries),
        'passed': len(passed),
        'failed': len(failed),
        'pass_rate': round(len(passed) / max(len(srt_entries), 1) * 100, 1),
        'failures': failed,
    }


def verify_rule_3(srt_entries, speech_segments, silence_segments,
                   min_silence_for_gap=2.0, max_same_sentence_gap=0.05):
    """Rule 3: Gaps between subtitles only where audio has >= 2s silence.

    Classification by gap size:
    - <= 50ms: same-sentence split (skip, always OK)
    - 50ms ~ 1.5s: natural inter-sentence breath pause (info, not a problem)
    - 1.5s ~ 2s: notable pause, check if silence exists (warn if no silence)
    - >= 2s: significant gap, MUST correspond to >= 2s audio silence (fail if not)

    Returns dict with verification results.
    """
    results = []

    for i in range(len(srt_entries) - 1):
        curr = srt_entries[i]
        nxt = srt_entries[i + 1]
        gap = nxt.start - curr.end

        if gap <= max_same_sentence_gap:
            # Continuous or near-continuous: same-sentence split, always OK
            continue

        # Find silence in this gap region
        silence = _get_silence_at(curr.end, nxt.start, silence_segments)
        silence_dur = silence.duration if silence else 0.0

        entry_result = {
            'between_entries': [curr.index, nxt.index],
            'gap_seconds': round(gap, 3),
            'curr_end': round(curr.end, 3),
            'next_start': round(nxt.start, 3),
        }

        if silence and silence_dur >= min_silence_for_gap:
            # Gap corresponds to real silence >= 2s: perfect
            entry_result['status'] = 'pass'
            entry_result['silence_duration'] = silence_dur
        elif gap < 1.5:
            # Small gap (< 1.5s): natural breath pause between sentences.
            # Whisper segments naturally have these gaps. Not a problem.
            entry_result['status'] = 'info'
            entry_result['reason'] = (
                f'Natural inter-sentence pause ({gap:.2f}s) — expected from '
                f'Whisper segmentation'
            )
        elif gap < min_silence_for_gap:
            # Medium gap (1.5s ~ 2s): notable but below threshold
            entry_result['status'] = 'warn'
            entry_result['reason'] = (
                f'Notable gap ({gap:.1f}s) without >= {min_silence_for_gap}s '
                f'silence (detected: {silence_dur:.1f}s)'
            )
        else:
            # Large gap (>= 2s) without matching silence: problem
            entry_result['status'] = 'fail'
            entry_result['reason'] = (
                f'Large gap ({gap:.1f}s) but no >= {min_silence_for_gap}s '
                f'silence detected (found: {silence_dur:.1f}s). '
                f'This gap may be artificial or audio has background noise.'
            )

        results.append(entry_result)

    passed = sum(1 for r in results if r['status'] == 'pass')
    info = sum(1 for r in results if r['status'] == 'info')
    warned = sum(1 for r in results if r['status'] == 'warn')
    failed = sum(1 for r in results if r['status'] == 'fail')

    return {
        'rule': 'Rule 3: Gaps only at >= 2s silence',
        'min_silence_seconds': min_silence_for_gap,
        'gaps_checked': len(results),
        'passed': passed,
        'info': info,
        'warnings': warned,
        'failed': failed,
        'details': [r for r in results if r['status'] in ('warn', 'fail')],
    }


def verify_rule_4_summary(rule_1, rule_3):
    """Rule 4: Overall pass/fail summary."""
    r1_ok = rule_1['failed'] == 0
    r3_ok = rule_3['failed'] == 0

    issues = []
    if not r1_ok:
        issues.append(
            f"Rule 1: {rule_1['failed']}/{rule_1['total']} subtitles "
            f"have start times not aligned with speech"
        )
    if not r3_ok:
        issues.append(
            f"Rule 3: {rule_3['failed']} gaps exist during active speech"
        )
    if rule_3['warnings'] > 0:
        issues.append(
            f"Rule 3: {rule_3['warnings']} gaps with < 2s silence (warnings)"
        )

    return {
        'rule': 'Rule 4: Overall verification',
        'overall_pass': r1_ok and r3_ok,
        'rule_1_pass': r1_ok,
        'rule_3_pass': r3_ok,
        'issues': issues,
        'summary': 'ALL RULES PASSED' if (r1_ok and r3_ok) else 'ISSUES FOUND',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Audio-subtitle verification: VAD analysis + SRT cross-reference'
    )
    parser.add_argument('audio_file', help='Audio/video file path')
    parser.add_argument('srt_file', help='SRT subtitle file path')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON report path (default: <srt>_verify.json)')
    parser.add_argument('--tolerance', type=int, default=500,
                        help='Start time tolerance in ms (default: 500)')
    parser.add_argument('--min-silence', type=float, default=2.0,
                        help='Minimum silence duration for gaps in seconds (default: 2.0)')
    parser.add_argument('--vad-threshold', type=float, default=0.65,
                        help='VAD confidence threshold 0-1 (default: 0.65). '
                             'Higher = stricter, better for noisy audio')

    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f'Error: Audio file not found: {args.audio_file}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.srt_file):
        print(f'Error: SRT file not found: {args.srt_file}', file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base = os.path.splitext(args.srt_file)[0]
        output_path = base + '_verify.json'

    # Step 1: Run VAD
    print('Running VAD analysis...', file=sys.stderr)
    speech_segments, total_duration = run_vad(args.audio_file,
                                              vad_threshold=args.vad_threshold)
    silence_segments = find_silence_segments(speech_segments, total_duration)

    print(f'Audio duration: {total_duration:.1f}s', file=sys.stderr)
    print(f'Speech segments: {len(speech_segments)}', file=sys.stderr)
    print(f'Silence segments (>= 0.5s): {len(silence_segments)}', file=sys.stderr)

    # Step 2: Parse SRT
    print('Parsing SRT...', file=sys.stderr)
    srt_entries = parse_srt(args.srt_file)
    print(f'SRT entries: {len(srt_entries)}', file=sys.stderr)

    # Step 3: Verify rules
    print('Verifying rules...', file=sys.stderr)
    rule_1 = verify_rule_1(srt_entries, speech_segments, args.tolerance)
    rule_3 = verify_rule_3(srt_entries, speech_segments, silence_segments,
                            args.min_silence)
    rule_4 = verify_rule_4_summary(rule_1, rule_3)

    # Step 4: Build report
    report = {
        'audio_file': os.path.basename(args.audio_file),
        'srt_file': os.path.basename(args.srt_file),
        'audio_duration_seconds': total_duration,
        'srt_entry_count': len(srt_entries),
        'vad': {
            'speech_segments': [asdict(s) for s in speech_segments],
            'silence_segments': [asdict(s) for s in silence_segments],
            'total_speech_seconds': round(
                sum(s.duration for s in speech_segments), 3
            ),
            'total_silence_seconds': round(
                sum(s.duration for s in silence_segments), 3
            ),
        },
        'verification': {
            'rule_1': rule_1,
            'rule_3': rule_3,
            'rule_4_summary': rule_4,
        },
    }

    # Step 5: Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\nReport saved to: {output_path}', file=sys.stderr)
    print(f'\n=== VERIFICATION RESULT: {rule_4["summary"]} ===', file=sys.stderr)
    print(f'Rule 1 (start time accuracy): {rule_1["pass_rate"]}% '
          f'({rule_1["passed"]}/{rule_1["total"]})', file=sys.stderr)
    print(f'Rule 3 (silence gaps): {rule_3["passed"]} passed, '
          f'{rule_3.get("info", 0)} info, '
          f'{rule_3["warnings"]} warnings, {rule_3["failed"]} failed',
          file=sys.stderr)

    # Exit code: 0 = all pass, 1 = issues found
    sys.exit(0 if rule_4['overall_pass'] else 1)


if __name__ == '__main__':
    main()
