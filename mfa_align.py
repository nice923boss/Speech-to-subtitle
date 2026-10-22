"""MFA (Montreal Forced Aligner) integration for character-level alignment.

Takes audio + optimized SRT, runs MFA forced alignment via conda subprocess,
parses TextGrid output, and produces SRT with precise timestamps.

MFA workflow:
1. Split audio into segments per SRT entry (ffmpeg)
2. Write transcript text files for each segment
3. Run `mfa align` via conda env subprocess
4. Parse TextGrid output for word-level timestamps
5. Build final SRT with MFA's precise timing
"""

import os
import sys
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8')

# ── Constants ──────────────────────────────────────────────────────────

MINIFORGE_DIR = os.path.join(str(Path.home()), "miniforge3")
MAMBA_BAT = os.path.join(MINIFORGE_DIR, "condabin", "mamba.bat")
MFA_ENV_NAME = "mfa"

# Supported MFA dictionaries (preference order for Chinese)
MFA_DICTIONARIES = [
    "mandarin_taiwan_mfa",
    "mandarin_china_mfa",
]
MFA_ACOUSTIC_MODEL = "mandarin_mfa"

# Audio segment padding (seconds)
SEGMENT_PAD_BEFORE = 0.15
SEGMENT_PAD_AFTER = 0.15

# Label MFA gives a word its dictionary does not contain.
UNK_LABEL = "<unk>"


@dataclass
class AlignedWord:
    """A single word/character with MFA-aligned timestamps."""
    text: str
    start_s: float
    end_s: float


@dataclass
class AlignedEntry:
    """An SRT entry with MFA-aligned timestamps."""
    index: int
    start_ms: int
    end_ms: int
    text: str
    # Word-level absolute times taken straight off the TextGrid, captured before
    # any gap rule touches start_ms/end_ms. This is what feeds the .words.json
    # sidecar. None means MFA produced no alignment for this entry.
    words: list[AlignedWord] | None = None


# ── MFA environment check ─────────────────────────────────────────────

# A cold `mamba run` took longer than the old 30 s limit (spike 0: 31.1 s, then
# 5.9 to 7.8 s warm), so the first run of the day silently lost MFA.
MFA_CHECK_TIMEOUT = 120

# One check per process. Callers are one-file-per-process workers, and the
# check used to run twice per file (mfa_refine_cues, then mfa_align).
_MFA_CHECK: tuple[bool, str] | None = None


def _run_tree(cmd: list[str], timeout: float, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run that kills the whole process tree on timeout.

    mamba.bat runs under cmd.exe and spawns mfa.exe plus its python workers.
    subprocess.run's timeout kills only the direct child, which left those
    running after the caller had given up (spike 0 counted 7 descendants).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def check_mfa_available() -> tuple[bool, str]:
    """Check if MFA conda environment is available and working.

    Returns (available, message). The result is cached for the process.
    """
    global _MFA_CHECK
    if _MFA_CHECK is None:
        _MFA_CHECK = _check_mfa()
    return _MFA_CHECK


def _check_mfa() -> tuple[bool, str]:
    if not os.path.isfile(MAMBA_BAT):
        return False, f"Miniforge not found at {MINIFORGE_DIR}"

    try:
        result = _run_tree(
            [MAMBA_BAT, "run", "-n", MFA_ENV_NAME, "mfa", "version"],
            timeout=MFA_CHECK_TIMEOUT, text=True,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"MFA {version}"
        return False, f"MFA env error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"MFA version check timed out (>{MFA_CHECK_TIMEOUT} s)"
    except Exception as e:
        return False, f"MFA check failed: {e}"


def _find_ffmpeg() -> str | None:
    """Find ffmpeg executable."""
    # Try scanner module first
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from scanner import find_ffmpeg
        return find_ffmpeg()
    except ImportError:
        pass
    # Fallback: check PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path_dir, "ffmpeg.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


# ── Audio segmentation ─────────────────────────────────────────────────

# A 30-minute AAC track decodes in under a second; this only stops a hung ffmpeg.
DECODE_TIMEOUT = 900


def _decode_audio(audio_path: str, ffmpeg_path: str, output_path: str) -> None:
    """Decode the whole input once to 16kHz mono WAV; segments are sliced from it.

    One ffmpeg run per SRT entry cost about 40 ms each: 600 cues on a
    30-minute file took 25 s, against under 1 s for one decode plus slicing.
    """
    result = subprocess.run(
        [ffmpeg_path, "-y", "-i", audio_path, "-vn",
         "-ar", "16000", "-ac", "1",
         "-acodec", "pcm_s16le",
         output_path],
        capture_output=True, timeout=DECODE_TIMEOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0 or not os.path.isfile(output_path):
        tail = result.stderr.decode('utf-8', 'replace').strip()[-300:]
        raise RuntimeError(f"ffmpeg could not decode audio: {tail}")


def _write_segment(source: wave.Wave_read, start_s: float, end_s: float,
                   output_path: str) -> bool:
    """Copy [start_s, end_s) of the decoded WAV into its own file for MFA."""
    rate = source.getframerate()
    first = int(start_s * rate)
    count = int(end_s * rate) - first
    if count <= 0:
        return False
    source.setpos(first)
    with wave.open(output_path, 'wb') as out:
        out.setparams(source.getparams())
        out.writeframes(source.readframes(count))
    return True


# ── Transcript preparation ─────────────────────────────────────────────

def _clean_text_for_mfa(text: str) -> str:
    """Clean SRT text for MFA alignment.

    MFA expects plain text without SRT formatting.
    Remove HTML tags, special chars, keep Chinese + basic punctuation.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove SRT line breaks within an entry
    text = text.replace('\n', ' ')
    # Remove characters MFA can't handle (keep Chinese, alphanumeric, spaces)
    # MFA dictionary has Chinese characters and basic punctuation
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaffa-zA-Z0-9\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── MFA execution ──────────────────────────────────────────────────────

def _run_mfa_align(corpus_dir: str, output_dir: str,
                   dictionary: str = None,
                   on_progress=None) -> bool:
    """Run MFA align command via conda subprocess.

    Returns True if alignment succeeded.
    """
    if dictionary is None:
        dictionary = MFA_DICTIONARIES[0]

    cmd = [
        MAMBA_BAT, "run", "-n", MFA_ENV_NAME,
        "mfa", "align",
        corpus_dir,
        dictionary,
        MFA_ACOUSTIC_MODEL,
        output_dir,
        "--clean",
        "--single_speaker",
        "--use_mp",
    ]

    if on_progress:
        on_progress("Running MFA alignment...")

    try:
        result = _run_tree(
            cmd,
            timeout=600,  # 10 min max
            text=True, encoding='utf-8', errors='replace',
        )

        if result.returncode == 0:
            if on_progress:
                on_progress("MFA alignment completed")
            return True

        stderr = result.stderr.strip()
        if on_progress:
            on_progress(f"MFA error: {stderr[:200]}")

        # Try fallback dictionary if primary fails
        if dictionary == MFA_DICTIONARIES[0] and len(MFA_DICTIONARIES) > 1:
            if on_progress:
                on_progress(f"Retrying with dictionary: {MFA_DICTIONARIES[1]}")
            return _run_mfa_align(corpus_dir, output_dir,
                                  dictionary=MFA_DICTIONARIES[1],
                                  on_progress=on_progress)
        return False

    except subprocess.TimeoutExpired:
        if on_progress:
            on_progress("MFA alignment timed out (>10 min)")
        return False
    except Exception as e:
        if on_progress:
            on_progress(f"MFA execution failed: {e}")
        return False


# ── TextGrid parsing ───────────────────────────────────────────────────

def _parse_textgrid(tg_path: str) -> list[AlignedWord]:
    """Parse MFA TextGrid output to extract word-level timestamps."""
    from praatio import textgrid

    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    words = []

    # MFA outputs tiers: "words" and "phones"
    # We want the "words" tier
    for tier_name in tg.tierNames:
        if "word" in tier_name.lower():
            tier = tg.getTier(tier_name)
            for interval in tier.entries:
                text = interval.label.strip()
                if text and text != "":  # Skip empty/silence intervals
                    words.append(AlignedWord(
                        text=text,
                        start_s=interval.start,
                        end_s=interval.end,
                    ))
            break

    return words


def _relabel_words_to_source(words: list[AlignedWord],
                             source_text: str) -> list[AlignedWord]:
    """Put the source transcript's own characters back onto MFA's word times.

    A TextGrid label is not the source text: the mandarin dictionary normalises
    a few glyphs (了→瞭, 裡→裏, 面→麵), any out-of-vocabulary word comes back as
    "<unk>", and a word MFA cannot place at all is simply absent. A sidecar
    built from those labels would stop matching the SRT text Quick subtitles
    aligns against, character for character, and its greedy matcher would drift.

    So MFA's segmentation and times are kept and the source characters are
    re-cut onto them: the label stream is diffed against the source, each word
    boundary is carried across (exactly on matching runs, linearly inside the
    stretches that differ), and each word takes the source characters between
    its boundaries. Words left with no characters are dropped. The result always
    concatenates back to the source text exactly.
    """
    src = source_text.replace(" ", "")
    if not words or not src:
        return words

    # One placeholder char per <unk> so it absorbs whatever the source has at
    # that position instead of matching the literal letters u-n-k.
    labels = ["�" if w.text == UNK_LABEL else w.text for w in words]
    stream = "".join(labels)

    # stream index -> src index, non-decreasing.
    pos = [0] * (len(stream) + 1)
    for _tag, i1, i2, j1, j2 in SequenceMatcher(None, stream, src,
                                                autojunk=False).get_opcodes():
        span = i2 - i1
        for k in range(i1, i2):
            pos[k] = j1 + (round((k - i1) * (j2 - j1) / span) if span else 0)
    pos[0] = 0
    pos[len(stream)] = len(src)
    for k in range(len(stream) - 1, -1, -1):
        pos[k] = min(pos[k], pos[k + 1])

    relabelled = []
    offset = 0
    for w, label in zip(words, labels):
        start, end = pos[offset], pos[offset + len(label)]
        offset += len(label)
        if end > start:
            relabelled.append(AlignedWord(text=src[start:end],
                                          start_s=w.start_s, end_s=w.end_s))
    return relabelled or words


# ── Core alignment function ────────────────────────────────────────────

def mfa_align(audio_path: str, srt_entries: list,
              ffmpeg_path: str = None,
              on_progress=None) -> list[AlignedEntry]:
    """Run MFA forced alignment on audio + SRT entries.

    Args:
        audio_path: Path to audio/video file
        srt_entries: List of SrtEntry objects (with text and approximate timing)
        ffmpeg_path: Path to ffmpeg executable
        on_progress: Optional callback function(message: str)

    Returns:
        List of AlignedEntry with MFA-precise timestamps
    """
    if ffmpeg_path is None:
        ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found - required for audio segmentation")

    available, msg = check_mfa_available()
    if not available:
        raise RuntimeError(f"MFA not available: {msg}")

    if on_progress:
        on_progress(f"MFA: {msg}")

    # Normalize: accept dicts (from srt_common.parse_srt) or objects with attributes
    normalized = []
    for entry in srt_entries:
        if isinstance(entry, dict):
            normalized.append(AlignedEntry(
                index=entry.get('idx', entry.get('index', 0)),
                start_ms=entry['start_ms'],
                end_ms=entry['end_ms'],
                text=entry['text'],
            ))
        else:
            normalized.append(entry)
    srt_entries = normalized

    # Create temporary working directories
    work_dir = tempfile.mkdtemp(prefix="mfa_align_")
    corpus_dir = os.path.join(work_dir, "corpus", "speaker")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Split audio and prepare corpus
        if on_progress:
            on_progress(f"Splitting audio into {len(srt_entries)} segments...")

        # Outside corpus_dir, or MFA would take it for an utterance.
        source_path = os.path.join(work_dir, "source.wav")
        _decode_audio(audio_path, ffmpeg_path, source_path)

        valid_segments = []
        with wave.open(source_path, 'rb') as source:
            audio_duration = source.getnframes() / source.getframerate()
            if audio_duration <= 0:
                raise RuntimeError(f"Decoded audio is empty: {audio_path}")

            for i, entry in enumerate(srt_entries):
                seg_name = f"{i:04d}"
                wav_path = os.path.join(corpus_dir, f"{seg_name}.wav")
                txt_path = os.path.join(corpus_dir, f"{seg_name}.lab")

                # Calculate segment boundaries with padding
                start_s = max(0, entry.start_ms / 1000.0 - SEGMENT_PAD_BEFORE)
                end_s = min(audio_duration, entry.end_ms / 1000.0 + SEGMENT_PAD_AFTER)

                if not _write_segment(source, start_s, end_s, wav_path):
                    if on_progress:
                        on_progress(f"  Warning: Failed to extract segment {i+1}")
                    continue

                # Write transcript
                clean_text = _clean_text_for_mfa(entry.text)
                if not clean_text:
                    os.remove(wav_path)
                    continue

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)

                valid_segments.append({
                    'index': i,
                    'seg_name': seg_name,
                    'entry': entry,
                    'pad_before': entry.start_ms / 1000.0 - start_s,
                    'abs_start_s': start_s,
                    'clean_text': clean_text,
                })

        if not valid_segments:
            raise RuntimeError("No valid segments could be prepared for MFA")

        if on_progress:
            on_progress(f"Prepared {len(valid_segments)}/{len(srt_entries)} segments")

        # Step 2: Run MFA alignment
        success = _run_mfa_align(
            os.path.dirname(corpus_dir),  # corpus parent (contains speaker/)
            output_dir,
            on_progress=on_progress,
        )

        if not success:
            raise RuntimeError("MFA alignment failed")

        # Step 3: Parse TextGrid outputs and build aligned entries
        if on_progress:
            on_progress("Parsing MFA alignment results...")

        aligned_entries = []
        mfa_aligned_count = 0

        for seg_info in valid_segments:
            seg_name = seg_info['seg_name']
            entry = seg_info['entry']
            pad_before = seg_info['pad_before']
            abs_start = seg_info['abs_start_s']

            # Look for TextGrid output
            tg_path = os.path.join(output_dir, "speaker", f"{seg_name}.TextGrid")
            if not os.path.isfile(tg_path):
                # MFA didn't align this segment - keep original timing
                aligned_entries.append(AlignedEntry(
                    index=entry.index,
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    text=entry.text,
                ))
                continue

            # Parse TextGrid
            words = _parse_textgrid(tg_path)
            if not words:
                aligned_entries.append(AlignedEntry(
                    index=entry.index,
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    text=entry.text,
                ))
                continue

            # Convert word timestamps from segment-relative to absolute
            # Segment was extracted starting at abs_start
            abs_words = [
                AlignedWord(text=w.text,
                            start_s=abs_start + w.start_s,
                            end_s=abs_start + w.end_s)
                for w in _relabel_words_to_source(words, seg_info['clean_text'])
            ]

            aligned_entries.append(AlignedEntry(
                index=entry.index,
                start_ms=int(abs_words[0].start_s * 1000),
                end_ms=int(abs_words[-1].end_s * 1000),
                text=entry.text,  # Keep original text (already optimized)
                words=abs_words,
            ))
            mfa_aligned_count += 1

        # Fill in any entries that weren't in valid_segments
        aligned_indices = {seg['index'] for seg in valid_segments}
        for i, entry in enumerate(srt_entries):
            if i not in aligned_indices:
                aligned_entries.append(AlignedEntry(
                    index=entry.index,
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    text=entry.text,
                ))

        # Sort by start time
        aligned_entries.sort(key=lambda e: e.start_ms)

        # Step 4: Fix overlaps only. Gap filling belongs to the last station
        # (Quick subtitles' gap mode); doing it here as well inflated every cue
        # end and pushed that inflation downstream. Word times were captured
        # above, so they stay clean regardless.
        from sentence_align import apply_gap_rules
        apply_gap_rules(aligned_entries, fill_gaps=False)

        if on_progress:
            on_progress(
                f"MFA aligned {mfa_aligned_count}/{len(srt_entries)} entries, "
                f"{len(srt_entries) - mfa_aligned_count} kept original timing"
            )

        return aligned_entries

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# ── Cue-level refinement wrapper (used by transcriber.py) ──────────────

def mfa_refine_cues(audio_path, cues, ffmpeg_path=None, on_progress=None):
    """Refine cue timings with MFA forced alignment.

    Args:
        cues: list of {"start": sec, "end": sec, "text": str}.

    Returns:
        (refined_cues, words) where refined_cues preserves each cue's text with
        MFA-precise start/end (seconds), and words is a word-level sidecar
        [{"word": str, "start": sec, "end": sec}] carrying MFA's per-word times
        (pre-gap-rule, so no inflated ends reach Quick subtitles). Returns None
        if MFA is unavailable or alignment fails, so the caller can fall back;
        the reason goes to on_progress first, so the fallback is never silent.
    """
    if not cues:
        return None
    available, msg = check_mfa_available()
    if not available:
        if on_progress:
            on_progress(f"MFA 無法使用，改用能量對齊：{msg}")
        return None

    entries = [{"idx": i,
                "start_ms": int(c["start"] * 1000),
                "end_ms": int(c["end"] * 1000),
                "text": c["text"]}
               for i, c in enumerate(cues)]
    try:
        aligned = mfa_align(audio_path, entries, ffmpeg_path=ffmpeg_path,
                            on_progress=on_progress)
    except Exception as e:
        if on_progress:
            on_progress(f"MFA 對齊失敗，改用能量對齊：{type(e).__name__}: {e}")
        return None

    by_idx = {e.index: e for e in aligned}
    refined = []
    words = []
    for i, c in enumerate(cues):
        e = by_idx.get(i)
        if e is not None and e.end_ms > e.start_ms:
            refined.append({"start": e.start_ms / 1000.0,
                            "end": e.end_ms / 1000.0,
                            "text": c["text"]})
        else:
            refined.append(dict(c))

        if e is not None and e.words:
            words.extend({"word": w.text, "start": w.start_s, "end": w.end_s}
                         for w in e.words)
        else:
            # MFA aligned nothing for this cue. Emit it as one sidecar entry so
            # the character timeline stays complete — a hole here would shift
            # this cue's text onto the next cue's timing downstream.
            r = refined[-1]
            words.append({"word": r["text"], "start": r["start"],
                          "end": r["end"]})
    return refined, words


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    """CLI for standalone MFA alignment testing."""
    import argparse

    # Import SRT utilities from the local srt_common module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from srt_common import parse_srt_file, write_srt, SrtEntry

    parser = argparse.ArgumentParser(
        description='MFA forced alignment for subtitle timing'
    )
    parser.add_argument('audio_file', help='Audio/video file')
    parser.add_argument('optimized_srt', help='Optimized SRT file')
    parser.add_argument('--output', '-o', required=True, help='Output SRT path')

    args = parser.parse_args()

    for path in [args.audio_file, args.optimized_srt]:
        if not os.path.isfile(path):
            print(f'Error: File not found: {path}', file=sys.stderr)
            sys.exit(1)

    # Parse optimized SRT
    entries = parse_srt_file(args.optimized_srt)
    print(f'Loaded {len(entries)} SRT entries', file=sys.stderr)

    # Run MFA alignment
    def progress(msg):
        print(f'  {msg}', file=sys.stderr)

    aligned = mfa_align(args.audio_file, entries, on_progress=progress)

    # Write output
    srt_entries = [SrtEntry(i+1, e.start_ms, e.end_ms, e.text)
                   for i, e in enumerate(aligned)]
    write_srt(args.output, srt_entries)
    print(f'\nOutput: {args.output}', file=sys.stderr)
    print(f'Aligned entries: {len(srt_entries)}', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
