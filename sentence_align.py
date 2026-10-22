"""Sentence-level subtitle alignment using MFA (Montreal Forced Aligner).

Usage:
    python sentence_align.py <audio_file> <optimized_srt> --method mfa --output <output.srt>

Inputs:
    audio_file     - Original audio/video file
    optimized_srt  - Step 4 output (AI-optimized text)

Output:
    Precise timestamps + optimized text = final SRT
"""

import os
import sys
import argparse

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding='utf-8')

import srt_common  # noqa: E402 — shared SRT parser


def _get(entry, key):
    """Get value from entry (dict or object with attributes)."""
    if isinstance(entry, dict):
        return entry[key]
    return getattr(entry, key)


def _set(entry, key, value):
    """Set value on entry (dict or object with attributes)."""
    if isinstance(entry, dict):
        entry[key] = value
    else:
        setattr(entry, key, value)


def apply_gap_rules(entries, gap_threshold_ms: int = 2000,
                    fill_gaps: bool = False) -> None:
    """Fix overlaps between adjacent entries (in-place), optionally close gaps.

    - Overlap (gap < 0): adjust current end to next start - 1
    - Small gap (0 < gap < threshold): extend current end to next start,
      only when `fill_gaps` is True.

    Gap filling defaults off: the line fills gaps once, at the last station
    (Quick subtitles' gap mode). Filling here as well inflated every cue end and
    carried that inflation downstream into the words sidecar.
    Supports both dict entries and dataclass/object entries.
    """
    for i in range(len(entries) - 1):
        curr = entries[i]
        nxt = entries[i + 1]
        gap_ms = _get(nxt, 'start_ms') - _get(curr, 'end_ms')

        if gap_ms < -10:
            _set(curr, 'end_ms', _get(nxt, 'start_ms') - 1)
        elif fill_gaps and 0 < gap_ms < gap_threshold_ms:
            _set(curr, 'end_ms', _get(nxt, 'start_ms'))


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description='Sentence-level subtitle alignment (MFA)'
    )
    parser.add_argument('audio_file', help='Original audio/video file')
    parser.add_argument('srt_files', nargs='+',
                        help='SRT file(s): optimized_srt')
    parser.add_argument('--output', '-o', required=True,
                        help='Output SRT path')
    parser.add_argument('--method', '-m', choices=['mfa'],
                        default='mfa',
                        help='Alignment method (mfa only)')

    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f'Error: File not found: {args.audio_file}', file=sys.stderr)
        sys.exit(1)

    optimized_srt = args.srt_files[-1]
    if not os.path.isfile(optimized_srt):
        print(f'Error: File not found: {optimized_srt}', file=sys.stderr)
        sys.exit(1)

    print('Parsing SRT files...', file=sys.stderr)
    opt_entries = srt_common.parse_srt_file(optimized_srt)

    print(f'Optimized: {len(opt_entries)} entries', file=sys.stderr)
    print('Using MFA forced alignment...', file=sys.stderr)
    from mfa_align import mfa_align

    def progress(msg):
        print(f'  {msg}', file=sys.stderr)

    aligned = mfa_align(args.audio_file, opt_entries, on_progress=progress)
    final = [{'idx': i + 1, 'start_ms': e.start_ms,
              'end_ms': e.end_ms, 'text': e.text}
             for i, e in enumerate(aligned)]

    srt_common.write_srt(args.output, final)
    print(f'\nOutput: {args.output}', file=sys.stderr)
    print(f'Final entries: {len(final)}', file=sys.stderr)

    improved = 0
    kept = 0
    for i, (opt, fin) in enumerate(zip(opt_entries, final)):
        if i >= len(final):
            break
        diff = abs(opt['start_ms'] - fin['start_ms'])
        if diff > 50:
            improved += 1
        else:
            kept += 1
    print(f'Timing: {improved} entries improved, {kept} entries unchanged',
          file=sys.stderr)

    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
