"""Measure how close each subtitle's start is to the real speech onset.

Extends the RMS-energy analysis of audio_energy_verify.py. For every SRT entry it
locates the nearest energy onset (silence -> speech transition) and reports the
signed error `srt_start - speech_onset` (positive = subtitle starts late,
negative = subtitle appears before the speaker starts).

Only entries that begin right after a pause have a physically measurable onset.
Entries that start in the middle of a continuous utterance (a line split without
a pause) are counted separately and excluded from the error statistics, because
energy cannot resolve a word boundary inside speech.

An independent second detector (spectral-flux onset detection, which reacts to
spectral change rather than to loudness) runs on the same audio so the instrument
itself can be cross-checked instead of trusted.

Usage:
    python start_accuracy_verify.py <audio_or_video> <srt> [--output report.json]
                                    [--label before] [--plot-dir DIR] [--plot-n 10]
"""

import argparse
import json
import os
import sys

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from srt_common import parse_srt_file  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402


SR = 16000
HOP = 160          # 10 ms frames
FRAME = 400        # 25 ms window


# ── Speech segmentation (detector A: percentile RMS threshold) ─────────

def detect_segments_rms(audio, rel_thr=0.30,
                        min_silence_s=0.30, min_speech_s=0.10):
    """Return [(start_s, end_s)] speech segments from an RMS dB threshold."""
    rms = librosa.feature.rms(y=audio, frame_length=FRAME, hop_length=HOP)[0]
    db = 20.0 * np.log10(rms + 1e-10)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    thr = floor + rel_thr * (peak - floor)

    mask = db > thr
    frame_s = HOP / SR
    mask = _close_gaps(mask, int(round(min_silence_s / frame_s)))
    mask = _drop_short(mask, int(round(min_speech_s / frame_s)))
    return _mask_to_segments(mask, frame_s), {
        'floor_db': round(floor, 2),
        'peak_db': round(peak, 2),
        'threshold_db': round(thr, 2),
    }


def _close_gaps(mask, max_gap_frames):
    """Fill False runs shorter than max_gap_frames (merges within-word pauses)."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) < max_gap_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _drop_short(mask, min_frames):
    """Remove True runs shorter than min_frames (kills clicks and noise spikes)."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_frames:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _mask_to_segments(mask, frame_s):
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i * frame_s, j * frame_s))
            i = j
        else:
            i += 1
    return segs


# ── Onset detection (detector B: spectral flux) ───────────────────────

def detect_onsets_flux(audio):
    """Independent detector: spectral-change onsets, no loudness threshold.

    Used only to cross-check where detector A places a speech start; it reacts
    to timbre change (a vowel appearing) instead of to level, so agreeing with
    it is evidence the energy threshold is not systematically off.
    """
    frames = librosa.onset.onset_detect(y=audio, sr=SR, hop_length=HOP,
                                        backtrack=True, units='frames')
    return librosa.frames_to_time(frames, sr=SR, hop_length=HOP).tolist()


def nearest_distance(reference_times, candidates, limit_s=0.5):
    """Distance from each reference time to the closest candidate within limit."""
    if not candidates:
        return []
    arr = np.asarray(candidates)
    out = []
    for t in reference_times:
        d = float(np.min(np.abs(arr - t)))
        if d <= limit_s:
            out.append(d)
    return out


# ── Per-entry onset matching ──────────────────────────────────────────

def _segment_containing(segments, t):
    for s, e in segments:
        if s <= t <= e:
            return (s, e)
        if s > t:
            break
    return None


def match_onsets(entries, segments, search_s=2.0, mid_speech_s=0.25):
    """Attach the nearest onset (segment start) to each SRT entry."""
    starts = [s for s, _ in segments]
    rows = []
    for entry in entries:
        t = entry['start_ms'] / 1000.0
        best, best_d = None, None
        for s in starts:
            d = abs(s - t)
            if best_d is None or d < best_d:
                best, best_d = s, d
            elif s > t and d > best_d:
                break

        containing = _segment_containing(segments, t)
        row = {
            'idx': entry['idx'],
            'text': entry['text'][:40],
            'srt_start_s': round(t, 3),
            'srt_end_s': round(entry['end_ms'] / 1000.0, 3),
        }

        if containing is not None and (t - containing[0]) > mid_speech_s:
            # Cue starts well inside an ongoing utterance: a line split without a
            # pause. Energy cannot locate that boundary, so it is not measurable.
            row['class'] = 'mid_speech'
            row['onset_s'] = round(containing[0], 3)
            row['error_s'] = round(t - containing[0], 3)
        elif best is None or best_d > search_s:
            row['class'] = 'no_onset_in_window'
            row['onset_s'] = None
            row['error_s'] = None
        else:
            row['class'] = 'onset'
            row['onset_s'] = round(best, 3)
            row['error_s'] = round(t - best, 3)

        rows.append(row)
    return rows


def error_stats(rows, cls='onset'):
    errs = [r['error_s'] for r in rows if r['class'] == cls and r['error_s'] is not None]
    if not errs:
        return {'count': 0}
    a = np.abs(np.array(errs))
    e = np.array(errs)
    return {
        'count': len(errs),
        'abs_median_s': round(float(np.median(a)), 3),
        'abs_mean_s': round(float(np.mean(a)), 3),
        'abs_p90_s': round(float(np.percentile(a, 90)), 3),
        'abs_max_s': round(float(np.max(a)), 3),
        'signed_median_s': round(float(np.median(e)), 3),
        'early_count': int(np.sum(e < -0.05)),   # subtitle before speech
        'late_count': int(np.sum(e > 0.05)),     # subtitle after speech
        'within_1_frame': int(np.sum(a <= 0.034)),
        'within_100ms': int(np.sum(a <= 0.100)),
        'within_200ms': int(np.sum(a <= 0.200)),
    }


# ── Manual-check plots ────────────────────────────────────────────────

def write_plots(audio, rows, out_dir, n, span=1.5):
    """Waveform strips around the N worst measurable cues for eyeball checking."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Titles carry the cue text, so the plots need a CJK-capable face.
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    os.makedirs(out_dir, exist_ok=True)
    measurable = [r for r in rows if r['class'] == 'onset']
    measurable.sort(key=lambda r: -abs(r['error_s']))
    picked = measurable[:n]

    written = []
    for r in picked:
        t = r['srt_start_s']
        a = max(0.0, t - span)
        b = min(len(audio) / SR, t + span)
        seg = audio[int(a * SR):int(b * SR)]
        times = np.linspace(a, b, len(seg))

        fig, ax = plt.subplots(figsize=(10, 2.6), dpi=110)
        ax.plot(times, seg, linewidth=0.5, color='#3a7bd5')
        ax.axvline(t, color='#e74c3c', linewidth=1.6,
                   label=f'SRT start {t:.3f}s')
        if r['onset_s'] is not None:
            ax.axvline(r['onset_s'], color='#27ae60', linewidth=1.6,
                       linestyle='--', label=f"energy onset {r['onset_s']:.3f}s")
        ax.set_xlim(a, b)
        ax.set_yticks([])
        ax.set_xlabel('seconds')
        ax.set_title(f"#{r['idx']}  error {r['error_s']:+.3f}s  {r['text']}",
                     fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        fig.tight_layout()
        path = os.path.join(out_dir, f"cue_{r['idx']:04d}.png")
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


# ── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Subtitle start-time accuracy against energy onsets')
    p.add_argument('audio_file')
    p.add_argument('srt_file')
    p.add_argument('--output', '-o', default=None)
    p.add_argument('--label', default='run')
    p.add_argument('--rel-thr', type=float, default=0.30)
    p.add_argument('--search', type=float, default=2.0,
                   help='max seconds between SRT start and a candidate onset')
    p.add_argument('--plot-dir', default=None)
    p.add_argument('--plot-n', type=int, default=10)
    args = p.parse_args()

    for path in (args.audio_file, args.srt_file):
        if not os.path.isfile(path):
            print(f'Error: file not found: {path}')
            sys.exit(1)

    if args.output is None:
        args.output = os.path.splitext(args.srt_file)[0] + '_start_accuracy.json'

    print(f'Loading audio: {args.audio_file}')
    audio, _ = librosa.load(args.audio_file, sr=SR, mono=True)
    print(f'Audio: {len(audio) / SR:.1f}s')

    entries = parse_srt_file(args.srt_file)
    print(f'SRT entries: {len(entries)}')

    seg_a, thr_info = detect_segments_rms(audio, rel_thr=args.rel_thr)
    flux = detect_onsets_flux(audio)
    print(f'Detector A (rms threshold): {len(seg_a)} segments  {thr_info}')
    print(f'Detector B (spectral flux): {len(flux)} onsets')

    rows_a = match_onsets(entries, seg_a, search_s=args.search)

    # Cross-check: does a spectral-flux onset sit where detector A starts speech?
    used = sorted({r['onset_s'] for r in rows_a
                   if r['class'] == 'onset' and r['onset_s'] is not None})
    diffs = nearest_distance(used, flux, limit_s=0.5)
    cross = {
        'onsets_checked': len(used),
        'matched_within_500ms': len(diffs),
        'median_abs_diff_s': round(float(np.median(diffs)), 3) if diffs else None,
        'p90_abs_diff_s': round(float(np.percentile(diffs, 90)), 3) if diffs else None,
        'agree_within_100ms': int(sum(1 for d in diffs if d <= 0.1)),
    }

    stats = error_stats(rows_a, 'onset')
    stats_mid = error_stats(rows_a, 'mid_speech')

    # Tail check: does the last cue stop near the last detected speech?
    last_end = entries[-1]['end_ms'] / 1000.0 if entries else 0.0
    speech_end = seg_a[-1][1] if seg_a else 0.0
    tail = {
        'last_cue_end_s': round(last_end, 3),
        'last_speech_end_s': round(speech_end, 3),
        'tail_error_s': round(last_end - speech_end, 3),
    }

    report = {
        'version': '1.0.0',
        'label': args.label,
        'audio_file': os.path.basename(args.audio_file),
        'srt_file': os.path.basename(args.srt_file),
        'parameters': {'rel_thr': args.rel_thr, 'search_s': args.search,
                       **thr_info},
        'counts': {
            'entries': len(entries),
            'measurable_onset': sum(1 for r in rows_a if r['class'] == 'onset'),
            'mid_speech': sum(1 for r in rows_a if r['class'] == 'mid_speech'),
            'no_onset_in_window': sum(1 for r in rows_a
                                      if r['class'] == 'no_onset_in_window'),
        },
        'start_error_onset_cues': stats,
        'start_error_mid_speech_cues': stats_mid,
        'detector_cross_check': cross,
        'tail': tail,
        'rows': rows_a,
    }

    if args.plot_dir:
        report['plots'] = write_plots(audio, rows_a, args.plot_dir, args.plot_n)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\nReport: {args.output}')
    print(f"Counts: {report['counts']}")
    print(f'Start error (onset cues): {stats}')
    print(f'Detector cross-check: {cross}')
    print(f'Tail: {tail}')


if __name__ == '__main__':
    main()
