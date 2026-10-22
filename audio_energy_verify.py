"""
Audio Energy Verification for Subtitle Pipeline
Uses librosa RMS energy to detect Whisper hallucinations and phantom subtitles.

Usage:
    python audio_energy_verify.py <audio_file> <srt_file> [--output suspects.json] [--speech-rate 3.5] [--rms-threshold 0.005]
"""

import argparse
import json
import os
import re
import sys

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure this script's directory is on sys.path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from srt_common import parse_srt_file  # noqa: E402

import librosa
import numpy as np


# ── Character Counting ──────────────────────────────────────────────

CJK_RANGE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')


def count_cjk_chars(text):
    return len(CJK_RANGE.findall(text))


def count_effective_chars(text):
    """Count characters that represent spoken content (CJK + English words)."""
    cjk = count_cjk_chars(text)
    # English words count as ~2 CJK chars equivalent (avg syllables)
    en_words = len(re.findall(r'[A-Za-z]+', text))
    return cjk + en_words * 2


# ── RMS Energy Extraction ──────────────────────────────────────────

def extract_segment_rms(audio, sr, start_ms, end_ms):
    start_sample = int(start_ms / 1000.0 * sr)
    end_sample = int(end_ms / 1000.0 * sr)

    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))

    if end_sample <= start_sample:
        return 0.0

    segment = audio[start_sample:end_sample]
    rms = np.sqrt(np.mean(segment ** 2))
    return float(rms)


# ── Suspect Detection ──────────────────────────────────────────────

def analyze_entries(entries, audio, sr, speech_rate, rms_threshold):
    results = []

    for entry in entries:
        duration_s = (entry['end_ms'] - entry['start_ms']) / 1000.0
        char_count = count_effective_chars(entry['text'])
        mean_rms = extract_segment_rms(audio, sr, entry['start_ms'], entry['end_ms'])

        expected_chars = duration_s * speech_rate
        char_density_ratio = char_count / expected_chars if expected_chars > 0 else 999.0

        suspect_reasons = []
        _CONF_RANK = {'high': 3, 'medium': 2, 'low': 1}
        confidence = None

        # Rule 1: Too many chars for duration (hallucination)
        if char_density_ratio > 2.0:
            suspect_reasons.append('hallucination_char_density')
            c = 'high' if char_density_ratio > 3.0 else 'medium'
            if confidence is None or _CONF_RANK[c] > _CONF_RANK[confidence]:
                confidence = c

        # Rule 2: Near-silence with text (phantom subtitle)
        if mean_rms < rms_threshold and char_count > 0:
            suspect_reasons.append('phantom_silence')
            c = 'high'
            if confidence is None or _CONF_RANK[c] > _CONF_RANK[confidence]:
                confidence = c

        # Rule 3: Impossibly fast speech
        if duration_s < 0.5 and char_count > 8:
            suspect_reasons.append('impossible_speed')
            c = 'high' if duration_s < 0.3 else 'medium'
            if confidence is None or _CONF_RANK[c] > _CONF_RANK[confidence]:
                confidence = c

        # Rule 4: Very sparse text for long duration (possible missing text)
        if char_density_ratio < 0.3 and duration_s > 1.0 and char_count > 0:
            suspect_reasons.append('sparse_text')
            c = 'low'
            if confidence is None or _CONF_RANK[c] > _CONF_RANK[confidence]:
                confidence = c

        entry_data = {
            'entry_index': entry['idx'],
            'text': entry['text'],
            'start_ms': entry['start_ms'],
            'end_ms': entry['end_ms'],
            'duration_s': round(duration_s, 3),
            'char_count': char_count,
            'expected_chars': round(expected_chars, 1),
            'char_density_ratio': round(char_density_ratio, 3),
            'mean_rms': round(mean_rms, 6),
        }

        if suspect_reasons:
            entry_data['suspect'] = True
            entry_data['reasons'] = suspect_reasons
            entry_data['confidence'] = confidence
        else:
            entry_data['suspect'] = False

        results.append(entry_data)

    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Audio energy verification for subtitle quality')
    parser.add_argument('audio_file', help='Path to audio/video file')
    parser.add_argument('srt_file', help='Path to SRT subtitle file')
    parser.add_argument('--output', '-o', default=None, help='Output suspects JSON path')
    parser.add_argument('--speech-rate', type=float, default=3.5,
                        help='Expected CJK characters per second (default: 3.5)')
    parser.add_argument('--rms-threshold', type=float, default=0.005,
                        help='RMS threshold below which audio is considered silence (default: 0.005)')
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    if not os.path.exists(args.srt_file):
        print(f"Error: SRT file not found: {args.srt_file}")
        sys.exit(1)

    # Default output path (based on audio file name, consistent with pipeline expectation)
    if args.output is None:
        base = os.path.splitext(args.audio_file)[0]
        args.output = f"{base}_suspects.json"

    print(f"Loading audio: {args.audio_file}")
    audio, sr = librosa.load(args.audio_file, sr=16000, mono=True)
    audio_duration_s = len(audio) / sr
    print(f"Audio loaded: {audio_duration_s:.1f}s, sample rate: {sr}")

    print(f"Parsing SRT: {args.srt_file}")
    entries = parse_srt_file(args.srt_file)
    print(f"Parsed {len(entries)} subtitle entries")

    print("Analyzing audio energy per segment...")
    all_entries = analyze_entries(entries, audio, sr, args.speech_rate, args.rms_threshold)

    suspects = [e for e in all_entries if e.get('suspect')]
    rms_values = [e['mean_rms'] for e in all_entries]

    output = {
        'version': '1.0.0',
        'audio_file': os.path.basename(args.audio_file),
        'srt_file': os.path.basename(args.srt_file),
        'parameters': {
            'speech_rate': args.speech_rate,
            'rms_threshold': args.rms_threshold,
        },
        'global_stats': {
            'total_entries': len(all_entries),
            'suspect_count': len(suspects),
            'mean_rms': round(float(np.mean(rms_values)), 6) if rms_values else 0,
            'median_rms': round(float(np.median(rms_values)), 6) if rms_values else 0,
            'audio_duration_s': round(audio_duration_s, 1),
        },
        'suspects': suspects,
        'all_entries': all_entries,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to: {args.output}")
    print(f"Total entries: {len(all_entries)}")
    print(f"Suspects found: {len(suspects)}")

    if suspects:
        print("\n--- Suspect Summary ---")
        for s in suspects:
            reasons = ', '.join(s['reasons'])
            print(f"  #{s['entry_index']} [{s['confidence']}] {reasons}: "
                  f"\"{s['text'][:30]}...\" "
                  f"(duration={s['duration_s']}s, chars={s['char_count']}, "
                  f"ratio={s['char_density_ratio']}, rms={s['mean_rms']})")


if __name__ == '__main__':
    main()
