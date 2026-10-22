"""Energy-based onset refinement for cue/word start times.

Whisper word/segment start times after a pause tend to land slightly inside the
preceding silence, so subtitles appear before the speaker actually starts. This
module snaps any start that falls in a low-energy (silence) region forward to the
real RMS-energy onset, so cues appear when speech begins. Starts already inside
speech are left untouched. Ends are preserved. The pass is best-effort: it
returns the input unchanged if librosa is unavailable or the audio can't load.
"""


def build_energy(audio_path, sr=16000, hop=160, frame=400,
                 floor_pct=10, peak_pct=95, rel_thr=0.15):
    """Load audio and return an RMS-energy context, or None on any failure."""
    try:
        import librosa
        import numpy as np
    except Exception:
        return None
    try:
        audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None
    if audio is None or len(audio) == 0:
        return None
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
    floor = float(np.percentile(rms, floor_pct))
    peak = float(np.percentile(rms, peak_pct))
    thr = floor + rel_thr * (peak - floor)
    return {"rms": rms, "sr": sr, "hop": hop, "thr": thr, "n": len(rms)}


def _rms_at(energy, t):
    f = int(t * energy["sr"] / energy["hop"])
    if 0 <= f < energy["n"]:
        return float(energy["rms"][f])
    return 0.0


def _onset_after(energy, t, cap):
    """First time >= t and < cap where RMS exceeds the threshold, else None."""
    f0 = max(0, int(t * energy["sr"] / energy["hop"]))
    f1 = min(energy["n"], int(cap * energy["sr"] / energy["hop"]) + 1)
    rms = energy["rms"]
    thr = energy["thr"]
    for f in range(f0, f1):
        if rms[f] > thr:
            return f * energy["hop"] / energy["sr"]
    return None


def snap_starts(items, energy, window=1.0, lead=0.0, min_dur=0.05):
    """Return a new list with each item's 'start' snapped forward to the speech
    onset when it begins in silence; 'end' preserved. Items are dicts with
    'start'/'end' in seconds. No-op when energy is None.

    window  - max seconds to search forward for an onset
    lead    - subtract a small lead so the cue appears just before the onset
    min_dur - never snap so close to 'end' that duration drops below this
    """
    if energy is None:
        return items
    out = []
    for it in items:
        new = dict(it)
        st = it["start"]
        en = it["end"]
        if _rms_at(energy, st) < energy["thr"]:
            cap = min(en - min_dur, st + window)
            if cap > st:
                onset = _onset_after(energy, st, cap)
                if onset is not None and onset > st:
                    new["start"] = max(0.0, onset - lead)
        out.append(new)
    return out
