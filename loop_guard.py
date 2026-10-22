"""Detect and repair Whisper repetition loops before anything downstream runs.

Whisper conditions each window on the text it produced for the previous one
(`condition_on_previous_text` defaults to True). Over music or singing it can
lock onto a canned phrase and repeat it for minutes, overwriting the real speech
underneath: 01.wav lost 601s to "中文字幕志願者 李宗盛" that way, in two runs of
17 and 30 consecutive identical cues.

Re-decoding only the affected span breaks the lock, because the span is decoded
against an empty context instead of the poisoned one. Measured on both of
01.wav's loop spans: the 17-cue loop came back as 112 distinct segments, the
30-cue loop as 88, and both recovered real conversation. So the default action
here is to replace the span, not to discard it. Discarding is the fallback for
when the re-decode loops as well.

Signals that did NOT work, so that nobody re-adds them: no_speech_prob and
avg_logprob do not separate music from speech in this material. Re-decoded loop
spans measured mean no_speech_prob 0.809 and 0.736, while ordinary dialogue from
the same file measured 0.701 and 0.620; avg_logprob was worse for one normal
span (-0.391) than for either loop span (-0.381, -0.299). There is no usable
"is this a song" test here. Whether the text repeats is the only reliable one.
"""

import s2tw_gate

# A loop has to be both long enough and repeated enough to be worth re-decoding.
# Real speech does repeat a little: 01.wav opens with "謝謝" three times in 4.0s,
# which is someone actually saying it three times. The measured hallucination
# loops were 17 cues over 267.6s and 30 cues over 333.6s, so these thresholds sit
# far from both populations.
MIN_REPEAT = 4
MIN_SPAN_SEC = 8.0

# Decode settings for the re-run. condition_on_previous_text=False is the one
# that matters (it is the loop's cause); the other two are cheap insurance.
REDECODE_KWARGS = dict(
    condition_on_previous_text=False,
    no_repeat_ngram_size=4,
    repetition_penalty=1.1,
)

_EPS = 1e-6


def find_loops(cues, min_repeat=MIN_REPEAT, min_span=MIN_SPAN_SEC):
    """Return [(first_index, last_index)] for each run of identical cue texts."""
    loops = []
    i = 0
    while i < len(cues):
        j = i
        while j + 1 < len(cues) and cues[j + 1]["text"] == cues[i]["text"]:
            j += 1
        span = cues[j]["end"] - cues[i]["start"]
        if j - i + 1 >= min_repeat and span >= min_span:
            loops.append((i, j))
        i = j + 1
    return loops


def _decode_span(model, audio_path, language, start, end):
    """Re-decode one span. Timestamps come back on the original audio timeline."""
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=8,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        clip_timestamps=[start, end],
        **REDECODE_KWARGS,
    )
    return list(segments)


def repair_loops(cues, words, model, audio_path, language, on_message=None):
    """Replace or drop every repetition loop in `cues`.

    Args:
        cues: [{"start", "end", "text"}], sorted by start.
        words: [{"word", "start", "end"}] covering the same audio.
        model: the already-loaded WhisperModel, reused for the re-decode.
        audio_path: the media file the cues came from.
        language: the language the first pass detected. Passing it explicitly
            avoids re-detecting on a span that may be mostly music.
        on_message: optional callback for one progress line per span.

    Returns:
        (cues, words, notes) with notes a list of log lines describing what
        happened to each span. Both lists are new; the inputs are not modified.
    """
    loops = find_loops(cues)
    if not loops:
        return cues, words, []

    cues = list(cues)
    words = list(words)
    notes = []

    def say(msg):
        notes.append(msg)
        if on_message:
            on_message(msg)

    say(f"重複迴圈偵測：{len(loops)} 段，開始重新解碼")

    # Last to first, so the earlier indices stay valid while splicing.
    for a, b in reversed(loops):
        start, end = cues[a]["start"], cues[b]["end"]
        span = end - start
        stamp = f"{start:.1f}s~{end:.1f}s（{span:.1f}s，{b - a + 1} 句）"
        say(f"  重新解碼 {stamp}「{cues[a]['text'][:20]}」")

        try:
            segments = _decode_span(model, audio_path, language, start, end)
        except Exception as e:
            say(f"    重新解碼失敗，保留原文字：{type(e).__name__}: {e}")
            continue

        new_cues = [{"start": s.start, "end": s.end,
                     "text": s2tw_gate.convert_guarded(s.text.strip())}
                    for s in segments if s.text.strip()]
        new_words = [{"word": s2tw_gate.convert_guarded(w.word),
                      "start": w.start, "end": w.end}
                     for s in segments for w in (s.words or [])]

        if not new_cues:
            action = "丟棄（重新解碼無輸出）"
        elif find_loops(new_cues):
            action = "丟棄（重新解碼仍然迴圈）"
            new_cues, new_words = [], []
        else:
            action = f"取代為 {len(new_cues)} 句"

        cues[a:b + 1] = new_cues
        words = [w for w in words
                 if not (start - _EPS <= w["start"] < end - _EPS)] + new_words
        say(f"    {action}")

    words.sort(key=lambda w: w["start"])
    return cues, words, notes
