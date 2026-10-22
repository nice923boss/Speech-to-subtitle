"""Conservative subtitle cue merge.

Keeps Whisper's own segment boundaries (which are already semantically good) and
only absorbs over-short fragments into a neighbour under strict, safe conditions:

  - at least one side is a short fragment (<= frag_len chars)
  - combined length stays <= max_merge
  - the inter-cue gap is small (<= merge_gap seconds): never merge across a pause
  - the earlier cue does not end on sentence-final punctuation

Whole cues are merged, so a word is never split. Input/output cues are
dicts: {'start': float_seconds, 'end': float_seconds, 'text': str}.
"""

SENT_END = "。！？!?…"


def _len(s):
    return len(s.strip())


def _join(a, b):
    """Join two cue texts; insert a space only between two ASCII word chars."""
    a, b = a.strip(), b.strip()
    if a and b and a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return a + " " + b
    return a + b


def merge_short_cues(cues, frag_len=6, max_merge=20, merge_gap=0.6):
    """Return a new cue list with over-short fragments conservatively merged."""
    result = []
    for c in cues:
        text = c["text"].strip()
        if not text:
            continue
        if result:
            prev = result[-1]
            gap = c["start"] - prev["end"]
            combined = _len(prev["text"]) + _len(text)
            prev_frag = _len(prev["text"]) <= frag_len
            cur_frag = _len(text) <= frag_len
            prev_ends_sentence = prev["text"].rstrip()[-1:] in SENT_END
            if (prev_frag or cur_frag) and combined <= max_merge \
                    and gap <= merge_gap and not prev_ends_sentence:
                prev["text"] = _join(prev["text"], text)
                prev["end"] = c["end"]
                continue
        result.append({"start": c["start"], "end": c["end"], "text": text})
    return result
