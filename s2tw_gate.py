"""Simplified -> Taiwan-traditional conversion gate (single source of truth).

Shared by transcriber.py (output gate) and convert_srt_s2tw.py (batch tool for
files produced before the gate existed).

Detection rule: a text holds residue when at least one of its characters is a
simplified-only character, i.e. a character s2tw rewrites on its own with no
context (头, 发, 条, 台, 历 ...) and that is not a MERGE_CHARS entry. When that is
the case the text is converted phrase by phrase, so context-sensitive pairs
still come out right (头发 -> 頭髮, 干燥 -> 乾燥, 面条 -> 麵條), but only the
phrases that hold the evidence are rewritten (see _phrase_fixes). A MERGE_CHARS
entry counts as evidence too, but only for its own slot and only when the
phrase table convicts it in context (see _merge_residue_positions).

The obvious rule, "converted != original means residue", is wrong here: s2tw is
not idempotent on already-traditional text. It also encodes Taiwan variant
preferences, so 了然 -> 瞭然 and 了解 -> 瞭解. Testing whole lines that way makes
correct traditional text look like residue and rewrites it: real output held
「然後就當兵了然後」 and the gate turned it into 「當兵瞭然後」. Every character
there is already traditional, so the character-level rule leaves it alone.

The character rule needs the same escape hatch, for the same reason one level
down. Simplification merged several traditional characters into a form that was
already a traditional character in its own right (一簡對多繁), so is_simplified_char
fires on characters that are perfectly correct where they stand. Measured on the
old rule: 萬里長城 -> 萬裡長城, 里長伯 -> 裡長伯, 岳飛 -> 嶽飛, 干擾 -> 幹擾,
濃郁 -> 濃鬱. The damage is worst at word granularity, where a lone 里 in the
.words.json sidecar is a one-character string and whole-string conversion always
differs, so the sidecar drifted away from the SRT it is supposed to match
character for character. MERGE_CHARS is that escape hatch.

The escape hatch has to be conditional, though, because 干 really is simplified
in 干淨 and 后 really is simplified in 后面. Excusing those characters outright
costs real conversions (干淨, 后面, 云彩 all stayed as typed). The only thing that
can tell 萬里 from 夢里 is s2tw's phrase table, and the table is keyed on
simplified phrases, so it never matches mixed text like 干淨 (干 simplified next
to 淨 traditional) and the blunt character rule answers instead. Normalising the
text to simplified first makes the key match: 干淨 -> 干净 -> 乾淨. See
_merge_residue_positions, which reads that normalised copy as evidence only.

When OpenCC cannot be imported the module stays importable but AVAILABLE is
False and every caller is expected to surface UNAVAILABLE_WARNING instead of
silently degrading to no conversion.
"""

import difflib

try:
    from opencc import OpenCC
    _S2TW = OpenCC("s2tw")
    # Orthographic only. tw2sp would also swap vocabulary (影片 -> 视频), which
    # does not round trip and would corrupt the text this is used to judge.
    _T2S = OpenCC("t2s")
    IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on local install
    _S2TW = None
    _T2S = None
    IMPORT_ERROR = f"{type(e).__name__}: {e}"

AVAILABLE = _S2TW is not None

UNAVAILABLE_WARNING = "警告：缺 OpenCC，輸出將不做簡轉繁（簡體字會原樣留在字幕裡）"


def convert(text: str) -> str:
    """Convert to Taiwan traditional. Identity when OpenCC is unavailable."""
    return _S2TW.convert(text) if AVAILABLE else text


# Characters simplification merged into a form that is itself standard
# traditional. s2tw rewrites each one when it stands alone, but each is correct
# Chinese where it stands, so seeing one is not evidence of residue:
#   里 公里、里長、萬里    后 皇后、后羿      云 人云亦云
#   干 干擾、天干、干戈    丑 丑時、丑角      划 划船、划算
#   几 茶几                咸 咸豐、老少咸宜  范 范仲淹（姓）
#   于 于右任（姓）        岳 岳父、岳飛      斗 北斗、斗笠、煙斗
#   涂 姓涂                郁 濃郁
#
# 台 is deliberately absent: the owner's decision is 臺, so 台北 -> 臺北 must keep
# firing. Variant-only forms (洼 帘 适 杰) are absent for the same reason. Their
# standard traditional character is the other one (窪 簾 適 傑), so rewriting them
# is the right call, and 周杰倫 is protected by the phrase table anyway.
#
# The excuse is conditional, not absolute: _merge_residue_positions still
# convicts one of these when the phrase table, shown a simplified copy of the
# same text, puts a different character in that slot. So 萬里 and 干擾 survive
# while 夢里 -> 夢裡 and 干淨 -> 乾淨 are still fixed.
MERGE_CHARS = frozenset("里后云干丑划几咸范于岳斗涂郁")

_SIMPLIFIED_CHARS: dict[str, bool] = {}


def is_simplified_char(char: str) -> bool:
    """True when s2tw rewrites this character on its own, with no context.

    Characters valid in traditional Chinese (了, 然, 面, 只, 系) map to
    themselves and are never treated as residue, even though s2tw may still
    prefer a variant for them inside a phrase.

    This is a property of s2tw, not a verdict on the character: a MERGE_CHARS
    entry is rewritten here and is still correct traditional. Callers deciding
    whether text holds residue must exclude MERGE_CHARS as residue() does.
    """
    cached = _SIMPLIFIED_CHARS.get(char)
    if cached is None:
        cached = _S2TW.convert(char) != char
        _SIMPLIFIED_CHARS[char] = cached
    return cached


def _merge_residue_positions(text: str) -> dict[int, str]:
    """Positions where a MERGE_CHARS character is genuinely simplified.

    A merge character means nothing on its own, so the only thing that can judge
    it is s2tw's phrase table. The table is keyed on simplified phrases and this
    text is usually mixed (干 simplified sitting next to 淨 traditional), so no
    key matches and the character rule answers instead. Converting the text to
    simplified first makes the key match: 干淨 -> 干净 -> 乾淨.

    The normalised copy is evidence, never output. Measured over the three real
    transcripts, using it as output rewrites correct traditional elsewhere in
    the same line: 裡面 -> 里面, 念書 -> 唸書, 盡量 -> 儘量, 回歸 -> 迴歸,
    讚嘆 -> 讚歎, 煙斗 -> 菸斗. Only slots holding a merge character are read
    back from it.

    A one-character string gives the phrase table no context to work with, so it
    is never evidence. That is the .words.json sidecar case: the cue makes the
    call and sync_words_to_cues carries it across.
    """
    if len(text) < 2:
        return {}
    normalised = _S2TW.convert(_T2S.convert(text))
    if len(normalised) != len(text):
        return {}
    return {i: normalised[i] for i, c in enumerate(text)
            if c in MERGE_CHARS and normalised[i] != c}


def _phrase_fixes(text: str, simplified: list[int]) -> dict[int, str]:
    """Replacements for the phrases that hold a simplified character.

    Converting the whole line with s2tw went wrong two ways on Whisper's mixed
    output. The phrase table never matches mixed text, so 頭发 came out 頭發 and
    复雜 came out 復雜 (the character default, not the phrase). And every other
    phrase on the line took s2tw's Taiwan variant preference, so 了解这個 came
    out 瞭解這個 although 了解 was already correct.

    Both are fixed by reading the normalised copy (simplified first, then s2tw,
    as _merge_residue_positions does) and taking only the runs of changed slots
    that contain a simplified character. 面条 changes as one run (麵條), so the
    context-dependent 面 is carried along; in 了解这個 the 了 -> 瞭 run holds no
    simplified character and is left alone. Merge characters are never written
    here: _merge_residue_positions alone decides those.
    """
    normalised = _S2TW.convert(_T2S.convert(text))
    if len(normalised) != len(text):
        # No slot mapping to read from. Character defaults are the safe floor.
        return {i: _S2TW.convert(text[i]) for i in simplified}

    evidence = set(simplified)
    fixes = {}
    i = 0
    while i < len(text):
        if normalised[i] == text[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(text) and normalised[j + 1] != text[j + 1]:
            j += 1
        if evidence.intersection(range(i, j + 1)):
            fixes.update((k, normalised[k]) for k in range(i, j + 1)
                         if text[k] not in MERGE_CHARS)
        i = j + 1
    return fixes


def residue(text: str) -> str | None:
    """Return the converted text when `text` still holds simplified characters."""
    if not AVAILABLE:
        return None
    merged = _merge_residue_positions(text)
    simplified = [i for i, c in enumerate(text)
                  if is_simplified_char(c) and c not in MERGE_CHARS]
    if not merged and not simplified:
        return None

    chars = list(text)
    for i, ch in merged.items():
        chars[i] = ch
    # Phrase conversion runs only when something outside MERGE_CHARS is
    # simplified. When the merge slots were the only evidence, converting the
    # rest as well would drag Taiwan variant preferences in with it (了解 ->
    # 瞭解) on a line that is otherwise already correct.
    if simplified:
        for i, ch in _phrase_fixes(text, simplified).items():
            chars[i] = ch
    fixed = "".join(chars)
    return fixed if fixed != text else None


def sync_words_to_cues(cues, words, text_key: str = "text",
                       word_key: str = "word") -> int:
    """Copy the cue stream's characters onto the word stream. Returns changes.

    Quick subtitles aligns the sidecar against the SRT character by character,
    and below a match ratio of 0.3 it silently drops to spreading subtitles
    evenly across the whole file, so every character the two streams disagree on
    is a timing risk. The two cannot be kept in step by judging them separately:
    a one-character word carries no context, and a merge character needs context
    to judge. So the cue decides and the words follow.

    Only equal-length substitutions are applied, over the streams with
    whitespace removed. Insertions and deletions are left alone: they are the
    known symbol-only difference between an SRT and its sidecar, and a word list
    that genuinely does not match these cues is better left intact than
    scrambled.
    """
    if not cues or not words:
        return 0

    origin, side = [], []
    for wi, word in enumerate(words):
        for ci, ch in enumerate(word[word_key]):
            if ch.isspace():
                continue
            side.append(ch)
            origin.append((wi, ci))
    cue_chars = [c for c in "".join(cue[text_key] for cue in cues)
                 if not c.isspace()]

    changed = 0
    ops = difflib.SequenceMatcher(None, cue_chars, side, autojunk=False)
    for tag, i1, i2, j1, j2 in ops.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for k in range(i2 - i1):
            wi, ci = origin[j1 + k]
            word = words[wi][word_key]
            words[wi][word_key] = word[:ci] + cue_chars[i1 + k] + word[ci + 1:]
            changed += 1
    return changed


def convert_guarded(text: str) -> str:
    """Convert only when there is evidence of residue. Identity otherwise.

    What raw Whisper output should be run through, rather than convert(). Whisper
    emits mostly-traditional text for zh with simplified characters sprinkled
    through it, so converting every segment unconditionally rewrites lines that
    were already correct: s2tw carries Taiwan variant preferences (了解 -> 瞭解,
    and 了然 -> 瞭然, which is wrong in 「當兵了然後」) and resolves merge
    characters with no evidence that they need resolving (萬里 -> 萬裡).
    """
    fixed = residue(text)
    return text if fixed is None else fixed


def convert_cues(cues, key: str = "text") -> int:
    """Convert `key` in place for a list of dicts. Returns how many changed."""
    changed = 0
    for cue in cues:
        fixed = residue(cue[key])
        if fixed is not None:
            cue[key] = fixed
            changed += 1
    return changed


def scan_text(content: str) -> list[tuple[int, str, str]]:
    """Line-by-line residue scan. Returns [(line_no, original, converted)]."""
    hits = []
    for i, line in enumerate(content.split("\n"), start=1):
        fixed = residue(line)
        if fixed is not None:
            hits.append((i, line, fixed))
    return hits


def status_line() -> str:
    """One-line conversion status for logs and completion messages."""
    if AVAILABLE:
        return "簡轉繁：已啟用（OpenCC s2tw）"
    return f"{UNAVAILABLE_WARNING}（原因：{IMPORT_ERROR}）"
