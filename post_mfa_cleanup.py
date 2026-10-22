"""
MFA 後處理腳本 — 放在 MFA 對齊之後、最終輸出之前。

功能：
1. 合併被拆斷的英文詞組（Computer S + cience → Computer Science）
2. 刪除行尾頓號
3. 合併過短碎片（< 3 字元）
4. 延伸空隙（< 2 秒貼齊下一句）
5. 最終驗證（贅詞殘留、過短條目、間隙違規）

合併規則：第一條 start_ms + 最後一條 end_ms，不影響其他條目時間軸。

用法：
  python post_mfa_cleanup.py <input.srt> [--output <output.srt>]
  若不指定 --output，預設覆蓋輸入檔。
"""

import sys, re, argparse, os, difflib

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure this script's directory is on sys.path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from srt_common import parse_srt, build_srt, write_srt, time_to_ms, ms_to_time, MAX_LINE, MAX_LINE2  # noqa: E402


# ==================== STEP 0: MFA TEXT LOSS RECOVERY ====================

def recover_mfa_text_loss(pre_entries, post_entries):
    """Compare pre-MFA and post-MFA entries, restore text that MFA dropped.

    Rules:
    - If pre text contains post text (post is substring of pre) → restore pre text
    - If post text contains pre text → MFA added content, keep post
    - If texts are completely different → skip (MFA reassigned content)
    - Don't restore if the 'lost' chars appear in adjacent post entries
      (means MFA moved them, not dropped them)

    Only modifies text, never touches timecodes.
    """
    # Build lookup from pre entries by matching start_ms (within 500ms tolerance)
    # This handles cases where entry counts differ slightly
    if len(pre_entries) == len(post_entries):
        # Perfect 1:1 match
        pre_lookup = {i: pre_entries[i] for i in range(len(pre_entries))}
        post_to_pre = {i: i for i in range(len(post_entries))}
    else:
        # Match by closest start_ms
        post_to_pre = {}
        used_pre = set()
        for pi in range(len(post_entries)):
            best_match = None
            best_diff = 501  # tolerance 500ms
            for qi in range(len(pre_entries)):
                if qi in used_pre:
                    continue
                diff = abs(post_entries[pi]['start_ms'] - pre_entries[qi]['start_ms'])
                if diff < best_diff:
                    best_diff = diff
                    best_match = qi
            if best_match is not None:
                post_to_pre[pi] = best_match
                used_pre.add(best_match)

    restored = 0
    for i in range(len(post_entries)):
        if i not in post_to_pre:
            continue
        pre_idx = post_to_pre[i]
        pre_text = pre_entries[pre_idx]['text'].strip()
        post_text = post_entries[i]['text'].strip()

        if pre_text == post_text:
            continue

        # Use difflib to precisely detect dropped characters
        if not pre_text or not post_text:
            continue

        # SequenceMatcher finds exact diff between pre and post
        sm = difflib.SequenceMatcher(None, post_text, pre_text)
        ratio = sm.ratio()

        # Only restore if texts are similar enough (post is subset of pre)
        if ratio < 0.5:
            continue  # Completely different texts — MFA reassigned content

        # Find chars in pre but not in post (true deletions)
        lost_chars = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'insert':
                # Present in pre (j1:j2) but missing from post
                lost_chars.append(pre_text[j1:j2])

        if not lost_chars:
            continue

        lost_text = ''.join(lost_chars).strip()
        if not lost_text:
            continue

        # Check if lost chars were moved to adjacent entries (not truly lost)
        moved = False
        if i > 0:
            prev_text = post_entries[i - 1]['text'].strip()
            if prev_text.endswith(lost_text) or lost_text in prev_text:
                moved = True
        if not moved and i + 1 < len(post_entries):
            next_text = post_entries[i + 1]['text'].strip()
            if next_text.startswith(lost_text) or lost_text in next_text:
                moved = True

        if not moved:
            post_entries[i]['text'] = pre_text
            restored += 1

    return post_entries, restored


# ==================== STEP 1: MERGE BROKEN ENGLISH ====================

def is_partial_english_end(text):
    """Check if entry ends with a partial English word (e.g., 'Computer S')"""
    text = text.rstrip()
    if not text:
        return False
    # Find the last whitespace-separated token
    tokens = text.split()
    if not tokens:
        return False
    last_token = tokens[-1]
    # It's a partial English word if:
    # - It's all ASCII letters
    # - It's 1-2 characters (too short to be a real word, e.g., 'S', 'Co')
    # - OR it doesn't look like a complete English word (no vowel pattern check needed,
    #   just check if next entry starts with lowercase continuation)
    if re.match(r'^[A-Za-z]{1,2}$', last_token):
        return True
    return False

def next_starts_with_english_continuation(text):
    """Check if text starts with lowercase English (continuation of a split word)"""
    text = text.lstrip()
    if not text:
        return False
    return bool(re.match(r'^[a-z]', text))

def merge_broken_english(entries):
    """Merge entries where English words were split across lines."""
    if not entries:
        return entries, 0

    merged_count = 0
    result = []
    i = 0

    while i < len(entries):
        entry = entries[i]

        # Look ahead: does current entry end with partial English?
        if i + 1 < len(entries):
            curr_text = entry['text'].rstrip()
            next_text = entries[i + 1]['text'].lstrip()

            should_merge = False

            # Case 1: Current ends with 1-2 char English token AND next starts with lowercase
            if is_partial_english_end(curr_text) and next_starts_with_english_continuation(next_text):
                should_merge = True

            # Case 2: Current ends with uppercase letter(s) and next starts with lowercase
            # e.g., "Computer S" + "cience 的工作"
            if not should_merge and curr_text and curr_text[-1].isupper():
                if next_starts_with_english_continuation(next_text):
                    should_merge = True

            if should_merge:
                # Merge: combine text, take first start + last end
                merged_text = curr_text + next_text
                new_entry = {
                    'idx': 0,
                    'start_ms': entry['start_ms'],
                    'end_ms': entries[i + 1]['end_ms'],
                    'text': merged_text,
                }
                # Check if NEXT next also needs merging (chain)
                entries[i + 1] = new_entry
                merged_count += 1
                i += 1
                continue

        result.append(entry)
        i += 1

    return result, merged_count


# ==================== STEP 2: REMOVE TRAILING PUNCTUATION ====================

def remove_trailing_enum_comma(entries):
    """Remove trailing 、 (enumeration comma) at end of entry text.

    Safety net: Step 3 (restructure.py) already removes these, but:
    - Step 0 text recovery may restore Step 4 AI-optimized text that contains trailing 、
    - Step 4 AI optimization itself may re-introduce trailing 、
    MFA 本身不修改文字內容，風險來自上游文字還原。
    """
    count = 0
    for e in entries:
        text = e['text'].rstrip()
        if text.endswith('、'):
            e['text'] = text[:-1].rstrip()
            count += 1
    return entries, count


# ==================== STEP 3: MERGE SHORT ENTRIES ====================

# 單一來源：從 srt_common 匯入（PURE_FILLERS 定義於本專案的 srt_common）
from srt_common import PURE_FILLERS  # noqa: F811

def is_english_dominant(text):
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    return not has_cjk and len(text.strip()) > 0

def remove_pure_fillers(entries):
    """Remove entries that are pure filler words."""
    result = []
    count = 0
    for e in entries:
        if e['text'].strip() in PURE_FILLERS:
            count += 1
            continue
        result.append(e)
    return result, count

def merge_short_entries(entries, min_chars=6):
    """Merge entries shorter than min_chars into adjacent entries.

    Thresholds aligned with restructure.py Step 3:
    - Fragment (≤6 chars) + combined ≤ MAX_LINE (18): single line
    - Fragment (≤6 chars) + combined ≤ MAX_LINE + MAX_LINE2 (40): allow split later
    Rule: first entry's start_ms + last entry's end_ms.
    """
    if not entries:
        return entries, 0

    result = []
    merged_count = 0
    i = 0

    while i < len(entries):
        entry = entries[i]
        text = entry['text'].strip()

        if is_english_dominant(text) or len(text) > min_chars:
            result.append(entry)
            i += 1
            continue

        # Try merge into next (threshold: MAX_LINE + MAX_LINE2 = 40)
        if i + 1 < len(entries):
            nxt = entries[i + 1]
            merged_text = text + nxt['text']
            if len(merged_text) <= MAX_LINE + MAX_LINE2:
                nxt['text'] = merged_text
                nxt['start_ms'] = entry['start_ms']
                merged_count += 1
                i += 1
                continue

        # Try merge into prev
        if result:
            prev = result[-1]
            merged_text = prev['text'] + text
            if len(merged_text) <= MAX_LINE + MAX_LINE2:
                prev['text'] = merged_text
                prev['end_ms'] = entry['end_ms']
                merged_count += 1
                i += 1
                continue

        # Can't merge, keep as is
        result.append(entry)
        i += 1

    return result, merged_count


# ==================== STEP 4: FILL GAPS ====================

def fill_gaps(entries, threshold_ms=2000):
    """Extend end time to fill gaps < threshold_ms."""
    count = 0
    for i in range(len(entries) - 1):
        gap = entries[i + 1]['start_ms'] - entries[i]['end_ms']
        if 0 < gap < threshold_ms:
            entries[i]['end_ms'] = entries[i + 1]['start_ms']
            count += 1
    return entries, count


# ==================== STEP 5: VALIDATION ====================

_CJK_SPACE_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf] [\u4e00-\u9fff\u3400-\u4dbf]')


def validate(entries, expect_gaps_filled=False):
    issues = []

    # Pure fillers remaining
    fillers = [e for e in entries if e['text'].strip() in PURE_FILLERS]
    if fillers:
        issues.append(f"Pure filler entries remaining: {len(fillers)}")

    # Short entries
    short = [e for e in entries if len(e['text'].strip()) < 3 and not is_english_dominant(e['text'])]
    if short:
        issues.append(f"Short entries (<3 chars): {len(short)}")
        for e in short[:5]:
            issues.append(f"  '{e['text']}'")

    # Gap violations. Only a violation when this run was asked to fill gaps:
    # by default the line leaves gaps alone here and closes them once, at the
    # last station (Quick subtitles' gap mode).
    if expect_gaps_filled:
        gap_v = 0
        for i in range(len(entries) - 1):
            gap = entries[i + 1]['start_ms'] - entries[i]['end_ms']
            if 0 < gap < 2000:
                gap_v += 1
        if gap_v:
            issues.append(f"Gap violations (0-2000ms unfilled): {gap_v}")

    # Overlap violations (10ms tolerance, consistent with run_pipeline.py)
    overlap_v = 0
    for i in range(len(entries) - 1):
        if entries[i + 1]['start_ms'] < entries[i]['end_ms'] - 10:
            overlap_v += 1
    if overlap_v:
        issues.append(f"Overlap violations: {overlap_v}")

    # Trailing enumeration comma
    trailing = [e for e in entries if e['text'].rstrip().endswith('、')]
    if trailing:
        issues.append(f"Trailing 、 remaining: {len(trailing)}")

    # Broken English (entry ending with 1-2 char English + next starting lowercase)
    broken_en = 0
    for i in range(len(entries) - 1):
        if is_partial_english_end(entries[i]['text']) and \
           next_starts_with_english_continuation(entries[i + 1]['text']):
            broken_en += 1
    if broken_en:
        issues.append(f"Broken English words: {broken_en}")

    # CJK space (Chinese chars separated by space — Whisper artifact or text recovery regression)
    cjk_space_v = 0
    for e in entries:
        if _CJK_SPACE_RE.search(e['text']):
            cjk_space_v += 1
    if cjk_space_v:
        issues.append(f"CJK space violations: {cjk_space_v}")

    return issues


# ==================== SUSPECTS CROSS-CHECK (2026-03-31) ====================

def check_suspects_post_mfa(entries, suspects_path):
    """Cross-reference post-MFA entries against audio energy suspects.

    High-confidence suspects with hallucination indicators are auto-removed:
      - phantom_silence (high): text over near-silence → delete
      - impossible_speed (high): impossibly fast speech → delete
      - hallucination_char_density (high): char density ratio > 3.0 → delete

    Medium/low confidence suspects are flagged but kept.
    """
    import json
    if not os.path.exists(suspects_path):
        print(f"Suspects file not found: {suspects_path}")
        return entries, 0, 0

    with open(suspects_path, 'r', encoding='utf-8') as f:
        suspects_data = json.load(f)

    # Build lookup: start_ms → suspect data (within 500ms tolerance)
    suspects_by_time = {}
    for s in suspects_data.get('suspects', []):
        suspects_by_time[s.get('start_ms', -9999)] = s

    AUTO_REMOVE_REASONS = {'phantom_silence', 'impossible_speed', 'hallucination_char_density'}

    flagged = 0
    removed = 0
    keep = []

    for entry in entries:
        matched_suspect = None
        for suspect_start, suspect in suspects_by_time.items():
            if abs(entry['start_ms'] - suspect_start) < 500:
                matched_suspect = suspect
                break

        if matched_suspect is None:
            keep.append(entry)
            continue

        reasons = set(matched_suspect.get('reasons', []))
        confidence = matched_suspect.get('confidence', 'low')

        if confidence == 'high' and reasons & AUTO_REMOVE_REASONS:
            removed += 1
            print(f"  AUTO-REMOVED #{entry['idx']}: \"{entry['text'][:40]}\" "
                  f"(reasons: {', '.join(reasons)})")
        else:
            flagged += 1
            print(f"  FLAGGED #{entry['idx']}: \"{entry['text'][:40]}\" "
                  f"[{confidence}] (reasons: {', '.join(reasons)})")
            keep.append(entry)

    return keep, flagged, removed


# ==================== STEP 6: CREW DIALOGUE DETECTION ====================

# Recording crew dialogue keywords — high confidence patterns that indicate
# behind-the-scenes discussion, not lecture content. Discovered from
# 08-TDCC退休金制度介紹 where ~57 entries were crew talk.
CREW_DIALOGUE_PATTERNS = [
    # Recording logistics
    r'重來|重弄|重錄|重新錄|再重新',
    r'再來一次|再錄一次',
    r'試打一下|試一下',
    r'這一?頁要?重',
    # Sound/noise issues
    r'椅子的?聲音|翻書|翻頁|收音.{0,4}靈敏',
    r'不要翻書|不要動|都不要',
    # Crew instructions
    r'剪好|剪掉|撒掉|弄過去',
    r'手幣要?放下|看.{0,2}鏡頭',
    r'不要馬上往下看',
    r'你可以把它解掉',
    r'有人在弄東西',
    # Format/technical
    r'表格.{0,4}紙的|影片.{0,4}橫的|放不進',
    r'連線檔|放進去會變',
    # After-recording wrap-up
    r'最後一頁.{0,4}不漏|念的唸完',
    r'講完謝謝之後',
]
_CREW_PATTERN = re.compile('|'.join(CREW_DIALOGUE_PATTERNS))


def detect_crew_dialogue(entries):
    """Detect and remove recording crew dialogue entries.

    Uses keyword pattern matching. High-confidence matches are auto-removed.
    Returns (kept_entries, removed_count).
    """
    keep = []
    removed = 0
    for entry in entries:
        text = entry['text'].strip()
        if _CREW_PATTERN.search(text):
            removed += 1
            print(f"  CREW-REMOVED #{entry['idx']}: \"{text[:50]}\"")
        else:
            keep.append(entry)
    return keep, removed


# ==================== STEP 7: SENTENCE-FINAL FILLER CLEANUP ====================

# Question words that justify keeping 呢 at sentence end
_QUESTION_WORDS = {'什麼', '為什麼', '怎麼', '嗎', '不然', '哪', '幾', '多少', '誰', '何'}


def remove_sentence_final_fillers(entries):
    """Remove sentence-final filler particles (呢) from non-question sentences.

    Rules:
    - 呢 mid-sentence (not at end): always remove
    - 呢 at sentence end + question word in text: keep (genuine question)
    - 呢 at sentence end + no question word: remove (oral filler)
    """
    count = 0
    for e in entries:
        text = e['text']
        if '呢' not in text:
            continue
        if is_english_dominant(text):
            continue

        original = text

        # Mid-sentence 呢: remove (呢 followed by more text)
        while '呢' in text and not text.rstrip().endswith('呢'):
            text = text.replace('呢', '', 1)
            text = re.sub(r'  +', ' ', text).strip()

        # Sentence-final 呢: only keep if text contains question word
        if text.rstrip().endswith('呢') and len(text.strip()) > 1:
            has_question = any(qw in text for qw in _QUESTION_WORDS)
            if not has_question:
                text = text.rstrip()[:-1].rstrip()

        if text != original:
            e['text'] = text
            count += 1

    return entries, count


# ==================== STEP 8: TEXT-BASED HALLUCINATION DETECTION ====================

def mark_homophone_candidates(entries):
    """Mark words that may be Whisper homophone substitutions.

    Uses the unified homophone dictionary from srt_common
    (single source of truth) instead of maintaining a local HOMOPHONE_PAIRS list.
    Outputs suspects for Step 9 human review.

    Returns:
        entries (unchanged), list of suspect dicts
    """
    try:
        from pypinyin import pinyin, Style
    except ImportError:
        print("  WARNING: pypinyin not installed, skipping homophone scan")
        return entries, []

    # Import from the local homophone dictionary (single source of truth)
    from srt_common import build_suspect_list
    suspect_pairs = build_suspect_list()  # [(wrong, right), ...] sorted long-first

    suspects = []
    for entry in entries:
        text = entry['text']
        for wrong, right in suspect_pairs:
            if wrong in text:
                chars = list(wrong)
                py = ' '.join([p[0] for p in pinyin(chars, style=Style.TONE)])
                suspects.append({
                    'idx': entry['idx'],
                    'word': wrong,
                    'suggestion': right,
                    'pinyin': py,
                    'context': text[:60],
                })

    return entries, suspects


def detect_text_hallucinations(entries):
    """Detect Whisper hallucinations based on text anomalies.

    Catches cases where audio has non-speech sounds but Whisper produces
    gibberish text with normal-sounding acoustic features (so audio energy
    verify doesn't flag them).

    Signals:
    - Repeated pronouns/words: 要我我是, 可以可以
    - Very low character overlap with surrounding entries (topic drift)
    - Known hallucination fragments
    """
    if len(entries) < 3:
        return entries, 0

    # Known Whisper hallucination fragments (language-agnostic gibberish)
    HALLUCINATION_FRAGMENTS = {
        '一隻小白鳥', '生蛋專的', '螺絲他其實有問題',
        '小白鳥', '我是一隻',
    }

    def _topic_chars(text):
        """Extract CJK characters for topic comparison."""
        return set(re.findall(r'[\u4e00-\u9fff]', text))

    removed = 0
    keep = []

    for i, entry in enumerate(entries):
        text = entry['text'].strip()
        is_hallucination = False

        # Check 1: Known hallucination fragments
        for frag in HALLUCINATION_FRAGMENTS:
            if frag in text:
                is_hallucination = True
                break

        # Check 2: Repeated pronoun/word stuttering (要我我是, 他是要我我)
        if not is_hallucination and re.search(r'(.{1,3})\1{2,}', text):
            # Triple+ repetition of 1-3 char pattern
            is_hallucination = True

        # Check 3: Topic drift — very low overlap with surrounding 5 entries
        if not is_hallucination and len(text) > 5:
            my_chars = _topic_chars(text)
            if my_chars:  # Only check CJK entries
                neighbor_chars = set()
                for j in range(max(0, i - 3), min(len(entries), i + 4)):
                    if j != i:
                        neighbor_chars |= _topic_chars(entries[j]['text'])
                if neighbor_chars:
                    overlap = len(my_chars & neighbor_chars) / len(my_chars)
                    # If less than 15% of chars appear in any neighbor → suspicious
                    if overlap < 0.15 and len(my_chars) > 3:
                        is_hallucination = True

        if is_hallucination:
            removed += 1
            print(f"  HALLUCINATION-REMOVED #{entry['idx']}: \"{text[:50]}\"")
        else:
            keep.append(entry)

    return keep, removed


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='MFA post-processing cleanup')
    parser.add_argument('input', help='Input SRT file (MFA output)')
    parser.add_argument('--pre', '-p', help='Pre-MFA SRT file (for text loss recovery)')
    parser.add_argument('--suspects', '-s', help='Suspects JSON from audio_energy_verify.py')
    parser.add_argument('--output', '-o', help='Output SRT file (default: overwrite input)')
    parser.add_argument('--fill-gaps', action='store_true',
                        help='Extend cue ends to close gaps < 2s. Off by default: '
                             'gap filling belongs to the last station (Quick subtitles).')
    args = parser.parse_args()

    output_path = args.output or args.input

    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = parse_srt(content)
    original_count = len(entries)
    print(f"Input: {original_count} entries")

    # Step 0: MFA text loss recovery (if pre-MFA file provided)
    if args.pre:
        with open(args.pre, 'r', encoding='utf-8') as f:
            pre_content = f.read()
        pre_entries = parse_srt(pre_content)
        entries, recovered = recover_mfa_text_loss(pre_entries, entries)
        print(f"MFA text loss recovered: {recovered}")

    # Step 0.1: Format fixes after text recovery
    # Text recovery (Step 0) may restore pre-Step-2 text that lacks CJK space removal
    # and number-unit spacing. Fix these at the source before downstream steps.
    format_fixed = 0
    for e in entries:
        old_text = e['text']
        new_text = old_text
        # Remove CJK spaces (loop for overlapping matches)
        prev = None
        while prev != new_text:
            prev = new_text
            new_text = re.sub(
                r'([\u4e00-\u9fff\u3400-\u4dbf]) ([\u4e00-\u9fff\u3400-\u4dbf])',
                r'\1\2', new_text,
            )
        if new_text != old_text:
            e['text'] = new_text
            format_fixed += 1
    if format_fixed:
        print(f"Format fixes (CJK spaces): {format_fixed}")

    # Step 0.5: Suspects cross-check (if suspects JSON provided)
    if args.suspects:
        entries, suspect_flags, suspect_removed = check_suspects_post_mfa(entries, args.suspects)
        print(f"Suspects: {suspect_removed} auto-removed, {suspect_flags} flagged")

    # Step 1: Merge broken English
    entries, en_merged = merge_broken_english(entries)
    # Run twice to catch chains
    entries, en_merged2 = merge_broken_english(entries)
    en_merged += en_merged2
    print(f"Broken English merged: {en_merged}")

    # Step 2: Remove trailing 、
    entries, trail_count = remove_trailing_enum_comma(entries)
    print(f"Trailing 、 removed: {trail_count}")

    # Step 3: Remove pure fillers + merge short
    entries, filler_count = remove_pure_fillers(entries)
    print(f"Pure fillers removed: {filler_count}")

    entries, short_merged = merge_short_entries(entries)
    print(f"Short entries merged: {short_merged}")

    # 那→哪 已在 Step 2 (text_fix.py) 處理，此處不再重複

    # Step 6: Crew dialogue detection
    entries, crew_removed = detect_crew_dialogue(entries)
    print(f"Crew dialogue removed: {crew_removed}")

    # Step 7: Sentence-final filler cleanup (呢)
    entries, filler_fixed = remove_sentence_final_fillers(entries)
    print(f"Sentence-final fillers cleaned: {filler_fixed}")

    # Step 8: Text-based hallucination detection
    entries, halluc_removed = detect_text_hallucinations(entries)
    print(f"Text hallucinations removed: {halluc_removed}")

    # Step 9-pre: Homophone candidate marking (for human review)
    entries, homophone_suspects = mark_homophone_candidates(entries)
    if homophone_suspects:
        print(f"\n=== HOMOPHONE SUSPECTS ({len(homophone_suspects)}) ===")
        for s in homophone_suspects:
            print(f"  #{s['idx']}: \"{s['word']}\" → \"{s['suggestion']}\" "
                  f"({s['pinyin']})  context: {s['context']}")
        # Write suspects to JSON for Step 9 reference
        import json
        suspects_out = os.path.splitext(args.input)[0] + '_homophone_suspects.json'
        with open(suspects_out, 'w', encoding='utf-8') as f:
            json.dump(homophone_suspects, f, ensure_ascii=False, indent=2)
        print(f"  → Saved to {suspects_out}")
    else:
        print("Homophone suspects: 0")

    # Step 4: Fill gaps (opt-in). Off by default so this tool does not re-inflate
    # cue ends that the line deliberately leaves clean for the last station.
    if args.fill_gaps:
        entries, gaps_filled = fill_gaps(entries)
        print(f"Gaps filled (<2s): {gaps_filled}")
    else:
        print("Gaps filled (<2s): 0 (--fill-gaps 未指定，填隙交給 Quick subtitles)")

    # Step 5: Validate
    issues = validate(entries, expect_gaps_filled=args.fill_gaps)
    if issues:
        print("\n=== VALIDATION ISSUES ===")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\nValidation: ALL PASSED")

    # Write output
    write_srt(output_path, entries)

    print(f"\nFinal: {len(entries)} entries (was {original_count})")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
