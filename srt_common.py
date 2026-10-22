"""
srt_common.py : self-contained SRT utilities for Speech-to-subtitle
===================================================================
This project is fully standalone. SRT parsing / serialization / time-code
conversion live in the local module ``srt_model``; the homophone correction
dictionary lives in ``homophone_dict``; the line-length limits and the
filler-word set are defined here. No external project is required at runtime.

Re-exports everything with the original names (and the legacy aliases
``time_to_ms`` / ``ms_to_time``) so existing callers keep working unchanged.

Usage (unchanged):
    from srt_common import parse_srt, build_srt, time_to_ms, ms_to_time, MAX_LINE
"""

import os
import sys

# Make the local sibling modules importable regardless of the caller's cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# SRT model: parsing / serialization / time-code conversion / text helpers.
from srt_model import (  # noqa: F401,E402
    SrtEntry,
    parse_srt, parse_srt_file, build_srt, write_srt,
    tc_to_ms as time_to_ms,    # alias kept for Speech-to-subtitle callers
    ms_to_tc as ms_to_time,    # alias kept for Speech-to-subtitle callers
    get_text, is_cjk, is_en,
)

# Homophone correction dictionary (used by post_mfa_cleanup Step 8).
from homophone_dict import build_suspect_list  # noqa: F401,E402

# ── Line-length limits (max CJK chars per line) ──
MAX_LINE = 19       # max CJK chars per single line
MAX_LINE2 = 22      # max second-line width (two-line fallback)

# ── Pure filler words (used by post_mfa_cleanup short-entry cleanup) ──
PURE_FILLERS = {
    '嗯', '嗯嗯', '嗯哼', '啊', '喔', '噢', '哦', '欸', '呃', '呢',
    '對', '對對', '對對對', '好', '好的', '是', '是的',
    '然後', '那', '就', '所以', '可是', '但是',
    'OK', 'ok', 'Ok', 'oh', '的', 'ok ok',
}
