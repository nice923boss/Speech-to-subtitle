"""
srt_model.py — Unified SRT entry model
=======================================
Single SRT entry class used by ALL pipeline and speech-processing modules.
Stores millisecond timestamps internally; provides timecode/lines properties
for backward compatibility with both pipeline and speech formats.

Supports dict-like access (entry['lines'], entry['start_ms']) so existing
code continues to work without modification.

Usage:
    from srt_model import SrtEntry, parse_srt, write_srt
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ─── Time code utilities ───

def tc_to_ms(tc: str) -> int:
    """'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' -> milliseconds"""
    h, m, rest = tc.split(':')
    s, ms = re.split(r'[,.]', rest)
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms[:3])


def ms_to_tc(ms: int) -> str:
    """milliseconds -> 'HH:MM:SS,mmm'"""
    if ms < 0:
        ms = 0
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_TS_RE = re.compile(
    r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})'
)


def parse_tc(tc: str) -> tuple[str, str]:
    """'HH:MM:SS,mmm --> HH:MM:SS,mmm' -> (start_tc, end_tc)"""
    parts = tc.split('-->')
    return parts[0].strip(), parts[1].strip()


def merge_tc(tc1: str, tc2: str) -> str:
    """Take start from tc1, end from tc2"""
    start, _ = parse_tc(tc1)
    _, end = parse_tc(tc2)
    return f"{start} --> {end}"


# ─── Text utilities ───

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def is_cjk(text: str) -> bool:
    """Text contains CJK characters"""
    return bool(_CJK_RE.search(text))


def is_en(text: str) -> bool:
    """Text is pure English (no CJK)"""
    return not is_cjk(text)


def get_text(entry) -> str:
    """Extract plain text from entry (supports both SrtEntry and legacy dict)"""
    if isinstance(entry, SrtEntry):
        return entry.text.strip()
    # Legacy dict support
    if 'text' in entry:
        return entry['text'].strip()
    return ''.join(entry.get('lines', [])).strip()


def cjk_len(text: str) -> int:
    """Count CJK characters in text"""
    return len(_CJK_RE.findall(text))


def char_len(text: str) -> int:
    """Display length of text (for line break decisions)"""
    return len(text)


# ─── SrtEntry dataclass ───

@dataclass
class SrtEntry:
    """Unified subtitle entry.

    Stores time as millisecond integers internally.
    Provides timecode string and lines list as computed properties
    for backward compatibility with both pipeline and speech formats.

    Supports dict-like access: entry['lines'], entry['start_ms'], etc.
    """
    idx: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def timecode(self) -> str:
        return f"{ms_to_tc(self.start_ms)} --> {ms_to_tc(self.end_ms)}"

    @timecode.setter
    def timecode(self, value: str):
        match = _TS_RE.match(value)
        if match:
            self.start_ms = tc_to_ms(match.group(1))
            self.end_ms = tc_to_ms(match.group(2))
        else:
            s, e = parse_tc(value)
            self.start_ms = tc_to_ms(s)
            self.end_ms = tc_to_ms(e)

    @property
    def lines(self) -> list[str]:
        if '\n' in self.text:
            return self.text.split('\n')
        return [self.text]

    @lines.setter
    def lines(self, value: list[str]):
        self.text = '\n'.join(value)

    @property
    def index(self) -> int:
        """Alias for idx (backward compat with speech-to-subtitle)."""
        return self.idx

    @index.setter
    def index(self, value: int):
        self.idx = value

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    # ─── Dict-like access for backward compatibility ───

    _KEYS = frozenset({'idx', 'index', 'start_ms', 'end_ms', 'text', 'timecode', 'lines'})

    def keys(self):
        return ('idx', 'index', 'start_ms', 'end_ms', 'text', 'timecode', 'lines')

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return key in self._KEYS

    def __getitem__(self, key: str):
        if key == 'idx' or key == 'index':
            return self.idx
        if key == 'start_ms':
            return self.start_ms
        if key == 'end_ms':
            return self.end_ms
        if key == 'text':
            return self.text
        if key == 'timecode':
            return self.timecode
        if key == 'lines':
            return self.lines
        raise KeyError(key)

    def __setitem__(self, key: str, value):
        if key == 'idx' or key == 'index':
            self.idx = value
        elif key == 'start_ms':
            self.start_ms = value
        elif key == 'end_ms':
            self.end_ms = value
        elif key == 'text':
            self.text = value
        elif key == 'timecode':
            self.timecode = value
        elif key == 'lines':
            self.lines = value
        else:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._KEYS

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# ─── Convenience functions on entries (backward compat) ───

def start_ms(entry) -> int:
    """Get entry start time in ms"""
    if isinstance(entry, SrtEntry):
        return entry.start_ms
    if 'start_ms' in entry:
        return entry['start_ms']
    s, _ = parse_tc(entry['timecode'])
    return tc_to_ms(s)


def end_ms(entry) -> int:
    """Get entry end time in ms"""
    if isinstance(entry, SrtEntry):
        return entry.end_ms
    if 'end_ms' in entry:
        return entry['end_ms']
    _, e = parse_tc(entry['timecode'])
    return tc_to_ms(e)


def duration_ms(entry) -> int:
    """Get entry duration in ms"""
    return end_ms(entry) - start_ms(entry)


# ─── SRT parsing / serialization ───

def parse_srt(content: str) -> list[SrtEntry]:
    """Parse SRT string into list of SrtEntry."""
    entries: list[SrtEntry] = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_match = _TS_RE.match(lines[1].strip())
        if not time_match:
            continue
        text = '\n'.join(l for l in lines[2:] if l.strip())
        entries.append(SrtEntry(
            idx=idx,
            start_ms=tc_to_ms(time_match.group(1)),
            end_ms=tc_to_ms(time_match.group(2)),
            text=text,
        ))
    return entries


def parse_srt_file(path: str) -> list[SrtEntry]:
    """Read and parse SRT file."""
    with open(path, encoding='utf-8-sig') as f:
        return parse_srt(f.read())


def build_srt(entries: list[SrtEntry]) -> str:
    """Serialize entry list to SRT string (auto-renumbered)."""
    parts: list[str] = []
    for i, e in enumerate(entries, 1):
        parts.append(str(i))
        if isinstance(e, SrtEntry):
            parts.append(e.timecode)
        elif 'timecode' in e:
            parts.append(e['timecode'])
        else:
            parts.append(f"{ms_to_tc(e['start_ms'])} --> {ms_to_tc(e['end_ms'])}")
        if isinstance(e, SrtEntry):
            parts.extend(e.lines)
        else:
            parts.extend(e.get('lines', [e.get('text', '')]))
        parts.append('')
    return '\n'.join(parts) + '\n'


def write_srt(path: str, entries: list[SrtEntry]) -> None:
    """Serialize and write SRT file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(build_srt(entries))


# ─── Aliases (backward compat with srt_utils names) ───

read_srt = parse_srt_file
serialize_srt = build_srt


# ─── Entry operations ───

def split_entry(entry: SrtEntry, split_pos: int) -> tuple[SrtEntry, SrtEntry]:
    """Split entry at character position. Time allocated by char ratio."""
    text = get_text(entry)
    t1, t2 = text[:split_pos], text[split_pos:]
    s = start_ms(entry)
    e = end_ms(entry)
    dur = e - s
    total = len(text)
    mid = s + int(len(t1) / total * dur) if total > 0 else s

    # 切割後修剪兩端空白：避免「在 AI ...」被切成「在」+「 AI ...」時殘留前導空格
    t1 = t1.strip()
    t2 = t2.strip()
    e1 = SrtEntry(idx=0, start_ms=s, end_ms=mid, text=t1)
    e2 = SrtEntry(idx=0, start_ms=mid, end_ms=e, text=t2)
    return e1, e2


def split_two_line_entry(entry: SrtEntry) -> list[SrtEntry]:
    """Split a two-line entry into two single-line entries by char ratio."""
    lines = entry.lines
    if len(lines) <= 1:
        return [entry]
    l1, l2 = lines[0], lines[1]
    if not l1.strip() or not l2.strip():
        return [entry]
    if is_en(l1) and is_en(l2):
        return [entry]

    s = entry.start_ms
    e = entry.end_ms
    dur = e - s
    total = len(l1) + len(l2)
    mid = s + int(len(l1) / total * dur) if total > 0 else s

    return [
        SrtEntry(idx=0, start_ms=s, end_ms=mid, text=l1),
        SrtEntry(idx=0, start_ms=mid, end_ms=e, text=l2),
    ]


def merge_entries(e1: SrtEntry, e2: SrtEntry) -> SrtEntry:
    """Merge two entries (take e1 start, e2 end, concatenate text)."""
    return SrtEntry(
        idx=0,
        start_ms=start_ms(e1),
        end_ms=end_ms(e2),
        text=get_text(e1) + get_text(e2),
    )


# ─── Gate mechanism ───

class GateError(Exception):
    """Gate check failed, pipeline halts"""
    def __init__(self, step: str, failures: list[str]):
        self.step = step
        self.failures = failures
        super().__init__(f"[{step}] Gate failed: {'; '.join(failures)}")


def run_gate(step_name: str, checks: list[tuple[str, bool, str]]) -> None:
    """Execute gate checks. checks: [(name, passed, fail_msg), ...]"""
    failures = [msg for _, passed, msg in checks if not passed]
    if failures:
        raise GateError(step_name, failures)
    print(f"  ► GATE [{step_name}]: 全部通過 ({len(checks)} 項檢查)")


# ─── Exports ───

__all__ = [
    'SrtEntry',
    'tc_to_ms', 'ms_to_tc', 'parse_tc', 'merge_tc',
    'start_ms', 'end_ms', 'duration_ms',
    'is_cjk', 'is_en', 'get_text', 'cjk_len', 'char_len',
    'parse_srt', 'parse_srt_file', 'build_srt', 'write_srt',
    'read_srt', 'serialize_srt',
    'split_entry', 'split_two_line_entry', 'merge_entries',
    'GateError', 'run_gate',
]
