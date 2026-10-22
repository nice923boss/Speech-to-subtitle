"""s2tw_gate against the real OpenCC tables. Skipped when OpenCC is missing."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s2tw_gate as g  # noqa: E402

pytestmark = pytest.mark.skipif(not g.AVAILABLE, reason="OpenCC not installed")

# None means the text must come back untouched.
CASES = [
    ("頭发", "頭髮"),          # phrase table must win over the 發 default
    ("复雜", "複雜"),
    ("面条", "麵條"),
    ("一只猫", "一隻貓"),
    ("日历", "日曆"),
    ("獲得冠军", "獲得冠軍"),
    ("收获", "收穫"),
    ("头发", "頭髮"),
    ("干净", "乾淨"),
    ("台北", "臺北"),
    ("当兵了然后", "當兵了然後"),
    ("为了解决这个问题", "為了解決這個問題"),
    ("了解这個", "了解這個"),  # 了解 is already correct, must not become 瞭解
    ("了解决", "了解決"),
    ("當兵了然後", None),       # all traditional, s2tw would give 瞭然
    ("萬里長城", None),         # merge characters correct where they stand
    ("里長伯", None),
    ("岳飛", None),
    ("干擾", None),
    ("濃郁", None),
    ("夢里", "夢裡"),           # merge characters genuinely simplified
    ("干淨", "乾淨"),
    ("后面", "後面"),
    # Known limits: no simplified evidence, so these stay as typed.
    ("吃面條", None),
    ("一只貓", None),
]


@pytest.mark.parametrize("text,want", CASES)
def test_residue(text, want):
    assert g.residue(text) == want


def test_convert_guarded_identity_on_clean_text():
    assert g.convert_guarded("然後就當兵了然後") == "然後就當兵了然後"


def test_sync_words_follows_cues():
    cues = [{"text": "夢裡的頭髮"}]
    words = [{"word": "夢"}, {"word": "里"}, {"word": "的"}, {"word": "頭"},
             {"word": "发"}]
    assert g.sync_words_to_cues(cues, words) == 2
    assert "".join(w["word"] for w in words) == "夢裡的頭髮"


def test_sync_words_ignores_insertions():
    cues = [{"text": "你好，世界"}]
    words = [{"word": "你好"}, {"word": "世界"}]
    assert g.sync_words_to_cues(cues, words) == 0
    assert [w["word"] for w in words] == ["你好", "世界"]
