"""Batch-convert existing SRT files (and their .words.json sidecars) to Taiwan
traditional.

For files produced before transcriber.py gained the output gate. The sidecar is
converted in the same pass on purpose: Quick subtitles aligns subtitle text
against the sidecar characters, so converting only one side would break the
character match and degrade the alignment.

Every file is backed up to `<name>.bak` before it is rewritten (an existing .bak
is never overwritten, so re-running the tool cannot destroy the original).

Usage:
    python convert_srt_s2tw.py <folder-or-file> [...] [--dry-run] [--no-backup]
                               [--report out.json]
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

import s2tw_gate  # noqa: E402


def collect_srt_files(targets):
    """Expand folders recursively into a sorted list of .srt paths."""
    found = []
    for target in targets:
        if os.path.isfile(target):
            if target.lower().endswith('.srt'):
                found.append(os.path.abspath(target))
        elif os.path.isdir(target):
            for root, _dirs, files in os.walk(target):
                for name in files:
                    if name.lower().endswith('.srt'):
                        found.append(os.path.abspath(os.path.join(root, name)))
        else:
            print(f'略過（找不到）: {target}')
    return sorted(set(found))


def backup(path):
    """Copy `path` to `path`.bak unless a backup already exists."""
    bak = path + '.bak'
    if os.path.exists(bak):
        return bak, False
    with open(path, 'rb') as src, open(bak, 'wb') as dst:
        dst.write(src.read())
    return bak, True


def convert_text_file(path, dry_run, make_backup):
    """Convert an SRT (or any UTF-8 text file) in place. Returns a result dict."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    hits = s2tw_gate.scan_text(content)
    result = {'path': path, 'residue_lines': len(hits), 'changed': False,
              'backup': None, 'samples': [h[1][:40] for h in hits[:3]],
              'text': content}
    if not hits or dry_run:
        return result

    if make_backup:
        result['backup'] = backup(path)[0]
    # Rewrite only the lines that hold residue. Converting the whole file would
    # also apply s2tw's Taiwan variant preferences to lines that are already
    # correct (了然 -> 瞭然), i.e. change text nobody asked to change.
    lines = content.split('\n')
    for line_no, _original, fixed in hits:
        lines[line_no - 1] = fixed
    result['text'] = '\n'.join(lines)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(result['text'])
    result['changed'] = True
    return result


def cue_texts(content):
    """The text lines of an SRT, with the index and timing lines dropped."""
    out = []
    for block in content.replace('\r\n', '\n').strip().split('\n\n'):
        lines = block.split('\n')
        if len(lines) > 2:
            out.append('\n'.join(lines[2:]))
    return out


def convert_words_sidecar(path, dry_run, make_backup, srt_text):
    """Convert the `word` field of a .words.json sidecar. Returns a result dict."""
    with open(path, 'r', encoding='utf-8') as f:
        words = json.load(f)

    changed_words = 0
    for w in words:
        fixed = s2tw_gate.residue(w.get('word', ''))
        if fixed is not None:
            w['word'] = fixed
            changed_words += 1

    # A word judged on its own has no context, so it cannot resolve a merge
    # character the way the cue containing it can. Without this the tool would
    # leave the sidecar disagreeing with the SRT it just converted, which is the
    # drift it exists to repair.
    cues = [{'text': t} for t in cue_texts(srt_text)]
    synced = s2tw_gate.sync_words_to_cues(cues, words)

    result = {'path': path, 'residue_words': changed_words, 'synced': synced,
              'changed': False, 'backup': None}
    if not (changed_words or synced) or dry_run:
        return result

    if make_backup:
        result['backup'] = backup(path)[0]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False)
    result['changed'] = True
    return result


def main():
    p = argparse.ArgumentParser(
        description='批次把既有 SRT 與 .words.json 轉成臺灣正體（OpenCC s2tw）')
    p.add_argument('targets', nargs='+', help='資料夾或 .srt 檔（資料夾會遞迴）')
    p.add_argument('--dry-run', action='store_true', help='只掃描不改檔')
    p.add_argument('--no-backup', action='store_true', help='不產生 .bak 備份')
    p.add_argument('--report', default=None, help='把明細寫成 JSON')
    args = p.parse_args()

    if not s2tw_gate.AVAILABLE:
        print(s2tw_gate.status_line())
        print('無法轉換，請先安裝 opencc：python -m pip install opencc-python-reimplemented')
        sys.exit(1)

    srt_files = collect_srt_files(args.targets)
    if not srt_files:
        print('找不到任何 .srt 檔')
        sys.exit(1)

    print(f'掃描 {len(srt_files)} 個 SRT'
          + ('（dry-run，不寫檔）' if args.dry_run else ''))

    results = {'srt': [], 'words': []}
    srt_changed = words_changed = 0
    for srt_path in srt_files:
        r = convert_text_file(srt_path, args.dry_run, not args.no_backup)
        results['srt'].append(r)
        if r['residue_lines']:
            srt_changed += 1
            state = '待轉' if args.dry_run else '已轉'
            print(f"  [{state}] {r['residue_lines']:3d} 行  {srt_path}")
            for s in r['samples']:
                print(f'          例: {s}')

        sidecar = os.path.splitext(srt_path)[0] + '.words.json'
        if os.path.isfile(sidecar):
            w = convert_words_sidecar(sidecar, args.dry_run, not args.no_backup,
                                      r['text'])
            results['words'].append(w)
            if w['residue_words'] or w['synced']:
                words_changed += 1
                state = '待轉' if args.dry_run else '已轉'
                print(f"  [{state}] {w['residue_words']:3d} 詞"
                      f"（另對齊 SRT {w['synced']} 字）  {sidecar}")

    print(f'\n完成：SRT {srt_changed}/{len(srt_files)} 有簡體殘留，'
          f"sidecar {words_changed}/{len(results['words'])} 有簡體殘留")
    if not args.dry_run and (srt_changed or words_changed):
        print('原檔已備份為同名 .bak' if not args.no_backup else '未建立備份（--no-backup）')

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'明細: {args.report}')


if __name__ == '__main__':
    main()
