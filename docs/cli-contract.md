# Speech-to-subtitle 外部呼叫契約（protocol 1）

[Doc] 外部程式呼叫 Speech-to-subtitle 的唯一介面是 `transcribe_worker.py`。
`transcribe_cli.py` 是舊的手動用 CLI，模型挑選規則不同（挑檔案最大的），不要拿來整合。

## 為什麼是子行程

CTranslate2／cuDNN 在 Windows 偶發原生例外直接 abort()，Python 接不到。
每次轉錄一個子行程，崩潰只死子行程，呼叫端看 exit code 與最後一行 JSON 即可判斷。

## 執行方式

用 Speech-to-subtitle 的 Python（`pyvenv.cfg` 的 `home`，目前是 Python312）執行：

```
<python.exe> <STS 資料夾>\transcribe_worker.py <job.json>
<python.exe> <STS 資料夾>\transcribe_worker.py --check
```

- 工作目錄不限，worker 自行把所在資料夾加入 `sys.path`。
- 不要用 `pythonw_speech2sub.exe`：STS 的 stop.bat 以映像名稱強殺，會連帶殺掉外部呼叫的轉錄。
- 環境變數建議加 `PYTHONIOENCODING=utf-8`（worker 本身已 reconfigure stdout 為 UTF-8）。
- 取消：`taskkill /T /F /PID <pid>`。worker 會再起 MFA（mamba、mfa）子行程，只殺父行程會留下孤兒。

## job.json 欄位（TranscribeOptions）

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `input_file` | str | 是 | 音檔或影片路徑，faster-whisper 以 PyAV 解碼 |
| `output_srt` | str | 是 | 輸出 SRT 路徑；空字串時寫在 input 旁同名 `.srt` |
| `model_path` | str | 否 | 省略時用 GUI 相同的預設（large-v3 優先，掃描 HF 快取、STS config 的 `extra_scan_dirs`／`whisper_model_dir`、`WHISPER_MODEL_DIR`） |
| `language` | str 或 null | 否 | null 為自動偵測。外部片段建議 null：強制 `zh` 會跳過英文講者（spike 0 實測） |
| `device` | str | 否 | 固定 `cuda` |
| `compute_type` | str | 否 | 預設 `float16` |
| `initial_prompt` | str 或 null | 否 | 領域關鍵字 |
| `use_mfa` | bool | 否 | 預設 true；MFA 不可用時改用能量對齊，並在 progress 訊息說明原因 |

未知欄位會讓 worker 以 `{"t":"error"}` 結束。

## stdout（每行一個 JSON，UTF-8）

```
{"t":"progress","phase":"loading|transcribing|writing","percent":0-100,"message":"...","segment_count":N}
{"t":"done","path":"<srt>","words":"<srt 去副檔名>.words.json"}
{"t":"error","message":"<ExceptionType>: <訊息>"}
```

- 成功：最後一行是 `done`，exit 0。
- Python 層失敗：最後一行是 `error`，exit 1。
- 原生崩潰：沒有 `done` 也沒有 `error`，exit code 非 0（常見 0xC0000409）。呼叫端要把「沒有 done」一律當失敗。
- 輸出檔以暫存檔加 `os.replace` 寫入；`output_srt` 已存在時先複製成 `<srt>.bak`（一代）。

### words.json

`[{"word": "...", "start": 秒, "end": 秒}, ...]`，文字已過簡轉繁閘門並與 SRT 字元一致，逗號已移除。

### SRT

標準 SRT，時間碼 `HH:MM:SS,mmm`，字幕文字無逗號。時間相對於 `input_file` 開頭。

## --check

印一行：

```
{"t":"check","protocol":1,
 "cuda":{"ok":true,"devices":1},
 "model":{"ok":true,"path":"..."},
 "ffmpeg":{"ok":true,"path":"..."},
 "mfa":{"ok":true,"message":"MFA 3.3.9"},
 "opencc":{"ok":true,"message":"簡轉繁：已啟用（OpenCC s2tw）"},
 "ok":true}
```

- `ok` 只看 `cuda` 與 `model`。MFA、ffmpeg、OpenCC 缺少時仍可轉錄，但品質降級（能量對齊、可能殘留簡體）。
- MFA 冷啟動檢查可達約 31 秒（spike 0 實測），呼叫端逾時請設 180 秒以上。
- exit 0 表示可轉錄，1 表示不可。

## 版本規則

`protocol` 只在既有欄位改變意義或被移除時遞增；新增欄位不遞增。呼叫端遇到不認得的 protocol 應拒絕呼叫並提示更新。
