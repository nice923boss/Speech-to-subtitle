# Speech-to-subtitle 語音轉字幕

把影片或音檔批次轉成 SRT 字幕的 Windows 桌面工具。以 faster-whisper（Whisper large-v3）轉錄，再用 Montreal Forced Aligner（MFA）或音訊能量校正時間軸，中文輸出經 OpenCC 轉成臺灣正體。

## 功能

- **批次佇列**：一次加入多個影片或音檔，依序轉錄。支援 mp4、mkv、avi、mov、wmv、flv、webm、mp3、wav、flac、aac、ogg、m4a、wma。
- **語言**：自動偵測（中英混合建議用這個），或指定中、英、日、韓、法、德、西、葡、俄、泰、越、阿拉伯文。
- **重複迴圈救援**：偵測 Whisper 在音樂、歌唱段落反覆輸出同一句的情形，只把那一段重新解碼。
- **時間軸校正**：有 MFA 時做詞級強制對齊；沒有 MFA 時，用音訊能量把字幕開頭對到實際開口的時間。
- **簡轉繁閘門**：偵測簡體殘留並以 OpenCC s2tw 轉換，SRT 與詞級時間檔的文字保持一致。
- **單檔獨立子行程**：每個檔案在自己的子行程轉錄，CUDA 原生崩潰只影響該檔，批次繼續跑下一個。
- **輸出**：字幕寫在原檔旁邊，`<檔名>.srt` 加詞級時間檔 `<檔名>.words.json`。覆寫既有 SRT 前會先留一份 `.bak`。

## 系統需求

| 項目 | 必要性 | 說明 |
|---|---|---|
| Windows 10／11 | 必要 | 啟動與安裝都是 Windows 批次檔 |
| NVIDIA 顯示卡與驅動程式 | 必要 | 轉錄只走 CUDA，沒有 CPU 模式 |
| Python 3.12 | 必要 | 找不到時，安裝腳本可用 winget 代裝 |
| Whisper 模型 | 必要 | 安裝腳本可代為下載 large-v3（約 3 GB） |
| ffmpeg | 建議 | MFA 對齊需要；沒有時改用能量對齊 |
| MFA | 選用 | 獨立 conda 環境，數 GB，需手動安裝（見下方） |

## 安裝

1. 下載程式：

   ```bash
   git clone https://github.com/nice923boss/Speech-to-subtitle.git
   ```

   或在 GitHub 頁面按 **Code → Download ZIP** 後解壓縮。

2. 雙擊資料夾裡的 `安裝依賴.bat`，它會依序：
   - 找 Python 3.12（`%LOCALAPPDATA%\Programs\Python\Python312` 或 `py -3.12`），找不到時詢問是否用 winget 安裝
   - 從這個 Python 複製啟動檔（`pythonw_speech2sub.exe`、`python312.dll`、`python3.dll`）並寫入 `pyvenv.cfg`
   - 安裝 `requirements.txt`；偵測到 NVIDIA 驅動程式時加裝 `requirements-gpu.txt`
   - 缺 ffmpeg 或 Whisper 模型時，詢問是否安裝或下載
   - 列出 CUDA、模型、ffmpeg、OpenCC、MFA 的檢查結果

3. 看到「可以執行 start.bat 開啟 App」即完成。

### 選用：安裝 MFA（詞級對齊）

App 會到 `%USERPROFILE%\miniforge3` 找名為 `mfa` 的 conda 環境：

1. 安裝 [Miniforge](https://conda-forge.org/download/) 到 `%USERPROFILE%\miniforge3`
2. `mamba create -n mfa -c conda-forge montreal-forced-aligner`
3. `mamba run -n mfa mfa model download acoustic mandarin_mfa`
4. `mamba run -n mfa mfa model download dictionary mandarin_taiwan_mfa`

## 使用方式

1. 執行 `start.bat` 開啟視窗（已開啟時不會開第二個）。
2. 按「新增檔案」加入影片或音檔。
3. 選模型（有 large-v3 時預設選它）。模型放在其他資料夾時，按「掃描資料夾」指定。
4. 選語言，按「開始批次轉錄」。
5. 結束：直接關視窗；卡住時執行 `stop.bat` 強制結束。

App 會在這些位置找 Whisper 模型：Hugging Face 快取、`%USERPROFILE%\.cache\whisper`、`%LOCALAPPDATA%\whisper`、程式資料夾下的 `models\`、環境變數 `WHISPER_MODEL_DIR`，以及按「掃描資料夾」加入過的資料夾。

## 命令列與其他工具

`transcribe_worker.py` 是給其他程式呼叫的轉錄介面：讀一個 JSON 工作檔，stdout 每行輸出一個 JSON 進度。完整規格見 [docs/cli-contract.md](docs/cli-contract.md)。

```bash
python transcribe_worker.py --check
```

```bash
python transcribe_worker.py job.json
```

其他獨立腳本，用法寫在各檔開頭：

- `convert_srt_s2tw.py`：把既有 SRT 與 `.words.json` 批次轉成臺灣正體（`--dry-run` 只掃描不改檔）。
- `audio_verify.py`、`audio_energy_verify.py`、`start_accuracy_verify.py`：比對字幕時間軸與音訊的驗證工具。

## 設定與記錄檔

都放在 `%USERPROFILE%\.speech-to-subtitle\`：

- `config.json`：掃描過的模型資料夾
- `logs\`：每日轉錄記錄
- `app.log`：啟動階段的錯誤

## 測試

用已安裝依賴的 Python 3.12 執行：

```bash
python -m pytest tests
```

## 已知限制

- 實機驗收只跑過 30 秒中文片段。30 分鐘以上長片的 MFA 耗時、暫存空間，以及 GPU 記憶體不足時的行為未實測。
- 驗收素材只有 MP4（AAC）。其他格式若整檔解碼失敗，會退回能量對齊。
- `安裝依賴.bat` 只在已裝好依賴的電腦跑過，新電腦首次安裝（安裝 Python、下載模型）的流程未實測。
- 未安裝 cuDNN 套件，依賴 ctranslate2 內附的 `cudnn64_9.dll`。ctranslate2 版本不同時，CUDA 可能載入失敗。
- MFA 詞級時間與 Whisper 句界各自計算，邊界詞的時間可能跨出字幕約 0.1 秒。
- 語言指定「中文」時，英文講者的段落會被略過；中英混合內容請用「自動偵測」。
- MFA 冷啟動檢查超過 120 秒時，該次轉錄會退回能量對齊。
- OpenCC s2tw 會把「台灣」轉成「臺灣」。
- `relink_launcher.bat` 假設 Python 裝在 `%LOCALAPPDATA%\Programs\Python\Python312`；裝在其他位置請改用 `安裝依賴.bat`。
- `安裝依賴.bat` 以 Big5（CP950）存檔，讓繁中 Windows 的 cmd 正確顯示中文。在 GitHub 網頁或非繁中系統上中文會顯示成亂碼，不影響執行。

## 授權

[MIT](LICENSE)。使用的第三方套件（faster-whisper、CTranslate2、OpenCC、Montreal Forced Aligner 等）依其各自授權。
