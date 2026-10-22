"""Speech-to-Subtitle: GUI application for Whisper-based transcription."""

# ── pythonw stdout/stderr guard (must be FIRST) ───────────────────────
import sys
import os
from pathlib import Path

if sys.stdout is None or sys.stderr is None:
    _log_dir = os.path.join(str(Path.home()), ".speech-to-subtitle")
    os.makedirs(_log_dir, exist_ok=True)
    _log = open(
        os.path.join(_log_dir, "app.log"),
        "a", buffering=1, encoding="utf-8",
    )
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log
# ──────────────────────────────────────────────────────────────────────

import subprocess
import threading
import json
import time
from tkinter import filedialog, messagebox

import customtkinter as ctk

import s2tw_gate
from scanner import (scan_whisper_models, find_ffmpeg, WhisperModel,
                     configured_scan_dirs, preferred_model)

# Transcription runs in a child process (transcribe_worker.py), so this process
# never loads CUDA at all — see _run_worker for why.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(APP_DIR, "transcribe_worker.py")
INSTALLER_BAT = os.path.join(APP_DIR, "安裝依賴.bat")

# ── Config persistence ──────────────────────────────────────────────

CONFIG_DIR = os.path.join(str(Path.home()), ".speech-to-subtitle")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
JOB_FILE = os.path.join(CONFIG_DIR, "job.json")
LOG_DIR = os.path.join(CONFIG_DIR, "logs")

# A worker error with one of these means the GPU stack itself is broken, so
# every later file in the batch would fail the same way after a model load.
FATAL_WORKER_ERRORS = ("cuda", "cudnn", "cublas", "failed to load model")

SUPPORTED_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",  # video
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",  # audio
)

LANGUAGES = [
    ("自動偵測（中英混合推薦）", None),
    ("中文 (zh)", "zh"),
    ("英文 (en)", "en"),
    ("日文 (ja)", "ja"),
    ("韓文 (ko)", "ko"),
    ("法文 (fr)", "fr"),
    ("德文 (de)", "de"),
    ("西班牙文 (es)", "es"),
    ("葡萄牙文 (pt)", "pt"),
    ("俄文 (ru)", "ru"),
    ("泰文 (th)", "th"),
    ("越南文 (vi)", "vi"),
    ("阿拉伯文 (ar)", "ar"),
]

AUTHOR = "黃政文"


def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ── Main Application ────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(WINDOW_TITLE)
        self.geometry("800x700")
        self.minsize(700, 620)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config_data = _load_config()
        self.models: list[WhisperModel] = []
        self.ffmpeg_path: str | None = None
        self.is_running = False
        self.cancel_requested = False
        self.file_queue: list[str] = []
        self._worker_proc: subprocess.Popen | None = None
        self._log_file = None
        self._log_error = None
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self._log_file = open(
                os.path.join(LOG_DIR, time.strftime("%Y-%m-%d") + ".log"),
                "a", buffering=1, encoding="utf-8")
        except OSError as e:
            self._log_error = f"無法建立記錄檔（{LOG_DIR}）：{e}"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if self._log_error:
            self._log(self._log_error)
        self._log(s2tw_gate.status_line())
        if not s2tw_gate.AVAILABLE:
            self.after(100, lambda: messagebox.showwarning(
                "簡轉繁未啟用",
                f"{s2tw_gate.UNAVAILABLE_WARNING}\n\n"
                f"原因：{s2tw_gate.IMPORT_ERROR}\n"
                "安裝方式：按「安裝依賴」（或執行資料夾裡的 安裝依賴.bat）",
            ))
        self._initial_scan()

    # ── UI Construction ──

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # ── Row 0: File queue ──
        frame_files = ctk.CTkFrame(self)
        frame_files.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="nsew")
        frame_files.grid_columnconfigure(0, weight=1)
        frame_files.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Header with label and buttons
        frame_file_header = ctk.CTkFrame(frame_files, fg_color="transparent")
        frame_file_header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        frame_file_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame_file_header, text="檔案佇列：").grid(
            row=0, column=0, padx=4, pady=4, sticky="w"
        )
        ctk.CTkButton(
            frame_file_header, text="新增檔案", width=90,
            command=self._browse_add_files,
        ).grid(row=0, column=1, padx=4, pady=4)
        ctk.CTkButton(
            frame_file_header, text="移除選取", width=90,
            command=self._remove_selected,
        ).grid(row=0, column=2, padx=4, pady=4)
        ctk.CTkButton(
            frame_file_header, text="清空全部", width=90,
            command=self._clear_queue,
        ).grid(row=0, column=3, padx=(4, 8), pady=4)

        # File list (using Textbox as a selectable list)
        self.file_listbox = ctk.CTkTextbox(
            frame_files, height=140,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.file_listbox.grid(row=1, column=0, padx=8, pady=(4, 8), sticky="nsew")
        self.file_listbox.configure(state="disabled")

        # ── Row 1: Model selection ──
        frame_model = ctk.CTkFrame(self)
        frame_model.grid(row=1, column=0, padx=12, pady=4, sticky="ew")
        frame_model.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame_model, text="模型：").grid(row=0, column=0, padx=8, pady=8)
        self.combo_model = ctk.CTkComboBox(
            frame_model, values=["掃描中..."], state="readonly", width=350,
        )
        self.combo_model.grid(row=0, column=1, padx=4, pady=8, sticky="ew")
        ctk.CTkButton(
            frame_model, text="掃描資料夾", width=100,
            command=self._scan_custom_folder,
        ).grid(row=0, column=2, padx=8, pady=8)
        ctk.CTkButton(
            frame_model, text="安裝依賴", width=90,
            command=self._open_installer,
        ).grid(row=0, column=3, padx=(0, 8), pady=8)

        # ── Row 2: Options ──
        frame_opts = ctk.CTkFrame(self)
        frame_opts.grid(row=2, column=0, padx=12, pady=4, sticky="ew")
        frame_opts.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(frame_opts, text="語言：").grid(row=0, column=0, padx=8, pady=8)
        lang_names = [name for name, _ in LANGUAGES]
        self.combo_lang = ctk.CTkComboBox(
            frame_opts, values=lang_names, state="readonly", width=180,
        )
        self.combo_lang.set(lang_names[0])
        self.combo_lang.grid(row=0, column=1, padx=4, pady=8)

        ctk.CTkLabel(frame_opts, text="裝置：").grid(row=0, column=2, padx=(16, 4), pady=8)
        self.label_device = ctk.CTkLabel(
            frame_opts, text="GPU (CUDA)", text_color="green",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.label_device.grid(row=0, column=3, padx=4, pady=8, sticky="w")

        self.label_ffmpeg = ctk.CTkLabel(
            frame_opts, text="ffmpeg: 偵測中...", text_color="gray",
        )
        self.label_ffmpeg.grid(row=0, column=4, padx=16, pady=8, sticky="e")

        # Simplified->traditional conversion status. Missing OpenCC must be
        # visible before a batch runs, not discovered in the finished subtitles.
        self.label_s2tw = ctk.CTkLabel(
            frame_opts,
            text=("簡轉繁：已啟用（OpenCC s2tw）" if s2tw_gate.AVAILABLE
                  else s2tw_gate.UNAVAILABLE_WARNING),
            text_color="green" if s2tw_gate.AVAILABLE else "#e74c3c",
            font=ctk.CTkFont(size=13, weight="normal" if s2tw_gate.AVAILABLE else "bold"),
        )
        self.label_s2tw.grid(row=1, column=0, columnspan=5, padx=8,
                             pady=(0, 8), sticky="w")

        # ── Row 3: Start / Cancel buttons ──
        frame_btns = ctk.CTkFrame(self, fg_color="transparent")
        frame_btns.grid(row=3, column=0, padx=12, pady=8, sticky="ew")
        frame_btns.grid_columnconfigure(0, weight=1)

        self.btn_start = ctk.CTkButton(
            frame_btns, text="開始批次轉錄", height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_transcription,
        )
        self.btn_start.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.btn_cancel = ctk.CTkButton(
            frame_btns, text="取消", height=40, width=100,
            fg_color="#c0392b", hover_color="#e74c3c",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._request_cancel, state="disabled",
        )
        self.btn_cancel.grid(row=0, column=1, padx=(4, 0))

        # ── Row 4: Progress bar + label ──
        frame_prog = ctk.CTkFrame(self, fg_color="transparent")
        frame_prog.grid(row=4, column=0, padx=12, pady=(0, 4), sticky="ew")
        frame_prog.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(frame_prog)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress.set(0)

        self.label_progress = ctk.CTkLabel(
            frame_prog, text="", text_color="gray",
            font=ctk.CTkFont(size=12),
        )
        self.label_progress.grid(row=0, column=1, padx=(8, 0))

        # ── Row 5: Log area ──
        self.log_box = ctk.CTkTextbox(
            self, height=160,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_box.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="nsew")
        self.grid_rowconfigure(5, weight=1)

        # ── Row 6: Author label ──
        self.label_author = ctk.CTkLabel(
            self, text=f"製作者：{AUTHOR}",
            text_color="gray", font=ctk.CTkFont(size=12),
        )
        self.label_author.grid(row=6, column=0, padx=12, pady=(0, 8), sticky="e")

    # ── File Queue Management ──

    def _refresh_file_list(self):
        """Redraw the file list display from self.file_queue."""
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", "end")
        for i, path in enumerate(self.file_queue):
            basename = os.path.basename(path)
            line = f"{i + 1}. {basename}\n"
            self.file_listbox.insert("end", line)
        self.file_listbox.configure(state="disabled")

    def _browse_add_files(self):
        """Open multi-file dialog and add selected files to queue."""
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        paths = filedialog.askopenfilenames(
            title="選擇影片或音訊檔案（可多選）",
            filetypes=[("媒體檔案", exts), ("所有檔案", "*.*")],
        )
        if not paths:
            return
        existing = set(self.file_queue)
        added = 0
        for p in paths:
            if p not in existing:
                self.file_queue.append(p)
                existing.add(p)
                added += 1
        self._refresh_file_list()
        if added:
            self._log(f"已加入 {added} 個檔案，佇列共 {len(self.file_queue)} 個。")

    def _remove_selected(self):
        """Remove files based on cursor selection in the textbox."""
        if not self.file_queue:
            return
        try:
            sel_start = self.file_listbox.index("sel.first")
            sel_end = self.file_listbox.index("sel.last")
            start_line = int(sel_start.split(".")[0]) - 1
            end_line = int(sel_end.split(".")[0])
            indices = list(range(start_line, min(end_line, len(self.file_queue))))
        except Exception:
            # No selection — remove last item
            if self.file_queue:
                indices = [len(self.file_queue) - 1]
            else:
                return
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.file_queue):
                removed = self.file_queue.pop(idx)
                self._log(f"已移除：{os.path.basename(removed)}")
        self._refresh_file_list()

    def _clear_queue(self):
        """Clear all files from the queue."""
        if not self.file_queue:
            return
        self.file_queue.clear()
        self._refresh_file_list()
        self._log("佇列已清空。")

    # ── Actions ──

    def _log(self, msg: str):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        if self._log_file:
            self._log_file.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def _scan_custom_folder(self):
        folder = filedialog.askdirectory(title="選擇要掃描 Whisper 模型的資料夾")
        if folder:
            self._log(f"正在掃描：{folder}")
            extra = [folder]
            prev = self.config_data.get("extra_scan_dirs", [])
            if folder not in prev:
                prev.append(folder)
                self.config_data["extra_scan_dirs"] = prev
                _save_config(self.config_data)
            self._run_model_scan(extra_dirs=extra)

    def _open_installer(self):
        try:
            os.startfile(INSTALLER_BAT)
        except OSError as e:
            messagebox.showerror("錯誤", f"無法開啟安裝程式：{INSTALLER_BAT}\n{e}")
            return
        self._log("已開啟「安裝依賴」視窗；裝完後請關閉並重開 App，新套件才會生效。")

    def _initial_scan(self):
        """Run initial scan for models and ffmpeg in a background thread."""
        def scan():
            self.ffmpeg_path = find_ffmpeg()
            if self.ffmpeg_path:
                ffmpeg_msg = f"ffmpeg 已找到：{self.ffmpeg_path}"
            else:
                ffmpeg_msg = "ffmpeg 未找到"
            self.after(0, lambda: self.label_ffmpeg.configure(
                text="ffmpeg: 已就緒" if self.ffmpeg_path else "ffmpeg: 未找到",
                text_color="green" if self.ffmpeg_path else "red",
            ))
            self.after(0, lambda: self._log(ffmpeg_msg))

            self._run_model_scan(extra_dirs=configured_scan_dirs(self.config_data))

        threading.Thread(target=scan, daemon=True).start()

    def _run_model_scan(self, extra_dirs: list[str] | None = None):
        """Scan for models and update the combo box."""
        def scan():
            self.after(0, lambda: self._log("正在掃描 Whisper 模型..."))
            found = scan_whisper_models(extra_dirs=extra_dirs)

            existing_paths = {m.path for m in self.models}
            for m in found:
                if m.path not in existing_paths:
                    self.models.append(m)
                    existing_paths.add(m.path)

            if self.models:
                names = [m.display_name() for m in self.models]
                self.after(0, lambda: self.combo_model.configure(values=names))
                # Same default the worker picks when no model is given
                default_idx = self.models.index(preferred_model(self.models))
                self.after(0, lambda: self.combo_model.set(names[default_idx]))
                self.after(0, lambda: self._log(f"找到 {len(self.models)} 個模型。"))
                for m in self.models:
                    self.after(0, lambda m=m: self._log(f"  - {m.display_name()} @ {m.path}"))
            else:
                self.after(0, lambda: self.combo_model.configure(values=["未找到模型"]))
                self.after(0, lambda: self.combo_model.set("未找到模型"))
                self.after(0, lambda: self._log(
                    "未找到 Whisper 模型。請使用「掃描資料夾」指定模型位置。"
                ))

        threading.Thread(target=scan, daemon=True).start()

    def _get_selected_model(self) -> WhisperModel | None:
        selected = self.combo_model.get()
        for m in self.models:
            if m.display_name() == selected:
                return m
        return None

    def _get_selected_language(self) -> str | None:
        selected = self.combo_lang.get()
        for name, code in LANGUAGES:
            if name == selected:
                return code
        return None

    def _request_cancel(self):
        """Stop the batch now: set the flag and kill the running worker tree."""
        self.cancel_requested = True
        self.btn_cancel.configure(state="disabled", text="取消中...")
        self._log("已要求取消，正在結束轉錄子行程。")
        self._kill_worker()

    def _kill_worker(self):
        """Kill the worker and everything under it (mamba, mfa, its workers).

        Killing only the worker would orphan the MFA tree, and an orphaned
        worker keeps the GPU and writes its SRT after the user gave up on it.
        """
        proc = self._worker_proc
        if proc is not None and proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)

    def _on_close(self):
        if self.is_running and not messagebox.askyesno(
                "轉錄進行中", "批次轉錄還在進行，關閉視窗會中止目前檔案。確定關閉？"):
            return
        self.cancel_requested = True
        self._kill_worker()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self.destroy()

    def _start_transcription(self):
        if self.is_running:
            return

        if not self.file_queue:
            messagebox.showerror("錯誤", "請先加入要轉錄的檔案。")
            return

        model = self._get_selected_model()
        if not model:
            messagebox.showerror("錯誤", "請選擇 Whisper 模型。\n可使用「掃描資料夾」尋找模型。")
            return

        # Validate all files exist
        missing = [f for f in self.file_queue if not os.path.isfile(f)]
        if missing:
            names = "\n".join(os.path.basename(f) for f in missing[:5])
            messagebox.showerror("錯誤", f"以下檔案不存在：\n{names}")
            return

        language = self._get_selected_language()
        device = "cuda"
        total = len(self.file_queue)

        self.is_running = True
        self.cancel_requested = False
        self.btn_start.configure(state="disabled", text="轉錄中...")
        self.btn_cancel.configure(state="normal", text="取消")
        self.progress.set(0)
        self.label_progress.configure(text=f"0 / {total}")

        # Snapshot the queue for this batch run
        batch_files = list(self.file_queue)

        def run():
            completed = 0
            failed = 0
            for idx, input_file in enumerate(batch_files):
                if self.cancel_requested:
                    self.after(0, lambda: self._log("\n批次轉錄已取消。"))
                    break

                basename = os.path.basename(input_file)
                output_srt = os.path.splitext(input_file)[0] + ".srt"
                file_num = idx + 1

                self.after(0, lambda fn=file_num, bn=basename: self._log(
                    f"\n{'='*50}\n"
                    f"[{fn}/{total}] {bn}\n"
                    f"{'='*50}"
                ))

                payload = {
                    "model_path": model.path,
                    "input_file": input_file,
                    "output_srt": output_srt,
                    "language": language,
                    "device": device,
                }

                try:
                    def on_progress(phase, percent, message, _idx=idx):
                        # Overall progress: completed files + current file fraction
                        file_fraction = percent / 100.0
                        overall = (_idx + file_fraction) / total
                        self.after(0, lambda o=overall: self.progress.set(o))
                        self.after(0, lambda: self.label_progress.configure(
                            text=f"{_idx} / {total}"
                        ))
                        self.after(0, lambda: self._log(f"  [{phase}] {message}"))

                    result_path = self._run_worker(payload, on_progress)
                    completed += 1
                    self.after(0, lambda rp=result_path: self._log(
                        f"  字幕檔已儲存：{rp}"
                    ))
                except Exception as e:
                    if self.cancel_requested:
                        self.after(0, lambda bn=basename: self._log(
                            f"  已中止：{bn}\n\n批次轉錄已取消。"))
                        break
                    failed += 1
                    self.after(0, lambda bn=basename, err=str(e): self._log(
                        f"  錯誤（{bn}）：{err}"
                    ))
                    if any(k in str(e).lower() for k in FATAL_WORKER_ERRORS):
                        self.after(0, lambda: self._log(
                            "\nGPU 或模型載入失敗，其餘檔案也會同樣失敗，批次停止。"))
                        break

            self.after(0, lambda: self._on_batch_done(completed, failed, total))

        threading.Thread(target=run, daemon=True).start()

    def _run_worker(self, payload: dict, on_progress) -> str:
        """Transcribe one file in a child process and return the SRT path.

        The Whisper native stack (CTranslate2 / cuDNN) can raise a C++ exception
        that never becomes a Python exception: the process is killed by abort()
        instead. That is what closed the whole window mid-batch — a try/except
        here could not have caught it. With the work in a child process, such a
        crash only kills the child and shows up as a non-zero exit code, which
        this method turns into an ordinary RuntimeError so the batch continues.
        """
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(JOB_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        proc = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, JOB_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._worker_proc = proc
        if self.cancel_requested:  # cancel landed between files
            self._kill_worker()

        # Drain stderr on its own thread: the worker's native libraries write
        # warnings there, and a full pipe would block it forever.
        stderr_tail = []

        def drain_stderr():
            for line in proc.stderr:
                stderr_tail.append(line.rstrip())
                del stderr_tail[:-20]

        threading.Thread(target=drain_stderr, daemon=True).start()

        result_path = None
        error_message = None
        # readline, not iteration: iterating a pipe uses a read-ahead buffer that
        # would hold progress lines back until it fills.
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("t")
            if kind == "progress":
                on_progress(msg["phase"], msg["percent"], msg["message"])
            elif kind == "done":
                result_path = msg["path"]
            elif kind == "error":
                error_message = msg["message"]

        code = proc.wait()
        self._worker_proc = None

        if error_message:
            raise RuntimeError(error_message)
        if result_path is None:
            tail = "\n".join(stderr_tail[-5:])
            raise RuntimeError(
                f"轉錄子進程異常結束（exit code {code} / 0x{code & 0xFFFFFFFF:08X}），"
                f"未產生字幕檔。"
                + (f"\n    子進程訊息：{tail}" if tail else "")
            )
        return result_path

    def _on_batch_done(self, completed: int, failed: int, total: int):
        self.is_running = False
        self.cancel_requested = False
        self.btn_start.configure(state="normal", text="開始批次轉錄")
        self.btn_cancel.configure(state="disabled", text="取消")
        self.progress.set(1.0 if failed == 0 else completed / max(total, 1))
        self.label_progress.configure(text=f"{completed} / {total}")

        summary = f"批次完成：成功 {completed} 個"
        if failed:
            summary += f"，失敗 {failed} 個"
        skipped = total - completed - failed
        if skipped:
            summary += f"，略過 {skipped} 個"
        # State the conversion status explicitly: a batch that ran without
        # OpenCC must say so here, not look identical to a converted batch.
        summary += f"\n{s2tw_gate.status_line()}"
        self._log(f"\n{summary}")
        self._send_notification(summary)
        messagebox.showinfo("批次轉錄完成", summary)

    def _send_notification(self, message: str):
        """Send push notification via ntfy.sh, off the UI thread.

        This used to run on the Tk thread with no time limit, so an offline
        network froze the window behind the completion dialog.
        """
        def send():
            try:
                subprocess.Popen(
                    [
                        "curl", "-s", "--max-time", "10", "-X", "POST",
                        "-H", "Title: Speech-to-Subtitle",
                        "-H", "Tags: white_check_mark",
                        "-H", "Content-Type: text/plain; charset=utf-8",
                        "--data-binary", f"@-",
                        "https://ntfy.sh/0pFTayhFKvCuXQE2A8UBY5NT8MYAjW",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                ).communicate(input=message.encode("utf-8"))
            except Exception as e:
                print(f"ntfy notification failed: {e}", file=sys.stderr)

        threading.Thread(target=send, daemon=True).start()


WINDOW_TITLE = "語音轉字幕工具（批次）"
MUTEX_NAME = "Global\\SpeechToSubtitle_SingleInstance_Mutex"


def _ensure_single_instance():
    """Win32 mutex 防多開；若已開啟，將既有視窗帶到前景後結束。"""
    try:
        import win32event, win32api, winerror, win32gui, win32con
    except ImportError:
        return None
    handle = win32event.CreateMutex(None, False, MUTEX_NAME)
    if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if hwnd:
            try:
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
        sys.exit(0)
    return handle  # 持有 handle 直到進程結束


def main():
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr is not None:
        sys.stderr.reconfigure(encoding="utf-8")

    # Write lockfile for start.bat duplicate detection
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".app.lock")
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    _mutex_handle = _ensure_single_instance()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
