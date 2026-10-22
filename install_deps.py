"""Install or repair what Speech-to-subtitle needs on this PC.

Run by the install .bat in this folder with the Python 3.12 the app should use:
1. Point pyvenv.cfg and the launcher trio (pythonw_speech2sub.exe,
   python312.dll, python3.dll) at this Python. The copies in the folder come
   from the PC they were made on, so a moved folder does not start without this.
2. pip install requirements.txt, plus requirements-gpu.txt when nvidia-smi
   exists. No --upgrade (see requirements.txt).
3. Offer winget for ffmpeg and a large-v3 download when either is missing, then
   print what transcribe_worker.py --check sees. MFA is reported, not
   installed: it is an optional conda environment of several GB.
"""
import filecmp
import json
import os
import shutil
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_COPIES = (
    ("pythonw.exe", "pythonw_speech2sub.exe"),
    ("python312.dll", "python312.dll"),
    ("python3.dll", "python3.dll"),
)
MFA_STEPS = (
    "  1. 安裝 Miniforge 到 %USERPROFILE%\\miniforge3（https://conda-forge.org/download/）\n"
    "  2. mamba create -n mfa -c conda-forge montreal-forced-aligner\n"
    "  3. mamba run -n mfa mfa model download acoustic mandarin_mfa\n"
    "  4. mamba run -n mfa mfa model download dictionary mandarin_taiwan_mfa"
)


def ask(question: str) -> bool:
    return input(f"{question} [Y/N] ").strip().lower().startswith("y")


def link_launcher() -> bool:
    home = sys.base_prefix
    cfg_path = os.path.join(APP_DIR, "pyvenv.cfg")
    cfg = f"home = {home}\n"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            current = f.read()
    except FileNotFoundError:
        current = ""
    try:
        if current != cfg:
            with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(cfg)
        for src_name, dst_name in LAUNCHER_COPIES:
            src = os.path.join(home, src_name)
            dst = os.path.join(APP_DIR, dst_name)
            if not (os.path.isfile(dst) and filecmp.cmp(src, dst, shallow=False)):
                shutil.copy2(src, dst)
    except PermissionError as e:
        print(f"[錯誤] 無法更新啟動檔：{e}\n"
              "       App 執行中時檔案會被鎖住，請先執行 stop.bat 再重跑本檔。")
        return False
    print(f"啟動檔已指向 {home}")
    return True


def pip_install(req_file: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "-r", os.path.join(APP_DIR, req_file)]
    return subprocess.call(cmd) == 0


def check_ffmpeg() -> None:
    from scanner import find_ffmpeg
    path = find_ffmpeg()
    if path:
        print(f"ffmpeg：{path}")
        return
    print("找不到 ffmpeg（MFA 對齊需要；沒有時轉錄照常，改用能量對齊）。")
    if shutil.which("winget") and ask("要用 winget 安裝 ffmpeg（Gyan.FFmpeg）嗎？"):
        if subprocess.call(["winget", "install", "-e", "--id", "Gyan.FFmpeg"]) == 0:
            print("ffmpeg 已安裝。PATH 在新開的視窗才生效：App 若開著請關閉後重開。")
            return
    print("手動安裝：到 https://www.gyan.dev/ffmpeg/builds/ 下載，解壓到 C:\\ffmpeg"
          "（App 會找 C:\\ffmpeg\\bin\\ffmpeg.exe），或把 bin 資料夾加入 PATH。")


def check_model() -> None:
    from scanner import scan_whisper_models, configured_scan_dirs, preferred_model
    model = preferred_model(scan_whisper_models(extra_dirs=configured_scan_dirs()))
    if model is not None:
        print(f"Whisper 模型：{model.path}")
        return
    print("找不到 Whisper 模型。")
    if not ask("要下載 large-v3 模型（約 3 GB，存到 Hugging Face 快取）嗎？"):
        print("略過。已有模型的話，開啟 App 按「掃描資料夾」選模型所在資料夾。")
        return
    from faster_whisper import download_model
    try:
        print(f"已下載到 {download_model('large-v3')}")
    except Exception as e:
        print(f"[錯誤] 模型下載失敗：{type(e).__name__}: {e}\n       請確認網路後重跑本檔。")


def worker_check() -> bool:
    print("檢查中（MFA 第一次啟動可能要 30 秒以上）...")
    proc = subprocess.run(
        [sys.executable, os.path.join(APP_DIR, "transcribe_worker.py"), "--check"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=APP_DIR)
    try:
        report = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        print(f"[錯誤] transcribe_worker.py --check 沒有回報結果：\n{proc.stderr[-1500:]}")
        return False

    cuda = report["cuda"]
    items = (
        ("CUDA 顯示卡（必要）", cuda["ok"],
         f"{cuda.get('devices')} 張" if cuda["ok"] else cuda.get("message", "找不到 NVIDIA 顯示卡")),
        ("Whisper 模型（必要）", report["model"]["ok"],
         report["model"].get("path") or report["model"].get("message")),
        ("ffmpeg", report["ffmpeg"]["ok"], report["ffmpeg"]["path"] or "未找到"),
        ("簡轉繁 OpenCC", report["opencc"]["ok"], report["opencc"]["message"]),
        ("MFA 詞級對齊（選用）", report["mfa"]["ok"], report["mfa"]["message"]),
    )
    for name, ok, detail in items:
        print(f"  {'[OK]' if ok else '[缺]'} {name}：{detail}")
    if not report["mfa"]["ok"]:
        print(f"MFA 未自動安裝（數 GB）。沒有時改用能量對齊；要裝的話：\n{MFA_STEPS}")
    return report["ok"]


def main() -> int:
    print(f"Python {sys.version.split()[0]}：{sys.executable}")
    if sys.version_info[:2] != (3, 12):
        print("[錯誤] 需要 Python 3.12：啟動檔 python312.dll 從它複製。")
        return 1

    print("\n[1/5] 更新啟動檔")
    launcher_ok = link_launcher()

    print("\n[2/5] 安裝核心套件")
    if not pip_install("requirements.txt"):
        print("[錯誤] 核心套件安裝失敗，請看上方訊息、確認網路後重跑。")
        return 1

    if shutil.which("nvidia-smi"):
        print("\n[3/5] 偵測到 NVIDIA 驅動程式，安裝 GPU 套件")
        if not pip_install("requirements-gpu.txt"):
            print("[錯誤] GPU 套件安裝失敗，請看上方訊息後重跑。")
            return 1
    else:
        print("\n[3/5] 找不到 nvidia-smi，跳過 GPU 套件。本 App 轉錄需要 NVIDIA 顯示卡與驅動程式。")

    print("\n[4/5] 檢查 ffmpeg 與 Whisper 模型")
    check_ffmpeg()
    check_model()

    print("\n[5/5] 最終檢查")
    ready = worker_check() and launcher_ok
    print("\n可以執行 start.bat 開啟 App。" if ready else "\n還有必要項目未完成，見上方 [缺] 或 [錯誤]。")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
