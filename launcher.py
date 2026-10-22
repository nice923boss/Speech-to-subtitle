"""Wrapper that catches ALL errors before app.py even imports."""
import sys
import os
import traceback

# Redirect stdout/stderr FIRST, before any other import
_LOG_DIR = os.path.join(os.path.expanduser("~"), ".speech-to-subtitle")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "app.log")
_log = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")

if sys.stdout is None:
    sys.stdout = _log
if sys.stderr is None:
    sys.stderr = _log

try:
    _log.write(f"[launcher] Starting app.py from {os.getcwd()}\n")
    _log.write(f"[launcher] Python: {sys.executable}\n")
    _log.write(f"[launcher] Version: {sys.version}\n")
    _log.flush()

    import app
    app.main()
except SystemExit as e:
    _log.write(f"[launcher] SystemExit: {e}\n")
except Exception:
    _log.write("[launcher] FATAL ERROR:\n")
    traceback.print_exc(file=_log)
finally:
    _log.flush()
