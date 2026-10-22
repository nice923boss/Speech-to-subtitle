"""Shared CUDA DLL preloading for Windows.

pip packages like nvidia-cublas-cu12 install DLLs under
site-packages/nvidia/*/bin/ which is NOT on the default DLL search
path on Windows (Python 3.8+). Both add_dll_directory AND preloading
via ctypes are needed for reliable operation.

Usage (must be called BEFORE importing CUDA-dependent libraries):
    import cuda_setup  # noqa: F401  — side-effect import
"""

import os
import sys
import site
import ctypes

if sys.platform == "win32":
    for _sp in site.getsitepackages() + [site.getusersitepackages()]:
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    os.add_dll_directory(_bin_dir)
                    for _dll in os.listdir(_bin_dir):
                        if _dll.endswith(".dll"):
                            try:
                                ctypes.CDLL(os.path.join(_bin_dir, _dll))
                            except OSError:
                                pass
