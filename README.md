"""
main.py — Entry point untuk Streamlit Cloud
Letakkan di root repo. Streamlit Cloud akan menjalankan file ini.
"""

import sys
import os

# Tambahkan root repo ke sys.path agar semua import bisa ditemukan
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Sekarang jalankan app — pakai importlib agar __file__ di app.py tetap benar
import importlib.util, types

app_path = os.path.join(ROOT, "streamlit_app", "app.py")
spec     = importlib.util.spec_from_file_location("app", app_path)
module   = importlib.util.module_from_spec(spec)
module.__file__ = app_path
sys.modules["app"] = module
spec.loader.exec_module(module)
