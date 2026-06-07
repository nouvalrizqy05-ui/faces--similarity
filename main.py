import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

for p in [ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

app_path = os.path.join(ROOT, "streamlit_app", "app.py")
spec     = importlib.util.spec_from_file_location("app", app_path)
module   = importlib.util.module_from_spec(spec)
module.__file__ = app_path
sys.modules["app"] = module
spec.loader.exec_module(module)
