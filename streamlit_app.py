"""
Streamlit Cloud entry point (repo root).
Live app: https://heartdiseasepredictiondemo.streamlit.app/
Deploy main file path: streamlit_app.py
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "app" / "streamlit_app.py"), run_name="__main__")
