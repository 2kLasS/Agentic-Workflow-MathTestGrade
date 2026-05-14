from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from grade_system.ui.gradio_app import APP_CSS, APP_THEME, build_app


def _launch_accepts_styling() -> bool:
    launch_params = inspect.signature(gr.Blocks.launch).parameters
    return "theme" in launch_params and "css" in launch_params


def main() -> None:
    launch_kwargs = {
        "server_name": os.getenv("GRADIO_HOST", "localhost"),
        "server_port": int(os.getenv("GRADIO_PORT", "7860")),
    }

    if _launch_accepts_styling():
        app = build_app()
        launch_kwargs.update(theme=APP_THEME, css=APP_CSS)
    else:
        app = build_app(theme=APP_THEME, css=APP_CSS)

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
