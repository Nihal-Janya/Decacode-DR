# check_demo_api.py (Refactored to return string output for Web UI)
import io
import contextlib
from gradio_client import Client

HF_SPACE = "bhimrazy/diabetic-retinopathy-detection"

def get_gradio_api_schema() -> str:
    """Captures demo.view_api() stdout so server.py can send it to the browser."""
    demo = Client(HF_SPACE)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.view_api()
    return buf.getvalue()

if __name__ == "__main__":
    print(get_gradio_api_schema())