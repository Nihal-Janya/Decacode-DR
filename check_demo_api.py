import io
from contextlib import redirect_stdout
from gradio_client import Client

SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"


def get_remote_api_info() -> str:
    """
    Connects to the Hugging Face Space and captures the API signature
    across both old and new versions of gradio_client.
    """
    demo = Client(SPACE_ID)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            api_info = demo.view_api(return_format="str")
        except TypeError:
            api_info = demo.view_api()

    captured = buffer.getvalue().strip()
    if api_info:
        return str(api_info)
    if captured:
        return captured
    return "Connected to Space, but no API schema text was returned."


if __name__ == "__main__":
    print(get_remote_api_info())