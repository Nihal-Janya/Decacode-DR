from gradio_client import Client

SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"


def get_remote_api_info() -> str:
    """
    Connects to the Hugging Face Space and returns the API signature
    so both the CLI and app.py web server can display it.
    """
    demo = Client(SPACE_ID)
    api_info = demo.view_api(return_format="str")
    return str(api_info)


if __name__ == "__main__":
    print(get_remote_api_info())
