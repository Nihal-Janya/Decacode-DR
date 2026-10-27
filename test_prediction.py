from pathlib import Path
from gradio_client import Client

try:
    from gradio_client import handle_file
except ImportError:
    handle_file = None

SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"


def run_retinal_prediction(image_path: str):
    """
    Runs prediction on a retinal image file path using the remote
    Hugging Face Space. Uses handle_file() for modern gradio_client
    compatibility and falls back gracefully if needed.
    """
    resolved_path = str(Path(image_path.strip('"')).resolve())

    if not Path(resolved_path).is_file():
        raise FileNotFoundError(f"Error: Image file not found at {resolved_path}")

    demo = Client(SPACE_ID)
    file_input = handle_file(resolved_path) if handle_file is not None else resolved_path

    try:
        result = demo.predict(
            file_input,
            api_name="/predict"
        )
    except Exception:
        # Fallback if the Space uses fn_index=0 instead of named '/predict'
        result = demo.predict(
            file_input,
            fn_index=0
        )

    return result


if __name__ == "__main__":
    user_input = input("Enter the full path to your retinal image: ")
    try:
        output = run_retinal_prediction(user_input)
        print("\nPrediction result:")
        print(output)
    except Exception as e:
        print(e)