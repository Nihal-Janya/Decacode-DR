from pathlib import Path
from gradio_client import Client

SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"


def run_retinal_prediction(image_path: str):
    """
    Runs prediction on a retinal image file path using the remote
    Hugging Face Space (/predict endpoint). Can be imported by app.py
    or executed directly from the terminal.
    """
    resolved_path = str(Path(image_path.strip('"')).resolve())

    if not Path(resolved_path).is_file():
        raise FileNotFoundError(f"Error: Image file not found at {resolved_path}")

    demo = Client(SPACE_ID)
    result = demo.predict(
        resolved_path,
        api_name="/predict"
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
