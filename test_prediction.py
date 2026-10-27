# test_prediction.py (Refactored for Modular Web + CLI Execution)
from gradio_client import Client
from pathlib import Path

HF_SPACE = "bhimrazy/diabetic-retinopathy-detection"

def run_retinal_prediction(image_path_str: str) -> dict:
    """Reusable function callable by server.py or CLI."""
    resolved_path = Path(image_path_str.strip('"')).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved_path}")

    demo = Client(HF_SPACE)
    result = demo.predict(str(resolved_path), api_name="/predict")
    return {
        "image_path": str(resolved_path),
        "endpoint": "/predict",
        "result": result
    }

if __name__ == "__main__":
    user_input = input("Enter the full path to your retinal image: ")
    try:
        output = run_retinal_prediction(user_input)
        print("\nPrediction result:")
        print(output["result"])
    except FileNotFoundError as err:
        print(f"Error: {err}")