
from gradio_client import Client
from pathlib import Path

demo = Client(
    "bhimrazy/diabetic-retinopathy-detection"
)

image_path = input("Enter the full path to your retinal image: ")
image_path = str(Path(image_path.strip('"')).resolve())

if not Path(image_path).is_file():
    print("Error: Image file not found.")
else:
    result = demo.predict(
    image_path,
    api_name="/predict"
)

    print(result)

    print("\nPrediction result:")
    print(result)
