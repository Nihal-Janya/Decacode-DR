from gradio_client import Client

demo = Client(
    "bhimrazy/diabetic-retinopathy-detection"
)

demo.view_api()