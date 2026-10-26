
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tempfile
from pathlib import Path
from gradio_client import Client


# 1. Configure the webpage
st.set_page_config(
    page_title="Retinal Image Prototype",
    page_icon="👁️",
    layout="centered"
)


# 2. Display the page introduction
st.title("Diabetic Retinopathy Screening Prototype")

st.write(
    "Upload a retinal photograph to inspect its "
    "basic image properties."
)

st.warning(
    "Educational prototype only. This application "
    "does not diagnose diabetic retinopathy."
)


# 3. Let the user upload an image
uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)


# 4. Process the image after upload
if uploaded_file is not None:

    try:
        # Open and decode the uploaded image
        image = Image.open(uploaded_file)
        image.load()

        # Reject unexpectedly large images
        max_pixels = 20_000_000

        if image.width * image.height > max_pixels:
            st.error(
                "The image is too large for this prototype."
            )
            st.stop()

        # Display the image
        st.success("Image opened successfully.")

        st.image(
            image,
            caption="Uploaded retinal image"
        )

        # Display file and image information
        st.write("File name:", uploaded_file.name)
        st.write("File size:", uploaded_file.size, "bytes")
        st.write("Image format:", image.format)
        st.write("Image dimensions:", image.size)
        st.write("Color mode:", image.mode)

        # Convert the image to RGB
        rgb_image = image.convert("RGB")

        # Convert pixels to a NumPy array
        pixels = np.array(rgb_image)

        # Calculate basic image statistics
        brightness = float(np.mean(pixels))
        contrast = float(np.std(pixels))

        st.subheader("Basic Image Statistics")

        st.write(
            "Mean pixel intensity:",
            round(brightness, 2)
        )

        st.write(
            "Pixel intensity standard deviation:",
            round(contrast, 2)
        )

        st.write("Array shape:", pixels.shape)

        # Demonstration brightness indicator
        st.subheader("Experimental Quality Indicator")

        if brightness < 50:
            st.warning(
                "Low average brightness. Inspect the image "
                "manually; this is not a diagnostic result."
            )
        else:
            st.info(
                "The average brightness is not below "
                "the example threshold. This does not "
                "establish clinical image quality."
            )

        # Explicitly avoid unsupported medical predictions
        st.info(
            "No AI model has been run. No disease "
            "prediction has been made."
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        st.error(
            "The uploaded file could not be decoded "
            "as a valid image. Please try another image."
        )

st.subheader("AI Diabetic Retinopathy Screening")

st.caption(
    "Experimental demonstration only. "
    "Not a medical diagnosis."
)

st.components.v1.iframe(
    "https://bhimrazy-diabetic-retinopathy-detection.hf.space",
    height=750,
    scrolling=True
)