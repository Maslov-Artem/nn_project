import json
import time
from functools import wraps
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from requests.models import MissingSchema
from torch import nn

from model.birds_classifier import bird_model
from model.preprocessing import preprocess

# Define the allowed image extensions
image_extensions = [
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "tif",
    "tiff",
    "svg",
    "webp",
    "ico",
]

# Load the class decoder from JSON file
with open("birds.json", "r") as file:
    decoder = json.load(file)


# Load the birds model
@st.cache_resource()
def load_birds_model() -> nn.Module:
    model = bird_model
    return model


model = load_birds_model()


# Decorator function to display execution time
def display_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time_placeholder = st.empty()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        time_placeholder.write(f"Execution Time: {execution_time:.2f} seconds.")
        return result

    return wrapper


# Function to make prediction
@display_time
def predict(image):
    model.eval()
    image = preprocess(image)
    pred = model(image)
    return decoder[str(pred.argmax().item())]


# Function to download image from URL
def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            return None
    except MissingSchema:
        return None


# Streamlit app
st.title("Bird Species Classifier")

# Description of the page
st.write(
    """
    This page allows you to classify bird species using ResNet152 image classifier model.
    You can either enter the URL of an image or upload an image file directly.
    """
)

# Image download section
st.header("Download Image from URL")

# User input for image URL
url = st.text_input("Enter the URL of the image:")

if st.button("Download Image"):
    if url:
        image = download_image(url)
        if image:
            prediction = predict(image)
            st.write("Prediction:", prediction)
            st.image(image, caption="Downloaded Image", use_column_width=True)
        else:
            st.error("Failed to download image. Please check the URL.")

# Image upload section
st.header("Upload Image Files")

# File uploader widget to upload images
images = st.file_uploader(
    label="Upload image", type=image_extensions, accept_multiple_files=True
)

if images:
    for image in images:
        image = Image.open(image)
        prediction = predict(image)
        st.write("Prediction:", prediction)
        st.image(image, caption="Uploaded Image", use_column_width=True)
