import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# This prevents crashes on headless servers like Streamlit Cloud
import matplotlib
matplotlib.use('Agg') 

import streamlit as st
from PIL import Image
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from inpaint import UNet, load_model, detect_scratches, multi_inpaint_image, save_inpainted_image

# Streamlit App
st.title("Scratch Detection and Image Inpainting")
st.write("Upload an image to detect scratches and apply inpainting.")

# Sidebar for settings
st.sidebar.header("Settings")

inpaint_radius = st.sidebar.slider("Inpainting Radius", min_value=1, max_value=15, value=7)
num_passes = st.sidebar.slider("Number of Inpainting Passes", min_value=1, max_value=30, value=5)

kernel_size = st.sidebar.slider("Kernel Size", min_value=1, max_value=10, value=3)
dilate_iterations = st.sidebar.slider("Dilation Iterations", min_value=1, max_value=10, value=3)
morph_iterations = st.sidebar.slider("Morphological Iterations", min_value=1, max_value=10, value=10)

inpaint_algorithm = st.sidebar.selectbox("Inpainting Algorithm", ["TELEA", "NS"])
confirm = st.sidebar.button("Confirm")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource # Caches the model so it doesn't reload on every interaction
def get_model():
    model = UNet(in_channels=1, out_channels=1)
    model_path = 'scratch_detection_model.pth'
    if not os.path.exists(model_path):
        st.error("Model file not found! Please check GitHub LFS settings.")
        return None
    try:
        model = load_model(model, model_path, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading model. It might be a Git LFS pointer. Error: {e}")
        return None

model = get_model()

# Upload Image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    # Resize large images to prevent "Killed" error on Cloud
    max_dimension = 1024
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension))
        st.warning(f"Image resized to {max_dimension}px to prevent memory crash on Cloud.")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to grayscale for processing
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    if confirm and model is not None:
        st.write("Processing the image...")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Detect scratches
        scratch_mask = detect_scratches(model, image_path, transform)

        # Save and display scratch mask
        scratch_mask_path = "scratch_mask.png"
        save_inpainted_image(scratch_mask, scratch_mask_path)
        st.image(scratch_mask, caption="Detected Scratch Mask", use_container_width=True, clamp=True, channels="GRAY")

        # Process Mask
        def custom_process_mask(mask):
            # Ensure kernel size is odd for OpenCV
            k_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = np.ones((k_size, k_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
            return mask
        
        scratch_mask = custom_process_mask(scratch_mask)
        
        # Save processed mask to ensure inpainting uses the THICKENED version
        cv2.imwrite(scratch_mask_path, scratch_mask)

        st.write("Inpainting the image...")
        inpaint_method = cv2.INPAINT_TELEA if inpaint_algorithm == "TELEA" else cv2.INPAINT_NS
        inpainted_image = multi_inpaint_image(image_path, scratch_mask_path, inpaint_method, inpaint_radius, num_passes)

        inpainted_image_path = "inpainted_image.png"
        save_inpainted_image(inpainted_image, inpainted_image_path)

        inpainted_image_rgb = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        inpainted_image_pil = Image.fromarray(inpainted_image_rgb)

        st.image(inpainted_image_pil, caption="Inpainted Image", use_container_width=True)
    elif confirm and model is None:
        st.error("Model failed to load.")
    else:
        st.write("Adjust the settings in the sidebar and click 'Confirm' to process the image.")
