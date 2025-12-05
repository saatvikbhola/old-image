import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import streamlit as st
from PIL import Image
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from inpaint import UNet, load_model, detect_scratches, process_mask, multi_inpaint_image, save_inpainted_image

# Streamlit App
st.title("Scratch Detection and Image Inpainting")
st.write("Upload an image to detect scratches and apply inpainting.")

# Sidebar for settings
st.sidebar.header("Settings")

# Slider for inpainting radius and number of passes
inpaint_radius = st.sidebar.slider("Inpainting Radius", min_value=1, max_value=15, value=7)
num_passes = st.sidebar.slider("Number of Inpainting Passes", min_value=1, max_value=30, value=5)

# Options for kernel size, dilation iterations, and morphological iterations
kernel_size = st.sidebar.slider("Kernel Size", min_value=1, max_value=10, value=3)
dilate_iterations = st.sidebar.slider("Dilation Iterations", min_value=1, max_value=10, value=3)
morph_iterations = st.sidebar.slider("Morphological Iterations", min_value=1, max_value=10, value=10)

# Dropdown to select the inpainting algorithm
inpaint_algorithm = st.sidebar.selectbox("Inpainting Algorithm", ["TELEA", "NS"])

# Button to confirm the settings
confirm = st.sidebar.button("Confirm")

# Load pre-trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=1, out_channels=1)
model_path = 'scratch_detection_model.pth'
model = load_model(model, model_path, device=device)

# Upload Image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to grayscale for processing
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    if confirm:
        # Detect scratches
        st.write("Processing the image...")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        scratch_mask = detect_scratches(model, image_path, transform)

        # Save and display scratch mask
        scratch_mask_path = "scratch_mask.png"
        save_inpainted_image(scratch_mask, scratch_mask_path)
        st.image(scratch_mask, caption="Detected Scratch Mask", use_container_width=True, clamp=True, channels="GRAY")

        # Modify the mask processing based on user inputs
        def custom_process_mask(mask):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Use user-defined kernel size
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)  # Apply dilation

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)  # Apply closing
            return mask
        
        # Apply the custom mask processing
        scratch_mask = custom_process_mask(scratch_mask)

        # Perform inpainting
        st.write("Inpainting the image...")
        inpaint_method = cv2.INPAINT_TELEA if inpaint_algorithm == "TELEA" else cv2.INPAINT_NS
        inpainted_image = multi_inpaint_image(image_path, scratch_mask_path, inpaint_method, inpaint_radius, num_passes)

        # Save and display the inpainted image
        inpainted_image_path = "inpainted_image.png"
        save_inpainted_image(inpainted_image, inpainted_image_path)

        # Convert the image to RGB format for display
        inpainted_image_rgb = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        inpainted_image_pil = Image.fromarray(inpainted_image_rgb)

        # Display the inpainted image in Streamlit
        st.image(inpainted_image_pil, caption="Inpainted Image", use_container_width=True)
    else:
        st.write("Adjust the settings in the sidebar and click 'Confirm' to process the image.")
