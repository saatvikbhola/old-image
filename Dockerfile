# Use the slim version of Python 3.12 (Debian-based, very slim and optimized for Python)
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install minimal system dependencies required for image processing (libGL is often needed, but we use headless OpenCV)
# We include libglib2.0-0 just in case it's needed by some cv2 operations, but keep it minimal to save space.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requirement/best practice)
RUN useradd -m -u 1000 user

# Change ownership of the application directory to the non-root user
RUN chown user:user /app

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy the requirements file first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .

# CRITICAL FOR SLIM IMAGE ON HUGGING FACE FREE TIER:
# Install the CPU-only version of PyTorch first. The default PyTorch wheel includes CUDA 
# which is several gigabytes and not supported/necessary on the free HF Spaces tier.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose the default port for Streamlit & Hugging Face Spaces
EXPOSE 7860

# Run the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
