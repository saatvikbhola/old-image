import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 1. Model Architecture ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.upconv_block(1024 + 512, 512)
        self.upconv3 = self.upconv_block(512 + 256, 256)
        self.upconv2 = self.upconv_block(256 + 128, 128)
        self.upconv1 = self.upconv_block(128 + 64, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        bottleneck_upsampled = F.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = self.upconv4(torch.cat([bottleneck_upsampled, enc4], dim=1))

        dec4_upsampled = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.upconv3(torch.cat([dec4_upsampled, enc3], dim=1))

        dec3_upsampled = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.upconv2(torch.cat([dec3_upsampled, enc2], dim=1))

        dec2_upsampled = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.upconv1(torch.cat([dec2_upsampled, enc1], dim=1))

        # Final convolution layer
        out = self.final_conv(dec1)
        return out

# --- 2. Helper Functions ---

def load_model(model, file_path, device="cpu"):
    model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    model.eval()
    return model

def detect_scratches(model, image_path, transform):
    device = next(model.parameters()).device
    model.eval()
    
    # Store the original size
    original_image = Image.open(image_path).convert('L')
    original_size = original_image.size  # (width, height)
    
    # Needs to be 256x256 for the model
    resized_image = original_image.resize((256, 256))
    
    image = transform(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output.squeeze(0))
        output = (output.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
    # Resize the 256x256 output back to the original image size
    # cv2.resize takes (width, height)
    output = cv2.resize(output, original_size, interpolation=cv2.INTER_LINEAR)

    return output

def process_mask(mask):
    # This default processing is kept for standalone use, 
    # but the app uses its own custom logic.
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    return mask

def multi_inpaint_image(image_path, mask_path, inpaint_method=cv2.INPAINT_TELEA, inpaint_radius=7, num_passes=2):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Threshold the mask to ensure binary
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # NOTE: I removed the call to 'process_mask(mask)' here.
    # Why? Because your app.py already processes the mask using the sliders
    # and saves it to 'mask_path'. If we process it again here, 
    # we override the slider settings!

    for _ in range(num_passes):
        image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

    return image

def save_inpainted_image(inpainted_image, save_path):
    if len(inpainted_image.shape) == 2: # It's a mask (grayscale)
        Image.fromarray(inpainted_image).save(save_path)
    else: # It's an image (BGR from OpenCV)
        inpainted_image_rgb = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(inpainted_image_rgb).save(save_path)
