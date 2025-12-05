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
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.upconv4(torch.cat([bottleneck, enc4], dim=1))
        dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=False)

        dec3 = self.upconv3(torch.cat([dec4, enc3], dim=1))
        dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=False)

        dec2 = self.upconv2(torch.cat([dec3, enc2], dim=1))
        dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=False)

        dec1 = self.upconv1(torch.cat([dec2, enc1], dim=1))
        dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=False)

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
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output.squeeze(0))
        output = (output.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

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
    # we override your slider settings!

    for _ in range(num_passes):
        image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

    return image

def save_inpainted_image(inpainted_image, save_path):
    if len(inpainted_image.shape) == 2: # It's a mask (grayscale)
        Image.fromarray(inpainted_image).save(save_path)
    else: # It's an image (BGR from OpenCV)
        inpainted_image_rgb = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(inpainted_image_rgb).save(save_path)

# --- END OF FILE --- 
# (Make sure there is NO code below this line)
