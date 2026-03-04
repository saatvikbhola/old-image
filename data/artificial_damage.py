"""
Artificial Damage Generator
============================
Adds realistic artificial scratches and damage to images using Cubic Bézier
curves. Processes all images in an input folder and outputs damaged images
along with their corresponding scratch masks.

Usage:
    1. Set INPUT_FOLDER, OUTPUT_FOLDER, and MASK_FOLDER paths below
    2. Run:  python artificial_damage.py
"""

import cv2
import numpy as np
import random
import os
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FOLDER = r"europeana_images"   # Input folder with original images
OUTPUT_FOLDER = r"images"            # Output folder for damaged images
MASK_FOLDER = r"mask"                # Output folder for scratch masks


# ============================================================================
# BEZIER CURVE UTILITY
# ============================================================================

class Bezier:
    """Cubic Bézier curves to generate realistic-looking scratches."""

    @staticmethod
    def TwoPoints(t, P1, P2):
        """Returns a point between P1 and P2, parameterized by t."""
        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')
        return (1 - t) * P1 + t * P2

    @staticmethod
    def Points(t, points):
        """Returns a list of points interpolated by the Bézier process."""
        newpoints = []
        for i1 in range(len(points) - 1):
            newpoints.append(Bezier.TwoPoints(t, points[i1], points[i1 + 1]))
        return newpoints

    @staticmethod
    def Point(t, points):
        """Returns a point interpolated by the Bézier process."""
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]

    @staticmethod
    def Curve(t_values, points):
        """Returns points interpolated by the Bézier process."""
        if not hasattr(t_values, '__iter__') or len(t_values) < 1:
            raise TypeError("`t_values` must be an iterable of integers or floats, of length greater than 0.")
        curve = np.empty((0, len(points[0])), dtype=float)
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
        return curve


# ============================================================================
# DAMAGE FUNCTIONS
# ============================================================================

def add_realistic_scratches(image, num_scratches=5):
    """Add Bézier-curve based scratches to an image and return the damaged image + mask."""
    scratched_image = image.copy()
    scratch_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = scratched_image.shape[:2]

    for _ in range(num_scratches):
        # Randomly select control points for the Bézier curve
        p0 = np.array([random.randint(0, width), random.randint(0, height)])
        p1 = np.array([random.randint(0, width), random.randint(0, height)])
        p2 = np.array([random.randint(0, width), random.randint(0, height)])
        p3 = np.array([random.randint(0, width), random.randint(0, height)])

        # Generate t values for the Bézier curve
        t_values = np.linspace(0, 1, num=100)

        # Calculate points on the Bézier curve
        curve_points = Bezier.Curve(t_values, [p0, p1, p2, p3])

        color = (255, 255, 255)

        # Draw the scratch on the image and update the mask
        for i in range(len(curve_points) - 1):
            start_point = tuple(map(int, curve_points[i]))
            end_point = tuple(map(int, curve_points[i + 1]))

            # Draw the scratch on the image
            cv2.line(scratched_image, start_point, end_point, color, thickness=random.randint(1, 1))

            # Update the mask: Mark the scratch region as white (255)
            cv2.line(scratch_mask, start_point, end_point, 255, thickness=random.randint(1, 1))

    return scratched_image, scratch_mask


def add_realistic_damage(image, num_scratches=5):
    """Apply realistic damage (scratches) to an image."""
    return add_realistic_scratches(image, num_scratches)


def add_dust_and_grain(image, intensity=0.2):
    """Add dust and grain noise for a vintage effect."""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(image, 1 - intensity, noise, intensity, 0)
    return noisy_image


# ============================================================================
# PROCESSING
# ============================================================================

def process_images(input_folder, output_folder, mask_folder):
    """Process all images in input_folder, applying artificial damage."""
    num_scratches = random.randint(2, 5)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Skipping {filename} (could not read)")
            continue

        # Add realistic damage
        damaged_image, damaged_image_mask = add_realistic_damage(image, num_scratches)

        # Optionally, add dust and grain for a vintage effect:
        # damaged_image = add_dust_and_grain(damaged_image, 0.15)

        # Save the damaged image and its mask
        output_path = os.path.join(output_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        cv2.imwrite(output_path, damaged_image)
        cv2.imwrite(mask_path, damaged_image_mask)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"Input:  {os.path.abspath(INPUT_FOLDER)}")
    print(f"Output: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"Masks:  {os.path.abspath(MASK_FOLDER)}")
    print()

    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MASK_FOLDER)
    print("\nDone!")
