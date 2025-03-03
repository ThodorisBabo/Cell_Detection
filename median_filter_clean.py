import cv2
import numpy as np
import os
from tqdm import tqdm

# ===================================================================
# 1. Load Image
# ===================================================================
def load_grayscale_image(filename):
    """
    Loads a grayscale image from a given filename.

    Parameters:
        filename (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded grayscale image.
    """
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: The file '{filename}' was not found.")
    return image

# ===================================================================
# 2. Apply Median Filter
# ===================================================================
def apply_median_filter(image, kernel_size=3):
    """
    Applies a median filter to reduce noise in an image.

    Parameters:
        image (numpy.ndarray): Grayscale image to be filtered.
        kernel_size (int): Size of the kernel for filtering. Default is 3.

    Returns:
        numpy.ndarray: Median-filtered image.
    """
    num_rows, num_cols = image.shape
    half_kernel = kernel_size // 2
    row_min, col_min = half_kernel, half_kernel
    row_max, col_max = num_rows - half_kernel - 1, num_cols - half_kernel - 1
    filtered_image = np.zeros(image.shape, dtype=np.uint8)

    for row in tqdm(range(num_rows), desc="Processing Image"):
        for col in range(num_cols):
            neighborhood_values = []

            if row < row_min or row > row_max or col < col_min or col > col_max:
                row_start = 0 if row < row_min else -half_kernel
                col_start = 0 if col < col_min else -half_kernel
                row_end = num_rows - row if row > row_max else half_kernel + 1
                col_end = num_cols - col if col > col_max else half_kernel + 1
            else:
                row_start, row_end = -half_kernel, half_kernel + 1
                col_start, col_end = -half_kernel, half_kernel + 1

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    neighborhood_values.append(image[row + r, col + c])

            # Compute median value and assign it to new image
            filtered_image[row, col] = np.median(neighborhood_values)

    return filtered_image

# ===================================================================
# 3. Save Image if Not Already Saved
# ===================================================================
def save_image_if_not_exists(filename, image):
    """
    Saves an image only if a file with the same name does not already exist.

    Parameters:
        filename (str): Name of the file to save.
        image (numpy.ndarray): Image to be saved.
    """
    if not os.path.exists(filename):
        cv2.imwrite(filename, image)
        print(f"Saved: {filename}")
    else:
        print(f"File already exists: {filename}")

# ===================================================================
# 4. Main Execution
# ===================================================================
filename = 'noisy_input.png'
output_filename = 'filtered_median_3x3.png'

# Load the original grayscale image
gray_img = load_grayscale_image(filename)

# Apply median filter to remove noise
filtered_img = apply_median_filter(gray_img, kernel_size=3)

# Display images
cv2.imshow('Noisy Input', gray_img)
cv2.imshow('Filtered Median 3x3', filtered_img)

# Save the filtered image if it does not exist
save_image_if_not_exists(output_filename, filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

