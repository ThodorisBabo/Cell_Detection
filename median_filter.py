import cv2
import numpy as np
from tqdm import tqdm

# ===================================================================
# 1. Load Image
# ===================================================================
filename = '9_noise.png'
gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', gray_img)
num_rows, num_cols = gray_img.shape

# ===================================================================
# 2. Initialize Variables
# ===================================================================
kernel_size = 3
half_kernel = kernel_size // 2
row_min, col_min = half_kernel, half_kernel
row_max, col_max = num_rows - half_kernel - 1, num_cols - half_kernel - 1
filtered_img = np.zeros(gray_img.shape, dtype=np.uint8)

# ===================================================================
# 3. Apply Median Filter with Progress Bar
# ===================================================================
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
                neighborhood_values.append(gray_img[row + r, col + c])

        # Compute median value and assign it to new image
        filtered_img[row, col] = np.median(neighborhood_values)

# ===================================================================
# 4. Display Filtered Image
# ===================================================================
cv2.imshow('Filtered', filtered_img)
cv2.waitKey(0)
