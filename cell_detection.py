import cv2
import numpy as np
from tqdm import tqdm
import os

# ===================================================================
# 1. Function to Compute Integral Image
# ===================================================================
def compute_integral_image(image):
    """
    Computes the integral image of the given grayscale image.

    Parameters:
        image (numpy.ndarray): Grayscale input image.

    Returns:
        numpy.ndarray: Integral image as an integer array.
    """
    rows, cols = image.shape
    integral_img = np.zeros((rows, cols), dtype=int)

    # Compute first row
    for col in range(cols):
        integral_img[0, col] = integral_img[0, col - 1] + image[0, col]

    # Compute first column
    for row in range(rows):
        integral_img[row, 0] = integral_img[row - 1, 0] + image[row, 0]

    # Compute the rest of the integral image
    for row in range(1, rows):
        for col in range(1, cols):
            integral_img[row, col] = (integral_img[row, col - 1] + integral_img[row - 1, col]
                                      - integral_img[row - 1, col - 1] + image[row, col])

    return integral_img


# ===================================================================
# 2. Function to Save Image if Not Already Saved
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

#===================================================================
# 2. Read and Preprocess Image
#===================================================================
img = cv2.imread('filtered_median_3x3.png', cv2.IMREAD_GRAYSCALE)
num_rows, num_cols = img.shape
_, thresholded_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
num_labels, labels = cv2.connectedComponents(thresholded_img)
binary_labels = ((labels != 0) * 255).astype(np.uint8)

#===================================================================
# 3. Process Each Connected Component
#===================================================================
bounding_boxes = []
for i in tqdm(range(num_labels), desc="Processing Components"):
    component_mask = (labels == i)* 255
    component_mask = component_mask.astype(np.uint8)

    x_min, x_max, y_min, y_max = None, None, None, None
    pixel_count = 0

    for r in range(num_rows):
        if max(component_mask[r, :]) == 255:
            pixel_count += np.sum(component_mask[r, :]) // 255
            if y_min is None:
                y_min = r
            y_max = r

    for c in range(num_cols):
        if max(component_mask[:, c]) == 255:
            if x_min is None:
                x_min = c
            x_max = c

    bounding_boxes.append([x_min, y_min, x_max, y_max, pixel_count])

#===================================================================
# 4. Filter Small Components
#===================================================================
x_size_total = sum(abs(b[0] - b[2]) for b in bounding_boxes)
y_size_total = sum(abs(b[1] - b[3]) for b in bounding_boxes)

avg_x_size = x_size_total // num_labels
avg_y_size = y_size_total // num_labels

size_threshold_x = avg_x_size
size_threshold_y = avg_y_size

small_components = [i for i, b in enumerate(bounding_boxes) if b[2] - b[0] < size_threshold_x and b[3] - b[1] < size_threshold_y]

for idx in small_components:
    bounding_boxes[idx] = None
    labels[labels == (idx + 1)] = 0

filtered_labels = ((labels != 0) * 255).astype(np.uint8)

#===================================================================
# 5. Draw Bounding Boxes and Labels
#===================================================================
color_img = np.dstack([img, img, img])
for i, bbox in enumerate(bounding_boxes):
    if bbox:
        x1, y1, x2, y2, _ = bbox
        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(color_img, str(i + 1), (x1 + 15, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

#===================================================================
# 6. Compute Integral Image and Measurements
#===================================================================
integral_img = compute_integral_image(img)

for i, bbox in enumerate(bounding_boxes):
    if bbox:
        x1, y1, x2, y2, area_px = bbox
        A = integral_img[y1, x1]
        B = integral_img[y1 - 1, x2] if y1 > 0 else 0
        C = integral_img[y2, x1 - 1] if x1 > 0 else 0
        D = integral_img[y2, x2]
        mean_gray_value = (D - B - C + A) / ((x2 - x1 + 1) * (y2 - y1 + 1))

        print(f"---- Region {i + 1}: ----")
        print(f"Area (px): {area_px}")
        print(f"Bounding Box Area (px): {(x2 - x1 + 1) * (y2 - y1 + 1)}")
        print(f"Mean gray level in bounding box: {mean_gray_value:.2f}")

# ===================================================================
# 7. Display and Save Images
# ===================================================================
# Show images in separate windows
cv2.imshow('Binary Threshold', binary_labels)
cv2.imshow('Filtered Regions', filtered_labels)
cv2.imshow('Final Processed', color_img)

# Save images if they do not exist
save_image_if_not_exists('binary_threshold.png', binary_labels)
save_image_if_not_exists('filtered_regions.png', filtered_labels)
save_image_if_not_exists('final_processed.png', color_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

