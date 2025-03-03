# Cell Detection from Noisy Microscopic Images

## Overview
This project focuses on detecting cells from grayscale microscopic images that contain significant noise. The approach involves **noise reduction, segmentation, connected components analysis, and feature extraction** to accurately identify and label individual cells.

## Features
- **Noise Reduction:** Uses a **median filter** to remove salt-and-pepper noise while preserving cell structures.
- **Segmentation & Connected Components:** Applies **thresholding** and **connected components labeling** to identify individual cells.
- **Bounding Box Extraction:** Extracts and filters bounding boxes around detected cells.
- **Integral Image Analysis:** Computes **area and intensity** values within bounding boxes for further analysis.
- **Final Visualization:** Displays and saves the processed image with labeled bounding boxes around detected cells.

## File Structure
```
├── median_filter.py         # Applies median filtering for noise reduction
├── cell_detection.py        # Detects and labels cells in the filtered image
├── images/                  # Contains example input and output images
├── README.md                # Project documentation
```

## Installation & Requirements
Ensure you have Python installed along with the following dependencies:
```bash
pip install opencv-python numpy tqdm
```

## Usage
1. **Run Median Filter to Reduce Noise:**
   ```bash
   python median_filter.py
   ```
   This will display and save the filtered image.

2. **Run Cell Detection and Labeling:**
   ```bash
   python cell_detection.py
   ```
   This will output an image with detected cells and their bounding boxes.

## Example Results
**Input Image (Noisy):**  
![Noisy Image](images/noisy_input.png)

**Output Image (Cells Detected):**  
![Detected Cells](images/final_processed.png)

