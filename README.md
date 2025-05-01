# Ball Bounce Counting Algorithm

## Overview

This project implements a ball tracking and bounce counting algorithm using different techniques for **real** and **simulated environments**.

---

## Methodology

### 1- Real Environment
- **Object Detection:** YOLOv8 for detecting balls and obstacles.
- **Tracking:** Switches to color-based tracking after initial detection for consistent IDs and efficiency.
- **Noise & Occlusion Handling:** Dynamically adjusts HSV ranges and uses a cooldown mechanism for bounce detection.

### 2- Simulated Environment
- **Ball Detection:** Contour analysis combined with Hough Circle Transform.
- **Tracking:** Registers detected balls by color and switches to color-based tracking.
- **Bounce Detection:** Stricter — counts bounces only in the lower 10% of the frame.

---

## Instructions

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### Required Libraries
Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, you can manually install the following libraries:
- `numpy`
- `opencv-python`
- `imutils`
- `ultralytics` (for YOLOv8)
- `argparse`

### Running the Program
To run the program, use the following command:

```bash
python main.py --environment [real|simulated] -v [video_path] -o [output_path]
```

Replace `[real|simulated]` with the desired environment (optional), `[video_path]` with the path to your input video (optional), and `[output_path]` with the path to save the output video (optional).

### Default Environment
The default environment is set to `real`. You can override this by specifying the `--environment` argument.

### Optional Output
You can specify an output file for saving the processed video using the `-o` or `--output` argument. For example:

```
python main.py --environment simulated -o output.mp4
```

This will save the processed video to `output.mp4`, in outputs folder.

---

## Strategies Used

### Noise Handling
- Gaussian blur and adaptive thresholding in simulations.
- Moving average smoothing for centroid positions.

### Obstacle Handling
- YOLO detects obstacles in real environments; excludes them from bounce detection.
- Ensures no overlap using bounding boxes.

### Multiple Ball Tracking
- Color similarity checks for ID consistency.
- Distance-based association matrix for centroid matching.

### Bounce Detection
- Minimum frame interval enforced to prevent double-counting.
- Region-restricted detection:
  - **Real:** Lower 32% of frame.
  - **Simulated:** Lower 10% of frame.

---

## Challenges Faced

- **HoughCircles limitations** in real environments; required DL + color-based matching.
- Maintaining **consistent ball IDs** under noise/occlusions — currently not fully reliable if the ball leaves view.
- **Differentiating balls and obstacles** in cluttered scenes.
- Handling **rapid trajectory changes** during bounces.
- **Time constraints** during midterms.

---

## Performance Metrics

### Accuracy
- **Real:** Reliable tracking if the ball’s color is unique and motion is moderate; bounce counting remains inconsistent.
- **Simulated:** Consistent tracking even with noise; parameters tunable for ball elasticity and scene conditions.

### Efficiency
- **Real:** YOLO detection runs periodically for balance.
- **Simulated:** Lightweight contour-based detection ensures real-time performance.

---

## Future Improvements

- Fine-tune YOLO with **custom training for better obstacle differentiation**.
- Implement **Kalman filters** for robust trajectory prediction.
- Explore **Mask R-CNN** based ball tracking.
- Optimize **color-based tracking** for high-res videos.
- Train a CNN/DL model to detect **floor surfaces**.
- Explore **floor-based detection** and measure ball-floor interactions.
- Improve accuracy validation and implement **continuous parameter optimization**.

---

