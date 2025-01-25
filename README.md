# False Positive Annotation Review Tool

## Overview
This tool allows quick review of model-generated false positive annotations in the COCO dataset format. Designed to streamline the process of manually reviewing and curating false positive predictions from machine learning models like Grounding DINO.

## Workflow
1. Generate false positives using IoU thresholding between model predictions and ground truth
2. Review and filter false positives using this tool
3. Merge accepted false positives with the original ground truth dataset

## Keyboard Shortcuts

### Navigation
- `a`: Accept current bounding box annotation
- `r`: Reject current bounding box annotation
- `n`: Skip current annotation without decision
- `p`: Go to previous annotation
- `e`: Edit annotation category
- `q`: Quit and save reviewed annotations

### Visual Indicators
- Blue box: Unreviewed annotation
- Green box: Accepted annotation
- Red box: Rejected annotation

## Prerequisites
- Python 3.x
- OpenCV
- tkinter

## Usage
1. Prepare your dataset and annotations
2. Update paths in `main()` function
3. Run the script
4. Review annotations interactively

## Example
```python
DATASET_PATH = "path/to/images"
INPUT_JSON = "path/to/false_positives.json"
OUTPUT_JSON = "reviewed_annotations.json"
```

## Note
Designed to save time by allowing quick review of model-generated false positives, minimizing manual annotation effort.
