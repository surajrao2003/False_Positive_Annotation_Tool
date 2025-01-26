# Update the following as per your model and dataset and run the script:

# 1) prediction_path: Path to model predictions JSON
# 2) ground_truth_path: Path to ground truth annotations JSON (in coco format)
# 3) output_path: Path to save false positives JSON (output)
# 4) pred_categories: List of object classes matching your dataset
# 5) iou_threshold: Adjust detection matching strictness (0.5 recommended)
# 6) score_threshold: Control false positive confidence threshold (0.3 recommended)


import json
import os

def convert_gt_to_pred_format(gt_bbox):
    """Convert ground truth bbox format [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    x, y, width, height = gt_bbox
    return [x, y, x + width, y + height]

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def create_category_mapping(gt_categories, pred_categories):
    """Create a mapping between ground truth and prediction category IDs."""
    mapping = {}
    for pred_idx, pred_category in enumerate(pred_categories):
        for gt_category in gt_categories:
            if gt_category['name'] == pred_category['name']:
                mapping[pred_idx] = gt_category['id']
                break
    return mapping

def find_false_positives(predictions, ground_truth, category_mapping, iou_threshold=0.5, score_threshold=0.35):
    """Find false positives above the score threshold."""
    false_positives = []
    gt_by_image = {gt['image_id']: [] for gt in ground_truth['annotations']}
    for gt in ground_truth['annotations']:
        gt_by_image[gt['image_id']].append(gt)
    
    image_map = {img['id']: img['file_name'] for img in ground_truth['images']}
    
    for pred in predictions:
        if pred["score"] < score_threshold:  # Skip predictions below the score threshold
            continue
        
        image_id = pred['image_id']
        pred_bbox = pred['bbox']
        pred_category = pred['category_id']
        
        # Map prediction category to ground truth category
        mapped_category = category_mapping.get(pred_category, None)
        
        gts = gt_by_image.get(image_id, [])
        matched = False
        for gt in gts:
            gt_bbox = gt['bbox']
            gt_bbox = convert_gt_to_pred_format(gt_bbox)
            gt_category = gt['category_id']
            if compute_iou(pred_bbox, gt_bbox) >= iou_threshold and mapped_category == gt_category:
                matched = True
                break
        
        if not matched:
            false_positives.append({
                "image_id": image_id,
                "file_name": image_map.get(image_id, "unknown"),
                "bbox": pred_bbox,
                "category_id": pred_category,
                "label_name": pred.get("label_name", "unknown"),
                "score": pred["score"]
            })
    
    return false_positives

def process_false_positives(config):
    """
    Process false positives detection with configurable parameters.
    
    Args:
        config (dict): Configuration dictionary containing paths and thresholds
    """
    # Load predictions and ground truth
    with open(config['prediction_path'], 'r') as f:
        predictions = json.load(f)

    with open(config['ground_truth_path'], 'r') as f:
        ground_truth = json.load(f)

    # Define prediction categories (customize as per your dataset)
    pred_categories = [
        {"id": idx, "name": name} 
        for idx, name in enumerate([
            "person", "rickshaw", "rickshaw van", "auto rickshaw", "truck",
            "pickup truck", "private car", "motorcycle", "bicycle", "bus",
            "micro bus", "covered van", "human hauler"
        ])
    ]

    # Ground truth categories from the ground truth JSON file
    gt_categories = ground_truth['categories']

    # Create category ID mapping
    category_mapping = create_category_mapping(gt_categories, pred_categories)

    # Find false positives
    false_positives = find_false_positives(
        predictions, 
        ground_truth, 
        category_mapping, 
        config['iou_threshold'], 
        config['score_threshold']
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)

    # Save false positives
    with open(config['output_path'], 'w') as f:
        json.dump(false_positives, f, indent=2)

    print(f"False positives with score >= {config['score_threshold']} saved to '{config['output_path']}'")

# Configuration - MODIFY THESE PATHS AND PARAMETERS
if __name__ == "__main__":
    config = {
        # Path to your model's prediction JSON file
        # This file should contain predictions with keys: image_id, bbox, category_id, score, label_name
        'prediction_path': '/path/to/your/predictions.json',
        
        # Path to ground truth annotations in COCO format
        # This file should contain image annotations with categories, bounding boxes
        'ground_truth_path': '/path/to/your/ground_truth.json',
        
        # Path where false positives will be saved
        'output_path': '/path/to/save/false_positives.json',
        
        # Intersection over Union (IoU) threshold for matching predictions
        'iou_threshold': 0.5,
        
        # Score threshold for filtering predictions
        # NOTE: In case you have already performed confidence score thresholding for your model, 
        # you can set score_threshold=0 to include all predictions
        'score_threshold': 0.30
    }
    
    process_false_positives(config)
