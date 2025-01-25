import json

def convert_gt_to_pred_format(gt_bbox):
    """Convert ground truth bbox format [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    x, y, width, height = gt_bbox
    return [x, y, x + width, y + height]

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
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
        # Add robust handling for different input formats
        score = pred.get('score')
        if isinstance(score, str):
            try:
                score = float(score)
            except ValueError:
                print(f"Skipping prediction with invalid score: {score}")
                continue
        
        if score is None or score < score_threshold:
            continue
        
        image_id = pred.get('image_id')
        bbox = pred.get('bbox', pred.get('box', None))
        
        if not bbox:
            print(f"Skipping prediction without bbox: {pred}")
            continue
        
        pred_category = pred.get('category_id', pred.get('label', None))
        if pred_category is None:
            print(f"Skipping prediction without category: {pred}")
            continue
        
        # Map prediction category to ground truth category
        mapped_category = category_mapping.get(pred_category, None)
        
        gts = gt_by_image.get(image_id, [])
        matched = False
        for gt in gts:
            gt_bbox = gt['bbox']
            gt_bbox = convert_gt_to_pred_format(gt_bbox)
            gt_category = gt['category_id']
            if compute_iou(bbox, gt_bbox) >= iou_threshold and mapped_category == gt_category:
                matched = True
                break
        
        if not matched:
            false_positives.append({
                "image_id": image_id,
                "file_name": image_map.get(image_id, "unknown"),
                "bbox": bbox,
                "category_id": pred_category,
                "label_name": pred.get("label_name", pred.get("label", "unknown")),
                "score": score
            })
    
    return false_positives

def convert_to_coco_format(false_positives, fixed_categories):
    """Convert false positives to COCO format."""
    category_lookup = {cat["name"]: cat["id"] for cat in fixed_categories}

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": fixed_categories
    }

    annotation_id = 1

    for item in false_positives:
        if not any(img["id"] == item["image_id"] for img in coco_data["images"]):
            coco_data["images"].append({
                "id": item["image_id"],
                "file_name": item["file_name"],
                "width": 1920,  # CHANGE THIS: Replace with actual image width if known
                "height": 1080  # CHANGE THIS: Replace with actual image height if known
            })

        adjusted_category_id = item["category_id"]
        if adjusted_category_id not in category_lookup.values():
            print(f"Warning: category_id {adjusted_category_id} is invalid. Skipping.")
            continue

        bbox = item["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": item["image_id"],
            "category_id": adjusted_category_id,
            "bbox": [bbox[0], bbox[1], width, height],
            "area": area,
            "iscrowd": 0,
            "score": item["score"]
        })
        annotation_id += 1

    return coco_data

def main():
    # Load input files
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
    
    # Handle different possible input formats
    if isinstance(predictions, dict) and 'annotations' in predictions:
        predictions = predictions['annotations']

    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth = json.load(f)

    # Extract ground truth categories
    gt_categories = ground_truth.get('categories', FIXED_CATEGORIES)

    # Create category ID mapping
    category_mapping = create_category_mapping(gt_categories, pred_categories)

    # Find false positives
    false_positives = find_false_positives(
        predictions, 
        ground_truth, 
        category_mapping, 
        IoU_THRESHOLD, 
        SCORE_THRESHOLD
    )

    # Convert to COCO format
    coco_data = convert_to_coco_format(false_positives, FIXED_CATEGORIES)

    # Save the COCO format data to a file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO format JSON saved to {OUTPUT_FILE}")
    print(f"Total false positives found: {len(coco_data['annotations'])}")

if __name__ == "__main__":
    # CONFIGURATION: Paths and Thresholds
    # CHANGE THESE PATHS TO MATCH YOUR DATASET
    PREDICTIONS_FILE = "/path/to/your/model_predictions.json"
    GROUND_TRUTH_FILE = "/path/to/your/ground_truth_annotations.json"
    OUTPUT_FILE = "/path/to/output/false_positives.json"
    
    # Thresholds for false positive detection
    IoU_THRESHOLD = 0.5  # Intersection over Union threshold
    SCORE_THRESHOLD = 0.30  # Confidence score threshold

    # MODIFY THIS LIST TO MATCH YOUR DATASET CATEGORIES
    FIXED_CATEGORIES = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "truck"},
        # Add more categories as needed
    ]

    # Create prediction categories for mapping
    pred_categories = [
        {"id": idx, "name": name} 
        for idx, name in enumerate([cat['name'] for cat in FIXED_CATEGORIES])
    ]

    """
    Main function to run false positive extraction.
    INSTRUCTIONS: 
    1. Update PREDICTIONS_FILE with path to your model predictions JSON
    2. Update GROUND_TRUTH_FILE with path to your ground truth annotations JSON
    3. Update OUTPUT_FILE with desired output path for false positives
    4. Modify FIXED_CATEGORIES to match your specific object classes
    5. Adjust IoU_THRESHOLD and SCORE_THRESHOLD as needed
    """

    # Run the main extraction process
    main()
