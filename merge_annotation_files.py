import json
from collections import defaultdict

def merge_coco_annotations(file1, file2, output_file):
    """
    Merge two COCO annotation files ensuring annotations of the same file_name
    are grouped under the same image_id.

    Parameters:
        file1 (str): Path to the first COCO annotation file (e.g., false positives).
        file2 (str): Path to the second COCO annotation file (e.g., ground truth).
        output_file (str): Path to save the merged COCO annotation file.
    """
    # Load both annotation files
    with open(file1, 'r') as f:
        coco1 = json.load(f)
    with open(file2, 'r') as f:
        coco2 = json.load(f)

    # Create a mapping of file_name to image_id from the first file
    file_name_to_id = {img['file_name']: img['id'] for img in coco1['images']}
    images = {img['id']: img for img in coco1['images']}  # id to image mapping

    # Start merging annotations
    merged_annotations = coco1['annotations'][:]  # Copy annotations from file1
    next_annotation_id = max(ann['id'] for ann in merged_annotations) + 1

    # Create a mapping for image_id in the second file
    coco2_image_mapping = {img['id']: img['file_name'] for img in coco2['images']}
    
    for ann in coco2['annotations']:
        # Find corresponding file_name in coco2
        file_name = coco2_image_mapping.get(ann['image_id'], None)
        if file_name is None:
            # If image_id is not found, skip the annotation
            continue
        
        # Find corresponding image_id in the merged file
        image_id = file_name_to_id.get(file_name, None)
        if image_id is None:
            # If the image is not in file1, add it
            new_image = next(img for img in coco2['images'] if img['id'] == ann['image_id'])
            new_image_id = max(images.keys()) + 1
            new_image['id'] = new_image_id
            images[new_image_id] = new_image
            file_name_to_id[file_name] = new_image_id
            image_id = new_image_id

        # Add annotation to the merged annotations
        new_annotation = ann.copy()
        new_annotation['id'] = next_annotation_id
        new_annotation['image_id'] = image_id
        merged_annotations.append(new_annotation)
        next_annotation_id += 1

    # Prepare the final merged file
    merged_coco = {
        "images": list(images.values()),
        "annotations": merged_annotations,
        "categories": coco1['categories']  # Assuming categories are the same
    }

    # Save to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_coco, f, indent=4)

    print(f"Merged annotations saved to {output_file}")

# ====== SETUP ======
# Replace these paths with your own file paths
file1 = "path/to/first_annotation_file.json"  # First COCO annotation file (e.g., false positives)
file2 = "path/to/second_annotation_file.json"  # Second COCO annotation file (e.g., ground truth)
output_file = "path/to/output_merged_file.json"  # Path to save the merged file # contains ground truth + false positives from model

# Merge annotations
merge_coco_annotations(file1, file2, output_file)
