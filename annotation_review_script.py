import json
import cv2
import os
import sys
from tkinter import Tk, simpledialog

# ======== CONFIGURATION INSTRUCTIONS ========
# Before using this script, update the following variables:
# 1. DATASET_PATH: Full path to the directory containing your image files
# 2. INPUT_JSON: Full path to your COCO-format annotation JSON file
# 3. OUTPUT_JSON: Desired path for saving filtered annotations
# 4. WINDOW_NAME: Optional - change if you want a different window title

# Path configurations (MODIFY THESE)
DATASET_PATH = "/path/to/your/image/dataset/directory"
INPUT_JSON = "/path/to/your/annotation/file.json"
OUTPUT_JSON = "/path/to/save/filtered_annotations.json"
WINDOW_NAME = "Annotation Review"

# Required dependencies:
# - Python 3.x
# - OpenCV (cv2)
# - Tkinter
# - json module

# Recommended setup:
# 1. Create a virtual environment
# 2. Install dependencies: 
#    pip install opencv-python

# Load the JSON data
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

# Map image IDs to file names for quick lookup
image_map = {img["id"]: img["file_name"] for img in images}

# Map category IDs to category names
category_map = {cat["id"]: cat["name"] for cat in categories}

# Map file names to image IDs for navigation
file_name_map = {img["file_name"]: img["id"] for img in images}

# Initialize variables for navigation
current_idx = 0
reviewed_ids = {}  # Dictionary to store decisions for annotations by ID

def find_start_index(start_file_name):
    """
    Find the start index of the image based on the file name.
    Returns the index of the first annotation for the given file.
    """
    if start_file_name in file_name_map:
        image_id = file_name_map[start_file_name]
        for idx, annotation in enumerate(annotations):
            if annotation["image_id"] == image_id:
                return idx
    else:
        print(f"Image file '{start_file_name}' not found.")
        return None

def edit_annotation(annotation):
    """
    Open a popup to edit the category ID of the current annotation.
    Prompts the user to enter a new category ID and updates the annotation if valid.
    """
    # Initialize Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the main window

    # Ask for new category ID
    new_category_id = simpledialog.askinteger(
        "Edit Annotation",
        f"Enter new category ID for annotation (current: {annotation['category_id']}):"
    )
    if new_category_id and new_category_id in category_map:
        annotation["category_id"] = new_category_id
        print(f"Updated category ID to {new_category_id} ({category_map[new_category_id]})")
    elif new_category_id:
        print(f"Invalid category ID: {new_category_id}")

def get_box_color(annotation_id):
    """
    Return the color for the bounding box based on its review status.
    Blue: Unreviewed, Green: Accepted, Red: Rejected.
    """
    if annotation_id not in reviewed_ids:
        return (255, 0, 0)  # Blue for unreviewed
    elif reviewed_ids[annotation_id] == "accept":
        return (0, 255, 0)  # Green for accepted
    else:  # "reject"
        return (0, 0, 255)  # Red for rejected

def log_problematic_image(image_file):
    """
    Log the problematic image filename to a text file, avoiding duplicates.
    """
    with open("problematic_images.txt", "a+") as log_file:
        log_file.seek(0)  # Move cursor to the beginning to read existing content
        existing_logs = log_file.read().splitlines()

        if image_file not in existing_logs:
            log_file.write(f"{image_file}\n")
            print(f"Logged image: {image_file}")
        else:
            print(f"Image {image_file} already logged.")

def display_annotation(idx):
    """
    Display the current annotation using OpenCV and provide options for review.
    """
    global current_idx
    current_idx = idx

    annotation = annotations[current_idx]
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    category_name = category_map[category_id]  # Get category name
    image_file = image_map[image_id]
    image_path = os.path.join(DATASET_PATH, image_file)

    # Load the image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    def draw_all_accepted_bboxes(base_img):
        """
        Draw all previously accepted bounding boxes on the image.
        """
        img_with_boxes = base_img.copy()
        for ann in annotations:
            if ann["image_id"] == image_id and ann["id"] in reviewed_ids and reviewed_ids[ann["id"]] == "accept":
                x, y, w, h = map(int, ann["bbox"])
                cv2.rectangle(
                    img_with_boxes, 
                    (x, y), 
                    (x + w, y + h), 
                    (255, 255, 0),  # Green color for accepted boxes
                    2
                )
                cv2.putText(
                    img_with_boxes,
                    f"{category_map[ann['category_id']]}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
        return img_with_boxes

    def draw_current_bbox():
        """
        Draw the bounding box and category label on the image, including accepted boxes.
        """
        img_copy = draw_all_accepted_bboxes(img)
        x, y, w, h = map(int, bbox)

        # Get color based on review status
        color = get_box_color(annotation["id"])

        # Draw the box and category label
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img_copy,
            f"{category_map[annotation['category_id']]}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Add image filename and box ID information with white background
        info_text = f"Image: {image_file} | Box ID: {annotation['id']}"

        # Get text size
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)

        # Calculate position for centered text
        img_width = img_copy.shape[1]
        text_x = (img_width - text_width) // 2
        text_y = 30

        # Draw white background rectangle
        padding = 5
        cv2.rectangle(img_copy, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (255, 255, 255),
                     -1)  # Filled rectangle

        # Draw text
        cv2.putText(
            img_copy,
            info_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            thickness
        )

        cv2.imshow(WINDOW_NAME, img_copy)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    # Draw the current bounding box
    draw_current_bbox()

    # Wait for key press (a = accept, r = reject, e = edit, q = quit, p = previous, n = next)
    while True:
        key = cv2.waitKey(1)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Saving and exiting...")
            save_results()
            cv2.destroyAllWindows()
            sys.exit()

        if key == ord("a"):
            reviewed_ids[annotation["id"]] = "accept"
            print(f"Accepted annotation {current_idx + 1}")
            current_idx += 1
            break
        elif key == ord("r"):
            reviewed_ids[annotation["id"]] = "reject"
            print(f"Rejected annotation {current_idx + 1}")
            current_idx += 1
            break
        elif key == ord("n"):
            print(f"Skipping annotation {current_idx + 1}")
            current_idx += 1
            break
        elif key == ord("e"):
            edit_annotation(annotation)
            draw_current_bbox()
        elif key == ord("p"):
            if current_idx > 0:
                current_idx -= 1
                print(f"Going back to annotation {current_idx + 1}")
                break
            else:
                print("Already at the first annotation.")
        elif key == ord("l"):
            log_problematic_image(image_file)
        elif key == ord("q"):
            print("Quitting early.")
            save_results()
            cv2.destroyAllWindows()
            sys.exit()

    if current_idx >= len(annotations):
        print("Finished reviewing all annotations.")
        save_results()
        cv2.destroyAllWindows()
        sys.exit()
    else:
        display_annotation(current_idx)

def save_results():
    """
    Save only the accepted annotations in the same COCO format.
    """
    filtered_annotations = [
        ann for ann in annotations if ann["id"] in reviewed_ids and reviewed_ids[ann["id"]] == "accept"
    ]

    output_data = {
        "images": images,
        "annotations": filtered_annotations,
        "categories": categories,
    }

    last_image_file = "N/A"
    if current_idx < len(annotations):
        last_image_id = annotations[current_idx]["image_id"]
        last_image_file = image_map.get(last_image_id, "Unknown")

    with open(OUTPUT_JSON, "w") as output_file:
        json.dump(output_data, output_file, indent=4)
        print(f"Results saved to {OUTPUT_JSON} with {len(filtered_annotations)} annotations.")
        print(f"Last processed image: {last_image_file}")

# Start reviewing annotations
try:
    while True:
        choice = input("Do you want to:\n1. Start from the first image\n2. Start from a specific image\nEnter your choice (1 or 2): ")

        if choice == "1":
            current_idx = 0
            break
        elif choice == "2":
            start_file_name = input("Enter the image file name to start from: ")
            current_idx = find_start_index(start_file_name)
            if current_idx is not None:
                break
            else:
                print("Please try again with a valid filename.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

    display_annotation(current_idx)

except KeyboardInterrupt:
    print("\nProgram interrupted by user. Saving and exiting...")
    save_results()
    cv2.destroyAllWindows()
    sys.exit()
