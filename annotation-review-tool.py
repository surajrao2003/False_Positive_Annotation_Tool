import json
import cv2
import os
import sys
from tkinter import Tk, simpledialog

class AnnotationReviewer:
    def __init__(self, dataset_path, input_json, output_json):
        """
        Initialize the annotation reviewer.
        
        :param dataset_path: Path to the directory containing images
        :param input_json: Path to the input COCO annotation JSON file
        :param output_json: Path to save the reviewed annotations
        """
        # Path configurations
        self.DATASET_PATH = dataset_path
        self.INPUT_JSON = input_json
        self.OUTPUT_JSON = output_json
        self.WINDOW_NAME = "Annotation Review"

        # Load the JSON data
        with open(self.INPUT_JSON, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        # Create mapping dictionaries
        self.image_map = {img["id"]: img["file_name"] for img in self.images}
        self.category_map = {cat["id"]: cat["name"] for cat in self.categories}
        self.file_name_map = {img["file_name"]: img["id"] for img in self.images}

        # Initialize review tracking
        self.current_idx = 0
        self.reviewed_ids = {}

    def find_start_index(self, start_file_name):
        """Find the start index of the image based on the file name."""
        if start_file_name in self.file_name_map:
            image_id = self.file_name_map[start_file_name]
            for idx, annotation in enumerate(self.annotations):
                if annotation["image_id"] == image_id:
                    return idx
        else:
            print(f"Image file '{start_file_name}' not found.")
            return None

    def edit_annotation(self, annotation):
        """Open a popup to edit the category ID of the current annotation."""
        root = Tk()
        root.withdraw()

        new_category_id = simpledialog.askinteger(
            "Edit Annotation",
            f"Enter new category ID for annotation (current: {annotation['category_id']}):"
        )
        if new_category_id and new_category_id in self.category_map:
            annotation["category_id"] = new_category_id
            print(f"Updated category ID to {new_category_id} ({self.category_map[new_category_id]})")
        elif new_category_id:
            print(f"Invalid category ID: {new_category_id}")

    def get_box_color(self, annotation_id):
        """Return the color for the bounding box based on its review status."""
        if annotation_id not in self.reviewed_ids:
            return (255, 0, 0)  # Blue for unreviewed
        elif self.reviewed_ids[annotation_id] == "accept":
            return (0, 255, 0)  # Green for accepted
        else:  # "reject"
            return (0, 0, 255)  # Red for rejected

    def display_annotation(self, idx):
        """Display the current annotation using OpenCV."""
        self.current_idx = idx
        annotation = self.annotations[self.current_idx]
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        image_file = self.image_map[image_id]
        image_path = os.path.join(self.DATASET_PATH, image_file)

        # Load the image
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return

        def draw_current_bbox():
            """Draw the bounding box and category label on the image."""
            img_copy = img.copy()
            x, y, w, h = map(int, bbox)
            
            color = self.get_box_color(annotation["id"])
            
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img_copy,
                f"{self.category_map[annotation['category_id']]}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            
            info_text = f"Image: {image_file} | Box ID: {annotation['id']}"
            
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            
            img_width = img_copy.shape[1]
            text_x = (img_width - text_width) // 2
            text_y = 30
            
            cv2.rectangle(img_copy, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (255, 255, 255),
                         -1)
            
            cv2.putText(
                img_copy,
                info_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
            
            cv2.imshow(self.WINDOW_NAME, img_copy)
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

        draw_current_bbox()

        while True:
            key = cv2.waitKey(1)
            
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Saving and exiting...")
                self.save_results()
                cv2.destroyAllWindows()
                sys.exit()
                
            if key == ord("a"):  # Accept
                self.reviewed_ids[annotation["id"]] = "accept"
                print(f"Accepted annotation {self.current_idx + 1}")
                self.current_idx += 1
                break
            elif key == ord("r"):  # Reject
                self.reviewed_ids[annotation["id"]] = "reject"
                print(f"Rejected annotation {self.current_idx + 1}")
                self.current_idx += 1
                break
            elif key == ord("n"):  # Next without decision
                print(f"Skipping annotation {self.current_idx + 1}")
                self.current_idx += 1
                break
            elif key == ord("e"):  # Edit
                self.edit_annotation(annotation)
                draw_current_bbox()
            elif key == ord("p"):  # Previous
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print(f"Going back to annotation {self.current_idx + 1}")
                    break
                else:
                    print("Already at the first annotation.")
            elif key == ord("q"):  # Quit early
                print("Quitting early.")
                self.save_results()
                cv2.destroyAllWindows()
                sys.exit()

        if self.current_idx >= len(self.annotations):
            print("Finished reviewing all annotations.")
            self.save_results()
            cv2.destroyAllWindows()
            sys.exit()
        else:
            self.display_annotation(self.current_idx)

    def save_results(self):
        """Save only the accepted annotations in the same COCO format."""
        filtered_annotations = [
            ann for ann in self.annotations if ann["id"] in self.reviewed_ids and self.reviewed_ids[ann["id"]] == "accept"
        ]

        output_data = {
            "images": self.images,
            "annotations": filtered_annotations,
            "categories": self.categories,
        }

        with open(self.OUTPUT_JSON, "w") as output_file:
            json.dump(output_data, output_file, indent=4)
            print(f"Results saved to {self.OUTPUT_JSON} with {len(filtered_annotations)} annotations.")

    def start_review(self):
        """Start the review process."""
        try:
            while True:
                choice = input("Do you want to:\n1. Start from the first image\n2. Start from a specific image\nEnter your choice (1 or 2): ")
                
                if choice == "1":
                    self.current_idx = 0
                    break
                elif choice == "2":
                    start_file_name = input("Enter the image file name to start from: ")
                    current_idx = self.find_start_index(start_file_name)
                    if current_idx is not None:
                        self.current_idx = current_idx
                        break
                    else:
                        print("Please try again with a valid filename.")
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            
            self.display_annotation(self.current_idx)
            
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Saving and exiting...")
            self.save_results()
            cv2.destroyAllWindows()
            sys.exit()

def main():
    """Main function to run the annotation review tool."""
    # Replace these with your actual paths or pass as command-line arguments
    DATASET_PATH = "path/to/your/image/directory"
    INPUT_JSON = "path/to/input/annotations.json"
    OUTPUT_JSON = "reviewed_annotations.json"

    reviewer = AnnotationReviewer(DATASET_PATH, INPUT_JSON, OUTPUT_JSON)
    reviewer.start_review()

if __name__ == "__main__":
    main()
