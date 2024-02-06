from transformers import pipeline, SamModel, SamProcessor
import torch
import numpy as np
from PIL import Image
import io
import os

# Initialize OWLv2 detector and SAM models
checkpoint = "google/owlvit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cpu")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def main(image_data, texts, threshold, result_folder="results"):
    """
    Main function to perform object detection and mask generation.

    Parameters:
        image: Image data.
        texts: Comma-separated candidate labels for object detection.
        threshold: Confidence threshold for object detection.

    Returns:
        tuple: Image data and result labels with masks.
    """
    texts = texts.split(",")

    # Load image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Perform object detection on the image
    predictions = detector(
        image,
        candidate_labels=texts,
        threshold=threshold
    )

    # Initialize a list to store the result labels
    result_labels = []
    for pred in predictions:
        # Extract bounding box coordinates
        box = pred["box"]
        score = pred["score"]
        label = pred["label"]
        box = [round(pred["box"]["xmin"], 2), round(pred["box"]["ymin"], 2), 
               round(pred["box"]["xmax"], 2), round(pred["box"]["ymax"], 2)]

        # Prepare inputs for mask generation using SAM
        inputs = sam_processor(
                image,
                input_boxes=[[[box]]],
                return_tensors="pt"
            ).to("cpu")
        
        # Perform mask generation using SAM model
        with torch.no_grad():
            outputs = sam_model(**inputs)
        
        # Post-process the generated mask
        mask = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0][0].numpy()
        mask = mask[np.newaxis, ...]

        # Append the mask and corresponding label to the result labels list
        result_labels.append((mask, label))

    # # Return the result labels dictionary
    # return image, result_labels

   # Create the result folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save the processed image to the result folder
    result_image_path = os.path.join(result_folder, "processed_image.jpeg")
    image.save(result_image_path)

    print("Processed image saved:", result_image_path)
    print("Result Labels:", result_labels)

if __name__ == "__main__":
    # Example usage (for integration to CVAT read concept document)
    image_path = "./sample_image.jpeg"
    with open(image_path, "rb") as f:
        image_data = f.read()

    texts = "cat,dog"
    threshold = 0.1

    main(image_data, texts=texts, threshold=threshold)
