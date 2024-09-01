import cv2
import numpy as np
import pandas as pd
import easyocr
from sklearn.cluster import DBSCAN
import argparse
import os
import glob
from collections import Counter
import re


def clean_numbers(li):
    # Preprocess each item in the list using the clean_text function
    processed_list = [clean_text(num) for num in li]
    
    # Calculate starts
    starts = [num[0] for num in processed_list if len(num)>1]  # Check for non-empty strings
    start_count = Counter(starts)
    
    # Ensure there is at least one most common start character
    if start_count:
        most_common, count = start_count.most_common(1)[0]
        
        # Check if the most common start character appears in at least 70% of the items
        if count / len(li) >= 0.5:
            # Remove the most common start character from items starting with it
            new_list = [re.sub(r'^' + re.escape(most_common), '', num) for num in processed_list]
            return new_list
    return processed_list

def clean_text(text):
    # Check if the first character is not a number, $, or minus sign and remove it
    if re.match(r'^[^0-9$-]', text):
        text = text[1:]  # Remove the first character

    text = re.sub(r'^[^0-9$]+', '', text)
    # Remove any unwanted character before a minus sign
    text = re.sub(r'(.)(?=-)', '', text)
    # Remove leading zeros if there's a minus sign followed by zeros and then a digit
    text = re.sub(r'(?<=^-)0+(?=\d)', '', text)
    text = re.sub(r'0+(?=\d)', '', text)
    if text == '':
      text = '0'
    
    return text

def split_image(image, num_rows, num_cols, overlap=0):
    """Split the image into smaller parts."""
    img_height, img_width = image.shape[:2]
    tile_height = img_height // num_rows
    tile_width = img_width // num_cols
    overlap_height = int(tile_height * overlap)
    overlap_width = int(tile_width * overlap)

    tiles = []
    coordinates = []
    for row in range(num_rows):
        for col in range(num_cols):
            y_start = max(row * tile_height - overlap_height, 0)
            y_end = min((row + 1) * tile_height + overlap_height, img_height)
            x_start = max(col * tile_width - overlap_width, 0)
            x_end = min((col + 1) * tile_width + overlap_width, img_width)
            tiles.append(image[y_start:y_end, x_start:x_end])
            coordinates.append((x_start, y_start))
    
    return tiles, coordinates, img_width, img_height

def perform_ocr_on_tiles(tiles, coordinates, reader):
    results = []
    for idx, (tile, (x_start, y_start)) in enumerate(zip(tiles, coordinates)):
        # Perform OCR on the original image
        ocr_result_original = reader.readtext(tile)
        for (bbox, text, conf) in ocr_result_original:
            bbox = np.array(bbox)
            bbox[:, 0] += x_start
            bbox[:, 1] += y_start
            results.append((bbox, text, conf))
    return results

def merge_nearby_results(results, iou_threshold=0.5):
    """Eliminate duplicate and proximate results using DBSCAN clustering based on bounding box centers."""
    if not results:
        print("No results found.")
        return [], [], []
    
    # Extracting text and bounding boxes from results
    texts = [result[1] for result in results]
    old_bboxes = [result[0] for result in results]
    new_texts = clean_numbers(texts)
    

    # Ensure bounding boxes are numpy arrays for processing
    bboxes = [np.array(bbox) if not isinstance(bbox, np.ndarray) else bbox for bbox in old_bboxes]

    # Calculate centers of bounding boxes
    centers = [(np.mean(bbox[:, 0]), np.mean(bbox[:, 1])) for bbox in bboxes]
    centers = np.array(centers)

    if len(centers) == 0:
        print("No centers found.")
        return [], [], []

    # Cluster centers using DBSCAN
    clustering = DBSCAN(eps=20, min_samples=1).fit(centers)
    labels = clustering.labels_
    
    # Collect unique results based on cluster labels
    unique_results = []
    unique_cen = []
    unique_texts = []
    seen_labels = set()

    for i, label in enumerate(labels):
        # Clean the text
        bbox, text, conf = results[i] # Unpack the tuple
        # Clean the text
        cleaned_text = new_texts[i]
        
        # Create a new tuple with the cleaned text
        modified_result = (bbox, cleaned_text, conf)

        # Check if the text contains more than one character
        if label not in seen_labels and len(text) > 1:
            seen_labels.add(label)
            unique_results.append(modified_result)
            unique_cen.append(centers[i])
            unique_texts.append(cleaned_text)

    return unique_results, unique_cen, unique_texts

def main(image_path, num_rows, num_cols, overlap):
    # Load the image
    image = cv2.imread(image_path)[50:222, 73:]
    scale_factor = 2

    # Prepare masks and filters
    BLACK_MIN = np.array([0, 0, 0], np.uint8)  
    BLACK_MAX = np.array([180, 255, 255], np.uint8)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(image, BLACK_MIN, BLACK_MAX)
    inverted_mask = cv2.bitwise_not(frame_threshed)
    
    # Resize the image to enhance details
    resized_image = cv2.resize(inverted_mask, (inverted_mask.shape[1] * scale_factor, inverted_mask.shape[0] * scale_factor), interpolation=cv2.INTER_AREA)
    resized_image_rotate = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

    # Split the image into parts
    tiles, coordinates, _, _ = split_image(resized_image, num_cols, num_rows, overlap)
    tiles_rotate, coordinates_rotate, _, _ = split_image(resized_image_rotate, num_rows, num_cols, overlap)
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on image parts
    results = perform_ocr_on_tiles(tiles, coordinates, reader)
    results_rotate = perform_ocr_on_tiles(tiles_rotate, coordinates_rotate, reader)
    
    merged_results, unique_cen, unique_texts = merge_nearby_results(results, iou_threshold=0.2)
    merged_results_rotate, unique_cen_rotate, unique_texts_rotate = merge_nearby_results(results_rotate, iou_threshold=0.5)
    
    unique_new_rotate = [array[::-1] for array in unique_cen_rotate]
    combined_cen = unique_cen + unique_new_rotate
    combined_texts = unique_texts + unique_texts_rotate
    combined = list(zip(combined_cen, combined_texts))
    
    # Sort the combined list based on the first element of each array
    sorted_combined = sorted(combined, key=lambda x: x[0][0])

    # Unpack the sorted lists, verifying they are not empty
    if sorted_combined:
        sorted_cen, sorted_texts_ = zip(*sorted_combined)
        frist_sen = 20
        sorted_texts = []
        for i in range(len(sorted_cen)):
          new_cen = int(sorted_cen[i][0])
          distance = ((new_cen - frist_sen)//38)-1
          sorted_texts.extend(['0']*distance)
          sorted_texts.append(sorted_texts_[i])
          frist_sen = new_cen
          if i == len(sorted_cen)-1:
            new_cen = resized_image.shape[1]
            distance = (new_cen - frist_sen-41)//38
            sorted_texts.extend(['0']*distance)

            
        return sorted_texts
    else:
        print("No results found to sort.")
        return []

def list_images(directory):
    """List .png and .jpg files in the specified directory."""
    # Create a pattern to match files
    patterns = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []

    # Loop through each pattern and extend the list with matched paths
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(directory, pattern)))

    return image_paths


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="OCR Image Processing")
    parser.add_argument("folder_path", type=str, help="Path to the folder images")
    parser.add_argument("--rows", type=int, default=20, help="Number of rows to split the image")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns to split the image")
    parser.add_argument("--overlap", type=float, default=0.3, help="Percentage of overlap between tiles (e.g., 0.1 for 10%)")
    args = parser.parse_args()
    images = list_images(args.folder_path)

    data = {}
    for index, img_path in enumerate(images):
        sorted_texts = main(img_path, args.rows, args.cols, args.overlap)
        data[img_path] = sorted_texts
        print(f"Processed {index + 1} of {len(images)} images.")
        print({img_path:sorted_texts})

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Image_Path', 'OCR_Text'])

    # Save the DataFrame to a CSV file
    df.to_csv('ocr_output.csv', index=False)

    print("OCR processing complete. Results saved to 'ocr_output.csv'.")


