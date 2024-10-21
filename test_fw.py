import os
import cv2
import pytesseract
from jiwer import wer, cer  # You might need to install jiwer using `pip install jiwer`

from preprocess_test import increase_contrast, get_grayscale, adaptive_threshold, smooth_image, dilate_then_erode

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Pre-process the image
    gray = get_grayscale(image)
    contrast_img = increase_contrast(gray)
    threshold_img = adaptive_threshold(contrast_img)
    smoothed_img = smooth_image(threshold_img)
    final_img = dilate_then_erode(smoothed_img, kernel_size=2)

    return final_img
    # return smoothed_img

def ocr_image(image):
    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯–'  # Bengali digits and hyphen
    text = pytesseract.image_to_string(image, lang='ben', config=custom_config)
    return text.strip()

def read_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read().strip()
    return gt_text

def evaluate_accuracy(image_folder, gt_folder):
    cer_scores = []
    wer_scores = []

    # List all the image files in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            gt_path = os.path.join(gt_folder, filename.replace('.png', '.gt.txt'))

            # Check if the corresponding ground truth file exists
            if not os.path.isfile(gt_path):
                print(f"Ground truth file not found for {filename}")
                continue

            # Preprocess the image and perform OCR
            preprocessed_image = preprocess_image(image_path)
            ocr_text = ocr_image(preprocessed_image)

            # Read the ground truth text
            ground_truth_text = read_ground_truth(gt_path)

            # Calculate CER and WER for the current file
            current_cer = cer(ground_truth_text, ocr_text)
            current_wer = wer(ground_truth_text, ocr_text)

            # Append the scores to the lists
            cer_scores.append(current_cer)
            wer_scores.append(current_wer)

            # Display results for each image
            print(f"File: {filename}")
            print(f"Ground Truth: {ground_truth_text}")
            print(f"OCR Output: {ocr_text}")
            print(f"CER: {current_cer:.4f}, WER: {current_wer:.4f}\n")

    # Calculate average CER and WER across all images
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0

    # Display final results
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")

# Usage example
if __name__ == '__main__':
    image_folder = 'number_plates'  # Folder containing .png files
    gt_folder = 'output_gt_txts'  # Folder containing .gt.txt files
    evaluate_accuracy(image_folder, gt_folder)
