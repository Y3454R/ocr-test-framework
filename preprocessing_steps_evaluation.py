import os
import cv2
import pytesseract
from jiwer import wer, cer  # You might need to install jiwer using `pip install jiwer`
from preprocess_test import increase_contrast, get_grayscale, adaptive_threshold, smooth_image, dilate_then_erode


def preprocess_image_steps(image):
    """
    Apply each preprocessing step to the image and return them as a list.
    """
    gray = get_grayscale(image)
    contrast_img = increase_contrast(gray)
    threshold_img = adaptive_threshold(contrast_img)
    smoothed_img = smooth_image(threshold_img)
    final_img = dilate_then_erode(smoothed_img, kernel_size=2)

    # Return the original image and all the steps
    return [image, gray, contrast_img, threshold_img, smoothed_img, final_img]


def ocr_image(image):
    """
    Perform OCR using Tesseract on the given image and return the text.
    """
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯–'  # Bengali digits and hyphen
    text = pytesseract.image_to_string(image, lang='ben', config=custom_config)
    return text.strip()


def read_ground_truth(gt_path):
    """
    Read the ground truth text from the corresponding .gt.txt file.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read().strip()
    return gt_text


def evaluate_accuracy_per_step(image_folder, gt_folder):
    """
    Evaluate CER and WER for the original image and after each preprocessing step.
    """
    cer_scores = {step: [] for step in range(6)}  # 6 stages: original + 5 preprocessing steps
    wer_scores = {step: [] for step in range(6)}

    # List all image files in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            gt_path = os.path.join(gt_folder, filename.replace('.png', '.gt.txt'))

            # Check if the corresponding ground truth file exists
            if not os.path.isfile(gt_path):
                print(f"Ground truth file not found for {filename}")
                continue

            # Read the image and ground truth text
            image = cv2.imread(image_path)
            ground_truth_text = read_ground_truth(gt_path)

            # Apply preprocessing steps and get intermediate images
            images = preprocess_image_steps(image)

            # Perform OCR for each step and calculate CER and WER
            for step, img in enumerate(images):
                ocr_text = ocr_image(img)

                # Calculate CER and WER for the current step
                current_cer = cer(ground_truth_text, ocr_text)
                current_wer = wer(ground_truth_text, ocr_text)

                # Append the scores to the lists for each step
                cer_scores[step].append(current_cer)
                wer_scores[step].append(current_wer)

                # Display results for each step
                step_name = ['Original', 'Grayscale', 'Contrast', 'Threshold', 'Smoothed', 'Dilated & Eroded'][step]
                print(f"File: {filename} | Step: {step_name}")
                print(f"Ground Truth: {ground_truth_text}")
                print(f"OCR Output: {ocr_text}")
                print(f"CER: {current_cer:.4f}, WER: {current_wer:.4f}\n")

    # Calculate average CER and WER across all images for each step
    for step in range(6):
        avg_cer = sum(cer_scores[step]) / len(cer_scores[step]) if cer_scores[step] else 0
        avg_wer = sum(wer_scores[step]) / len(wer_scores[step]) if wer_scores[step] else 0

        step_name = ['Original', 'Grayscale', 'Contrast', 'Threshold', 'Smoothed', 'Dilated & Eroded'][step]
        print(f"Average CER for {step_name}: {avg_cer:.4f}")
        print(f"Average WER for {step_name}: {avg_wer:.4f}\n")


# Usage example
if __name__ == '__main__':
    image_folder = 'number_plates'  # Folder containing .png files
    gt_folder = 'output_gt_txts'  # Folder containing .gt.txt files
    evaluate_accuracy_per_step(image_folder, gt_folder)
