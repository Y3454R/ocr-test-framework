import os
import cv2
import pytesseract
from jiwer import wer, cer
from sklearn.metrics import confusion_matrix
import numpy as np

def preprocess_image(image_path):
    # Read and preprocess the image as before
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def ocr_image(image):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯–'
    text = pytesseract.image_to_string(image, lang='ben', config=custom_config)
    return text.strip()

def read_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read().strip()
    return gt_text

def evaluate_accuracy_with_confusion_matrix(image_folder, gt_folder):
    # Initialize lists to store actual and predicted digits
    actual_digits = []
    predicted_digits = []

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

            # Make sure the ground truth and OCR results have the same length
            if len(ground_truth_text) == len(ocr_text):
                for gt_char, ocr_char in zip(ground_truth_text, ocr_text):
                    if gt_char in '০১২৩৪৫৬৭৮৯' and ocr_char in '০১২৩৪৫৬৭৮৯':
                        actual_digits.append(gt_char)
                        predicted_digits.append(ocr_char)

    # Create a confusion matrix
    labels = list('০১২৩৪৫৬৭৮৯')
    cm = confusion_matrix(actual_digits, predicted_digits, labels=labels)

    # Calculate misclassification percentages for each digit
    misclassifications = {}
    for i, label in enumerate(labels):
        total = sum(cm[i])
        if total > 0:
            for j, other_label in enumerate(labels):
                if i != j:
                    percent = (cm[i, j] / total) * 100
                    if label not in misclassifications:
                        misclassifications[label] = {}
                    misclassifications[label][other_label] = percent

    # Display the misclassification percentages in descending order
    for digit, mistakes in misclassifications.items():
        sorted_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)
        print(f"Digit '{digit}' mistakes:")
        for other_digit, percent in sorted_mistakes:
            print(f"  Mistaken as '{other_digit}': {percent:.2f}%")
        print()

# Usage example
if __name__ == '__main__':
    image_folder = 'number_plates'  # Folder containing .png files
    gt_folder = 'output_gt_txts'  # Folder containing .gt.txt files
    evaluate_accuracy_with_confusion_matrix(image_folder, gt_folder)
