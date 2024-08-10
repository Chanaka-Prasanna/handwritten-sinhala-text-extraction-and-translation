import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 28
NUM_CLASSES = 454  # Update with the actual number of classes

# Image Preprocessing Functions
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)

def preprocess_image(image):
    gray_image = convert_to_grayscale(image)
    noise_removed_image = remove_noise(gray_image)
    threshold_image = apply_adaptive_threshold(noise_removed_image)
    return threshold_image

def segment_image(image):
    kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes and sort by middle point
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter out small contours
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                bounding_boxes.append([x, y, w, h, cX])
    
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[4])  # Sort by center x-coordinate
    segmented_images = []

    for box in bounding_boxes:
        x, y, w, h, _ = box
        segment = image[y:y+h, x:x+w]
        if segment.shape[0] > 0 and segment.shape[1] > 0:
            resized_segment = cv2.resize(segment, (IMG_SIZE, IMG_SIZE))
            segmented_images.append(resized_segment)
    
    return segmented_images

def predict_character(image, model, label_encoder):
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)[0]

# Load Model and Label Encoder
model = load_model('alexnet_sinhala_character_recognition.h5')
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def process_input_image(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    
    # Segment image into individual characters
    segmented_images = segment_image(preprocessed_image)
    
    # Predict each segmented character
    results = []
    for segmented in segmented_images:
        result = predict_character(segmented, model, label_encoder)
        results.append(result)
    
    return ''.join(results)

def save_results_to_file(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(results)

if __name__ == "__main__":
    # Update this with your input image path
    input_image_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\test\my-img.jpeg'
    output_file_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\results\extracted_text.txt'
    
    # Process image and get extracted text
    extracted_text = process_input_image(input_image_path)
    
    # Save extracted text to a file
    save_results_to_file(extracted_text, output_file_path)
    print("Extracted text saved to", output_file_path)
    
    # Optionally display segmented characters
    image = cv2.imread(input_image_path)
    preprocessed_image = preprocess_image(image)
    segmented_images = segment_image(preprocessed_image)
    
    for i, segmented in enumerate(segmented_images):
        plt.subplot(1, len(segmented_images), i + 1)
        plt.imshow(segmented, cmap='gray')
        plt.title('Segmented {}'.format(i + 1))
        plt.axis('off')
    
    plt.show()
