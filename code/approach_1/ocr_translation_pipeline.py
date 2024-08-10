import cv2
from pytesseract import pytesseract
from deep_translator import GoogleTranslator

def process_image(image_path, threshold_output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise using Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Save the thresholded image
    cv2.imwrite(threshold_output_path, thresh1)

    print(f"Processed image saved at: {threshold_output_path}")

    return thresh1, img

def perform_dilation(thresh1, dilation_output_path):
    # Define the structuring element
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    # Perform dilation
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=3)
    cv2.imwrite(dilation_output_path, dilation)

    print(f"Dilated image saved at: {dilation_output_path}")

    return dilation

def translate_text(text):
    try:
        # Using GoogleTranslator from deep_translator
        translator = GoogleTranslator(source='si', target='en')
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return "Translation failed"

def extract_bounding_boxes_and_text(dilation, img, bounding_boxes_output_path):
    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Accumulate all extracted text
    combined_text = ""
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the bounding box area
        cropped = img[y:y+h, x:x+w]
        cv2.imwrite('rectanglebox.jpg', img)  # Save the image with bounding boxes (optional)

        # Extract text from the cropped image area
        text = pytesseract.image_to_string(cropped, lang='sin')
        
        # Accumulate text and remove new lines
        combined_text += text.strip() + " "

    # Clean up extra spaces and new lines
    combined_text = ' '.join(combined_text.split())

    # Translate the combined text
    translated_text = translate_text(combined_text)

    # Save the combined and translated text
    with open("text_output.txt", "w", encoding='utf-8') as file:
        file.write(combined_text)
    
    with open("translated_text_output.txt", "w", encoding='utf-8') as trans_file:
        trans_file.write(translated_text)

    # Save the image with bounding boxes
    cv2.imwrite(bounding_boxes_output_path, img)
    print(f"Image with bounding boxes saved at: {bounding_boxes_output_path}")

def main():
    # Define paths
    image_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\test\my-img.jpeg'
    threshold_output_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\results\img_thresholding.jpg'
    dilation_output_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\results\dilation_image.jpg'
    bounding_boxes_output_path = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\results\bounding_boxes_image.jpg'
    
    # Process the image
    thresh1, img = process_image(image_path, threshold_output_path)
    
    # Perform dilation
    dilation = perform_dilation(thresh1, dilation_output_path)
    
    # Extract bounding boxes and text
    extract_bounding_boxes_and_text(dilation, img, bounding_boxes_output_path)

if __name__ == "__main__":
    # Specify the path to the Tesseract executable (update this path as necessary)
    pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main()
