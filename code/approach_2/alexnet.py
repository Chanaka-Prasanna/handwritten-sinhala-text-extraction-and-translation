import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Constants
IMG_SIZE = 28
NUM_CLASSES = 454  # Update with the actual number of classes

# Load label mapping from CSV
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    label_mapping = dict(zip(df['id'], df['value']))
    return label_mapping

# Image Preprocessing Functions
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_threshold(image):
    _, threshold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return threshold_image

def preprocess_image(image):
    gray_image = convert_to_grayscale(image)
    noise_removed_image = remove_noise(gray_image)
    threshold_image = apply_threshold(noise_removed_image)
    resized_image = cv2.resize(threshold_image, (IMG_SIZE, IMG_SIZE))
    return resized_image

# Load Images
def load_images_from_folders(base_dir, label_mapping):
    images = []
    labels = []
    for class_folder in os.listdir(base_dir):
        class_folder_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_folder_path):
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                image = cv2.imread(image_path)
                preprocessed_image = preprocess_image(image)
                images.append(preprocessed_image)
                
                # Convert class_folder name to the true label
                folder_label = int(class_folder)
                labels.append(label_mapping.get(folder_label, 'Unknown'))
    return np.array(images), np.array(labels)

# Define AlexNet Model
def create_alexnet_model(num_classes):
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(IMG_SIZE, IMG_SIZE, 1), kernel_size=(7,7), strides=(2,2), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    # Second Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    # Third Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # Fourth Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # Fifth Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Code
if __name__ == "__main__":
    # Load label mapping
    label_mapping = load_label_mapping(r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\labels_with_ids.csv')
    
    # Load images and labels
    base_dir = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\train'  # Update with your dataset path
    images, labels = load_images_from_folders(base_dir, label_mapping)
    
    # Preprocess images
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded, num_classes=NUM_CLASSES)


    # Split dataset
    X_train = images
    y_train = labels_encoded
    
    # Create and train the model
    model = create_alexnet_model(NUM_CLASSES)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Save the model
    model.save('alexnet_sinhala_character_recognition.h5')
    print("Model saved to disk.")

    # Load images and labels for evaluation
    base_dir = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\test'  # Update with your dataset path
    images_t, labels_t = load_images_from_folders(base_dir, label_mapping)
    
    # Preprocess images
    images_t = images_t.astype('float32') / 255.0
    images_t = np.expand_dims(images_t, axis=-1)  # Add channel dimension
    
    # Encode labels
    labels_encoded_t = label_encoder.transform(labels_t)
    labels_encoded_t = to_categorical(labels_encoded_t, num_classes=NUM_CLASSES)
    
    # Split dataset
    X_test = images_t
    y_test = labels_encoded_t
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    
    # Predict on new images
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Display some results
    for i in range(5):
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title(f'Predicted: {label_encoder.inverse_transform([predicted_classes[i]])[0]}')
        plt.show()

    # Save the LabelEncoder for future predictions
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)
    print("Label encoder saved to disk.")
