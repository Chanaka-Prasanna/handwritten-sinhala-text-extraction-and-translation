import os
import pandas as pd

# Load label mapping from CSV
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    label_mapping = dict(zip(df['id'], df['value']))
    return label_mapping

# Load label mapping
label_mapping = load_label_mapping(r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\labels_with_ids.csv')

# Create ground truth text files
def create_ground_truth_files(base_dir, label_mapping, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_folder in os.listdir(base_dir):
        class_folder_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_folder_path):
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                image_base_name = os.path.splitext(image_name)[0]
                text_file_path = os.path.join(output_dir, image_base_name + '.txt')

                # Assuming the folder name is the class label
                folder_label = int(class_folder)
                ground_truth_text = label_mapping.get(folder_label, 'Unknown')

                # Write ground truth text to file
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(ground_truth_text)

# Set base and output directories
base_dir = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\data\train'
output_dir = r'D:\AI_ML_Internship_Assignment_Chanaka_Prasanna_Dissanayaka\fine-tune'

# Run the function
create_ground_truth_files(base_dir, label_mapping, output_dir)
