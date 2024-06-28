import os
import cv2
import pandas as pd
import numpy as np


def process_lfw_dataset(raw_data_path, processed_data_path, lfw_allnames_file):
    # Read people file
    people_data = pd.read_csv(lfw_allnames_file)

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    print("Starting image processing...")
    for index, row in people_data.iterrows():
        person = row.iloc[0]  # Accessing first column by position
        images_count_raw = row.iloc[1]  # Accessing second column by position

        # Check if images_count_raw is NaN
        if pd.isna(images_count_raw):
            print(
                f"Warning: Invalid or missing value for images count in row {index + 1}. Skipping.")
            continue

        images_count = int(images_count_raw)  # Convert to integer

        person_path = os.path.join(raw_data_path, person)

        if not os.path.exists(person_path):
            print(f"Warning: Directory for {person} not found")
            continue

        # Check if the directory contains any .jpg files
        jpg_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        if not jpg_files:
            print(f"\nError: No images found for {person}. Skipping.")
            continue

        print(f"\nProcessing images for {person}...")
        for i in range(1, images_count + 1):
            img_path = os.path.join(
                person_path, f"{person}_{str(i).zfill(4)}.jpg")
            image = cv2.imread(img_path)
            if image is None:
                print(
                    f"Image not found for {person}_{str(i).zfill(4)}.jpg. Skipping...")
                continue

            # Resizing the image
            processed_image = cv2.resize(image, (128, 128))

            # Save the processed image
            save_path = os.path.join(processed_data_path, person)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(
                save_path, f"{person}_{str(i).zfill(4)}.jpg"), processed_image)

        print(f"\nFinished processing for {person}")

    print("\n\n******************Processing complete.******************")


if __name__ == "__main__":
    raw_data_path = r'C:/Users/mdutt/Desktop/Face_REC/data/Raw_img/lfw/lfw-deepfunneled'
    processed_data_path = r'C:/Users/mdutt/Desktop/Face_REC/data/processed'
    lfw_allnames_file = r'C:/Users/mdutt/Desktop/Face_REC/Datasets/lfw_allnames.csv'

    process_lfw_dataset(raw_data_path, processed_data_path, lfw_allnames_file)
