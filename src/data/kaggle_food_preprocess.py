import os
import pandas as pd

# Function to remove rows where 'Image_Name' contains '#NAME'
def remove_invalid_image_names(csv_file_path, output_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove rows where the 'Image_Name' column contains '#NAME'
    df_cleaned = df[~df['Image_Name'].astype(str).str.contains('#NAME')]

    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(output_file_path, index=False)
    print(f"Cleaned CSV saved to {output_file_path}")

# Specify the path to your CSV file and the output file
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'KaggleFoodDataset', 'data.csv')
output_file_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'KaggleFoodDataset', 'data.csv')

# Call the function with the specified paths
remove_invalid_image_names(csv_file_path, output_file_path)
