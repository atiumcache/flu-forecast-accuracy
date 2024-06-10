import pandas as pd
import os

input_folder_path = './formatted-pf-predictions/'
output_folder_path = './formatted-pf-predictions/'


def standardize_file(file_path: str, output_folder: str):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert specified columns to strings without additional quotes
    df['output_type_id'] = df['output_type_id'].astype(str)
    df['location'] = df['location'].apply(lambda x: str(x).zfill(2))
    df['output_type'] = df['output_type'].astype(str)

    # Determine the output file path
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, file_name)

    # Save the modified dataframe to a new CSV
    df.to_csv(output_file_path, index=False)

    print(f"Converted file saved to {output_file_path}")

# Create the output directory if it does not exist
os.makedirs(output_folder_path, exist_ok=True)

# Process each file in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder_path, file_name)
        standardize_file(input_file_path, output_folder_path)
