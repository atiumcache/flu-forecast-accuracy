import pandas as pd
import os

input_folder_path = './formatted-pf-predictions/'


def standardize_file(file_path: str):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Remove duplicate rows based on specific columns, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=['location', 'target_end_date',
                                            'output_type_id'], keep='first')

    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(file_path, index=False)

    print(f"Cleaned file saved to {file_path}")


# Process each file in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_folder_path, file_name)
        standardize_file(input_file_path)
