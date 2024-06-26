"""
This script standardizes the PF prediction output to
match the format of the MCMC predictions.
"""

import pandas as pd
import os

input_folder_path = "./pf_1000_data/datasets/hosp_forecasts"
output_folder_path = "./formatted-pf-predictions/"


def fix_horizon_values(horizon_series):
    """The input horizon values are inverted. This fixes them."""
    # Define a mapping from original values to their inverted counterparts
    mapping = {1: 4, 2: 3, 3: 2, 4: 1}

    # Use the map function to replace values based on the mapping
    return horizon_series.map(mapping)


def standardize_file(file_path: str, output_folder: str) -> None:
    """
    Standardize the formatting of the output from the PF/Trend Forecasting
    pipeline. Cleans the file and saves it to an output folder defined as a
    global variable above.

    Args:
        file_path: a path to an input file.
        output_folder: the directory to save the standardized file.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    df = df.drop_duplicates(
        subset=["location", "target_end_date", "output_type_id"], keep="first"
    )

    # Convert specified columns to strings without additional quotes
    df["output_type_id"] = df["output_type_id"].astype(str)
    df["location"] = df["location"].apply(lambda x: str(x).zfill(2))
    df["output_type"] = df["output_type"].astype(str)

    df["horizon"] = fix_horizon_values(df["horizon"])

    # Determine the output file path
    new_file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, new_file_name)

    # Save the modified dataframe to a new CSV
    df.to_csv(output_file_path, index=False)

    print(f"Converted file saved to {output_file_path}")


# Create the output directory if it does not exist
os.makedirs(output_folder_path, exist_ok=True)

# Process each file in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith(".csv"):
        input_file_path = os.path.join(input_folder_path, file_name)
        standardize_file(input_file_path, output_folder_path)
