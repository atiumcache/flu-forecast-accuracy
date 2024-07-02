"""
Example command line usage:
    python3 generate_wis_data.py -l '01' --mcmc

The example runs the script on location 01, using the data from the MCMC folder.
"""

from typing import Dict

import pandas as pd
import argparse
import numpy as np
from pandas.tseries.offsets import DateOffset


def main():
    args = parse_arguments()
    prediction_method = check_which_pred_method(args)
    location_code = args.location_code

    location_to_state = map_loc_codes()

    all_data = gather_all_hosp_data()
    single_state_data = get_state_hosp_data(all_data, location_to_state, location_code)
    weekly_state_data = convert_hosp_to_weekly(single_state_data)
    hosp_rate_of_change_df = get_rates_of_change(weekly_state_data)
    hosp_roc_and_wis_df = join_hosp_data_and_wis(
        hosp_rate_of_change_df, prediction_method, location_code, location_to_state
    )
    df = add_z_score_col(hosp_roc_and_wis_df)
    df = add_five_point_stencil_derivative(df, "prev_week_hosp", "1st_deriv_stencil")
    df = add_week_of_year(df)
    df = add_mov_avg_and_lag(df)
    df = add_avg_wis(df)
    df.to_csv("./hosp_roc/" + location_code + ".csv")


def add_avg_wis(df):
    # Calculate average WIS score
    average_wis = df["1wk_WIS"].mean()

    # Create binary labels
    df["is_wis_above_avg"] = (df["1wk_WIS"] > average_wis).astype(int)

    return df


def add_mov_avg_and_lag(df):
    # Calculate moving average WIS
    df["moving_avg_WIS"] = df["1wk_WIS"].rolling(window=3).mean()

    # Calculate lagged 1wk WIS
    df["lagged_1wk_WIS"] = df["1wk_WIS"].shift(1)

    return df


def add_week_of_year(df):
    # Convert date column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Add week of the year column
    df["week_of_year"] = df["date"].dt.isocalendar().week

    return df


def add_z_score_col(df):
    # Calculate the average WIS (mean of 1wk, 2wk, 3wk, 4wk)
    df["avg_wis"] = df[["1wk_WIS", "2wk_WIS", "3wk_WIS", "4wk_WIS"]].mean(axis=1)

    # Calculate the standard deviation of WIS values
    df["std_wis"] = df[["1wk_WIS", "2wk_WIS", "3wk_WIS", "4wk_WIS"]].std(axis=1)

    # Calculate the absolute Z-score for each WIS value
    df["abs_z_1wk_WIS"] = np.abs((df["1wk_WIS"] - df["avg_wis"]) / df["std_wis"])
    df["abs_z_2wk_WIS"] = np.abs((df["2wk_WIS"] - df["avg_wis"]) / df["std_wis"])
    df["abs_z_3wk_WIS"] = np.abs((df["3wk_WIS"] - df["avg_wis"]) / df["std_wis"])
    df["abs_z_4wk_WIS"] = np.abs((df["4wk_WIS"] - df["avg_wis"]) / df["std_wis"])

    return df


def get_state_wis_data(pred_method, loc_code, location_to_state):
    dir_path = "./" + pred_method + "_" + "accuracy_results/"
    loc_abbreviation = location_to_state[loc_code]
    csv_filename = loc_abbreviation + ".csv"
    full_path = dir_path + csv_filename
    state_wis_data = pd.read_csv(full_path)
    return state_wis_data


def join_hosp_data_and_wis(
    hosp_rate_of_change_data: pd.DataFrame,
    pred_method: str,
    loc_code: str,
    location_to_state: Dict,
):

    state_wis_df = get_state_wis_data(pred_method, loc_code, location_to_state)

    state_wis_df["date"] = pd.to_datetime(state_wis_df["date"])
    state_wis_df.set_index("date", inplace=True)

    merged_df = pd.merge(state_wis_df, hosp_rate_of_change_data, on="date", how="inner")

    merged_df = merged_df.drop(columns=["state_abbrev", "state_code"])
    return merged_df


def convert_hosp_to_weekly(daily_state_data: pd.DataFrame) -> pd.DataFrame:
    """
    Our hospital data comes in as daily new flu cases.
    This function sums the daily data and returns weekly data.
    """
    dates_df = pd.read_csv("./datasets/target_dates.csv")
    target_dates = [pd.to_datetime(date) for date in dates_df["date"]]

    # Ensure the date column is in datetime format
    daily_state_data["date"] = pd.to_datetime(daily_state_data["date"])
    daily_state_data.set_index("date", inplace=True)

    weekly_df = pd.DataFrame(columns=["date", "prev_week_hosp"])

    for target_date in target_dates:
        # Calculate the start date of the week (7 days prior to the target date)
        start_date = target_date - DateOffset(days=6)

        # Sum the 'prev_week_hosp' values for the week
        weekly_sum = daily_state_data.loc[
            start_date:target_date, "previous_day_admission_influenza_confirmed"
        ].sum()

        # Append the result to weekly_df
        weekly_df = weekly_df._append(
            {"date": target_date, "prev_week_hosp": weekly_sum}, ignore_index=True
        )

    return weekly_df


def get_rates_of_change(state_data: pd.DataFrame) -> pd.DataFrame:
    state_data["1_week_roc"] = state_data["prev_week_hosp"].pct_change(periods=1)
    state_data["2_week_roc"] = state_data["prev_week_hosp"].pct_change(periods=2)
    return state_data


def add_five_point_stencil_derivative(
    df: pd.DataFrame, column_name: str, new_column_name: str
) -> pd.DataFrame:
    """
    Adds a column to the DataFrame with the five-point stencil derivative of the specified column.

    Args:
        df: The input DataFrame.
        column_name: The name of the column for which to calculate the derivative.
        new_column_name: The name of the new column to store the derivative.

    Returns:
        pd.DataFrame: The DataFrame with the new column added.
    """

    def five_point_stencil(values: np.ndarray) -> np.ndarray:
        n = len(values)
        derivatives = np.zeros(n)

        # Use five-point stencil for interior points
        derivatives[2 : n - 2] = (
            -values[4:]
            + 8 * values[3 : n - 1]
            - 8 * values[1 : n - 3]
            + values[: n - 4]
        ) / 12

        # Forward difference for the first point
        derivatives[0] = values[1] - values[0]

        # Central difference for the second point
        derivatives[1] = (values[2] - values[0]) / 2

        # Central difference for the second-to-last point
        derivatives[n - 2] = (values[n - 1] - values[n - 3]) / 2

        # Backward difference for the last point
        derivatives[n - 1] = values[n - 1] - values[n - 2]

        return derivatives

    df[new_column_name] = five_point_stencil(df[column_name].values)
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--location_code",
        required=True,
        type=str,
        help="Location code corresponding to a state or territory. See `./locations.csv`.\nExample: '01'",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--particle-filter", action="store_true", help="Analyze Particle Filter"
    )
    group.add_argument("--mcmc", action="store_true", help="Analyze MCMC")

    args = parser.parse_args()
    return args


def check_which_pred_method(args) -> str:
    if args.particle_filter:
        return "pf"
    elif args.mcmc:
        return "mcmc"
        # Add your MCMC related code here


def gather_all_hosp_data() -> pd.DataFrame:

    # Process reported hospitalization data.
    full_hosp_data = pd.read_csv(
        "./datasets/COVID_Reported_Data.csv"
    )
    full_hosp_data = full_hosp_data[
        ["date", "state", "previous_day_admission_influenza_confirmed"]
    ].sort_values(["state", "date"])
    full_hosp_data["date"] = pd.to_datetime(full_hosp_data["date"])

    return full_hosp_data


def map_loc_codes() -> Dict:
    # Read location data.
    locations = pd.read_csv("datasets/locations.csv").iloc[
        1:
    ]  # skip first row (national ID)
    # Map location codes to state abbreviations.
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))

    return location_to_state


def get_state_hosp_data(
    full_hosp_data: pd.DataFrame, location_to_state: dict, state_code: str
) -> pd.DataFrame:
    """Filters a single state's hospitalization data from the full dataset."""
    state_abbrev = location_to_state[state_code]
    return full_hosp_data[full_hosp_data["state"] == state_abbrev]


if __name__ == "__main__":
    main()
