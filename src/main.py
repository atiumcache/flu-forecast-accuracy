"""
This script features the main logic for calculating
1, 2, 3, and 4-week WIS scores for 28 day time series predictions.
"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import date, timedelta
import os


INPUT_FOLDER = "../prediction_data/pmcmc_wis/test_predictions_20241006/"
OUTPUT_FOLDER = "../prediction_data/pmcmc_wis/accuracy_results_test_20241124/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# INPUT_FOLDER = "./LosAlamos_NAU-CModel_Flu/"
# OUTPUT_FOLDER = "./mcmc_accuracy_results/"


def main() -> None:
    """Main function to execute the script on all locations."""

    # Read location data.
    locations = pd.read_csv("../datasets/locations.csv").iloc[
        1:
    ]  # skip first row (national ID)

    # Map location codes to state abbreviations.
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))

    # Process reported hospitalization data.
    full_hosp_data = pd.read_csv("../datasets/COVID_Reported_Data.csv")
    full_hosp_data = full_hosp_data[
        ["date", "state", "previous_day_admission_influenza_confirmed"]
    ].sort_values(["state", "date"])
    full_hosp_data["date"] = pd.to_datetime(full_hosp_data["date"])

    # Run forecast accuracy on all locations.
    for state_code in location_to_state.keys():
        if state_code != "04":
            continue
        print("Running forecast accuracy on location code", state_code)
        one_state_all_scores_to_csv(
            state_code, INPUT_FOLDER, full_hosp_data, location_to_state
        )

    print(f"Completed all locations.")
    return


def IS(alpha: float, predL: float, predU: float):
    """
    Calculates Interval Score. Helper function for WIS.

    Args:
        alpha: 1 - the difference between quantile marks.
        predL: Predicted value for lower quantile.
        predU: Predicted value for upper quantile.

    Returns:
        Interval Score function.
    """
    return (
        lambda y: (predU - predL)
        + 2 / alpha * (y < predL) * (predL - y)
        + 2 / alpha * (y > predU) * (y - predU)
    )


def WIS(y_obs: float, qtlMark: list[float], predQTL: list[float]) -> float:
    """
    Calculates a Weighted Interval Score based on this paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

    Args:
        y_obs: Observed hospitalization data points.
        qtlMark: The quantile marks used in forecast data file.
        predQTL: Predicted values for each quantile.

    Returns:
        WIS score.
    """
    if len(qtlMark) % 2 == 0:
        raise ValueError(
            "Check the quantile marks: either no median defined, or not in symmetric central QTL form."
        )

    NcentralizedQT = (len(qtlMark) - 1) // 2 + 1
    alpha_list = [1 - (qtlMark[-1 - i] - qtlMark[i]) for i in range(NcentralizedQT)]
    weight_list = [alpha / 2 for alpha in alpha_list]

    output = abs(y_obs - predQTL[NcentralizedQT - 1]) / 2
    for i in range(NcentralizedQT - 1):
        output += weight_list[i] * IS(alpha_list[i], predQTL[i], predQTL[-1 - i])(y_obs)

    return output / (NcentralizedQT - 0.5)


def get_target_dates_list(forecast_df: pd.DataFrame) -> list:
    """Returns a list of target forecast dates."""
    filtered_forecast_df = forecast_df[forecast_df["horizon"] != -1]
    dates_list = filtered_forecast_df["target_end_date"].unique().tolist()
    sorted_dates = sorted(dates_list)
    return sorted_dates


def get_state_hosp_data(
    full_hosp_data: pd.DataFrame, location_to_state: dict, state_code: str
) -> pd.DataFrame:
    """Filters a single state's hospitalization data from the full dataset."""
    state_abbrev = location_to_state[state_code]
    return full_hosp_data[full_hosp_data["state"] == state_abbrev]


def one_state_one_week_WIS(
    forecast_df: pd.DataFrame,
    state_code: str,
    full_hosp_data: pd.DataFrame,
    location_to_state: dict,
) -> dict:
    """
    Generates one state's WIS scores for a single week's predictions.

    Args:
        forecast_df: DataFrame containing the forecast data.
        state_code: The location code for the current state.
        full_hosp_data: DataFrame containing full hospitalization data.
        location_to_state: Dictionary mapping location codes to state abbreviations.

    Returns:
        WIS scores and state info.
    """
    state_hosp_data = get_state_hosp_data(full_hosp_data, location_to_state, state_code)
    target_dates = get_target_dates_list(forecast_df)
    predict_from_date = str(date.fromisoformat(target_dates[0]) - timedelta(days=7))

    quantiles = np.zeros((23, 4))
    reported_data = np.zeros(4)
    wis_scores = np.zeros(4)

    for n_week_ahead in range(4):
        target_date = target_dates[n_week_ahead]

        week_observation = compute_week_observation(state_hosp_data, target_date)
        reported_data[n_week_ahead] = compute_week_observation(
            state_hosp_data, target_date
        )

        df_state_forecast = forecast_df[
            (forecast_df["location"] == state_code)
            & (forecast_df["target_end_date"] == target_date)
        ]
        quantiles = df_state_forecast["output_type_id"].astype(float).to_numpy()
        predictions = df_state_forecast["value"].astype(float).to_numpy()

        try:
            wis_scores[n_week_ahead] = round(
                np.sum(np.nan_to_num(WIS(week_observation, quantiles, predictions))), 2
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Check the {location_to_state[state_code]} data file.\n")

    return {
        "state_code": state_code,
        "state_abbrev": location_to_state[state_code],
        "date": predict_from_date,
        "1wk_WIS": wis_scores[0],
        "2wk_WIS": wis_scores[1],
        "3wk_WIS": wis_scores[2],
        "4wk_WIS": wis_scores[3],
    }


def compute_week_observation(state_hosp_data, target_date):
    """
    Computes the hospitalization counts for a given week.

    This is necessary because the hospitalization data is provided daily,
    but we are making weekly predictions.
    """
    observation = 0
    state_hosp_data.loc[:, "date"] = pd.to_datetime(state_hosp_data["date"])
    target_date_obj = pd.to_datetime(target_date)

    for i in range(7):
        current_date = target_date_obj - timedelta(days=i)

        # Check if the current date is in the DataFrame
        filtered_data = state_hosp_data.loc[
            state_hosp_data["date"] == current_date,
            "previous_day_admission_influenza_confirmed",
        ]

        if filtered_data.empty:
            print(f"No data found for date: {current_date}")
            continue

        try:
            observation += filtered_data.values[0]
        except IndexError as e:
            print(f"An error occurred: {e}")
            print(f"Data for date {current_date} seems to be missing.")
            return None

    return observation


def one_state_all_scores_to_csv(
    state_code: str,
    forecast_path: str,
    full_hosp_data: pd.DataFrame,
    location_to_state: dict,
) -> None:
    """
    Generates all WIS scores for one state.
    Exports scores to a csv file.

    Args:
        state_code: The location code for the current state.
        forecast_path: Path to the forecast data files.
        full_hosp_data: DataFrame containing full hospitalization data.
        location_to_state: Dictionary mapping location codes to state abbreviations.
    """
    state_df = pd.DataFrame()

    forecast_files = sorted(
        [file for file in listdir(forecast_path) if isfile(join(forecast_path, file))]
    )

    for file in forecast_files:
        all_forecast_data = pd.read_csv(join(forecast_path, file))
        all_forecast_data = all_forecast_data[
            all_forecast_data["output_type"] == "quantile"
        ]
        all_forecast_data["location"] = all_forecast_data["location"].apply(
            lambda x: str(x).zfill(2)
        )

        weekly_scores = one_state_one_week_WIS(
            all_forecast_data, state_code, full_hosp_data, location_to_state
        )
        state_df = pd.concat(
            [state_df, pd.DataFrame([weekly_scores])], ignore_index=True
        )

    state_csv_path = join(OUTPUT_FOLDER, f"{location_to_state[state_code]}.csv")
    state_df.to_csv(state_csv_path, index=False)


if __name__ == "__main__":
    main()
