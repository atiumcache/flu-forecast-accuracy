import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pymmwr as pm
import seaborn as sns
import csv


INPUT_FOLDER = "./LosAlamos_NAU-CModel_Flu/"


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
    dates_list = forecast_df["target_end_date"].unique().tolist()
    return dates_list


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

    quantiles = np.zeros((23, 4))
    reported_data = np.zeros(4)
    wis_scores = np.zeros(4)

    for n_week_ahead in range(4):
        target_date = target_dates[n_week_ahead]

        observation = sum(
            state_hosp_data.loc[
                state_hosp_data["date"]
                == str(date.fromisoformat(target_date) - timedelta(days=i)),
                "previous_day_admission_influenza_confirmed",
            ].values[0]
            for i in range(7)
        )
        reported_data[n_week_ahead] = observation

        df_state_forecast = forecast_df[
            (forecast_df["location"] == state_code)
            & (forecast_df["target_end_date"] == target_date)
        ]
        quantiles = df_state_forecast["output_type_id"].astype(float).to_numpy()
        predictions = df_state_forecast["value"].astype(float).to_numpy()

        try:
            wis_scores[n_week_ahead] = round(
                np.sum(np.nan_to_num(WIS(observation, quantiles, predictions))), 2
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Check the {location_to_state[state_code]} data file.\n")

    return {
        "state_code": state_code,
        "state_abbrev": location_to_state[state_code],
        "date": target_dates[0],
        "1wk_WIS": wis_scores[0],
        "2wk_WIS": wis_scores[1],
        "3wk_WIS": wis_scores[2],
        "4wk_WIS": wis_scores[3],
    }


def one_state_all_scores_to_csv(
    state_code: str,
    forecast_path: str,
    full_hosp_data: pd.DataFrame,
    location_to_state: dict,
):
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

        weekly_scores = one_state_one_week_WIS(
            all_forecast_data, state_code, full_hosp_data, location_to_state
        )
        state_df = pd.concat(
            [state_df, pd.DataFrame([weekly_scores])], ignore_index=True
        )

    state_csv_path = join(
        "./mcmc_accuracy_results/", f"{location_to_state[state_code]}.csv"
    )
    state_df.to_csv(state_csv_path, index=False)


def main():
    """Main function to execute the script."""
    locations = pd.read_csv("./locations.csv").iloc[1:]  # skip first row (national ID)
    print(f"Number of Locations: {len(locations)}")

    # Map location codes to state abbreviations.
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))

    # Process reported hospitalization data.
    full_hosp_data = pd.read_csv(
        "./COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv"
    )
    full_hosp_data = full_hosp_data[
        ["date", "state", "previous_day_admission_influenza_confirmed"]
    ].sort_values(["state", "date"])
    full_hosp_data["date"] = pd.to_datetime(full_hosp_data["date"])

    # Run forecast accuracy on all locations.
    for state_code in location_to_state.keys():
        print("Running forecast accuracy on location code", state_code, end="\r")
        one_state_all_scores_to_csv(
            state_code, INPUT_FOLDER, full_hosp_data, location_to_state
        )


if __name__ == "__main__":
    main()
