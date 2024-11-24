"""
This module contains functions for plotting WIS
scores over time.
"""
from os import listdir
from os.path import isfile, join
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os


def plot_state_comparison_avg(
    method_1_path: str,
    method_2_path: str,
    method_1_name: str,
    method_2_name: str,
    save: bool = False,
) -> None:
    """
    Compares WIS scores from MCMC and Particle Filter methods over time for a state.

    Args:
        method_1_path: relative path to csv file containing MCMC WIS scores.
        method_2_path: relative path to csv file containing Particle Filter WIS scores.
        method_1_name: the first method name to use in the plot title.
        method_2_name: the second method name to use in the plot title.
        save: When `True`, saves the plot to `./plots/`.
    """
    mcmc_data = pd.read_csv(method_1_path)
    pf_data = pd.read_csv(method_2_path)

    mcmc_data["date"] = pd.to_datetime(mcmc_data["date"])
    pf_data["date"] = pd.to_datetime(pf_data["date"])

    # Dropping date from MCMC because PF doesn't have that data
    index_to_drop = mcmc_data[mcmc_data["date"] == pd.to_datetime("2023-10-14")].index
    mcmc_data = mcmc_data.drop(index_to_drop)

    state_name = mcmc_data["state_abbrev"][1]

    warm_palette = sns.color_palette("Oranges", 4)
    cool_palette = sns.color_palette("Blues", 4)
    tab_palette = sns.color_palette("tab10")

    plt.figure(figsize=(8, 4.5), dpi=200)

    # Plot MCMC WIS scores
    sns.lineplot(
        x="date",
        y="avg_wis",
        data=mcmc_data,
        linewidth=1.8,
        label=f"{method_1_name} Avg. WIS",
        color=tab_palette[0],
        linestyle="-",
    )
    sns.lineplot(
        x="date",
        y="avg_wis",
        data=pf_data,
        linewidth=1.8,
        label=f"{method_2_name} Avg. WIS",
        color=tab_palette[1],
        linestyle="-",
    )

    plt.title(
        f"Avg. WIS Score Over Time :: {method_1_name} vs {method_2_name} Forecast :: {state_name}"
    )
    plt.xlabel("Date")
    plt.ylabel("WIS")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"./plots/{state_name}_WIS_comparison.png")
    else:
        plt.show()


def plot_state_comparison(
    mcmc_csv_path: str, pf_csv_path: str, save: bool = False
) -> None:
    """
    Compares WIS scores from MCMC and Particle Filter methods over time for a state.

    Args:
        mcmc_csv_path: relative path to csv file containing MCMC WIS scores.
        pf_csv_path: relative path to csv file containing Particle Filter WIS scores.
        save: When `True`, saves the plot to `./plots/`.
    """
    mcmc_data = pd.read_csv(mcmc_csv_path)
    pf_data = pd.read_csv(pf_csv_path)

    mcmc_data["date"] = pd.to_datetime(mcmc_data["date"])
    pf_data["date"] = pd.to_datetime(pf_data["date"])

    state_name = mcmc_data["state_abbrev"][1]

    warm_palette = sns.color_palette("Oranges", 4)
    cool_palette = sns.color_palette("Blues", 4)

    plt.figure(figsize=(8.5, 4.5), dpi=200)

    # Plot MCMC WIS scores
    sns.lineplot(
        x="date",
        y="1wk_WIS",
        data=mcmc_data,
        linewidth=1.5,
        label="MCMC 1-Week WIS",
        color=warm_palette[0],
        linestyle="-",
    )
    sns.lineplot(
        x="date",
        y="2wk_WIS",
        data=mcmc_data,
        linewidth=1.5,
        label="MCMC 2-Week WIS",
        color=warm_palette[1],
        linestyle="--",
    )
    sns.lineplot(
        x="date",
        y="3wk_WIS",
        data=mcmc_data,
        linewidth=1.5,
        label="MCMC 3-Week WIS",
        color=warm_palette[2],
        linestyle="-.",
    )
    sns.lineplot(
        x="date",
        y="4wk_WIS",
        data=mcmc_data,
        linewidth=1.5,
        label="MCMC 4-Week WIS",
        color=warm_palette[3],
        linestyle=":",
    )

    # Plot Particle Filter WIS scores
    sns.lineplot(
        x="date",
        y="1wk_WIS",
        data=pf_data,
        linewidth=1.5,
        label="PF 1-Week WIS",
        color=cool_palette[0],
        linestyle="-",
    )
    sns.lineplot(
        x="date",
        y="2wk_WIS",
        data=pf_data,
        linewidth=1.5,
        label="PF 2-Week WIS",
        color=cool_palette[1],
        linestyle="--",
    )
    sns.lineplot(
        x="date",
        y="3wk_WIS",
        data=pf_data,
        linewidth=1.5,
        label="PF 3-Week WIS",
        color=cool_palette[2],
        linestyle="-.",
    )
    sns.lineplot(
        x="date",
        y="4wk_WIS",
        data=pf_data,
        linewidth=1.5,
        label="PF 4-Week WIS",
        color=cool_palette[3],
        linestyle=":",
    )

    plt.title(f"WIS Scores Over Time :: MCMC vs PF Forecast :: {state_name}")
    plt.xlabel("Date")
    plt.ylabel("WIS")
    plt.legend(title="Forecast Horizon")
    plt.grid(True)

    if save:
        plt.savefig(f"./plots/{state_name}_WIS_comparison.png")

    plt.show()


def plot_one_state(state_csv: str, method: str, save: bool = False) -> None:
    """
    Displays a plot of one state's WIS scores over time.

    Args:
        state_csv: relative path to csv file containing WIS scores.
        method: a string for the method (e.g. MCMC, PF), used in the plot title.
        save: When `True`, saves the plot to `./plots/`.
    """
    data = pd.read_csv(state_csv)

    data["date"] = pd.to_datetime(data["date"])
    state_name = data["state_abbrev"][1]

    palette = sns.color_palette("colorblind")

    plt.figure(figsize=(10, 4), dpi=600)
    sns.lineplot(
        x="date",
        y="1wk_WIS",
        data=data,
        linewidth=1.5,
        label="1-Week WIS",
        color=palette[0],
        linestyle="-",
    )
    sns.lineplot(
        x="date",
        y="2wk_WIS",
        data=data,
        linewidth=1.5,
        label="2-Week WIS",
        color=palette[1],
        linestyle="--",
    )
    sns.lineplot(
        x="date",
        y="3wk_WIS",
        data=data,
        linewidth=1.5,
        label="3-Week WIS",
        color=palette[2],
        linestyle="-.",
    )
    sns.lineplot(
        x="date",
        y="4wk_WIS",
        data=data,
        linewidth=1.5,
        label="4-Week WIS",
        color=palette[4],
        linestyle=":",
    )

    plt.title(f"WIS Scores Over Time :: {method} Forecast :: " + state_name)
    plt.xlabel("Date")
    plt.ylabel("WIS")
    plt.legend(title="Forecast Horizon")
    plt.grid(True)
    plt.show()

    if save == True:
        filepath = "./plots/" + data["state_abbrev"][1] + "_WIS_scores_plot.csv"
        plt.savefig(filepath)


def plot_average_of_all_states(data_folder_path: str):
    """
    Displays a plot of the average WIS scores of all states over time.

    Args:
        data_folder_path: relative path to folder containing all accuracy results.
    """
    data_list = []
    wis_columns = ["1wk_WIS", "2wk_WIS", "3wk_WIS", "4wk_WIS"]

    # Read each CSV file in the directory
    for filename in listdir(data_folder_path):
        if filename.endswith(".csv"):
            # Load the data
            file_path = join(data_folder_path, filename)
            data = pd.read_csv(file_path)

            # Convert 'date' column to datetime format
            data["date"] = pd.to_datetime(data["date"])

            for col in wis_columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            # Check if any row has all zeros in WIS columns
            # If so, don't append/use it, because that indicates missing data.
            if not (data[wis_columns] == 0).all(axis=1).any():
                data_list.append(data)

    # Concatenate all dataframes
    combined_data = pd.concat(data_list)

    # Group by date and calculate the mean for each WIS
    average_data = combined_data.groupby("date")[wis_columns].mean().reset_index()

    # Plot
    palette = sns.color_palette("colorblind")

    plt.figure(figsize=(10, 4), dpi=600)
    sns.lineplot(
        x="date",
        y="1wk_WIS",
        data=average_data,
        label="Average 1-Week WIS",
        color=palette[0],
        linestyle="-",
        linewidth=1.5,
    )
    sns.lineplot(
        x="date",
        y="2wk_WIS",
        data=average_data,
        label="Average 2-Week WIS",
        color=palette[1],
        linestyle="--",
        linewidth=1.5,
    )
    sns.lineplot(
        x="date",
        y="3wk_WIS",
        data=average_data,
        label="Average 3-Week WIS",
        color=palette[2],
        linestyle="-.",
        linewidth=1.5,
    )
    sns.lineplot(
        x="date",
        y="4wk_WIS",
        data=average_data,
        label="Average 4-Week WIS",
        color=palette[4],
        linestyle=":",
        linewidth=1.5,
    )

    plt.title("Average WIS Scores Over Time :: All States :: MCMC Forecast")
    plt.xlabel("Date")
    plt.ylabel("Average WIS")
    plt.legend(title="Forecast Horizon")
    plt.grid(True)
    plt.show()


def calculate_improvement_scores():
    pf_folder_path = "../prediction_data/pf_accuracy_results/"
    mcmc_folder_path = "../prediction_data/mcmc_accuracy_results/"
    all_improvements_df = pd.DataFrame()
    columns_to_compare = ["1wk_WIS", "2wk_WIS", "3wk_WIS", "4wk_WIS"]

    for filename in os.listdir(mcmc_folder_path):
        mcmc_df = pd.read_csv(mcmc_folder_path + filename)
        pf_df = pd.read_csv(pf_folder_path + filename)

        location_abbrev = mcmc_df["state_abbrev"][1]
        location_code = mcmc_df["state_code"][1]

        # make sure we have the same dates for each method
        for date in mcmc_df["date"]:
            if date not in pf_df["date"]:
                index_to_drop = mcmc_df[mcmc_df["date"] == date].index
                mcmc_df = mcmc_df.drop(index_to_drop)
        for date in pf_df["date"]:
            if date not in mcmc_df["date"]:
                index_to_drop = pf_df[pf_df["date"] == date].index
                pf_df = pf_df.drop(index_to_drop)

        improvement_df = pd.DataFrame()
        improvement_df[["state_code", "state_abbrev"]] = pf_df[
            ["state_code", "state_abbrev"]
        ]

        # Calculate improvement: (pf_wis - mcmc_wis) / mcmc_wis
        for col in columns_to_compare:
            improvement_df[col + "_improvement"] = (
                pf_df[col] - mcmc_df[col]
            ) / mcmc_df[col]

        all_improvements_df = pd.concat(
            [all_improvements_df, improvement_df], ignore_index=True
        )

    return all_improvements_df
