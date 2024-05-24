import pandas as pd
from os import listdir
from os.path import isfile, join
from sys import path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_one_state(state_csv):
    data = pd.read_csv(state_csv)

    data['date'] = pd.to_datetime(data['date'])
    state_name = data['state_abbrev'][1]

    palette = sns.color_palette("colorblind")  

    plt.figure(figsize=(10, 4))
    sns.lineplot(x='date', y='1wk_WIS', data=data, linewidth=2, label='1-Week WIS', color=palette[0], linestyle='-')
    sns.lineplot(x='date', y='2wk_WIS', data=data, linewidth=2, label='2-Week WIS', color=palette[1], linestyle='--')
    sns.lineplot(x='date', y='3wk_WIS', data=data, linewidth=2, label='3-Week WIS', color=palette[2], linestyle='-.')
    sns.lineplot(x='date', y='4wk_WIS', data=data, linewidth=2, label='4-Week WIS', color=palette[4], linestyle=':')

    plt.title('WIS Scores Over Time :: MCMC Forecast :: ' + state_name)
    plt.xlabel('Date')
    plt.ylabel('WIS')
    plt.legend(title='Forecast Horizon')
    plt.grid(True)
    plt.show()

    return None


def plot_average_of_all_states(data_folder_path):

    data_list = []
    wis_columns = ['1wk_WIS', '2wk_WIS', '3wk_WIS', '4wk_WIS']

    # Read each CSV file in the directory
    for filename in listdir(data_folder_path):
        if filename.endswith('.csv'):
            # Load the data
            file_path = join(data_folder_path, filename)
            data = pd.read_csv(file_path)

            # Convert 'date' column to datetime format
            data['date'] = pd.to_datetime(data['date'])

            for col in wis_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Check if any row has all zeros in WIS columns
            # If so, don't append/use it, because that indicates missing data. 
            if not (data[wis_columns] == 0).all(axis=1).any():
                data_list.append(data)

    # Concatenate all dataframes
    combined_data = pd.concat(data_list)

    # Group by date and calculate the mean for each WIS
    average_data = combined_data.groupby('date')[wis_columns].mean().reset_index()

    # Plot
    palette = sns.color_palette("colorblind")  

    plt.figure(figsize=(10, 4))
    sns.lineplot(x='date', y='1wk_WIS', data=average_data, label='Average 1-Week WIS', color=palette[0], linestyle='-', linewidth=2)
    sns.lineplot(x='date', y='2wk_WIS', data=average_data, label='Average 2-Week WIS', color=palette[1], linestyle='--', linewidth=2)
    sns.lineplot(x='date', y='3wk_WIS', data=average_data, label='Average 3-Week WIS', color=palette[2], linestyle='-.', linewidth=2)
    sns.lineplot(x='date', y='4wk_WIS', data=average_data, label='Average 4-Week WIS', color=palette[4], linestyle=':', linewidth=2)

    plt.title('Average WIS Scores Over Time :: All States :: MCMC Forecast')
    plt.xlabel('Date')
    plt.ylabel('Average WIS')
    plt.legend(title='Forecast Horizon')
    plt.grid(True)
    plt.show()

    return None


def sanity_check_plot():
    from matplotlib.pyplot import cm

    colors = cm.plasma(np.linspace(0,1,12))


    for i in range(11):

        plt.fill_between(range(4), np_quantiles[i,:], np_quantiles[22-i,:], facecolor = colors[11-i], zorder = i)

    plt.scatter(range(4), np_forecast, zorder=30)
    plt.title('4 weeks ahead WIS score' + str(np_wis))