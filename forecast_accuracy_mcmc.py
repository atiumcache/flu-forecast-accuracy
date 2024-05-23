import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sys import path
import matplotlib.pyplot as plt
import pymmwr as pm
from datetime import datetime
from datetime import date
from datetime import timedelta
import seaborn as sns


location_to_state = {}
global full_hosp_data


def IS(alpha, predL, predU):
    """
    Calculates Interval Score. 
    Helper function for WIS function below.

    Args:
        alpha (_type_): _description_
        predL (_type_): _description_
        predU (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return lambda y: (predU - predL) + 2/alpha*(y < predL)*(predL - y) + 2/alpha*(y > predU)*(y-predU)


def WIS(y_obs, qtlMark, predQTL):
    """
    Calculates a Weighted Interval Score based on this paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

    Args:
        y_obs (List): _description_
        qtlMark (List): The quartile marks used in forecast data file.
        predQTL (List): _description_

    Returns:
        float: The WIS score.
    """    

    is_well_defined = np.mod(len(qtlMark), 2) != 0
    
    NcentralizedQT = (len(qtlMark)-1)//2 + 1
    
    alpha_list = np.zeros(NcentralizedQT)
    weight_list = np.zeros(NcentralizedQT)
    
    for i in range(NcentralizedQT):  
        is_well_defined = is_well_defined & (np.abs(-1.0 + qtlMark[i] + qtlMark[-1-i]) < 1e-8)
        alpha_list[i] = 1 - (qtlMark[-1-i] - qtlMark[i])
        weight_list[i] = alpha_list[i]/2
        
    if is_well_defined:
        #print(alpha_list)
        #print(qtlMark)
        #print(NcentralizedQT)
        
        output = 1.0/2 * np.abs(y_obs - predQTL[NcentralizedQT - 1])
        
        for i in range(NcentralizedQT - 1):
            output += weight_list[i] * IS(alpha_list[i], predQTL[i], predQTL[-1 - i])(y_obs)
            #print(alpha_list[i], predQTL[i],predQTL[-1-i])
            
        return output/(NcentralizedQT - 1/2)
    
    else:
        print('Check the quantile marks: either no median defined, or not in symmetric central QTL form.')


def get_target_dates_list(forecast_df):
    return forecast_df['target_end_date'].unique()


def get_state_hosp_data(state_code):
    '''Filters a single state's hospitalization data from the full dataset.'''
    global full_hosp_data
    state_abbrev = location_to_state[str(state_code).zfill(2)]
    return full_hosp_data[full_hosp_data['state'] == state_abbrev]


def one_state_one_week_WIS(forecast_df, state_code_input):
    """
    Generates one state's WIS scores for a single week's predictions.

    Args:
        forecast_df (Dataframe): Contains the forecast data.
        state_code (Int): The location code for the current state.

    Returns:
        float: WIS score
    """    
    state_code = str(state_code_input).zfill(2)
    quantiles = np.zeros((23,4))
    wis = np.zeros(4)

    state_hosp_data = get_state_hosp_data(state_code)

    target_dates = get_target_dates_list(forecast_df)

    for n_week_ahead in range(4):
        target_date = target_dates[n_week_ahead]

        # Sum the daily reported cases to find the weekly observation.
        observation = 0
        for i in range(7):
            target_date = str(date.fromisoformat(target_date) + timedelta(days=-i))
            daily_count = state_hosp_data.loc[state_hosp_data['date'] == target_date].values[0, 2]
            observation += daily_count

        df_state = forecast_df[forecast_df['location'].to_numpy() == state_code]
        df_state_forecast = df_state[df_state['target_end_date'] == target_dates[n_week_ahead]]
        quantiles = df_state_forecast['output_type_id'].astype(float).to_numpy()
        predictions = df_state_forecast['value'].astype(float).to_numpy()
        
        try:
            wis[n_week_ahead] = np.round(
                                np.sum(np.nan_to_num(
                                    WIS(observation, 
                                        quantiles, 
                                        predictions))),
                                        2)
        except:
            # if an error occurs, print the corresponding data
            print("An error occured. Output will include a row of 0's.")
            print("Check the", location_to_state[state_code], "data file.\n")

    # Send scores to csv
    one_week_scores = {'state_code': state_code, 'state_abbrev': location_to_state[state_code], 'date': target_dates[0], '1wk_WIS': wis[0],'2wk_WIS': wis[1], '3wk_WIS': wis[2], '4wk_WIS': wis[3]}

    return one_week_scores


def one_state_all_scores(state_code):
    """
    Generates all WIS scores for one state. 
    Uses the 'one_state_one_week' function to generate each week's score.
    Exports scores to a csv file.

    Args:
        state_code (int): The location code for the current state.
    """    
    state_code = str(state_code).zfill(2)
    
    state_df = pd.DataFrame(columns=[])

    forecast_path = './LosAlamos_NAU-CModel_Flu/'
    forecast_files = [file for file in listdir(forecast_path) if isfile(join(forecast_path, file))]
    forecast_files.sort()

    for file in forecast_files:
        all_forecast_data = pd.read_csv(forecast_path + file)
        all_forecast_data = all_forecast_data[all_forecast_data['output_type'] == 'quantile']

        weekly_scores = one_state_one_week_WIS(all_forecast_data, state_code)
        state_df = pd.concat([state_df, pd.DataFrame([weekly_scores])], ignore_index=True)
    
    # Export to CSV
    state_csv_path = './mcmc_accuracy_results/' + location_to_state[state_code] + '.csv'
    state_df.to_csv(state_csv_path)

    return None


def main():
    '''Import locations'''
    locations = pd.read_csv('./locations.csv',skiprows=0) 
    locations = locations.drop([0]) #skip first row (national ID)
    print("Number of Locations:", len(locations))

    '''Map locations codes to state abbreviations.'''
    for index, row in locations.iterrows():
        location_number = row['location']
        abbreviation = row['abbreviation']
        location_to_state.update({location_number: abbreviation})

    '''Extract hospitalization data'''
    global full_hosp_data
    full_hosp_data = pd.read_csv('./COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv') 
    full_hosp_data = full_hosp_data[['date','state','previous_day_admission_influenza_confirmed']].sort_values(['state','date'])
    full_hosp_data['date'] = pd.to_datetime(full_hosp_data['date'])

    '''Generate WIS data for all states. Exports all scores to csv files.'''
    for state in location_to_state.keys():
        one_state_all_scores(state)


if __name__ == "__main__":
    main()