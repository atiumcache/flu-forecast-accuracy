from typing import Tuple

import numpy as np
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta


def get_lat_long_from_loc_code(loc_code: str) -> Tuple[float, float]:
    """Returns the lat/long coordinates in a tuple, given a location code."""
    loc_code = loc_code.zfill(2)
    df = pd.read_csv("./datasets/locations_with_capitals.csv")
    lat = df["latitude"].loc[df["location"] == loc_code].values[0]
    long = df["longitude"].loc[df["location"] == loc_code].values[0]
    return lat, long


def get_daily_data_point(
    lat: float, long: float, date: str, weather_type: str
) -> float:
    """Returns a mean temperature for a given location and date.

    Args:
        lat: latitude for the location
        long: longitude for the location
        date: YYYY-MM-DD
        weather_type: the data type (temperature, humidity, etc.)

    Returns:
        Float. Mean temperature (celsius) for given date and location
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    weather_options = {
        "mean_temp": "temperature_2m_mean",
        "precip_hours": "precipitation_hours",
        "rel_humidity": "relative_humidity_2m",
        "radiation": "shortwave_radiation_sum",
        "sunshine": "sunshine_duration",
        "wind_speed": "wind_speed_10m_max"
    }

    # hourly_weather are variables that return hourly data
    # that needs to be averaged
    hourly_weather = ["rel_humidity"]

    if weather_type not in weather_options:
        types_string = "\n".join(weather_options.keys())
        raise ValueError(
            f"Type {type} is not supported.\n Valid weather types are: {types_string}"
        )

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"

    if weather_type in hourly_weather:
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": date,
            "end_date": date,
            "hourly": weather_options[weather_type],
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_data = hourly.Variables(0).ValuesAsNumpy()
        return max(hourly_data)
    else:
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": date,
            "end_date": date,
            "daily": weather_options[weather_type],
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_data = daily.Variables(0).ValuesAsNumpy()
        return daily_data[0]


def get_weekly_forecast_avg_temp(lat: float, long: float, start_date: str) -> float:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": start_date,
        # end date is 1 week ahead
        "end_date": (pd.to_datetime(start_date) + timedelta(days=7)).strftime(
            "%Y-%m-%d"
        ),
        "daily": "temperature_2m_max",
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    return np.average(daily_temperature_2m_max)


def get_avg_weekly_forecast_from_loc_code(loc_code: str, date: str) -> float:
    lat, long = get_lat_long_from_loc_code(loc_code)
    avg_weekly_temp = get_weekly_forecast_avg_temp(lat, long, date)
    return avg_weekly_temp

def openmeteo_data(lat, long):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": "2023-08-10",
        "end_date": "2023-10-28",
        "daily": [
            "temperature_2m_mean",
            "sunshine_duration",
            "wind_speed_10m_max",
            "shortwave_radiation_sum"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = \
    responses[
        0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_sunshine_duration = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(3).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data[
        "temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data[
        "sunshine_duration"] = daily_sunshine_duration
    daily_data[
        "wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data[
        "shortwave_radiation_sum"] = daily_shortwave_radiation_sum

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe
