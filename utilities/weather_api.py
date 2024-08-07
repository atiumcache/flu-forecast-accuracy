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
    df = pd.read_csv("../datasets/locations_with_capitals.csv")
    lat = df['latitude'].loc[df['location'] == loc_code].values[0]
    long = df['longitude'].loc[df['location'] == loc_code].values[0]
    return lat, long


def get_single_day_mean_temp(lat: float, long: float, date: str) -> float:
    """Returns a mean temperature for a given location and date.

    Args:
        lat: latitude for the location
        long: longitude for the location
        date: YYYY-MM-DD

    Returns:
        Float. Mean temperature (celsius) for given date and location
    """
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
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_mean"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    return daily_temperature_2m_mean[0]


def get_single_day_precip_hours(lat: float, long: float, date: str) -> float:
    """Returns a precipitation hours for a given location and date.

    Args:
        lat: latitude for the location
        long: longitude for the location
        date: YYYY-MM-DD

    Returns:
        Float. Precipitation hours for given date and location.
    """
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
        "start_date": date,
        "end_date": date,
        "daily": "precipitation_hours"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_precipitation_hours = daily.Variables(0).ValuesAsNumpy()
    return daily_precipitation_hours[0]


def get_max_rel_humidity(lat: float, long: float, date: str) -> float:
    """Returns the maximum relative humidity for a given location and date.

    Args:
        lat: latitude for the location
        long: longitude for the location
        date: YYYY-MM-DD

    Returns:
        Float. Maximum relative humidity for given date and location.
    """
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
        "start_date": date,
        "end_date": date,
        "hourly": "relative_humidity_2m"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_relative_humidity_2m = hourly.Variables(0).ValuesAsNumpy()
    return max(hourly_relative_humidity_2m)


def get_weekly_forecast_avg_temp(lat: float, long: float, start_date: str) -> float:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
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
        "end_date": (pd.to_datetime(start_date) + timedelta(
            days=7)).strftime('%Y-%m-%d'),
        "daily": "temperature_2m_max"
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
