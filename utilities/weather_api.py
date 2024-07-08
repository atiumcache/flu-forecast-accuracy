from typing import Tuple
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


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
        mean temperature (celsius) for given date and location
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
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean

    daily_dataframe = pd.DataFrame(data=daily_data)
    print(daily_dataframe)
