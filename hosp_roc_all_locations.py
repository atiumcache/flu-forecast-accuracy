import subprocess
from typing import Dict
import pandas as pd
import multiprocessing as mp


def main():
    location_to_state_map = map_loc_codes()

    with mp.Pool(mp.cpu_count()) as pool:
        """
        Runs the hosp_roc.py script on each location.
        Results are output to csv files in `./hosp_roc`.
        """
        pool.map(run_script, location_to_state_map.keys())


def run_script(loc_code):
    subprocess.run(['python3', 'hosp_roc.py', '-l', loc_code, '--mcmc'])


def map_loc_codes() -> Dict:
    # Read location data.
    locations = pd.read_csv("datasets/locations.csv").iloc[1:]  # skip first row (national ID)
    # Map location codes to state abbreviations.
    location_to_state = dict(zip(locations["location"], locations["abbreviation"]))
    return location_to_state


if __name__ == "__main__":
    main()
