import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from request_data import activities_summary_data
from count_settings import strava_api_settings



if __name__ == "__main__":
    # Load parameters from YAML file
    params = yaml.safe_load(open("params.yaml", "r"))
    athlete_name = params["personal_data"]["name"]

    # Get Strava API access token
    access_token = strava_api_settings()
    print("Strava API access token obtained successfully.")

    # Fetch all activities summary data
    all_activities_summary = activities_summary_data(access_token)
    print("Athlete activities summary data fetched successfully.")

    name = athlete_name.replace(" ", "")
    # read athlete metadata
    with open(f"data/archivos_json_actividades/{name}_athlete_run_metadata.json",  "r") as f:
        athlete_run_metadata = json.load(f)