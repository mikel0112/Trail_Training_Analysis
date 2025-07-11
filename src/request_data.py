import requests
import yaml
import json
import os
import time
from count_settings import strava_api_settings

def activities_summary_data(access_token):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    all_activities_summary = []
    request_page_number = 1
    
    while True:
        params = {
            "per_page": 200,
            "page": request_page_number
        }
        response = requests.get(url, headers=headers, params=params).json()
        request_page_number += 1
        
        if len(response) == 0:
            break

        if not all_activities_summary:
            all_activities_summary = response
        else:
            all_activities_summary.extend(response)
        
    return all_activities_summary

def get_activity_data(activity_id, access_token):

    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = headers = {"Authorization": f"Bearer {access_token}"}
    keys = ['time', 'distance', 'altitude', 'heartrate', 'cadence', 'watts', 'grade_smooth','velocity_smooth']
    keys = ','.join(keys)
    param = {'keys': keys, 'key_by_type': True}

    response = requests.get(url, headers=headers, params=param).json()
    return response

def fetch_activities_data(list_activity_ids, athlete_run_metadata, old_list, name, access_token):

    RATE_LIMIT = 70
    TIME_FRAME = 18*60
    request_count = 0  # Counter for the number of requests
    start_time = time.time()  # Start time to track time frame
    max_retries = 10  # maximo 10 veces para pasar 700 actividades

    for activity_id in list_activity_ids:
        # Skip activity if it already exists in the metadata
        # Check if we are exceeding the rate limit
        if request_count >= RATE_LIMIT:
            elapsed_time = time.time() - start_time  # Time since first request in this batch
            if elapsed_time < TIME_FRAME:
                # Calculate how much time to wait
                wait_time = TIME_FRAME - elapsed_time
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)

                # Reset the counter and start time after waiting
                request_count = 0
                start_time = time.time()
                max_retries -= 1
                if max_retries <= 0:
                    print("Max retries reached. Exiting the loop.")
                    break

        # Check if the activity ID is already in the metadata
        if str(activity_id) in old_list:
            continue
        else:
            # Fetch the activity data (assuming get_activity_data is defined elsewhere)
            try:
                activity_data = get_activity_data(activity_id, access_token)
                if activity_data.get("message") == 'Rate Limit Exceeded':
                    continue
                else:
                    athlete_run_metadata[f"{name}"][activity_id] = activity_data
                # Increment the request count after a successful request
                request_count += 1
            except Exception as e:
                print(f"Error fetching data for activity {activity_id}: {e}")
                continue

    return athlete_run_metadata

def run_activities_full_data(name, run_activities_id, file_path, access_token):
        
    old_list = []
    # Check if the file exists before attempting to open it
    if os.path.exists(file_path):
        print('el path existe')

        with open(file_path, 'r') as json_file:
            athlete_run_metadata = json.load(json_file)
        
        for key in athlete_run_metadata[f"{name}"]:
            old_list.append(key)
        # if it 
        print(f"El anterior json tenia {len(old_list)} actividades.")
    else:
        print("el archivo no existe lo creamos")
        athlete_run_metadata = {f"{name}":{}}

    athlete_run_metadata = fetch_activities_data(run_activities_id, athlete_run_metadata, old_list, name, access_token)
    return athlete_run_metadata

if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml", "r"))
    athlete_name = params["personal_data"]["name"]

    access_token = strava_api_settings()
    print("Strava API access token obtained successfully.")

    all_activities_summary = activities_summary_data(access_token)
    run_activities_id = [activity['id'] for activity in all_activities_summary if activity['type'] == 'Run']
    print("Athlete activities summary data fetched successfully.")

    name = athlete_name.replace(" ", "")
    file_path = f"data/archivos_json_actividades/{name}_athlete_run_metadata.json"
    athlete_run_metadata = run_activities_full_data(athlete_name, run_activities_id, file_path, access_token)
    print("Athlete run metadata fetched successfully.")

    with open(file_path, 'w') as json_file:
        json.dump(athlete_run_metadata, json_file, indent=4)