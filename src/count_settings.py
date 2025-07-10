import yaml
import json
import requests

def strava_api_settings():
    
    """
    Function to read Strava API settings from a YAML file and 
    a JSON payload file. It returns the athlete's access token.
    """
    params = yaml.safe_load(open("params.yaml", "r"))

    athlete_name = params["personal_data"]["name"]

    with open('data/datos_payload.json', 'r') as json_file:
        payload_data = json.load(json_file)
    
    token_uri = "https://www.strava.com/api/v3/oauth/token"
    payload = {
        'client_id':payload_data[athlete_name]['client_id'],
        'client_secret': payload_data[athlete_name]['client_secret'],
        'refresh_token': payload_data[athlete_name]['refresh_token'],
        'grant_type':'refresh_token',
        'f':'json'
    }

    response = requests.post(token_uri, data=payload, verify=False).json()

    access_token = response['access_token']

    return access_token