import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from request_data import activities_summary_data
from count_settings import strava_api_settings
from yearly_intensity_distribution import dataframes_dict

def separate_trail_road_activities(diccionario_dataframes):
    # guardamos en un diccionario solo las actividades con un desnivel >300
    diccionario_rendimientos_trail = {}
    diccionario_rendmientos_llano = {}
    for key in diccionario_dataframes:
    # guardamos en una variable desnivel_pos la suma de la diferencia de altitude siempre que sea positiva
        if 'altitude' in diccionario_dataframes[key]['datos'].columns:
            desnivel_pos = diccionario_dataframes[key]['datos']['altitude'].diff().where(diccionario_dataframes[key]['datos']['altitude'].diff() > 0).sum()
            if desnivel_pos > 300:
                diccionario_rendimientos_trail[key] = diccionario_dataframes[key]
            elif diccionario_dataframes[key]['datos']['distancia'].max()>10:
                diccionario_rendmientos_llano[key] = diccionario_dataframes[key]

    return diccionario_rendimientos_trail, diccionario_rendmientos_llano

def separate_trail_disciplines(diccionario_rendimientos_trail, cadencia_max, cadencia_min, ascenso_max, ascenso_min):
    diccionario_sub_correr = {}
    for key in diccionario_rendimientos_trail:
    #comprobamos que existe cadencia y cuesta porcentaje
        if 'cadencia' in diccionario_rendimientos_trail[key]['datos'].columns and\
            'cuesta_porcentaje' in diccionario_rendimientos_trail[key]['datos'].columns and\
            'heartrate' in diccionario_rendimientos_trail[key]['datos'].columns and\
            'watts' in diccionario_rendimientos_trail[key]['datos'].columns:
            df = diccionario_rendimientos_trail[key]['datos']
            df['ascenso'] = df['altitude'].diff()
            # loopear
            # Apply the condition to keep only rows where the conditions are met
            df_filtered = df[(df['ascenso'] > ascenso_min) & (df['ascenso'] < ascenso_max) & (df['cadencia'] > cadencia_min) & (df['cadencia'] < cadencia_max)]
            diccionario_sub_correr[key] = key
            diccionario_sub_correr[key] = {}
            diccionario_sub_correr[key]['datos'] = df_filtered
            diccionario_sub_correr[key]['fecha'] = diccionario_rendimientos_trail[key]['fecha']

    return diccionario_sub_correr

def dict_discipline_samples(diccionario_sub_correr):
    # guardar en diccionarios subidas por pulsos, duraciones y porcentaje
    running_uphills = {}
    for key in diccionario_sub_correr:
        df = diccionario_sub_correr[key]['datos']
        if df.empty:
            continue
        else:
            index_diff = np.diff(df.index)
            split_points = np.where(index_diff > 5)[0] + 1


            # separar las subidas cuando haya un salto en el index
            activity_run_uphills = {}

            # first index value
            start_pos = 0
            n_uphill = 0
            for point in split_points:
                end_pos = point
                start_index = df.index[start_pos]
                end_index = df.index[end_pos-1]
                if end_index - start_index > 400:
                    n_uphill += 1
                    #print(f"previous value: {start_index}, next value {end_index}")
                    activity_run_uphills[n_uphill] = df.iloc[start_pos:end_pos]
                    start_pos = end_pos
                    break
                else:
                    start_pos = end_pos
            final_index = df.index[-1]
            start_index = df.index[start_pos]
            if (final_index - start_index > 400) and (final_index - start_index < 6000):
                activity_run_uphills[n_uphill + 1] = df.iloc[start_pos:]
            # si el diccionario de actividad no esta vacio
            if activity_run_uphills:
                # datos de activity a diccionario global running_uphills
                running_uphills[diccionario_sub_correr[key]['fecha']] = activity_run_uphills

    return running_uphills

def dict_discipline_stats(running_uphills):
    # running uphill stats
    running_uphills_stats = {}
    for key in running_uphills.keys():
        for uphill in running_uphills[key].keys():
            df = running_uphills[key][uphill]
            running_uphills_stats[key] = {}
            running_uphills_stats[key][uphill] = {}
            running_uphills_stats[key][uphill]['tiempo'] = round(df['tiempo'].max()-df['tiempo'].min(),2)
            running_uphills_stats[key][uphill]['distancia'] = round(df['distancia'].max()-df['distancia'].min(),2)
            running_uphills_stats[key][uphill]['ritmo'] = round(float(df['ritmo'].mean()),2)
            running_uphills_stats[key][uphill]['pulso'] = round(float(df['heartrate'].mean()),2)
            running_uphills_stats[key][uphill]['potencia'] = round(float(df['watts'].mean()),2)
            running_uphills_stats[key][uphill]['cuesta_porcentaje'] = round(float(df['cuesta_porcentaje'].mean()),2)
            running_uphills_stats[key][uphill]['desnivel'] = round(df['altitude'].max()-df['altitude'].min(),2)
            running_uphills_stats[key][uphill]['velocidad_vertical m/h'] = round(running_uphills_stats[key][uphill]['desnivel']/(running_uphills_stats[key][uphill]['tiempo']/60),2)

    return running_uphills_stats

def create_personal_best_dict(running_uphills_stats, discipline, personal_bests):
    
    for key in running_uphills_stats:
        year = key[:4]
        for uphill in running_uphills_stats[key].keys():
            pulso = running_uphills_stats[key][uphill]['pulso']
            if pulso >= 110 and pulso < 120:
                if discipline not in personal_bests[year]['pulso 110_120']:
                    personal_bests[year]['pulso 110_120'][discipline] = running_uphills_stats[key][uphill]
                    personal_bests[year]['pulso 110_120'][discipline]['fecha'] = key
                else:
                    best_vv = personal_bests[year]['pulso 110_120'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 110_120'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 120 and pulso < 130:
                if discipline not in personal_bests[year]['pulso 120_130']:
                    personal_bests[year]['pulso 120_130'][discipline] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 120_130'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 120_130'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 130 and pulso < 140:
                if discipline not in personal_bests[year]['pulso 130_140']:
                    personal_bests[year]['pulso 130_140'][discipline] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 130_140'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 130_140'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 140 and pulso < 150:
                if discipline not in personal_bests[year]['pulso 140_150']:
                    personal_bests[year]['pulso 140_150'][discipline] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 140_150'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 140_150'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 150 and pulso < 160:
                if discipline not in personal_bests[year]['pulso 150_160']:
                    personal_bests[year]['pulso 150_160'][discipline] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 150_160'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 150_160'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 160 and pulso < 170:
                if discipline not in personal_bests[year]['pulso 160_170']:
                    personal_bests[year]['pulso'] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 160_170'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 160_170'][discipline] = running_uphills_stats[key][uphill]
            elif pulso >= 170 and pulso < 180:
                if discipline not in personal_bests[year]['pulso 170_180']:
                    personal_bests[year]['pulso 170_180'][discipline] = running_uphills_stats[key][uphill]
                else:
                    best_vv = personal_bests[year]['pulso 170_180'][discipline]['potencia']
                    last_vv = running_uphills_stats[key][uphill]['potencia']
                    if last_vv > best_vv:
                        personal_bests[year]['pulso 170_180'][discipline] = running_uphills_stats[key][uphill]

    return personal_bests

def pbs_df(personal_bests):
    # generar tabla
    rows = []
    for year, zonas in personal_bests.items():
        for zona, actividades in zonas.items():
            for actividad, datos in actividades.items():
                if isinstance(datos, dict):
                    row = {
                        'Año': int(year),
                        'Zona': zona,
                        'Actividad': actividad,
                        'Tiempo': datos['tiempo'],
                        'Distancia': datos['distancia'],
                        'Ritmo': datos['ritmo'],
                        'Pulso': datos['pulso'],
                        'Potencia': datos['potencia'],
                        'Desnivel': datos['desnivel'],
                        'Velocidad Vertical': datos['velocidad_vertical m/h']
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['Zona', 'Actividad'])

    return df

def plot_pbs(df):
    # Preparar zonas
    zones = df['Zona'].unique()
    n_zones = len(zones)

    # Crear subplots
    fig, axes = plt.subplots(nrows=n_zones, figsize=(12, 5 * n_zones), sharex=False)

    # Asegurar que axes sea iterable
    if n_zones == 1:
        axes = [axes]

    for idx, zone in enumerate(zones):
        ax1 = axes[idx]
        df_zone = df[df['Zona'] == zone]

        activities = [a for a in df_zone['Actividad'].unique() if a != 'downhill']
        years = sorted(df_zone['Año'].unique())
        x = np.arange(len(years))
        width = 0.8 / len(activities)

        for i, activity in enumerate(activities):
            df_activity = df_zone[df_zone['Actividad'] == activity]
            df_activity = df_activity.groupby('Año')['Potencia'].sum().reindex(years).fillna(0)

            bars = ax1.bar(x + i * width, df_activity.values, width, label=activity)

            # Añadir valores encima de cada barra
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0f}',
                            ha='center', va='bottom', fontsize=8)

        # Configurar ejes
        ax1.set_title(f"Zona: {zone}")
        ax1.set_ylabel("Potencia (Uphill)")
        ax1.set_xticks(x + width * (len(activities)-1)/2)
        ax1.set_xticklabels(years, rotation=45)
        ax1.set_xlabel("Año")  # <- ahora se muestra en todos los subplots
        ax1.legend()

    # Título global
    fig.suptitle("Potencia por Año y Actividad por Zona", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Guardar figura
    plt.savefig("outputs/personal_bests/discipline_pbs_by_year.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load parameters from YAML file
    params = yaml.safe_load(open("params.yaml", "r"))
    weight = params["personal_data"]["weight"]
    max_heartrate = params["personal_data"]["max_pulse"]
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
    
    # Create a dictionary of DataFrames
    dfs_dict = dataframes_dict(athlete_run_metadata, all_activities_summary, athlete_name)

    # Separate trail and road activities
    diccionario_rendimientos_trail, diccionario_rendmientos_llano = separate_trail_road_activities(dfs_dict)

    # uphills running
    diccionario_uphills_running = separate_trail_disciplines(diccionario_rendimientos_trail, 500.0, 75.0, 500.0, 0.0)
    running_uphills = dict_discipline_samples(diccionario_uphills_running)
    running_uphills_stats = dict_discipline_stats(running_uphills)

    # uphills hiking
    diccionario_uphills_hiking = separate_trail_disciplines(diccionario_rendimientos_trail, 75.0, 0.0, 500.0, 0.0)
    hiking_uphills = dict_discipline_samples(diccionario_uphills_hiking)
    hiking_uphills_stats = dict_discipline_stats(hiking_uphills)

    # downhills running
    diccionario_downhills_running = separate_trail_disciplines(diccionario_rendimientos_trail, 500.0, 70.0, 0.0, -500.0)
    running_downhills = dict_discipline_samples(diccionario_downhills_running)
    running_downhills_stats = dict_discipline_stats(running_downhills)

    # personal bests
    # dejar anualmente cada 10 pulsaciones la que mejor velocidad vertical tenga
    personal_bests = {
        '2025' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2024' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2023' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2022' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2021' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2020' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2019' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2018' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2017' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2016' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
        '2015' : {
            'pulso 110_120': {},
            'pulso 120_130': {},
            'pulso 130_140': {},
            'pulso 140_150': {},
            'pulso 150_160': {},
            'pulso 160_170': {},
            'pulso 170_180': {},
        },
    }

    personal_bests = create_personal_best_dict(running_uphills_stats, 'uphill_run', personal_bests)
    personal_bests = create_personal_best_dict(hiking_uphills_stats, 'uphill_hike', personal_bests)
    personal_bests = create_personal_best_dict(running_downhills_stats, 'downhill_run', personal_bests)

    # save personal bests
    os.makedirs('outputs/personal_bests', exist_ok=True)
    with open(f"outputs/personal_bests/{name}_personal_bests.json", "w") as f:
        json.dump(personal_bests, f, indent=4)

    # plot personal bests
    personal_bests_df = pbs_df(personal_bests)
    plot_pbs(personal_bests_df)

    