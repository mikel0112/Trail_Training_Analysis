import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime
from html2image import Html2Image
from request_data import activities_summary_data
from count_settings import strava_api_settings

def dataframes_dict(athlete_run_metadata, all_activities_summary, name):
    list_ids_all_activities = []
    for activity in all_activities_summary:
        list_ids_all_activities.append(activity['id'])
    list_ids_athlete_run_metadata = []
    for key in athlete_run_metadata[name]:
        list_ids_athlete_run_metadata.append(key)
    
    #añadir fecha desde all_activities al diccionario de cada actividad dentro de athlete_run_metadata
    for activity in all_activities_summary:
        for key in athlete_run_metadata[name]:
            if activity['id'] == int(key):
                # Convertir la cadena a un objeto de fecha y hora
                fecha_convertida = datetime.strptime(activity['start_date_local'], "%Y-%m-%dT%H:%M:%SZ")

                # Extraer solo año, mes y día y formatearlo como "YYYY-MM-DD"
                fecha_formateada = fecha_convertida.strftime("%Y-%m-%d")
                athlete_run_metadata[name][key]['fecha'] = fecha_formateada

    # crear dataframe por actividad y guardarlo en un diccionario
    columnas_df_activity = ['tiempo', 'distancia', 'altitud', 'pulso', 'cadencia', 'potencia', 'pendiente','ritmo']
    diccionario_dataframes = {}
    list_of_ids = []
    for activity in list_ids_athlete_run_metadata:
    # Verificar si existen los datos antes de asignarlos
    # Obtener los datos (si existen) y garantizar que sea una lista, si no devolver una lista vacía
    #fecha = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('fecha', []) or []
        tiempo = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('time', {}).get('data', []) or []
        distancia = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('distance', {}).get('data', []) or []
        altitude = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('altitude', {}).get('data', []) or []
        heartrate = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('heartrate', {}).get('data', []) or []
        cadencia = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('cadence', {}).get('data', []) or []
        watts = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('watts', {}).get('data', []) or []
        cuesta_porcentaje = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('grade_smooth', {}).get('data', []) or []
        ritmo = athlete_run_metadata[f"{name}"].get(f"{activity}", {}).get('velocity_smooth', {}).get('data', []) or []
        # Crear el DataFrame solo con las columnas que contienen datos
        data_dict = {
            'tiempo': tiempo,
            'distancia': distancia,
            'altitude': altitude,
            'heartrate': heartrate,
            'cadencia': cadencia,
            'watts': watts,
            'cuesta_porcentaje': cuesta_porcentaje,
            'ritmo': ritmo
        }

        # Filtrar solo las columnas que tienen datos (listas no vacías)
        data_dict = {col: data for col, data in data_dict.items() if data}
        # Crear DataFrame con las columnas filtradas
        df_activity = pd.DataFrame(data_dict)
        if 'distancia' in df_activity.columns:
            df_activity['distancia']=df_activity['distancia']/1000
        if 'tiempo' in df_activity.columns:
            df_activity['tiempo']=df_activity['tiempo']/60
        # convertir velocidad min/km
        if 'ritmo' in df_activity.columns:
            df_activity['ritmo'] = 1000/(df_activity['ritmo']*60)

        diccionario_dataframes[activity] = {}
        diccionario_dataframes[activity]['fecha'] = athlete_run_metadata[name][activity]['fecha']
        diccionario_dataframes[activity]['datos'] = df_activity
    
    # Ordenar el diccionario por fecha
    diccionario_dataframes = dict(sorted(
        diccionario_dataframes.items(),
        key=lambda item: datetime.strptime(item[1]['fecha'], "%Y-%m-%d"),
        reverse=True
    ))

    return diccionario_dataframes

def power_curve(diccionario_dataframes):
    # CURVA DE POTENCIA Y CP

    # Definimos la duración en segundos para la curva de potencia
    durations = [5, 10, 30, 60, 180, 300, 600, 900, 1200,1800, 3600, 7200, 10800, 14400]  # Ejemplo: 5s, 10s, 30s, 1min, 3min, 5min, 10min, 15min, 20min, 30min, 1h, 2h, 3h, 4h
    # Creamos un DataFrame para almacenar los resultados de la curva de potencia
    power_curve_data = []
    # curva de potencia
    for activity in diccionario_dataframes:
        # cambiar para que solo coja las actividades 1 año previas a la fecha actual
        current_date = datetime.now()
        # Subtract one year
        one_year_ago = current_date - relativedelta(years=1)
        formatted_date = one_year_ago.strftime("%Y-%m-%d")
        if diccionario_dataframes[activity]['fecha'] > formatted_date:
            max_power = {}
            for duration in durations:
                if 'watts' in diccionario_dataframes[activity]['datos'].columns:
                    max_power[duration] = diccionario_dataframes[activity]['datos']['watts'].rolling(duration).mean().max()
                    # Añadimos los resultados al DataFrame
                    power_curve_data.append(max_power)

    # Convertir los resultados en un DataFrame
    numero_actividades = len(power_curve_data)
    power_curve_df = pd.DataFrame(power_curve_data, index=[f'Actividad {i+1}' for i in range(numero_actividades)])

    # Transponer el DataFrame para mejor visualización
    power_curve_df = power_curve_df.T

    # Calcular la potencia conjunta tomando el máximo entre las actividades
    combined_power_curve = power_curve_df.max(axis=1)
    # pasar el index a columna
    combined_power_curve = combined_power_curve.reset_index()
    combined_power_curve.columns = ['Duración', 'Potencia Máxima']
    combined_power_curve['Work_capacity'] = combined_power_curve['Potencia Máxima']*combined_power_curve['Duración']

    # calculo de la potencia critica
    CP = ((combined_power_curve['Work_capacity'][7] - combined_power_curve['Work_capacity'][4])/(combined_power_curve['Duración'][7]- combined_power_curve['Duración'][4])).round(0)
    work_capacity = (combined_power_curve['Work_capacity'][5] - (CP*combined_power_curve['Duración'][5])).round(0)

    # Mostrar los resultados
    print(f'Critical Power: {CP} W')
    print(f'Work Capacity (W\'): {work_capacity} J')

    return combined_power_curve, CP

# Define the style function to color based on comparison
def highlight_greater(row):
    # Initialize a list with empty styles for all columns
    styles = ['' for _ in row.index]

    # Compare the % columns between 'pro_cyclist' and 'athlete'
    pro_cyclist_index = row.index.get_loc(('pro_cyclist', '%'))
    athlete_index = row.index.get_loc(('athlete', '%'))

    if row[('pro_cyclist', '%')] > row[('athlete', '%')]:
        styles[pro_cyclist_index] = 'background-color: lightgreen;'
        styles[athlete_index] = 'background-color: lightcoral;'
    else:
        styles[pro_cyclist_index] = 'background-color: lightcoral;'
        styles[athlete_index] = 'background-color: lightgreen;'

    return styles
def power_comp_with_cyclists(weight, combined_power_curve):
    # COMPARACIÓN PERFIL POTENCIA CON CICLISTAS
    # datos ciclistas
    potencias_ciclistas = [820, 531, 481, 453, 427, 398, 355, 338, 325]
    w_kg_ciclistas = [11.33, 7.65, 7, 6.59, 6.24, 5.76, 5.12, 4.84, 4.63]
    tiempo_minutos = [1, 5, 10, 20, 30, 60, 120, 180, 240]

    df_comparacion_pot_ciclistas = pd.DataFrame({
        'Potencia (W)': potencias_ciclistas,
        'W/kg': w_kg_ciclistas,
        'Tiempo (min)': tiempo_minutos
    })
    df_comparacion_pot_ciclistas.set_index('Tiempo (min)', inplace=True)
    df_comparacion_pot_ciclistas['%'] = ((df_comparacion_pot_ciclistas['Potencia (W)'] / df_comparacion_pot_ciclistas['Potencia (W)'][60]) * 100).round(2)

    # datos atleta
    combined_power_curve['Duración'] = combined_power_curve['Duración']/60
    combined_power_curve.drop([0,1,2,4,7], inplace=True)
    combined_power_curve.drop(columns=['Work_capacity'], inplace=True)
    combined_power_curve.set_index('Duración', inplace=True)
    combined_power_curve['Potencia Máxima'] = combined_power_curve['Potencia Máxima'].round(0)
    ##################PESO ATLETA#########################################
    peso_atleta = weight
    ######################################################################
    combined_power_curve['W/kg'] = (combined_power_curve['Potencia Máxima']/peso_atleta).round(2)
    combined_power_curve.rename(columns={'Potencia Máxima': 'Potencia (W)'}, inplace=True)
    combined_power_curve['%'] = ((combined_power_curve['Potencia (W)'] / combined_power_curve['Potencia (W)'][60]) * 100).round(2)
    df_comparacion_pot_ciclistas = pd.concat([df_comparacion_pot_ciclistas, combined_power_curve],axis=1)
    # Define the new multi-level column structure
    multi_index = pd.MultiIndex.from_tuples([
        ('pro_cyclist', 'Potencia (W)'),
        ('pro_cyclist', 'W/kg'),
        ('pro_cyclist', '%'),
        ('athlete', 'Potencia (W)'),
        ('athlete', 'W/kg'),
        ('athlete', '%')
    ])
    # Set the new multi-level column structure
    df_comparacion_pot_ciclistas.columns = multi_index

    # poner color a los porcentajes
    # Apply the style function row-wise
    styled_df = df_comparacion_pot_ciclistas.style.apply(highlight_greater, axis=1)

    return styled_df

def tiempo_zonas_potencia(diccionario_dataframes, list_años, df_zonas_potencia, critical_power):

    for año in list_años:
        tiempo_zona_1 = 0
        tiempo_zona_2 = 0
        tiempo_zona_3 = 0
        tiempo_zona_4 = 0
        tiempo_zona_5 = 0

        for activity in diccionario_dataframes:
            if 'watts' in diccionario_dataframes[activity]['datos'].columns:
                for potencia in diccionario_dataframes[activity]['datos']['watts']:
                    if diccionario_dataframes[activity]['fecha'][0:4] ==  año:
                        if potencia <= CP*0.56:
                            tiempo_zona_1 += 1
                        elif potencia <= CP*0.75:
                            tiempo_zona_2 += 1
                        elif potencia <= CP*0.9:
                            tiempo_zona_3 += 1
                        elif potencia <= CP*1.05:
                            tiempo_zona_4 += 1
                        else:
                            tiempo_zona_5 += 1
        df_zonas_potencia.loc[año] = [tiempo_zona_1,tiempo_zona_2,tiempo_zona_3,tiempo_zona_4,tiempo_zona_5]
    return df_zonas_potencia
def power_annual_distribution(diccionario_dataframes, CP):
    df_zonas_potencia = pd.DataFrame(columns=['Z1','Z2','Z3','Z4','Z5'])

    list_años = []
    for key in diccionario_dataframes:
        año = diccionario_dataframes[key]['fecha'][0:4]
        if año not in list_años and 'watts' in diccionario_dataframes[key]['datos'].columns :
            list_años.append(año)
    
    df_zonas_potencia = tiempo_zonas_potencia(diccionario_dataframes, list_años, df_zonas_potencia, CP)
    df_zonas_potencia = (df_zonas_potencia/3600).round(2) #pasar de segundos a horas
    df_zonas_potencia['Total'] = df_zonas_potencia.sum(axis=1)
    # pasar a porcentaje añadiendo columna
    df_zonas_potencia['Z1%'] = ((df_zonas_potencia['Z1']/df_zonas_potencia['Total'])*100).round(2)
    df_zonas_potencia['Z2%'] = ((df_zonas_potencia['Z2']/df_zonas_potencia['Total'])*100).round(2)
    df_zonas_potencia['Z3%'] = ((df_zonas_potencia['Z3']/df_zonas_potencia['Total'])*100).round(2)
    df_zonas_potencia['Z4%'] = ((df_zonas_potencia['Z4']/df_zonas_potencia['Total'])*100).round(2)
    df_zonas_potencia['Z5%'] = ((df_zonas_potencia['Z5']/df_zonas_potencia['Total'])*100).round(2)

    df_zonas_potencia.drop(columns=['Z1','Z2','Z3','Z4','Z5'])
    #ordenamos las columnas
    df_zonas_potencia = df_zonas_potencia[['Z1%','Z2%','Z3%','Z4%','Z5%']]

    return df_zonas_potencia

def tiempo_zonas_pulso(diccionario_dataframes, list_años, df_zonas_pulso, pulso_max):

    for año in list_años:
        tiempo_zona_1 = 0
        tiempo_zona_2 = 0
        tiempo_zona_3 = 0
        tiempo_zona_4 = 0
        tiempo_zona_5 = 0
        for activity in diccionario_dataframes:
            if 'heartrate' in diccionario_dataframes[activity]['datos'].columns:
                for pulso in diccionario_dataframes[activity]['datos']['heartrate']:
                    if diccionario_dataframes[activity]['fecha'][0:4] ==  año:
                        if pulso <= pulso_max*0.72:
                            tiempo_zona_1 += 1
                        elif pulso <= pulso_max*0.82:
                            tiempo_zona_2 += 1
                        elif pulso <= pulso_max*0.87:
                            tiempo_zona_3 += 1
                        elif pulso <= pulso_max*0.92:
                            tiempo_zona_4 += 1
                        else:
                            tiempo_zona_5 += 1
        df_zonas_pulso.loc[año] = [tiempo_zona_1,tiempo_zona_2,tiempo_zona_3,tiempo_zona_4,tiempo_zona_5]
    return df_zonas_pulso

def heartrate_annual_distribution(diccionario_dataframes, max_heartrate):

    df_zonas_pulso = pd.DataFrame(columns=['Z1','Z2','Z3','Z4','Z5'])
    pulso_max = max_heartrate #pulso maximo
    list_años = []
    for key in diccionario_dataframes:
        año = diccionario_dataframes[key]['fecha'][0:4]
        if año not in list_años and 'heartrate' in diccionario_dataframes[key]['datos'].columns:
            list_años.append(año)

    df_zonas_pulso = tiempo_zonas_pulso(diccionario_dataframes, list_años, df_zonas_pulso, pulso_max)
    df_zonas_pulso = (df_zonas_pulso/3600).round(2) #pasar de segundos a horas
    df_zonas_pulso['Total'] = df_zonas_pulso.sum(axis=1)
    # pasar a porcentaje añadiendo columna
    df_zonas_pulso['Z1%'] = ((df_zonas_pulso['Z1']/df_zonas_pulso['Total'])*100).round(2)
    df_zonas_pulso['Z2%'] = ((df_zonas_pulso['Z2']/df_zonas_pulso['Total'])*100).round(2)
    df_zonas_pulso['Z3%'] = ((df_zonas_pulso['Z3']/df_zonas_pulso['Total'])*100).round(2)
    df_zonas_pulso['Z4%'] = ((df_zonas_pulso['Z4']/df_zonas_pulso['Total'])*100).round(2)
    df_zonas_pulso['Z5%'] = ((df_zonas_pulso['Z5']/df_zonas_pulso['Total'])*100).round(2)

    df_zonas_pulso.drop(columns=['Z1','Z2','Z3','Z4','Z5'])
    #ordenamos las columnas
    df_zonas_pulso = df_zonas_pulso[['Z1%','Z2%','Z3%','Z4%','Z5%']]

    return df_zonas_pulso

def unify_power_and_heartrate(df_zonas_potencia, df_zonas_pulso):
    #añadimos fila de super encabezado
    df_zonas_potencia.columns = pd.MultiIndex.from_product([['Potencia'], df_zonas_potencia.columns])
    df_zonas_pulso.columns = pd.MultiIndex.from_product([['Pulso'], df_zonas_pulso.columns])

    #concatenamos los dataframes
    df_zonas = pd.concat([df_zonas_pulso, df_zonas_potencia], axis=1)
    df_zonas = df_zonas.round(2)
    return df_zonas

def plot_yearly_intensity_distribution(df_zonas):
    # GRAFICA DE LAS ZONAS

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Pulso subplot
    pulso_ax = axes[0]
    pulso_bars = df_zonas['Pulso'].plot(kind='bar', stacked=True, ax=pulso_ax, colormap='Blues')
    pulso_ax.set_title('Pulso Zones')
    pulso_ax.set_ylabel('Percentage %')
    pulso_ax.legend(title='Zones')

    # Adding cell values as markers to the Pulso bars
    for i in range(len(df_zonas)):
        heights = df_zonas['Pulso'].iloc[i].values
        cumulative_heights = np.cumsum(heights)

        for j in range(len(heights)):
            # Center position for the marker
            center = cumulative_heights[j] - (heights[j] / 2)
            # Annotate with the value
            pulso_ax.annotate(f'{heights[j]:.2f}', xy=(i, center), ha='center', va='center', fontsize=10, color='black')
    # Move legend outside of the graph
    pulso_ax.legend(title='Zones', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Potencia subplot
    potencia_ax = axes[1]
    potencia_bars = df_zonas['Potencia'].plot(kind='bar', stacked=True, ax=potencia_ax, colormap='Oranges')
    potencia_ax.set_title('Potencia Zones')
    potencia_ax.legend(title='Zones')

    # Adding cell values as markers to the Potencia bars
    for i in range(len(df_zonas)):
        heights = df_zonas['Potencia'].iloc[i].values
        cumulative_heights = np.cumsum(heights)

        for j in range(len(heights)):
            # Center position for the marker
            center = cumulative_heights[j] - (heights[j] / 2)
            # Annotate with the value
            potencia_ax.annotate(f'{heights[j]:.2f}', xy=(i, center), ha='center', va='center', fontsize=10, color='black')
    # Move legend outside of the graph
    potencia_ax.legend(title='Zones', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs('outputs/yearly_intensity_distribution', exist_ok=True)
    grafico_zonas_path = "outputs/yearly_intensity_distribution/grafico_distribucion_zonas.png"
    plt.savefig(grafico_zonas_path)
    plt.show()

def individualized_zones(pulso_max, CP):
    # Tabla Valores Individualizados Zonas
    df_zonas_pulso_potencia = pd.DataFrame(columns=['Pulso', 'Potencia'], index=['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    #calculo valor maximo zona pulso
    df_zonas_pulso_potencia.at['Z1', 'Pulso'] = pulso_max*0.72
    df_zonas_pulso_potencia.at['Z2', 'Pulso'] = pulso_max*0.82
    df_zonas_pulso_potencia.at['Z3', 'Pulso'] = pulso_max*0.87
    df_zonas_pulso_potencia.at['Z4', 'Pulso'] = pulso_max*0.92
    df_zonas_pulso_potencia.at['Z5', 'Pulso'] = pulso_max
    #calculo valor maximo zona potencia
    df_zonas_pulso_potencia.at['Z1', 'Potencia'] = CP*0.56
    df_zonas_pulso_potencia.at['Z2', 'Potencia'] = CP*0.75
    df_zonas_pulso_potencia.at['Z3', 'Potencia'] = CP*0.9
    df_zonas_pulso_potencia.at['Z4', 'Potencia'] = CP*1.05
    df_zonas_pulso_potencia.at['Z5', 'Potencia'] = CP*1.2
    df_zonas_pulso_potencia = df_zonas_pulso_potencia.round(0).astype(int)

    return df_zonas_pulso_potencia

def table_individualized_zones(df_zonas_pulso_potencia):
    # Prepare table data with hierarchical column headers
    cell_text = []

    # Add the first level of columns (Category1, Category2)
    first_row = [''] + [col for col in df_zonas_pulso_potencia.columns]
    cell_text.append(first_row)

    # Add the main data rows, including the index as the first column
    for idx, row in zip(df_zonas_pulso_potencia.index, df_zonas_pulso_potencia.values):
        cell_text.append([idx] + list(row))

    # Define column labels (empty since we are manually adding rows for headers)
    col_labels = [''] * len(cell_text[0])

    # Plot the table
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')

    # Set the font size for the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Save the plot as an image file
    plt.savefig("outputs/yearly_intensity_distribution/tabla_zonas_intensidad.png", dpi=300, bbox_inches='tight')
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

    # print CP and W'
    combined_power_curve, CP = power_curve(dfs_dict)

    # Compare with professional cyclists
    comparison_df = power_comp_with_cyclists(weight, combined_power_curve)

    # yearly intensity distribution power
    df_zonas_potencia = power_annual_distribution(dfs_dict, CP)

    # yearly intensity distribution pulse
    df_zonas_pulso = heartrate_annual_distribution(dfs_dict, max_heartrate)

    # yearly intensity distribution power and pulse
    df_zonas = unify_power_and_heartrate(df_zonas_potencia, df_zonas_pulso)

    # plot yearly intensity distribution power and pulse
    plot_yearly_intensity_distribution(df_zonas)

    # generare individualized training zones
    indivividual_zones = individualized_zones(max_heartrate, CP)

    # table individualized zones
    table_individualized_zones(indivividual_zones)