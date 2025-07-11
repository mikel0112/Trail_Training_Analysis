import pandas as pd
import yaml
import matplotlib.pyplot as plt
import os
from request_data import activities_summary_data
from count_settings import strava_api_settings

def total_df(activities_summary):
    # Crear un DataFrame con los datos solicitados
    df_total = pd.DataFrame(activities_summary, columns=['start_date', 'type', 'distance', 'moving_time', 'total_elevation_gain', 'average_watts', 'average_heartrate','kilojoules'])

    # Establecer 'start_date' como índice
    df_total.set_index('start_date', inplace=True)

    # Cambiar los nombres de algunas columnas
    df_total.rename(columns={
        'type': 'Tipo_actividad',
        'distance': 'Distancia',
        'moving_time': 'Tiempo_segundos',
        'total_elevation_gain': 'Desnivel',
        'average_watts': 'Pot_media',
        'average_heartrate': 'Pulso_medio',
        'kilojoules':'Kilojulios'
    }, inplace=True)
    # Cambiar el nombre del índice
    df_total.index.name = 'Fecha'
    # Asegurarte de que el índice está en formato datetime
    df_total.index = pd.to_datetime(df_total.index, format='ISO8601', dayfirst=True)
    # Cambiar el formato del índice para que solo muestre día, mes y año
    df_total.index = df_total.index.strftime('%d-%m-%Y')

    # Cambios de unidades
    df_total['Distancia'] = df_total['Distancia'].apply(lambda x: x / 1000)
    df_total['Tiempo_horas'] = df_total['Tiempo_segundos'].apply(lambda x: x / 3600)
    #meter en una lista los tipos de actividad
    lista_tipos_de_actividad = df_total['Tipo_actividad'].to_list()
    lista_tipos_de_actividad = list(set(lista_tipos_de_actividad))
    #eliminar columnas no necesarias
    df_total.drop(columns=['Tiempo_segundos'],inplace=True)
    df_total_resumen = df_total.copy()
    df_total_resumen.drop(columns=['Pot_media', 'Pulso_medio'], inplace=True)

    return df_total_resumen, lista_tipos_de_actividad

def group_annually(df_total_resumen, lista_tipos_de_actividad):

    # Asegúrate de que la columna 'Fecha' esté en el formato datetime.
    # Si la columna de fecha es el índice, primero conviértela a datetime.
    df_total_resumen.index = pd.to_datetime(df_total_resumen.index, format='%d-%m-%Y', errors='raise')

    # Lista de tipos de actividad
    lista_tipos_actividad = ['Rowing', 'Walk', 'Hike', 'VirtualRide', 'Run', 'WeightTraining', 'Ride']

    # Paso 1: Filtrar el DataFrame por los tipos de actividad deseados
    df_filtered = df_total_resumen[df_total_resumen['Tipo_actividad'].isin(lista_tipos_actividad)]

    # Paso 2: Agrupar anualmente por 'Tipo_actividad' y sumar las columnas especificadas
    df_grouped_activity = df_filtered.groupby('Tipo_actividad').resample('YE').sum(numeric_only=True)

    # Paso 3: Mantener solo las columnas 'Distancia', 'Desnivel' y 'Tiempo_horas'
    df_grouped_activity = df_grouped_activity[['Distancia', 'Desnivel', 'Tiempo_horas']]

    # Paso 4: Cambiar el índice a solo el año
    df_grouped_activity.index = pd.MultiIndex.from_tuples(
        [(year.year, tipo) for (tipo, year) in df_grouped_activity.index],
        names=['Año', 'Tipo_actividad']
    )

    # Paso 1: Agrupar anualmente y sumar las columnas 'Distancia', 'Desnivel' y 'Tiempo_horas'.
    df_grouped_annual = df_total_resumen.resample('YE').sum(numeric_only=True)

    # Paso 2: Mantener solo las columnas 'Distancia', 'Desnivel' y 'Tiempo_horas'.
    df_grouped_annual = df_grouped_annual[['Distancia', 'Desnivel', 'Tiempo_horas']]

    # Paso 3: Cambiar el índice a solo el año.
    df_grouped_annual.index = df_grouped_annual.index.year
    df_grouped_annual['Tipo_actividad'] = 'Total'
    df_grouped_annual['Año'] = df_grouped_annual.index
    df_grouped_annual.reset_index(drop=True, inplace=True)
    columns_order = ['Año','Tipo_actividad', 'Distancia', 'Desnivel', 'Tiempo_horas']
    df_grouped_annual = df_grouped_annual[columns_order]
    
    return df_grouped_activity, df_grouped_annual


def split_dataframe_by_activity(df):
    """
    Esta función divide el DataFrame en varios DataFrames cuando el tipo de actividad cambia.
    :param df: DataFrame con columnas 'Distancia', 'Desnivel', 'Tiempo_horas', 'Año', 'Tipo_actividad'
    :return: Diccionario con los DataFrames divididos
    """
    # Restablecer el índice para asegurarnos de que 'Tipo_actividad' sea una columna
    df = df.reset_index()

    # Diccionario para almacenar los DataFrames separados
    activity_dfs = {}

    # Obtener la lista de actividades únicas
    actividades = df['Tipo_actividad'].unique()

    for actividad in actividades:
        # Filtrar el DataFrame por la actividad actual
        activity_df = df[df['Tipo_actividad'] == actividad]
        activity_dfs[actividad] = activity_df

    return activity_dfs

def df_total_finished(activity_dfs):
    
    #concatenar los dfs
    concatenated_df = pd.concat(activity_dfs.values(), axis=0, ignore_index=True)

    # Pivotar el DataFrame
    df_total_completo = concatenated_df.pivot_table(index='Año', columns='Tipo_actividad', values=['Distancia', 'Desnivel', 'Tiempo_horas'], fill_value=0)

    # Move 'Total' column to the end for each superindex
    new_columns = []
    for key in df_total_completo.columns.levels[0]:
        cols = list(df_total_completo[key].columns)
        cols.remove('Total')
        cols.append('Total')
        for col in cols:
            new_columns.append((key, col))
    df_total_completo = df_total_completo.reindex(columns=pd.MultiIndex.from_tuples(new_columns))
    df_total_completo = df_total_completo.round(2)
    
    return df_total_completo

def totals_table(df_total_completo):
    # Prepare table data with hierarchical column headers
    cell_text = []

    # Add the first level of columns (Category1, Category2)
    first_row = [''] + [col[0] for col in df_total_completo.columns]
    cell_text.append(first_row)

    # Add the second level of columns (Subcategory1, Subcategory2)
    second_row = ['Index'] + [col[1] for col in df_total_completo.columns]
    cell_text.append(second_row)

    # Add the main data rows, including the index as the first column
    for idx, row in zip(df_total_completo.index, df_total_completo.values):
        cell_text.append([idx] + list(row))

    # Define column labels (empty since we are manually adding rows for headers)
    col_labels = [''] * len(cell_text[0])

    # Plot the table
    fig, ax = plt.subplots(figsize=(28, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')

    # Set the font size for the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Save the plot as an image file
    os.makedirs("outputs/totales", exist_ok=True)  # Ensure the outputs directory exists
    plt.savefig("outputs/totales/table.png", dpi=300, bbox_inches='tight')
    plt.show()

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

    # make a df
    df_total, lista_tipos_de_actividad = total_df(all_activities_summary)

    # df grouped by year
    df_grouped_activities, df_grouped_total = group_annually(df_total, lista_tipos_de_actividad)

    # Split the DataFrame by activity type
    activity_dfs = split_dataframe_by_activity(df_grouped_activities)
    activity_dfs['Total'] = df_grouped_total

    # Create the final DataFrame with all activities
    df_total_finish = df_total_finished(activity_dfs)

    # Create and save the totals table
    totals_table(df_total_finish)
