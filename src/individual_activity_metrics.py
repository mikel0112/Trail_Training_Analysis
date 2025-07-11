import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from request_data import activities_summary_data
from count_settings import strava_api_settings
from yearly_intensity_distribution import dataframes_dict, power_curve

def metrics_dict(diccionario_dataframes):
    #guardamos todas las actividades con potencia y pulso en un diccionario
    # y teniendo en cuenta que el ritmo medio de la actividad sea inferior a 9min/km
    #ademas si hay varias actividades en la misma fecha solo dejamos la mas larga
    diccionario_calculo_metricas = {}
    lista_fechas = []
    for key in diccionario_dataframes:
        if 'watts' in diccionario_dataframes[key]['datos'].columns and\
            'heartrate' in diccionario_dataframes[key]['datos'].columns:
            df = diccionario_dataframes[key]['datos']
            df.replace(np.inf, 0, inplace=True)
            ritmo_medio = df['ritmo'].mean()
            if ritmo_medio < 9:
                diccionario_calculo_metricas[key] = diccionario_dataframes[key]

    # Diccionario para almacenar la actividad más larga por fecha
    actividades_por_fecha = {}

    for key in diccionario_calculo_metricas:
        actividad = diccionario_calculo_metricas[key]
        fecha = actividad['fecha']
        datos = actividad['datos']

        # Si la fecha ya existe en el diccionario de actividades
        if fecha in actividades_por_fecha:
            # Comparar la longitud de la actividad actual con la almacenada
            if len(datos) > len(actividades_por_fecha[fecha]['datos']):
                actividades_por_fecha[fecha] = actividad
        else:
            actividades_por_fecha[fecha] = actividad

    diccionario_calculo_metricas = actividades_por_fecha

    return diccionario_calculo_metricas

def metrics_dataframe(diccionario_calculo_metricas):
    #dataframe con todas las fechas
    lista_fechas = []
    for key in diccionario_calculo_metricas:
        if diccionario_calculo_metricas[key]['fecha'] not in lista_fechas:
            lista_fechas.append(diccionario_calculo_metricas[key]['fecha'])
    df_metricas = pd.DataFrame(index=lista_fechas)

    return df_metricas

class ActivityMetrics:
    def __init__(self, diccionario_calculo_metricas, df_metricas, diccionario_dataframes):
        self.diccionario_calculo_metricas = diccionario_calculo_metricas
        self.df_metricas = df_metricas
        self.diccionario_dataframes = diccionario_dataframes
    
    def Aerobic_Decoupling(self):
        for key in self.diccionario_calculo_metricas:
            df = self.diccionario_calculo_metricas[key]['datos']
            df.replace(np.inf, 0, inplace=True)
            mitad = len(df)//2
            primera_mitad = df.iloc[:mitad]
            segunda_mitad = df.iloc[mitad:]
            pri_mit_ritmo = primera_mitad['ritmo'].mean()
            seg_mit_ritmo = segunda_mitad['ritmo'].mean()
            pri_mit_pulso = primera_mitad['heartrate'].mean()
            seg_mit_pulso = segunda_mitad['heartrate'].mean()
            pri_mit_potencia = primera_mitad['watts'].mean()
            seg_mit_potencia = segunda_mitad['watts'].mean()
            fecha = self.diccionario_calculo_metricas[key]['fecha']
            self.df_metricas.loc[self.diccionario_calculo_metricas[key]['fecha'], 'Pace Aerobic Decoupling'] = (pri_mit_ritmo/pri_mit_pulso)/(seg_mit_ritmo/seg_mit_pulso)
            self.df_metricas.loc[self.diccionario_calculo_metricas[key]['fecha'], 'Power Aerobic Decoupling'] = (pri_mit_potencia/pri_mit_pulso)/(seg_mit_potencia/seg_mit_pulso)

        return self
    
    def NP_Normalized_Power(self):
        for key in diccionario_calculo_metricas:
            df = self.diccionario_calculo_metricas[key]['datos']
            df.replace(np.inf, 0, inplace=True)
            # Paso 1: Calcular la media móvil de 30 segundos de la potencia
            rolling_mean_30s = df['watts'].rolling(window=30, min_periods=1).mean()
            # Paso 2: Elevar cada valor de la media móvil a la cuarta potencia
            power_fourth = rolling_mean_30s ** 4

            # Paso 3: Calcular el promedio de estos valores elevados
            average_power_fourth = power_fourth.mean()

            # Paso 4: Tomar la cuarta raíz del promedio obtenido
            normalized_power = average_power_fourth ** 0.25
            self.df_metricas.loc[self.diccionario_calculo_metricas[key]['fecha'], 'NP'] = normalized_power

        return self
    
    def Efficiency_Factor(self):
        for key in diccionario_calculo_metricas:
            df = self.diccionario_calculo_metricas[key]['datos']
            df.replace(np.inf, 0, inplace=True)
            self.df_metricas.loc[self.diccionario_calculo_metricas[key]['fecha'], 'Average Heart Rate'] = df['heartrate'].mean()
            self.df_metricas['Efficiency Factor'] = self.df_metricas['NP']/self.df_metricas['Average Heart Rate']

        return self
    
    def plot_efficiency_factor(self):
        # Filtrar los datos para los últimos tres años desde la fecha actual
        fecha_actual = pd.to_datetime('today')
        tres_anos_atras = fecha_actual - pd.DateOffset(years=3)

        # Asegurarse de que el índice es datetime
        self.df_metricas.index = pd.to_datetime(self.df_metricas.index)

        # Filtrar el DataFrame para obtener los últimos tres años
        df_filtrado = self.df_metricas[self.df_metricas.index >= tres_anos_atras]

        # Crear la gráfica de línea
        plt.figure(figsize=(10, 6))
        plt.plot(df_filtrado.index, df_filtrado['Efficiency Factor'], marker='o', color='b', linestyle='-', linewidth=2, markersize=6)

        # Calcular la línea de tendencia (regresión lineal)
        # Convertir las fechas en números para hacer la regresión
        # Usar la diferencia en días
        fechas_numericas = (df_filtrado.index - df_filtrado.index.min()).days

        # Realizar la regresión lineal
        coeficientes = np.polyfit(fechas_numericas, df_filtrado['Efficiency Factor'], 1)
        polinomio = np.poly1d(coeficientes)

        # Crear los valores ajustados para la línea de tendencia
        linea_tendencia = polinomio(fechas_numericas)

        # Agregar la línea de tendencia a la gráfica
        plt.plot(df_filtrado.index, linea_tendencia, color='r', linestyle='--', linewidth=2, label='Línea de Tendencia')

        # Etiquetas y título
        plt.title('Efficiency Factor en los Últimos 3 Años con Línea de Tendencia', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Efficiency Factor', fontsize=12)

        # Rotar las fechas en el eje X para mayor legibilidad
        plt.xticks(rotation=45)

        # Mostrar la leyenda
        plt.legend()
        # Mostrar la gráfica
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("outputs/running_evolution/eficiencia.png")
        plt.show()

    def IF_Intensity_Factor(self):
        combined_df, CP = power_curve(self.diccionario_dataframes)
        self.df_metricas['IF'] = self.df_metricas['NP']/CP
        return self
    
    def VI_Variability_Index(self):
        for key in self.diccionario_calculo_metricas:
            df = self.diccionario_calculo_metricas[key]['datos']
            df.replace(np.inf, 0, inplace=True)
            self.df_metricas.loc[self.diccionario_calculo_metricas[key]['fecha'], 'Average Power'] = df['watts'].mean()
            self.df_metricas['Variability Index'] = self.df_metricas['NP']/self.df_metricas['Average Power']
            #eliminar la columna de average power
            self.df_metricas.drop(columns=['Average Power'], inplace=True)

        return self
    

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

    # Create a dictionary of metrics
    diccionario_calculo_metricas = metrics_dict(dfs_dict)

    # Create a DataFrame for metrics
    df_metricas = metrics_dataframe(diccionario_calculo_metricas)

    # Create an instance of ActivityMetrics
    activity_metrics = ActivityMetrics(diccionario_calculo_metricas, df_metricas, dfs_dict)

    # Calculate metrics
    activity_metrics.Aerobic_Decoupling()
    activity_metrics.NP_Normalized_Power()
    activity_metrics.Efficiency_Factor()
    activity_metrics.IF_Intensity_Factor()
    activity_metrics.VI_Variability_Index()
    activity_metrics.plot_efficiency_factor()