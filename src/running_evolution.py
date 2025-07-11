import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from request_data import activities_summary_data
from count_settings import strava_api_settings

def running_df(activities_summary):
    # Filtrar solo las actividades de tipo 'Run'
    run_activities = [activity for activity in activities_summary if activity['type'] == 'Run']

    # Crear un DataFrame con los datos solicitados
    df_total_run = pd.DataFrame(run_activities, columns=['start_date', 'distance', 'moving_time', 'total_elevation_gain', 'average_watts', 'average_heartrate','kilojoules'])

    # Establecer 'start_date' como índice
    df_total_run.set_index('start_date', inplace=True)

    # Cambiar los nombres de algunas columnas
    df_total_run.rename(columns={
        'distance': 'Distancia',
        'moving_time': 'Tiempo_segundos',
        'total_elevation_gain': 'Desnivel',
        'average_watts': 'Pot_media',
        'average_heartrate': 'Pulso_medio',
        'kilojoules':'Kilojulios'
    }, inplace=True)
    # Cambiar el nombre del índice
    df_total_run.index.name = 'Fecha'
    # Asegurarte de que el índice está en formato datetime
    df_total_run.index = pd.to_datetime(df_total_run.index)

    # Cambiar el formato del índice para que solo muestre día, mes y año
    df_total_run.index = df_total_run.index.strftime('%d-%m-%Y')
    # Cambios de unidades
    df_total_run['Distancia'] = df_total_run['Distancia'].apply(lambda x: x / 1000)
    df_total_run['Tiempo_minutos'] = df_total_run['Tiempo_segundos'].apply(lambda x: x / 60)
    #añadir columna de ritmo (min/km) y ratio de desnivel (m+/km)
    df_total_run['Ritmo']=df_total_run['Tiempo_minutos']/df_total_run['Distancia']
    df_total_run['Ratio_Desn']=df_total_run['Desnivel']/df_total_run['Distancia']

    return df_total_run

def df_pulso_potencia_comp(df_total_run):
    #solo usando datos de potencia
    df_potencias = df_total_run.dropna(subset=['Pot_media'])
    #ritmos entre 5min/km y 10min/km
    df_potencias = df_potencias[(df_potencias['Ritmo']>=5)& (df_potencias['Ritmo']<=10)]

    # Diccionario con dataframes por pulso
    datos = {}
    lista_ratio_desnivel = [0, 40, 60, 80, 100, 200]

    # Crear los dataframes filtrados
    for j in range(len(lista_ratio_desnivel) - 1):
        datos[f'ratio: {lista_ratio_desnivel[j+1]}m+/km'] = {}
        for i in range(105, 205, 10):
            df_temp = df_potencias[
                (df_potencias['Pulso_medio'] >= i) &
                (df_potencias['Pulso_medio'] < i + 10) &
                (df_potencias['Ratio_Desn'] >= lista_ratio_desnivel[j]) &
                (df_potencias['Ratio_Desn'] < lista_ratio_desnivel[j+1])
            ]

            # Solo agregar si no está vacío
            if not df_temp.empty:
                datos[f'ratio: {lista_ratio_desnivel[j+1]}m+/km'][f'df_ppm_max{i+10}'] = df_temp

    # Eliminar claves con DataFrames vacíos
    for ratio_key in list(datos.keys()):
        sub_dict = datos[ratio_key]
        empty_keys = [key for key, df in sub_dict.items() if df.empty]  # Lista de claves con DataFrames vacíos
        for key in empty_keys:
            del sub_dict[key]  # Eliminar DataFrame vacío
        if not sub_dict:  # Si el sub-diccionario está vacío después de eliminar
            del datos[ratio_key]  # Eliminar la clave de ratio
    
    # Diccionario para guardar los resultados del groupby
    groupby_quarterly_dict = {}

    # Iterar sobre los DataFrames en el diccionario
    for key, grupos in datos.items():
        groupby_quarterly_dict[key] = {}
        for group, df in grupos.items():
            # Asegurarte de que estás trabajando sobre una copia explícita del DataFrame
            df = df.copy()  # Hacemos una copia para evitar warnings

            # Verificar el tipo de índice y convertirlo si es necesario
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index, dayfirst=True)
                except Exception as e:
                    continue  # Omitir este grupo si no se puede convertir

            # Verificar que el índice se haya convertido correctamente
            if pd.api.types.is_datetime64_any_dtype(df.index):
                # Crear una nueva columna para el trimestre
                df['Trimestre'] = df.index.to_period('Q').start_time

                # Agrupar por la columna 'Trimestre' y calcular la media (incluyendo las columnas numéricas)
                quarterly_mean = df.groupby('Trimestre').mean(numeric_only=True)

                # Guardar el DataFrame resultante en el diccionario
                groupby_quarterly_dict[key][group] = quarterly_mean
            else:
                print(f"Advertencia: El índice del DataFrame de {group} en {key} sigue sin ser de tipo datetime. Se omitirá este grupo.")
    return groupby_quarterly_dict

def comp_tipico_rodaje_suave(df_total_run):
    # Rango de distancia y ganancia de elevación
    distance_range = 0.5  # Distancia en km
    elevation_range = 80   # Elevación en metros

    # Almacenar las actividades similares
    similar_activities = []
    activity_ids = set()  # Usar un conjunto para rastrear actividades únicas

    # Comparar cada actividad con las demás
    for i, row in df_total_run.iterrows():
        for j, other_row in df_total_run.iterrows():
            if i != j:  # No comparar la actividad consigo misma
                distance_diff = abs(row['Distancia'] - other_row['Distancia'])
                elevation_diff = abs(row['Desnivel'] - other_row['Desnivel'])
                if distance_diff <= distance_range and elevation_diff <= elevation_range:
                    # Usar el índice de la actividad como un identificador único
                    activity_id = (row['Distancia'], row['Desnivel'], row['Pot_media'], row['Pulso_medio'])

                    if activity_id not in activity_ids:
                        activity_ids.add(activity_id)  # Agregar el identificador al conjunto
                        # Añadir la actividad como un diccionario y la fecha
                        similar_activities.append(row.to_dict())
                        similar_activities[-1]['Fecha'] = row.name  # Añadir la fecha desde el índice

    # Crear un DataFrame con las actividades similares
    similar_df = pd.DataFrame(similar_activities)

    # Establecer la columna 'Fecha' como índice
    similar_df.set_index('Fecha', inplace=True)

    # Agrupar las actividades por distancia y desnivel, contando el número de actividades
    grouped = similar_df.groupby(['Distancia', 'Desnivel']).size().reset_index(name='count')

    # Encontrar el grupo con el mayor número de actividades
    max_group = grouped.loc[grouped['count'].idxmax()]

    # Filtrar el DataFrame de actividades similares que estén dentro del rango del grupo más común
    final_similar_activities = similar_df[
        (abs(similar_df['Distancia'] - max_group['Distancia']) <= distance_range) &
        (abs(similar_df['Desnivel'] - max_group['Desnivel']) <= elevation_range)
    ]

    # Eliminar duplicados en el DataFrame final
    df_easy = final_similar_activities.drop_duplicates()

    # Diccionario con DataFrames por pulso
    datos_easy = {}
    for i in range(105, 205, 10):
        # Filtrar DataFrame basado en el pulso medio
        filtered_df = df_easy[(df_easy['Pulso_medio'] > i) & (df_easy['Pulso_medio'] < i + 10)]

        # Solo añadir al diccionario si el DataFrame no está vacío
        if not filtered_df.empty:
            datos_easy[f'df_easy{i+10}_ppm_max'] = filtered_df
    
    # Diccionario para guardar los resultados del groupby
    groupby_quarterly_dict_easy = {}

    # Iterar sobre los DataFrames en el diccionario
    for key, df in datos_easy.items():
        # Asegurarte de que estás trabajando sobre una copia explícita del DataFrame
        df = df.copy()  # Hacemos una copia para evitar warnings

        # Convertir el índice 'Fecha' a tipo datetime si no lo es ya
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, dayfirst=True)

        # Crear una nueva columna para el trimestre como un objeto datetime
        df['Trimestre'] = df.index.to_period('Q').start_time  # Usar el índice en lugar de la columna

        # Agrupar por la columna 'Trimestre' y calcular la media (incluyendo las columnas numéricas)
        quarterly_mean_easy = df.groupby('Trimestre').mean(numeric_only=True)

        # Guardar el DataFrame resultante en el diccionario
        groupby_quarterly_dict_easy[key] = quarterly_mean_easy
    
    return groupby_quarterly_dict_easy
def plot_multiple_potencia_pulso(dfs_dict):
    """
    Esta función recibe un diccionario de DataFrames, donde cada DataFrame tiene columnas
    'Pot_media' y 'Pulso_medio', ya indexados por 'Fecha'. Genera una figura con múltiples gráficos
    de líneas (uno para cada DataFrame), mostrando la Potencia Media y el Pulso Medio
    a lo largo del tiempo en dos ejes diferentes (doble escala), con marcadores circulares,
    los valores en cada punto y la línea de tendencia de la potencia.

    Además, añade una cuadrícula negra y fondo gris para cada gráfico.

    :param dfs_dict: Diccionario con los nombres como clave y DataFrames como valor.
    """

    # Filtrar y almacenar los DataFrames que tienen columnas de interés
    filtered_dfs = {}
    for key_ratio, dfs in dfs_dict.items():
        filtered_dfs[key_ratio] = {}
        for name, df in dfs.items():
            pot_valid_count = df['Pot_media'].notna().sum()
            pulso_valid_count = df['Pulso_medio'].notna().sum()

            # Guardar solo DataFrames con al menos 5 datos válidos para ambas variables
            if pot_valid_count >= 5 and pulso_valid_count >= 5:
                filtered_dfs[key_ratio][name] = (df, key_ratio)

    # Número total de DataFrames mantenidos
    total_dfs_per_ratio = {key_ratio: len(dfs) for key_ratio, dfs in filtered_dfs.items()}
    num_ratios = len(total_dfs_per_ratio)
    total_dfs = sum(total_dfs_per_ratio.values())

    # Comprobar si hay DataFrames válidos para graficar
    if total_dfs == 0:
        print("No hay suficientes datos válidos para graficar.")
        return

    # Crear una figura con múltiples subplots (uno por DataFrame), organizados en columnas por key_ratio
    fig, axes = plt.subplots(nrows=max(total_dfs_per_ratio.values()), ncols=num_ratios, figsize=(60, 30))  # Ancho 60

    # Aplanar el array de ejes si es necesario
    axes = np.array(axes).reshape(max(total_dfs_per_ratio.values()), num_ratios)

    # Inicializar un contador para la fila actual
    current_row = 0

    # Iterar sobre los DataFrames filtrados y los ejes (uno por gráfico)
    for col, (key_ratio, dfs) in enumerate(filtered_dfs.items()):
        current_row = 0  # Reset row counter for each ratio
        m=0
        for name, (df, _) in dfs.items():
            ax = axes[current_row, col]  # Obtener el eje actual
            ax2 = ax.twinx()  # Crear un segundo eje Y (doble escala) para el pulso

            # Graficar Pot_media con líneas, marcador circular, y color azul en el eje original
            ax.plot(df.index.astype(str), df['Pot_media'], marker='o', label='Potencia Media', color='b')
            ax.set_ylabel('Potencia Media', color='b')  # Etiqueta del eje Y para Potencia

            # Añadir etiquetas de los valores en cada punto para Pot_media si son finitos
            for i, val in enumerate(df['Pot_media']):
                if np.isfinite(val):  # Verificar si el valor es finito
                    ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', color='blue')

            # Calcular la línea de tendencia para Pot_media (ajuste lineal)
            valid_indices = df['Pot_media'].notna()  # Máscara booleana para índices válidos
            valid_potencia = df['Pot_media'][valid_indices]  # Filtrar valores válidos

            if len(valid_potencia) > 1:  # Asegurarse de tener suficientes datos
                x_vals = np.arange(len(df.index))  # Crear un array de índices
                x_valid = x_vals[valid_indices]  # Obtener índices válidos
                coef = np.polyfit(x_valid, valid_potencia, 1)  # Ajuste lineal
                poly = np.poly1d(coef)
                trendline = poly(x_vals)  # Calcular la línea de tendencia para todos los índices

                # Dibujar la línea de tendencia en negro
                ax.plot(df.index.astype(str), trendline, color='black', linestyle='--', label='Tendencia Potencia')

            # Graficar Pulso_medio con líneas, marcador circular, y color rojo en el segundo eje Y
            ax2.plot(df.index.astype(str), df['Pulso_medio'], marker='o', label='Pulso Medio', color='r')
            ax2.set_ylabel('Pulso Medio', color='r')  # Etiqueta del eje Y para Pulso

            # Añadir etiquetas de los valores en cada punto para Pulso_medio si son finitos
            for i, val in enumerate(df['Pulso_medio']):
                if np.isfinite(val):  # Verificar si el valor es finito
                    ax2.text(i, val, f'{val:.1f}', ha='center', va='bottom', color='red')

            # Configurar la cuadrícula de color negro solo en el eje original
            ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='blue')

            # Establecer el color de fondo del gráfico en gris para el eje original
            ax.set_facecolor('lightgray')

            # Añadir etiquetas y título
            ax.set_xlabel('Fecha')
            if m == 0:
                ax.set_title(f'{key_ratio} máximo', fontsize=20, fontweight='bold',pad=30)
                titulo = f'Potencia Media y Pulso Medio a lo largo del tiempo ({name})'
                ax.text(0.5, 1.02, titulo,transform=ax.transAxes, fontsize=12, ha='center' )
            else:
                ax.set_title(f'Potencia Media y Pulso Medio a lo largo del tiempo ({name})')
            m=i+1
            # Ajustar la rotación de las etiquetas de fecha en el eje X
            ax.tick_params(axis='x', rotation=45)

            # Incrementar el índice de la fila para el siguiente DataFrame
            current_row += 1

        # Hacer invisibles las filas que no tienen datos
        for row in range(current_row, max(total_dfs_per_ratio.values())):
            axes[row, col].set_visible(False  )  # Ocultar subplots vacíos

    # Agregar un título a la figura completa y moverlo hacia arriba
    plt.suptitle('Relación Pulso/Potencia en distintos ratios de desnivel', fontsize=40, fontweight='bold')  # Ajustar y para mover el título hacia arriba

    # Ajustar los espacios entre subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el rectángulo para el título

    # Guardar el gráfico como archivo JPG
    os.makedirs("outputs/running_evolution", exist_ok=True)  # Ensure the outputs directory exists
    plt.savefig(f"outputs/running_evolution/Relaciones_Potencia_Pulso.png", dpi=100)
    plt.show()

def plot_comp_rodaje_tipico(dfs_dict):
    """
    Esta función recibe un diccionario de DataFrames, donde cada DataFrame tiene columnas
    'Pot_media', 'Pulso_medio' y 'Tiempo_minutos', ya indexados por 'Fecha'. Genera una figura
    con múltiples gráficos de líneas (una para cada DataFrame), mostrando la Potencia Media,
    el Pulso Medio y el Tiempo a lo largo del tiempo en dos ejes diferentes (doble escala),
    con marcadores circulares, los valores en cada punto y la línea de tendencia del tiempo.

    Además, añade una cuadrícula negra y fondo gris para cada gráfico.

    :param dfs_dict: Diccionario con los nombres como clave y DataFrames como valor.
    """

    # Filtrar los DataFrames que tienen al menos 5 datos válidos para ambas variables
    filtered_dfs = {
        name: df for name, df in dfs_dict.items()
        if df['Pot_media'].notna().sum() >= 5 and df['Pulso_medio'].notna().sum() >= 5 and df['Tiempo_minutos'].notna().sum() >= 5
    }

    # Número de DataFrames filtrados
    num_dfs = len(filtered_dfs)

    # Comprobar si hay DataFrames válidos para graficar
    if num_dfs == 0:
        print("No hay suficientes datos válidos para graficar.")
        return

    # Crear una figura con múltiples subplots (una fila por cada DataFrame)
    fig, axes = plt.subplots(nrows=num_dfs, ncols=1, figsize=(10, 18))

    # Si hay solo un DataFrame, convertir 'axes' a lista para que el código sea consistente
    if num_dfs == 1:
        axes = [axes]

    # Iterar sobre los DataFrames filtrados y los ejes (uno por gráfico)
    for ax, (name, df) in zip(axes, filtered_dfs.items()):
        # Crear un segundo eje Y (doble escala) para el pulso
        ax2 = ax.twinx()

        # Graficar Pot_media con líneas, marcador circular, y color azul en el eje original
        ax.plot(df.index.astype(str), df['Pot_media'], marker='o', label='Potencia Media', color='b')
        ax.set_ylabel('Potencia Media', color='b')  # Etiqueta del eje Y para Potencia

        # Añadir etiquetas de los valores en cada punto para Pot_media si son finitos
        for i, val in enumerate(df['Pot_media']):
            if np.isfinite(val):  # Verificar si el valor es finito
                ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', color='blue')

        # Calcular la línea de tendencia para Pot_media (ajuste lineal)
        valid_indices = df['Pot_media'].notna()  # Máscara booleana para índices válidos
        valid_potencia = df['Pot_media'][valid_indices]  # Filtrar valores válidos

        if len(valid_potencia) > 1:  # Asegurarse de tener suficientes datos
            x_vals = np.arange(len(df.index))  # Crear un array de índices
            x_valid = x_vals[valid_indices]  # Obtener índices válidos
            coef = np.polyfit(x_valid, valid_potencia, 1)  # Ajuste lineal
            poly = np.poly1d(coef)
            trendline = poly(x_vals)  # Calcular la línea de tendencia para todos los índices

            # Dibujar la línea de tendencia en negro
            ax.plot(df.index.astype(str), trendline, color='blue', linestyle='--', label='Tendencia Potencia')

        # Graficar Pulso_medio con líneas, marcador circular, y color rojo en el segundo eje Y
        ax2.plot(df.index.astype(str), df['Pulso_medio'], marker='o', label='Pulso Medio', color='r')
        ax2.set_ylabel('Pulso Medio', color='r')  # Etiqueta del eje Y para Pulso

        # Añadir etiquetas de los valores en cada punto para Pulso_medio si son finitos
        for i, val in enumerate(df['Pulso_medio']):
            if np.isfinite(val):  # Verificar si el valor es finito
                ax2.text(i, val, f'{val:.1f}', ha='center', va='bottom', color='red')

        # Graficar Tiempo_minutos con líneas y color azul en el segundo eje Y
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Mover el segundo eje Y un poco a la derecha
        ax3.plot(df.index.astype(str), df['Tiempo_minutos'], marker='o', label='Tiempo (min)', color='orange', alpha=0.7)
        ax3.set_ylabel('Tiempo (min)', color='orange')  # Etiqueta del eje Y para Tiempo

        # Añadir etiquetas de los valores en cada punto para Tiempo si son finitos
        for i, val in enumerate(df['Tiempo_minutos']):
            if np.isfinite(val):  # Verificar si el valor es finito
                ax3.text(i, val, f'{val:.1f}', ha='center', va='bottom', color='orange')

        # Calcular la línea de tendencia para Tiempo_minutos (ajuste lineal)
        valid_time_indices = df['Tiempo_minutos'].notna()  # Máscara booleana para índices válidos
        valid_time = df['Tiempo_minutos'][valid_time_indices]  # Filtrar valores válidos

        if len(valid_time) > 1:  # Asegurarse de tener suficientes datos
            x_valid_time = x_vals[valid_time_indices]  # Obtener índices válidos
            coef_time = np.polyfit(x_valid_time, valid_time, 1)  # Ajuste lineal
            poly_time = np.poly1d(coef_time)
            trendline_time = poly_time(x_vals)  # Calcular la línea de tendencia para todos los índices

            # Dibujar la línea de tendencia en negro
            ax3.plot(df.index.astype(str), trendline_time, color='orange', linestyle='--', label='Tendencia Tiempo')

        # Configurar la cuadrícula de color negro solo en el eje original
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='black')

        # Establecer el color de fondo del gráfico en gris para el eje original
        ax.set_facecolor('lightgray')

        # Añadir etiquetas y título
        ax.set_xlabel('Fecha')
        ax.set_title(f'Potencia Media, Pulso Medio y Tiempo a lo largo del tiempo ({name})')

        # Ajustar la rotación de las etiquetas de fecha en el eje X
        ax.tick_params(axis='x', rotation=45)

    # Ajustar los espacios entre subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Dejar espacio suficiente para el título
    # Agregar un título a la figura completa y moverlo hacia arriba
    plt.suptitle('Comparación tiempo/pulso/potencia recorrido típico', fontsize=20, fontweight='bold')  # Ajustar y para mover el título hacia arriba
    # Guardar el gráfico como archivo JPG
    plt.savefig(f"outputs/running_evolution/Comparacion_Rodaje_Tipico.png", dpi=100)
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

    # make a df for running activities
    df_running = running_df(all_activities_summary)

    # dict grouped by quarter year
    groupby_quarterly_dict = df_pulso_potencia_comp(df_running)

    # Compare typical easy run activities
    groupby_quarterly_dict_easy = comp_tipico_rodaje_suave(df_running)
    # make plots
    plot_multiple_potencia_pulso(groupby_quarterly_dict)
    plot_comp_rodaje_tipico(groupby_quarterly_dict_easy)
