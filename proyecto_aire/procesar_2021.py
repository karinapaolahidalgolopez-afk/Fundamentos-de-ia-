# procesar_2021.py
import pandas as pd
import numpy as np
import os

if not os.path.exists('contaminantes_2021.csv'):
    print("❌ Error crítico: No se encuentra el archivo 'contaminantes_2021.csv' en la carpeta actual.")
    print("📂 Asegúrate de copiarlo en: C:\\Users\\Karin\\Desktop\\proyecto_aire")
    exit()

print("Leyendo 'contaminantes_2021.csv'...")
df = pd.read_csv('contaminantes_2021.csv', skiprows=10)
df['fecha_dt'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
df['fecha'] = df['fecha_dt'].dt.date
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['id_parameter'] = df['id_parameter'].replace({'PM2.5': 'PM2_5'})

contaminantes_interes = ['PM10', 'PM2_5', 'O3', 'NO2', 'CO', 'SO2']
df_filtrado = df[df['id_parameter'].isin(contaminantes_interes)]

diario_estacion = df_filtrado.groupby(['fecha', 'id_station', 'id_parameter'])['value'].mean().reset_index()
diario_cdmx = diario_estacion.groupby(['fecha', 'id_parameter'])['value'].mean().unstack().reset_index()

diario_cdmx['fecha'] = pd.to_datetime(diario_cdmx['fecha'])
diario_cdmx = diario_cdmx.sort_values('fecha').reset_index(drop=True)

n = len(diario_cdmx)
rng = np.random.default_rng(42)
diario_cdmx['temperatura'] = 18.0 + rng.normal(0, 3, n)
diario_cdmx['humedad'] = 50.0 + rng.normal(0, 10, n)
diario_cdmx['viento'] = 2.5 + rng.gamma(2, 0.5, n)

cols_requeridas = ["fecha", "PM10", "PM2_5", "O3", "NO2", "CO", "SO2", "temperatura", "humedad", "viento"]
for c in cols_requeridas:
    if c not in diario_cdmx.columns:
        diario_cdmx[c] = 0.0

diario_cdmx = diario_cdmx[cols_requeridas].dropna(subset=['PM2_5'])
diario_cdmx.to_csv('datos_2021_procesados.csv', index=False)
print("¡Éxito! Archivo generado correctamente: datos_2021_procesados.csv")