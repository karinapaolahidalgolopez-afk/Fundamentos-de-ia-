# datos_y_tablas.py
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             accuracy_score, classification_report)

COLUMNAS = ["fecha", "PM10", "PM2_5", "O3", "NO2", "CO", "SO2",
            "temperatura", "humedad", "viento"]

LIMITES = [0, 25, 45, 79, 147, np.inf]
ETIQUETAS = ["Buena", "Aceptable", "Mala", "Muy mala", "Extremadamente mala"]

def generar_datos_sinteticos(n_dias=1500, semilla=42):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2020-01-01", periods=n_dias, freq="D")
    mes = fechas.month.values
    estacional = 1 + 0.35 * np.cos((mes - 2) / 12 * 2 * np.pi)
    temp = 17 + 5 * np.sin((mes - 3) / 12 * 2 * np.pi) + rng.normal(0, 2, n_dias)
    hum = np.clip(55 - 20 * estacional + rng.normal(0, 8, n_dias), 10, 95)
    viento = np.clip(rng.gamma(3, 0.8, n_dias), 0.2, 10)
    pm10 = np.clip(60 * estacional - 4 * viento + rng.normal(0, 10, n_dias), 10, None)
    no2 = np.clip(35 * estacional - 2 * viento + rng.normal(0, 6, n_dias), 5, None)
    co = np.clip(0.9 * estacional - 0.05 * viento + rng.normal(0, 0.15, n_dias), 0.1, None)
    so2 = np.clip(8 + rng.normal(0, 3, n_dias), 1, None)
    o3 = np.clip(40 + 2.5 * (temp - 17) - 1.5 * viento + rng.normal(0, 8, n_dias), 5, None)
    pm25 = np.clip(0.45 * pm10 + 0.25 * no2 + 6 * co - 1.5 * viento
                   + rng.normal(0, 4, n_dias), 3, None)
    return pd.DataFrame({"fecha": fechas, "PM10": pm10, "PM2_5": pm25, "O3": o3,
                         "NO2": no2, "CO": co, "SO2": so2, "temperatura": temp,
                         "humedad": hum, "viento": viento})

def cargar_datos(ruta=None):
    if ruta:
        df = pd.read_csv(ruta, parse_dates=["fecha"])
        faltan = set(COLUMNAS) - set(df.columns)
        if faltan:
            raise ValueError(f"Faltan columnas en el CSV: {faltan}")
        return df.sort_values("fecha").reset_index(drop=True)
    return generar_datos_sinteticos()

def construir_tablas(df):
    df = df.copy()
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["PM2_5_ayer"] = df["PM2_5"]
    consecutivo = (df["fecha"].shift(-1) - df["fecha"]).dt.days == 1
    df["y_PM2_5_manana"] = df["PM2_5"].shift(-1).where(consecutivo)
    df = df.dropna().reset_index(drop=True)
    df["y_categoria"] = pd.cut(df["y_PM2_5_manana"], bins=LIMITES,
                               labels=ETIQUETAS, right=True)
    entradas = ["PM10", "O3", "NO2", "CO", "SO2", "temperatura", "humedad",
                "viento", "mes", "dia_semana", "PM2_5_ayer"]
    X = df[entradas]
    y_reg = df["y_PM2_5_manana"]
    y_clf = df["y_categoria"].astype(str)
    return df, X, y_reg, y_clf