# app_streamlit.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calidad del aire CDMX", page_icon="🌫️")
st.title("🌫️ Predicción de calidad del aire - CDMX")
st.caption("Clasifica la categoría de PM2.5 del día siguiente a partir de las mediciones de hoy.")

try:
    pkg = joblib.load("modelo.joblib")
except FileNotFoundError:
    st.error("No se encontró modelo.joblib. Ejecuta primero: python entrenar_modelos.py")
    st.stop()

st.sidebar.write(f"**Modelo activo:** {pkg['nombre']}")
etiquetas = {"PM10": "PM10 (µg/m³)", "O3": "Ozono (ppb)", "NO2": "NO2 (ppb)", "CO": "CO (ppm)",
             "SO2": "SO2 (ppb)", "temperatura": "Temperatura (°C)", "humedad": "Humedad (%)",
             "viento": "Viento (m/s)", "mes": "Mes", "dia_semana": "Día de la semana (0=lunes)",
             "PM2_5_ayer": "PM2.5 de hoy (µg/m³)"}

valores = {}
cols = st.columns(2)
for i, c in enumerate(pkg["columnas"]):
    lo, hi, med = (float(pkg["rangos"][c][k]) for k in ("min", "max", "50%"))
    with cols[i % 2]:
        if c in ("mes", "dia_semana"):
            valores[c] = st.number_input(etiquetas.get(c, c), int(lo), int(hi), int(med))
        else:
            valores[c] = st.slider(etiquetas.get(c, c), lo * 0.5, hi * 1.5, med)

if st.button("Predecir calidad del aire", type="primary"):
    fila = pd.DataFrame([valores])[pkg["columnas"]]
    prediccion = pkg["modelo"].predict(fila)[0]
    st.subheader(f"Categoría prevista para mañana: {prediccion}")
    if hasattr(pkg["modelo"], "predict_proba"):
        prob = pd.Series(pkg["modelo"].predict_proba(fila)[0], index=pkg["clases"])
        st.bar_chart(prob)