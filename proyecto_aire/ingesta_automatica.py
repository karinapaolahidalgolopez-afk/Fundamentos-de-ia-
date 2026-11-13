"""
Módulo de ingesta automática de datos en tiempo real (AQICN / Open-Meteo).
"""
import urllib.request, json

def obtener_aire(token):
    if not token: return {}
    endpoints = ["mexico", "ciudad de mexico", "mexico/merced"]
    for ep in endpoints:
        try:
            url = f"https://api.waqi.info/feed/{ep}/?token={token}"
            req = urllib.request.urlopen(url, timeout=5)
            res = json.loads(req.read().decode())
            if res.get("status") == "ok":
                iaqi = res["data"].get("iaqi", {})
                return {
                    "PM2_5_aqi": iaqi.get("pm25", {}).get("v"),
                    "PM10_aqi": iaqi.get("pm10", {}).get("v"),
                    "O3_aqi": iaqi.get("o3", {}).get("v"),
                    "NO2_aqi": iaqi.get("no2", {}).get("v"),
                    "CO_aqi": iaqi.get("co", {}).get("v"),
                    "SO2_aqi": iaqi.get("so2", {}).get("v"),
                }
        except Exception as e:
            print(f"Error con endpoint {ep}: {e}")
            continue
    return {}

def obtener_clima():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=19.43&longitude=-99.13&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        req = urllib.request.urlopen(url, timeout=5)
        res = json.loads(req.read().decode())
        cur = res.get("current", {})
        return {
            "temperatura": cur.get("temperature_2m", 20.0),
            "humedad": cur.get("relative_humidity_2m", 50.0),
            "viento": cur.get("wind_speed_10m", 2.5)
        }
    except Exception:
        return {"temperatura": 20.0, "humedad": 50.0, "viento": 2.5}

def aqi_a_concentracion(param, aqi):
    if aqi is None: return 20.0
    return float(aqi) * 0.5