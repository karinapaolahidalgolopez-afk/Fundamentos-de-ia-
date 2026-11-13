# entrenar_modelos.py
import os, sys, joblib, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from datos_y_tablas import cargar_datos, construir_tablas, LIMITES, ETIQUETAS

os.makedirs("figuras", exist_ok=True)
ruta = sys.argv[1] if len(sys.argv) > 1 else None
datos = cargar_datos(ruta)

datos = datos.drop_duplicates()
datos, X, y_reg, y = construir_tablas(datos)

corte = int(len(X) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]

def pipe(modelo):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("esc", StandardScaler()), ("modelo", modelo)])

candidatos = {
    "KNN": (pipe(KNeighborsClassifier()),
            {"modelo__n_neighbors": [5, 11, 21]}),
    "SVM": (pipe(SVC(probability=True, class_weight="balanced", random_state=0)),
            {"modelo__C": [0.5, 1, 5], "modelo__kernel": ["rbf", "linear"]}),
    "Árbol de Decisión": (pipe(DecisionTreeClassifier(class_weight="balanced", random_state=0)),
                          {"modelo__max_depth": [3, 5, 8]}),
    "Random Forest": (pipe(RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1)),
                      {"modelo__n_estimators": [100, 300], "modelo__max_depth": [None, 8]}),
}

filas, ajustados = [], {}
for nombre, (p, grid) in candidatos.items():
    gs = GridSearchCV(p, grid, cv=TimeSeriesSplit(4), scoring="f1_macro", n_jobs=-1).fit(Xtr, ytr)
    pred = gs.predict(Xte)
    ajustados[nombre] = gs.best_estimator_
    filas.append({"Modelo": nombre,
                  "Accuracy": accuracy_score(yte, pred),
                  "Precision": precision_score(yte, pred, average="macro", zero_division=0),
                  "Recall": recall_score(yte, pred, average="macro", zero_division=0),
                  "F1": f1_score(yte, pred, average="macro", zero_division=0),
                  "Mejores parámetros": str(gs.best_params_)})

tabla = pd.DataFrame(filas).sort_values("F1", ascending=False).reset_index(drop=True)
tabla.to_csv("tabla_comparativa.csv", index=False)

ganador = tabla.loc[0, "Modelo"]
ConfusionMatrixDisplay.from_estimator(ajustados[ganador], Xte, yte, xticks_rotation=30)
plt.title(f"Matriz de confusión - {ganador}"); plt.tight_layout()
plt.savefig("figuras/matriz_confusion.png", dpi=120); plt.close()

joblib.dump({"modelo": ajustados[ganador], "columnas": list(X.columns),
             "clases": list(ajustados[ganador].classes_), "nombre": ganador,
             "rangos": X.describe().loc[["min", "max", "50%"]].to_dict()}, "modelo.joblib")