#!/usr/bin/env python3
"""
Detección de Anomalías de Sensor — PTAR Bello / Simulador COP-05
====================================================================

Tercer caso de uso de la Propuesta de Transición a Machine Learning (Sección 3.3).
Clasifica el estado de los transmisores de nivel LT-051004A/B en 3 clases —
sano / deriva gradual / fallo duro — a partir del residual entre ambos sensores
y su tendencia reciente, para detectar una deriva ANTES de que dispare la alarma
binaria de "fallo de medida" que ya existe.

ARCHIVOS DE ENTRADA (los que exporta el botón "Detección de Anomalías de Sensor"
del Generador Automático de Datasets ML, en la pestaña EOP-05 del simulador):
    telemetria_anomalias_sensor_*.csv
    episodios_anomalias_sensor_*.csv

Para correrlo en VS Code sin argumentos: coloca ambos CSV en la misma carpeta
que este script, renómbralos a "telemetria.csv" y "episodios.csv" (o edita los
valores por defecto más abajo), y presiona "Run".

También se puede correr por línea de comandos con rutas propias:
    python3 deteccion_anomalias_sensor_modelo.py --telemetria telemetria_anomalias_sensor_2590s.csv --episodios episodios_anomalias_sensor_2590s.csv

ADVERTENCIA (se repite intencionalmente en cada entrega de este proyecto):
    Dataset SINTÉTICO del simulador COP-05 — ver Sección 2.2 de la Propuesta de
    Transición a ML. Este script valida el pipeline de punta a punta, no certifica
    un modelo listo para producción. Antes de cualquier despliegue real, debe
    reentrenarse contra datos reales del historiador de planta.

HALLAZGO QUE ESTE SCRIPT DEMUESTRA NUMÉRICAMENTE (no lo escondas si lo corres):
    Un modelo entrenado solo con el residual entre LT-A y LT-B detecta la deriva
    gradual muy bien, pero es CIEGO al fallo duro — porque un sensor "caído" sigue
    reportando un valor cercano a la física real (el simulador no lo pone en NaN),
    así que el residual no se dispara. Un sistema real necesitaría una señal de
    disponibilidad/calidad de la medida, no solo la magnitud del residual, para
    cubrir ambos casos.
"""

import argparse
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

CLASES = ["sano", "deriva", "fallo_duro"]
ROLLING_WINDOW = 5  # muestras (25 s a 5 s/muestra)
FEATURES = ["ltA_medido", "ltB_medido", "lt_residual", "residual_roll_mean", "residual_slope"]
UMBRAL_BASELINE = 0.15  # m — el mismo umbral de aviso ya implementado en el simulador (sin histéresis, por simplicidad)


# ------------------------------------------------------------------------- #
# 1. Carga, etiquetado y features                                           #
# ------------------------------------------------------------------------- #

def cargar_y_preparar(path_telemetria: str) -> pd.DataFrame:
    """Carga la telemetría y construye el target de 3 clases directamente desde
    las columnas de verdad de terreno que exporta el simulador (sensorA_fallo,
    sensorB_fallo, sensorA_deriva_activa, sensorB_deriva_activa) — no hace falta
    cruzar por tiempo contra el CSV de episodios para etiquetar."""
    tel = pd.read_csv(path_telemetria)
    requeridas = ["t", "ltA_medido", "ltB_medido", "lt_residual",
                  "sensorA_fallo", "sensorB_fallo", "sensorA_deriva_activa", "sensorB_deriva_activa"]
    faltantes = [c for c in requeridas if c not in tel.columns]
    if faltantes:
        raise ValueError(
            f"Faltan columnas {faltantes}. Este script requiere telemetría generada "
            f"con la versión del simulador que incluye las columnas de verdad de "
            f"terreno de sensores (posterior a la incorporación de sensorA_fallo, etc.)."
        )

    def clase(row):
        if row["sensorA_fallo"] or row["sensorB_fallo"]:
            return 2  # fallo_duro
        if row["sensorA_deriva_activa"] or row["sensorB_deriva_activa"]:
            return 1  # deriva
        return 0  # sano

    tel = tel.sort_values("t").reset_index(drop=True)
    tel["label"] = tel.apply(clase, axis=1)

    tel["residual_roll_mean"] = tel["lt_residual"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    tel["residual_slope"] = tel["lt_residual"].diff(ROLLING_WINDOW).fillna(0) / ROLLING_WINDOW
    return tel


# ------------------------------------------------------------------------- #
# 2. División temporal train/test usando el CSV de episodios                #
# ------------------------------------------------------------------------- #

def dividir_train_test(tel: pd.DataFrame, path_episodios: str, n_episodios_prueba: int = 2):
    """Corte temporal, no aleatorio: entrena con la primera parte de la sesión y
    evalúa contra los últimos N episodios de falla (por defecto 2), que el modelo
    nunca ve durante el entrenamiento — evita fuga de datos entre muestras
    consecutivas y prueba generalización a severidades/tipos de falla nuevos."""
    ep = pd.read_csv(path_episodios).sort_values("inicio_t")
    if len(ep) < n_episodios_prueba + 1:
        raise ValueError(
            f"Solo hay {len(ep)} episodios en el CSV — se necesitan al menos "
            f"{n_episodios_prueba + 1} para dejar alguno en entrenamiento."
        )
    episodios_prueba = ep.iloc[-n_episodios_prueba:]
    corte_t = episodios_prueba["inicio_t"].min()

    print("Episodios reservados para prueba (nunca vistos en entrenamiento):")
    print(episodios_prueba[["causa", "inicio_t", "fin_t"]].to_string(index=False))
    print(f"\nCorte temporal: entrenamiento con t < {corte_t}, prueba con t >= {corte_t}")

    train = tel[tel["t"] < corte_t]
    test = tel[tel["t"] >= corte_t]
    return train, test


# ------------------------------------------------------------------------- #
# 3. Entrenamiento, evaluación y comparación contra la línea base            #
# ------------------------------------------------------------------------- #

def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame):
    Xtr, ytr = train[FEATURES], train["label"]
    Xte, yte = test[FEATURES], test["label"]

    print(f"\nEntrenamiento: {len(Xtr)} muestras — clases: {ytr.value_counts().sort_index().to_dict()}")
    print(f"Prueba:        {len(Xte)} muestras — clases: {yte.value_counts().sort_index().to_dict()}")

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print("\n--- Reporte de clasificación (episodios nunca vistos en entrenamiento) ---")
    presentes = sorted(set(yte) | set(pred))
    print(classification_report(yte, pred, labels=presentes, zero_division=0,
                                 target_names=[CLASES[i] for i in presentes]))
    print("Matriz de confusión (filas = real, columnas = predicho; orden: sano, deriva, fallo_duro):")
    print(confusion_matrix(yte, pred, labels=[0, 1, 2]))

    recall_fallo_duro = None
    if (yte == 2).sum() > 0:
        recall_fallo_duro = ((pred == 2) & (yte == 2)).sum() / (yte == 2).sum()
        print(f"\n>>> Recall específico sobre 'fallo_duro': {recall_fallo_duro:.2f} <<<")
        if recall_fallo_duro < 0.5:
            print(">>> HALLAZGO ESPERADO: el modelo, basado en el residual, es prácticamente")
            print(">>> ciego al fallo duro — ver la explicación en el docstring de este script.")

    # Comparación binaria contra el umbral fijo que ya usa el simulador
    real_anomalia = (yte > 0).astype(int)
    baseline_pred = (test["lt_residual"] > UMBRAL_BASELINE).astype(int)
    modelo_pred_bin = (pred > 0).astype(int)
    recall_baseline = (baseline_pred[real_anomalia == 1] == 1).mean() if (real_anomalia == 1).any() else float("nan")
    recall_modelo = (modelo_pred_bin[real_anomalia == 1] == 1).mean() if (real_anomalia == 1).any() else float("nan")
    print(f"\nComparación binaria '¿hay alguna anomalía?' (deriva o fallo, sin distinguir tipo):")
    print(f"  Línea base (umbral fijo {UMBRAL_BASELINE} m, sin histéresis): recall = {recall_baseline:.2f}")
    print(f"  Modelo Random Forest:                                        recall = {recall_modelo:.2f}")

    return clf, test.assign(pred=pred)


# ------------------------------------------------------------------------- #
# 4. Visualizaciones                                                        #
# ------------------------------------------------------------------------- #

def graficar_importancia(clf, path_salida="feature_importance_sensores.png"):
    if not HAS_PLOTTING:
        return
    importancias = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: x[1])
    nombres, valores = zip(*importancias)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(nombres, valores, color="#5B3F94")
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Detección de Anomalías de Sensor — Importancia de Features")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=130)
    plt.close(fig)
    print(f"\n[guardado] {path_salida}")


def graficar_residual_con_clases(tel: pd.DataFrame, path_salida="residual_vs_clases.png"):
    if not HAS_PLOTTING:
        return
    colores = {0: "#2ecc71", 1: "#f2a02f", 2: "#e74c3c"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tel["t"], tel["lt_residual"], color="#333333", linewidth=0.8, zorder=1)
    for c, color in colores.items():
        sub = tel[tel["label"] == c]
        ax.scatter(sub["t"], sub["lt_residual"], color=color, s=8, label=CLASES[c], zorder=2)
    ax.axhline(UMBRAL_BASELINE, color="#999999", linestyle=":", linewidth=1, label=f"Umbral línea base ({UMBRAL_BASELINE} m)")
    ax.set_xlabel("Tiempo simulado (s)")
    ax.set_ylabel("Residual |LT-A − LT-B| (m)")
    ax.set_title("Residual de sensores vs. clase real — nótese el fallo duro sin residual visible")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_salida, dpi=130)
    plt.close(fig)
    print(f"[guardado] {path_salida}")


# ------------------------------------------------------------------------- #
# main                                                                       #
# ------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Modelo de detección de anomalías de sensor sobre datos del simulador COP-05.")
    parser.add_argument("--telemetria", default="telemetria.csv", help="CSV de telemetría (default: telemetria.csv)")
    parser.add_argument("--episodios", default="episodios.csv", help="CSV de episodios de falla (default: episodios.csv)")
    parser.add_argument("--episodios-prueba", type=int, default=2, help="Nº de últimos episodios a reservar para prueba (default: 2)")
    args = parser.parse_args()

    print("=" * 70)
    print("DETECCIÓN DE ANOMALÍAS DE SENSOR — Simulador COP-05, PTAR Bello")
    print("=" * 70)
    print("ADVERTENCIA: dataset sintético — ver docstring del script antes de interpretar resultados.\n")

    tel = cargar_y_preparar(args.telemetria)
    print(f"Telemetría cargada: {len(tel)} filas.")
    print(f"Distribución de clases en toda la sesión: "
          f"{dict(zip(CLASES, tel['label'].value_counts().sort_index().tolist()))}")

    train, test = dividir_train_test(tel, args.episodios, args.episodios_prueba)
    clf, test_pred = entrenar_y_evaluar(train, test)

    print("\nImportancia de features (mayor a menor):")
    for f, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:20s} {imp:.3f}")

    graficar_importancia(clf)
    graficar_residual_con_clases(tel)

    print("\n" + "=" * 70)
    print("Pipeline verificado de punta a punta. El hallazgo sobre el fallo duro")
    print("es real y esperado — no es un error del script. Para cubrir ese caso,")
    print("un sistema real necesitaría una señal de disponibilidad de la medida")
    print("además del residual (ver Sección 3.3 de la Propuesta de Transición a ML).")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
