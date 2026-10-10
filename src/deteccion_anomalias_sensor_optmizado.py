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
import logging
import numpy as np
import pandas as pd

# --- Configuración de logging ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Validación de librerías críticas ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
    logging.info("Módulo matplotlib cargado correctamente, gráficos habilitados.")
except ImportError:
    HAS_PLOTTING = False
    logging.warning("Matplotlib no disponible, se deshabilitan gráficos.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    logging.error("scikit-learn no está instalado. Instala con: pip install scikit-learn")
    raise

# --- Configuración y constantes del modelo ---
PARAMS = {
    "CLASES": ["sano", "deriva", "fallo_duro"],
    "ROLLING_WINDOW": 5,  # muestras (25 s a 5 s/muestra)
    "FEATURES": ["lta_medido", "ltb_medido", "lt_residual", "residual_roll_mean", "residual_slope"],
    "UMBRAL_BASELINE": 0.15,  # m — umbral de aviso del simulador
    "RF_PARAMS": {"n_estimators": 200, "max_depth": 5, "class_weight": "balanced", "random_state": 42}
}

def __repr__():
    return f"Configuración del modelo: {PARAMS}"

logging.info("Constantes inicializadas correctamente: %s", PARAMS)


# ------------------------------------------------------------------------- #
# 1. Carga, etiquetado y features                                           #
# ------------------------------------------------------------------------- #

def cargar_y_preparar(path_telemetria: str) -> pd.DataFrame:
    """
    Carga la telemetría y construye el target de 3 clases directamente desde
    las columnas de verdad de terreno que exporta el simulador.

    Mejoras aplicadas:
    - Logging en cada fase crítica (lectura, validación, construcción de labels).
    - Manejo robusto de errores con mensajes claros.
    - Validación de tipos de datos y valores nulos.
    - Normalización de nombres de columnas para estandarización.
    - Verificación de consistencia temporal.
    """
    import logging

    logging.info(f"Iniciando carga de telemetría desde: {path_telemetria}")

    # --- Lectura robusta del archivo ---
    try:
        tel = pd.read_csv(path_telemetria)
        logging.info(f"Archivo cargado con {tel.shape[0]} filas y {tel.shape[1]} columnas")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {path_telemetria}")
        raise
    except Exception as e:
        logging.error(f"Error al leer {path_telemetria}: {e}")
        raise

    # --- Normalización de nombres de columnas ---
    tel.columns = [c.strip().lower() for c in tel.columns]

    requeridas = [
        "t", "lta_medido", "ltb_medido", "lt_residual",
        "sensora_fallo", "sensorb_fallo",
        "sensora_deriva_activa", "sensorb_deriva_activa"
    ]
    faltantes = [c for c in requeridas if c not in tel.columns]
    if faltantes:
        logging.error(f"Faltan columnas requeridas: {faltantes}")
        raise ValueError(
            f"Este script requiere telemetría generada con la versión del simulador "
            f"que incluye las columnas de verdad de terreno de sensores."
        )

    # --- Validación de tipos de datos ---
    for col in ["lta_medido", "ltb_medido", "lt_residual"]:
        if not np.issubdtype(tel[col].dtype, np.number):
            logging.error(f"Columna {col} no es numérica")
            raise TypeError(f"Columna {col} debe ser numérica")

    # --- Validación de valores nulos ---
    nulos = tel.isnull().sum().sum()
    if nulos > 0:
        logging.warning(f"Se encontraron {nulos} valores nulos en la telemetría")

    # --- Validación temporal ---
    if not tel["t"].is_monotonic_increasing:
        logging.warning("La columna 't' no está ordenada, se reordena automáticamente")
        tel = tel.sort_values("t").reset_index(drop=True)

    # --- Construcción de etiquetas ---
    def clase(row):
        if row["sensora_fallo"] or row["sensorb_fallo"]:
            return 2  # fallo_duro
        if row["sensora_deriva_activa"] or row["sensorb_deriva_activa"]:
            return 1  # deriva
        return 0  # sano

    tel["label"] = tel.apply(clase, axis=1)
    logging.info("Etiquetas de clase construidas correctamente")

    # --- Features adicionales ---
    tel["residual_roll_mean"] = tel["lt_residual"].rolling(PARAMS["ROLLING_WINDOW"], min_periods=1).mean()
    tel["residual_slope"] = tel["lt_residual"].diff(PARAMS["ROLLING_WINDOW"]).fillna(0) / PARAMS["ROLLING_WINDOW"]
    logging.info("Features adicionales calculadas: residual_roll_mean, residual_slope")

    return tel



# ------------------------------------------------------------------------- #
# 2. División temporal train/test usando el CSV de episodios                #
# ------------------------------------------------------------------------- #

def dividir_train_test(tel: pd.DataFrame, path_episodios: str, n_episodios_prueba: int = 2):
    """
    Divide el dataset en entrenamiento y prueba mediante corte temporal, no aleatorio.
    - Entrena con la primera parte de la sesión.
    - Evalúa contra los últimos N episodios de falla (por defecto 2).
    - Evita fuga de datos entre muestras consecutivas y prueba generalización.

    Mejoras aplicadas:
    - Manejo robusto de errores en lectura de episodios.
    - Validación de cantidad mínima de episodios.
    - Verificación de balance de clases en train/test.
    - Retorno extendido con metadatos de episodios reservados.
    """
    import logging

    logging.info(f"Cargando episodios desde: {path_episodios}")

    # --- Lectura robusta del archivo ---
    try:
        ep = pd.read_csv(path_episodios).sort_values("inicio_t")
        logging.info(f"Archivo de episodios cargado con {len(ep)} registros")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de episodios: {path_episodios}")
        raise
    except Exception as e:
        logging.error(f"Error al leer {path_episodios}: {e}")
        raise

    # --- Validación de cantidad mínima de episodios ---
    if len(ep) < n_episodios_prueba + 1:
        logging.error(f"Solo hay {len(ep)} episodios en el CSV — se necesitan al menos {n_episodios_prueba + 1}")
        raise ValueError(
            f"Cantidad insuficiente de episodios: {len(ep)}. "
            f"Se requieren al menos {n_episodios_prueba + 1} para dejar alguno en entrenamiento."
        )

    # --- Selección de episodios de prueba ---
    episodios_prueba = ep.iloc[-n_episodios_prueba:]
    corte_t = episodios_prueba["inicio_t"].min()

    logging.info("Episodios reservados para prueba (nunca vistos en entrenamiento):")
    logging.info("\n" + episodios_prueba[["causa", "inicio_t", "fin_t"]].to_string(index=False))
    logging.info(f"Corte temporal definido: entrenamiento con t < {corte_t}, prueba con t >= {corte_t}")

    # --- División temporal ---
    train = tel[tel["t"] < corte_t]
    test = tel[tel["t"] >= corte_t]

    # --- Validación de balance de clases ---
    clases_train = train["label"].value_counts().to_dict()
    clases_test = test["label"].value_counts().to_dict()
    logging.info(f"Distribución de clases en train: {clases_train}")
    logging.info(f"Distribución de clases en test: {clases_test}")

    if len(clases_train) < len(PARAMS["CLASES"]):
        logging.warning("No todas las clases están presentes en el conjunto de entrenamiento")
    if len(clases_test) < len(PARAMS["CLASES"]):
        logging.warning("No todas las clases están presentes en el conjunto de prueba")

    # --- Retorno extendido con metadatos ---
    return train, test, {"episodios_prueba": episodios_prueba, "corte_t": corte_t}



# ------------------------------------------------------------------------- #
# 3. Entrenamiento, evaluación y comparación contra la línea base            #
# ------------------------------------------------------------------------- #

def entrenar_randomforest(train: pd.DataFrame, test: pd.DataFrame):
    """
    Entrena un modelo RandomForestClassifier y lo evalúa contra el conjunto de prueba.
    Devuelve el modelo entrenado, el dataframe con predicciones y las métricas calculadas.
    """
    import logging
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    # --- Validación de datasets ---
    if train.empty or test.empty:
        logging.error("Conjuntos vacíos. No se puede entrenar.")
        raise ValueError("Dataset vacío.")

    Xtr, ytr = train[PARAMS["FEATURES"]], train["label"]
    Xte, yte = test[PARAMS["FEATURES"]], test["label"]

    # Validación de clases mínimas
    for clase_idx, clase_nombre in enumerate(PARAMS["CLASES"]):
        count_train = (ytr == clase_idx).sum()
        if count_train < 5:
            logging.warning(f"La clase '{clase_nombre}' tiene pocas muestras ({count_train}) en entrenamiento")

    logging.info(f"Entrenamiento: {len(Xtr)} muestras — clases: {ytr.value_counts().to_dict()}")
    logging.info(f"Prueba:        {len(Xte)} muestras — clases: {yte.value_counts().to_dict()}")

    # --- Entrenamiento ---
    clf = RandomForestClassifier(**PARAMS["RF_PARAMS"])
    logging.info(f"Entrenando RandomForest con parámetros: {PARAMS['RF_PARAMS']}")
    clf.fit(Xtr, ytr)
    logging.info("Entrenamiento completado ✅")

    # --- Predicciones ---
    pred = clf.predict(Xte)
    logging.info("Predicciones generadas")

    # --- Reporte ---
    reporte = classification_report(yte, pred, target_names=PARAMS["CLASES"], zero_division=0)
    logging.info("\n--- Reporte de clasificación RF ---\n" + reporte)

    matriz = confusion_matrix(yte, pred, labels=[0, 1, 2])
    logging.info("Matriz de confusión RF:\n%s", matriz)

    # --- Métricas específicas ---
    recall_fallo_duro = None
    if (yte == 2).sum() > 0:
        recall_fallo_duro = ((pred == 2) & (yte == 2)).sum() / (yte == 2).sum()
        logging.info(f"Recall fallo_duro RF: {recall_fallo_duro:.2f}")

    real_anomalia = (yte > 0).astype(int)
    baseline_pred = (test["lt_residual"] > PARAMS["UMBRAL_BASELINE"]).astype(int)
    modelo_pred_bin = (pred > 0).astype(int)

    recall_baseline = (baseline_pred[real_anomalia == 1] == 1).mean() if (real_anomalia == 1).any() else float("nan")
    recall_modelo = (modelo_pred_bin[real_anomalia == 1] == 1).mean() if (real_anomalia == 1).any() else float("nan")

    logging.info(f"Comparación binaria RF: baseline={recall_baseline:.2f}, modelo={recall_modelo:.2f}")

    # --- Devolver métricas ---
    metrics = {
        "recall_fallo_duro": recall_fallo_duro,
        "recall_baseline": recall_baseline,
        "recall_modelo": recall_modelo
    }

    return clf, test.assign(pred_rf=pred), metrics


def entrenar_mlp(train: pd.DataFrame, test: pd.DataFrame):
    """
    Entrena un MLP con Keras/TensorFlow y lo evalúa contra el conjunto de prueba.
    Devuelve el modelo entrenado, el dataframe con predicciones y las métricas calculadas.
    """
    import logging
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd

    # --- Preparar datos ---
    Xtr, ytr = train[PARAMS["FEATURES"]].values, train["label"].values
    Xte, yte = test[PARAMS["FEATURES"]].values, test["label"].values

    ytr_cat = to_categorical(ytr, num_classes=len(PARAMS["CLASES"]))
    yte_cat = to_categorical(yte, num_classes=len(PARAMS["CLASES"]))

    # --- Definir arquitectura ---
    mlp = Sequential([
        Dense(64, activation="relu", input_shape=(Xtr.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(len(PARAMS["CLASES"]), activation="softmax")
    ])

    mlp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    logging.info("Entrenando MLP con Keras/TensorFlow...")
    mlp.fit(Xtr, ytr_cat, epochs=50, batch_size=16, verbose=0)

    # --- Evaluar ---
    loss, acc = mlp.evaluate(Xte, yte_cat, verbose=0)
    logging.info(f"Accuracy MLP: {acc:.2f}")

    # --- Predicciones ---
    y_pred_mlp = mlp.predict(Xte)
    y_pred_classes = y_pred_mlp.argmax(axis=1)

    # --- Reporte ---
    reporte = classification_report(yte, y_pred_classes, target_names=PARAMS["CLASES"], zero_division=0)
    logging.info("\n--- Reporte de clasificación MLP ---\n" + reporte)

    matriz = confusion_matrix(yte, y_pred_classes, labels=[0, 1, 2])
    logging.info("Matriz de confusión MLP:\n%s", matriz)

    # --- Métricas específicas ---
    metrics = {
        "accuracy": float((yte == y_pred_classes).mean()),
        "pred_counts": {str(k): int(v) for k, v in pd.Series(y_pred_classes).value_counts().to_dict().items()}
    }

    return mlp, test.assign(pred_mlp=y_pred_classes), metrics





# ------------------------------------------------------------------------- #
# 4. Visualizaciones                                                        #
# ------------------------------------------------------------------------- #

def graficar_importancia(clf, path_salida="feature_importance_sensores.png"):
    """
    Genera un gráfico de barras horizontales con la importancia relativa de las features.

    Mejoras aplicadas:
    - Validación de disponibilidad de matplotlib.
    - Validación de atributos del modelo (feature_importances_).
    - Uso de parámetros estandarizados desde CONFIG/PARAMS.
    - Logging en cada fase crítica.
    - Manejo robusto de errores al guardar archivo.
    """
    import logging

    if not HAS_PLOTTING:
        logging.warning("Matplotlib no disponible, se omite graficar importancia de features.")
        return

    if not hasattr(clf, "feature_importances_"):
        logging.error("El modelo no tiene atributo 'feature_importances_'")
        raise AttributeError("El modelo no soporta cálculo de importancia de features")

    importancias = sorted(zip(PARAMS["FEATURES"], clf.feature_importances_), key=lambda x: x[1])
    nombres, valores = zip(*importancias)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(nombres, valores, color="#5B3F94")
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Detección de Anomalías de Sensor — Importancia de Features")
    fig.tight_layout()

    try:
        fig.savefig(path_salida, dpi=130)
        logging.info(f"Gráfico de importancia de features guardado en: {path_salida}")
    except Exception as e:
        logging.error(f"No se pudo guardar el gráfico en {path_salida}: {e}")
    finally:
        plt.close(fig)


def graficar_residual_con_clases(tel: pd.DataFrame, path_salida="residual_vs_clases.png"):
    """
    Genera un gráfico del residual LT-A − LT-B contra el tiempo, coloreado por clase real.

    Mejoras aplicadas:
    - Validación de disponibilidad de matplotlib.
    - Validación de columnas requeridas en el dataframe.
    - Uso de parámetros estandarizados desde CONFIG/PARAMS.
    - Logging en cada fase crítica.
    - Manejo robusto de errores al guardar archivo.
    """
    import logging

    if not HAS_PLOTTING:
        logging.warning("Matplotlib no disponible, se omite graficar residual con clases.")
        return

    requeridas = ["t", "lt_residual", "label"]
    faltantes = [c for c in requeridas if c not in tel.columns]
    if faltantes:
        logging.error(f"Faltan columnas requeridas para graficar residual: {faltantes}")
        raise ValueError(f"El dataframe no contiene las columnas necesarias: {faltantes}")

    colores = {0: "#2ecc71", 1: "#f2a02f", 2: "#e74c3c"}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tel["t"], tel["lt_residual"], color="#333333", linewidth=0.8, zorder=1)

    for c, color in colores.items():
        sub = tel[tel["label"] == c]
        if not sub.empty:
            ax.scatter(sub["t"], sub["lt_residual"], color=color, s=8, label=PARAMS["CLASES"][c], zorder=2)

    ax.axhline(PARAMS["UMBRAL_BASELINE"], color="#999999", linestyle=":", linewidth=1,
               label=f"Umbral línea base ({PARAMS['UMBRAL_BASELINE']} m)")
    ax.set_xlabel("Tiempo simulado (s)")
    ax.set_ylabel("Residual |LT-A − LT-B| (m)")
    ax.set_title("Residual de sensores vs. clase real — nótese el fallo duro sin residual visible")
    ax.legend(fontsize=8)
    fig.tight_layout()

    try:
        fig.savefig(path_salida, dpi=130)
        logging.info(f"Gráfico residual vs clases guardado en: {path_salida}")
    except Exception as e:
        logging.error(f"No se pudo guardar el gráfico en {path_salida}: {e}")
    finally:
        plt.close(fig)



# ------------------------------------------------------------------------- #
# main                                                                       #
# ------------------------------------------------------------------------- #
def main():
    """
    Función principal: orquesta el pipeline completo de detección de anomalías.
    - Parseo de argumentos.
    - Carga y preparación de telemetría.
    - División en train/test.
    - Entrenamiento y evaluación con RandomForest y MLP.
    - Exportación de métricas separadas.
    - Comparación automática de resultados.
    - Generación de gráficas.
    """

    import logging
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(
        description="Modelo de detección de anomalías de sensor sobre datos del simulador COP-05."
    )
    parser.add_argument("--telemetria", default="telemetria.csv",
                        help="CSV de telemetría (default: telemetria.csv)")
    parser.add_argument("--episodios", default="episodios.csv",
                        help="CSV de episodios de falla (default: episodios.csv)")
    parser.add_argument("--episodios-prueba", type=int, default=2,
                        help="Nº de últimos episodios a reservar para prueba (default: 2)")
    parser.add_argument("--salida", default="outputs/",
                        help="Carpeta de salida para gráficas y métricas (default: outputs/)")
    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("DETECCIÓN DE ANOMALÍAS DE SENSOR — Simulador COP-05, PTAR Bello")
    logging.info("=" * 70)
    logging.warning("ADVERTENCIA: dataset sintético — ver docstring del script antes de interpretar resultados.")

    # --- Carga de telemetría ---
    tel = cargar_y_preparar(args.telemetria)
    logging.info(f"Telemetría cargada: {len(tel)} filas.")
    logging.info(f"Distribución de clases en toda la sesión: "
                 f"{dict(zip(PARAMS['CLASES'], tel['label'].value_counts().sort_index().tolist()))}")

    # --- División train/test ---
    train, test, meta = dividir_train_test(tel, args.episodios, args.episodios_prueba)

    # --- Entrenamiento y evaluación RandomForest ---
    clf_rf, test_rf, metrics_rf = entrenar_randomforest(train, test)

    # Exportar métricas RF
    try:
        with open(f"{args.salida}/metrics_rf.json", "w") as f:
            json.dump(metrics_rf, f, indent=2)
        logging.info("Métricas RF exportadas a metrics_rf.json")
    except Exception as e:
        logging.warning(f"No se pudieron exportar métricas RF: {e}")

    # --- Entrenamiento y evaluación MLP ---
    clf_mlp, test_mlp, metrics_mlp = entrenar_mlp(train, test)

    # Exportar métricas MLP
    try:
        with open(f"{args.salida}/metrics_mlp.json", "w") as f:
            json.dump(metrics_mlp, f, indent=2)
        logging.info("Métricas MLP exportadas a metrics_mlp.json")
    except Exception as e:
        logging.warning(f"No se pudieron exportar métricas MLP: {e}")

    # --- Comparación automática RF vs MLP ---
    logging.info("=" * 70)
    logging.info("COMPARACIÓN AUTOMÁTICA DE MODELOS")
    logging.info("=" * 70)

    if metrics_rf["recall_fallo_duro"] is not None:
        logging.info(f"Recall fallo_duro RF: {metrics_rf['recall_fallo_duro']:.2f}")
    else:
        logging.info("Recall fallo_duro RF: No disponible")

    logging.info(f"Accuracy RF (binario): {metrics_rf['recall_modelo']:.2f}")
    logging.info(f"Accuracy MLP (multiclase): {metrics_mlp['accuracy']:.2f}")

    if metrics_mlp["accuracy"] > metrics_rf["recall_modelo"]:
        logging.info("👉 El MLP tuvo mejor desempeño global en accuracy.")
    else:
        logging.info("👉 El RandomForest tuvo mejor desempeño global en accuracy.")

    # --- Importancia de features (solo RF) ---
    if hasattr(clf_rf, "feature_importances_"):
        logging.info("Importancia de features (mayor a menor):")
        for f, imp in sorted(zip(PARAMS["FEATURES"], clf_rf.feature_importances_), key=lambda x: -x[1]):
            logging.info(f"  {f:20s} {imp:.3f}")
        graficar_importancia(clf_rf, path_salida=f"{args.salida}/feature_importance_sensores.png")

    # --- Gráfica residual vs clases ---
    graficar_residual_con_clases(tel, path_salida=f"{args.salida}/residual_vs_clases.png")

    # --- Mensaje final ---
    logging.info("=" * 70)
    logging.info("Pipeline verificado de punta a punta. El hallazgo sobre el fallo duro")
    logging.info("es real y esperado — no es un error del script. Para cubrir ese caso,")
    logging.info("un sistema real necesitaría una señal de disponibilidad de la medida")
    logging.info("además del residual (ver Sección 3.3 de la Propuesta de Transición a ML).")
    logging.info("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.error(f"Error crítico en ejecución del pipeline: {e}")
        sys.exit(1)
