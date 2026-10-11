#!/usr/bin/env python3
"""
Detección de Anomalías de Sensor — PTAR Bello / Simulador COP-05
====================================================================
Pipeline orientado a redes neuronales (MLP con Keras).
Clasifica el estado de los transmisores LT-051004A/B en tres clases:
- sano
- deriva gradual
- fallo duro
"""

import argparse
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight



# --- Configuración de logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Reproducibilidad ---
np.random.seed(42)
tf.random.set_seed(42)

# --- Parámetros globales ---
PARAMS = {
    "CLASES": ["sano", "deriva", "fallo_duro"],
    "ROLLING_WINDOW": 5,
    "FEATURES": ["lta_medido", "ltb_medido", "lt_residual", "residual_roll_mean", "residual_slope"],
    "UMBRAL_BASELINE": 0.15,
    "EPOCHS": 30,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001
}


def cargar_y_preparar(path_telemetria: str) -> pd.DataFrame:
    """
    Carga telemetría y construye etiquetas de 3 clases.
    Incluye validación de columnas, tipos y consistencia temporal.
    """
    import logging

    tel = pd.read_csv(path_telemetria)

    # Normalizar nombres de columnas a minúsculas
    tel = tel.rename(columns={
        "ltA_medido": "lta_medido",
        "ltB_medido": "ltb_medido",
        "sensorA_fallo": "sensora_fallo",
        "sensorB_fallo": "sensorb_fallo",
        "sensorA_deriva_activa": "sensora_deriva_activa",
        "sensorB_deriva_activa": "sensorb_deriva_activa"
    })

    requeridas = ["t", "lta_medido", "ltb_medido", "lt_residual",
              "sensora_fallo", "sensorb_fallo",
              "sensora_deriva_activa", "sensorb_deriva_activa"]

    faltantes = [c for c in requeridas if c not in tel.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Validación de tipos numéricos
    for col in ["lta_medido", "ltb_medido", "lt_residual"]:
        if not np.issubdtype(tel[col].dtype, np.number):
            raise TypeError(f"La columna {col} debe ser numérica")

    # Construcción de etiquetas
    def clase(row):
        if row["sensora_fallo"] or row["sensorb_fallo"]:
            return 2  # fallo_duro
        if row["sensora_deriva_activa"] or row["sensorb_deriva_activa"]:
            return 1  # deriva
        return 0  # sano

    tel = tel.sort_values("t").reset_index(drop=True)
    tel["label"] = tel.apply(clase, axis=1)

    # Features adicionales
    tel["residual_roll_mean"] = tel["lt_residual"].rolling(PARAMS["ROLLING_WINDOW"], min_periods=1).mean()
    tel["residual_slope"] = tel["lt_residual"].diff(PARAMS["ROLLING_WINDOW"]).fillna(0) / PARAMS["ROLLING_WINDOW"]

    logging.info(f"Telemetría cargada: {tel.shape[0]} filas, {tel.shape[1]} columnas")
    logging.info(f"Distribución de clases: {tel['label'].value_counts().to_dict()}")

    return tel


def dividir_train_test(tel: pd.DataFrame, path_episodios: str, n_episodios_prueba: int = 2):
    """
    Divide dataset en train/test mediante corte temporal.
    - Entrena con la primera parte de la sesión.
    - Evalúa contra los últimos N episodios de falla (por defecto 2).
    """
    import logging

    ep = pd.read_csv(path_episodios).sort_values("inicio_t")

    # Validación de columnas
    for col in ["inicio_t", "fin_t", "causa"]:
        if col not in ep.columns:
            raise ValueError(f"Falta columna requerida en episodios: {col}")

    # Validación de cantidad mínima
    if len(ep) < n_episodios_prueba + 1:
        raise ValueError(f"Solo hay {len(ep)} episodios — se necesitan al menos {n_episodios_prueba + 1} para dividir.")

    # Selección de episodios de prueba
    episodios_prueba = ep.iloc[-n_episodios_prueba:]
    corte_t = episodios_prueba["inicio_t"].min()

    logging.info("Episodios reservados para prueba (nunca vistos en entrenamiento):")
    logging.info("\n" + episodios_prueba[["causa", "inicio_t", "fin_t"]].to_string(index=False))
    logging.info(f"Corte temporal definido: entrenamiento con t < {corte_t}, prueba con t >= {corte_t}")

    # División temporal
    train = tel[tel["t"] < corte_t]
    test = tel[tel["t"] >= corte_t]

    # Distribución de clases
    logging.info(f"Distribución de clases en train: {train['label'].value_counts().to_dict()}")
    logging.info(f"Distribución de clases en test: {test['label'].value_counts().to_dict()}")

    return train, test, {"episodios_prueba": episodios_prueba, "corte_t": corte_t}

# def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame):
#     """
#     Entrena un MLP con Keras y evalúa contra test.
#     Devuelve el modelo, el historial de entrenamiento, el dataframe con predicciones y las métricas calculadas.
#     """
#     import logging

#     Xtr, ytr = train[PARAMS["FEATURES"]].values, train["label"].values
#     Xte, yte = test[PARAMS["FEATURES"]].values, test["label"].values

#     # --- Definir arquitectura ---
#     model = keras.models.Sequential([
#         keras.layers.Dense(64, activation="relu", input_shape=(Xtr.shape[1],)),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(32, activation="relu"),
#         keras.layers.Dense(len(PARAMS["CLASES"]), activation="softmax")
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=PARAMS["LEARNING_RATE"]),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     logging.info("Entrenando MLP con Keras/TensorFlow...")
#     history = model.fit(
#         Xtr, ytr,
#         epochs=PARAMS["EPOCHS"],
#         batch_size=PARAMS["BATCH_SIZE"],
#         validation_data=(Xte, yte),
#         verbose=0
#     )

#     # --- Predicciones ---
#     pred = np.argmax(model.predict(Xte), axis=1)

#     # --- Reporte ---
#     reporte = classification_report(yte, pred, target_names=PARAMS["CLASES"], zero_division=0)
#     logging.info("\n--- Reporte de clasificación MLP ---\n" + reporte)

#     matriz = confusion_matrix(yte, pred, labels=[0, 1, 2])
#     logging.info("Matriz de confusión MLP:\n%s", matriz)

#     # --- Métricas específicas ---
#     metrics = {
#         "accuracy": float((yte == pred).mean()),
#         "pred_counts": {str(k): int(v) for k, v in pd.Series(pred).value_counts().to_dict().items()}
#     }

#     return model, history, test.assign(pred=pred), metrics


def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame):
    """
    Entrena un MLP con Keras y evalúa contra test.
    Incluye normalización de features y class_weight para balancear clases.
    """
    # --- Features y labels ---
    Xtr, ytr = train[PARAMS["FEATURES"]].values, train["label"].values
    Xte, yte = test[PARAMS["FEATURES"]].values, test["label"].values

    # --- Normalización ---
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # --- Definir arquitectura ---
    model = keras.models.Sequential([
        keras.layers.Input(shape=(Xtr.shape[1],)),   # mejor práctica
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(len(PARAMS["CLASES"]), activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PARAMS["LEARNING_RATE"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --- Pesos de clase (dar más peso al fallo duro) ---
 #   class_weights = {0: 1.0, 1: 1.0, 2: 5.0}

    # Calcular pesos automáticamente
    classes = np.unique(ytr)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    class_weights = dict(zip(classes, class_weights_array))

    logging.info(f"Pesos de clase calculados: {class_weights}")

    logging.info("Entrenando MLP con Keras/TensorFlow...")
    history = model.fit(
        Xtr, ytr,
        epochs=PARAMS["EPOCHS"],
        batch_size=PARAMS["BATCH_SIZE"],
        validation_data=(Xte, yte),
        class_weight=class_weights,
        verbose=0
    )

    # --- Predicciones ---
    pred = np.argmax(model.predict(Xte), axis=1)

    # --- Reporte ---
    reporte = classification_report(yte, pred, target_names=PARAMS["CLASES"], zero_division=0)
    logging.info("\n--- Reporte de clasificación MLP ---\n" + reporte)

    matriz = confusion_matrix(yte, pred, labels=[0, 1, 2])
    logging.info("Matriz de confusión MLP:\n%s", matriz)

    # --- Métricas específicas ---
    metrics = {
        "accuracy": float((yte == pred).mean()),
        "pred_counts": {str(k): int(v) for k, v in pd.Series(pred).value_counts().to_dict().items()}
    }

    return model, history, test.assign(pred=pred), metrics


import matplotlib.pyplot as plt
import logging

def graficar_curvas(history, path_salida="curvas_entrenamiento.png"):
    """
    Gráfica curvas de pérdida y precisión del entrenamiento.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # --- Pérdida ---
    ax[0].plot(history.history.get("loss", []), label="train")
    ax[0].plot(history.history.get("val_loss", []), label="val")
    ax[0].set_title("Curva de pérdida")
    ax[0].set_xlabel("Épocas")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # --- Precisión ---
    ax[1].plot(history.history.get("accuracy", []), label="train")
    ax[1].plot(history.history.get("val_accuracy", []), label="val")
    ax[1].set_title("Curva de precisión")
    ax[1].set_xlabel("Épocas")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path_salida, dpi=130)
    plt.close(fig)
    logging.info(f"Gráfico de curvas de entrenamiento guardado en: {path_salida}")


def main():
    import argparse, json, sys, logging

    parser = argparse.ArgumentParser(description="Modelo MLP para detección de anomalías de sensor.")
    parser.add_argument("--telemetria", default="telemetria.csv")
    parser.add_argument("--episodios", default="episodios.csv")
    parser.add_argument("--episodios-prueba", type=int, default=2)
    parser.add_argument("--salida", default="outputs/", help="Directorio de salida para métricas y gráficas")
    args = parser.parse_args()

    # --- Cargar datos ---
    tel = cargar_y_preparar(args.telemetria)
    train, test, meta = dividir_train_test(tel, args.episodios, args.episodios_prueba)

    # --- Entrenar y evaluar MLP ---
    model, history, test_pred, metrics_mlp = entrenar_y_evaluar(train, test)

    # --- Graficar curvas de entrenamiento ---
    graficar_curvas(history, path_salida=f"{args.salida}/curvas_entrenamiento.png")

    # --- Exportar métricas MLP ---
    try:
        with open(f"{args.salida}/metrics_mlp.json", "w") as f:
            json.dump(metrics_mlp, f, indent=2)
        logging.info("Métricas MLP exportadas a metrics_mlp.json")
    except Exception as e:
        logging.warning(f"No se pudieron exportar métricas MLP: {e}")

    # --- Mostrar métricas en consola ---
    logging.info("=== Resultados MLP ===")
    for k, v in metrics_mlp.items():
        logging.info(f"{k}: {v}")

    logging.info("Pipeline completado con MLP. Hallazgo esperado: el modelo sigue siendo ciego al fallo duro si solo usa residual.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
