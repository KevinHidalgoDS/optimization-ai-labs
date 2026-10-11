#%%
import sys
import time

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import seaborn as sns
from src.utils import logger as log

logger = log.LOGGER
config = log.CONFIG
logger.info("Python %s on %s", sys.version, sys.platform)

# %matplotlib inline  # solo jupyter
plt.style.use("ggplot")
rng = np.random.default_rng(123)

# Constantes
CLASES = ["sano", "deriva", "fallo_duro"]
ROLLING_WINDOW = 5  # muestras (25 s a 5 s/muestra)
FEATURES = ["ltA_medido", "ltB_medido", "lt_residual", "residual_roll_mean", "residual_slope"]
UMBRAL_BASELINE = 0.15  # m — el mismo umbral de aviso ya implementado en el simulador (sin histéresis, por simplicidad)
file_path = Path(config.paths.data.raw) / "telemetria.csv"
if not file_path.exists():
    from src.config import PROJECT_PATH
    file_path = PROJECT_PATH / config.paths.data.raw / "telemetria.csv"

#%%
# --------------------
# Funciones auxiliares
# --------------------


def cargar_y_preparar(path_telemetria: str|None, tel: pd.DataFrame|None, rolling_window: int = 10) -> pd.DataFrame:
    """
    Carga la telemetría y construye el target de 3 clases basado en la verdad de terreno.

    Utiliza las columnas exportadas por el simulador (sensorA_fallo, sensorB_fallo,
    sensorA_deriva_activa, sensorB_deriva_activa) para clasificar el estado sin necesidad
    de cruzar datos por tiempo contra un CSV externo de episodios.

    Args:
        path_telemetria (str): Ruta al archivo CSV que contiene la telemetría del simulador.
        tel (pd.DataFrame): dataframe previo.
        rolling_window (int, optional): Tamaño de la ventana móvil para el cálculo de
            medias y pendientes. Por defecto es 10.

    Returns:
        pd.DataFrame: DataFrame ordenado cronológicamente con la nueva columna 'label'
            (0: sano, 1: deriva, 2: fallo_duro) y características móviles calculadas.

    Raises:
        FileNotFoundError: Si el archivo especificado en `path_telemetria` no existe.
        pd.errors.EmptyDataError: Si el archivo CSV está vacío.
        ValueError: Si faltan columnas requeridas en el archivo cargado.
        Exception: Para capturar y registrar cualquier otro error crítico en tiempo de ejecución.
    """
    # =========================================================================
    # Bloque 1: Carga de datos y control de acceso al archivo
    # =========================================================================
    logger.info(f"Iniciando la carga de telemetría desde: {path_telemetria}")
    try:
        if path_telemetria:
            tel = pd.read_csv(path_telemetria)
            logger.debug(f"Archivo CSV cargado exitosamente. Dimensiones: {tel.shape}")
        if not isinstance(tel, pd.DataFrame):
            raise ValueError(
                "Se debe proporcionar 'path_telemetria' o un DataFrame válido en 'tel'."
            )
        logger.debug(f"DataFrame listo. Dimensiones: {tel.shape}")
    except FileNotFoundError as e:
        logger.error(f"No se encontró el archivo de telemetría en la ruta: {path_telemetria, e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"El archivo CSV de telemetría está completamente vacío: {path_telemetria, e}")
        raise
    except Exception as e:
        logger.critical(f"Error inesperado al intentar leer el CSV: {str(e)}")
        raise

    # =========================================================================
    # Bloque 2: Validación de esquema y datos requeridos
    # =========================================================================
    logger.debug("Validando la integridad del esquema de columnas.")
    requeridas = [
        "t",
        "ltA_medido",
        "ltB_medido",
        "lt_residual",
        "sensorA_fallo",
        "sensorB_fallo",
        "sensorA_deriva_activa",
        "sensorB_deriva_activa",
    ]
    faltantes = [c for c in requeridas if c not in tel.columns]

    if faltantes:
        mensaje_error = (
            f"Faltan columnas requeridas: {faltantes}. Este script necesita "
            f"telemetría generada con una versión del simulador que incluya "
            f"columnas de verdad de terreno de sensores."
        )
        logger.error(mensaje_error)
        raise ValueError(mensaje_error)

    # =========================================================================
    # Bloque 3: Procesamiento temporal y etiquetado de clases (Target)
    # =========================================================================
    def clase(row: pd.Series) -> int:
        """Determina la clase objetivo evaluando booleanos de la fila actual."""
        if row["sensorA_fallo"] or row["sensorB_fallo"]:
            return 2  # fallo_duro
        if row["sensorA_deriva_activa"] or row["sensorB_deriva_activa"]:
            return 1  # deriva
        return 0  # sano

    logger.info("Ordenando datos temporalmente y generando variable objetivo ('label').")
    try:
        tel = tel.sort_values("t").reset_index(drop=True)
        tel["label"] = tel.apply(clase, axis=1)
        logger.debug("Columna 'label' generada correctamente.")
    except KeyError as e:
        logger.error(f"Llave no encontrada durante la asignación de clases: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error procesando la lógica de clases: {str(e)}")
        raise

    # =========================================================================
    # Bloque 4: Ingeniería de características (Feature Engineering)
    # =========================================================================
    logger.info(f"Calculando estadísticas de ventana móvil (window_size={rolling_window}).")
    try:
        # Promedio móvil del residual
        tel["residual_roll_mean"] = (
            tel["lt_residual"].rolling(window=rolling_window, min_periods=1).mean()
        )
        # Pendiente (diferencia) sobre la ventana
        tel["residual_slope"] = (
            tel["lt_residual"].diff(periods=rolling_window).fillna(0) / rolling_window
        )
        logger.debug("Cálculo de 'residual_roll_mean' y 'residual_slope' finalizado.")
    except TypeError as e:
        logger.warning(f"Error de tipos calculando ventanas móviles (revisar NaNs o strings): {e}")
        raise
    except Exception as e:
        logger.error(f"Fallo durante la ingeniería de características: {str(e)}")
        raise

    logger.info("La carga y preparación del DataFrame se completó con éxito.")
    return tel


def column_overview(data_frame: pd.DataFrame) -> pd.DataFrame:
    dtypes = data_frame.dtypes.astype(str).to_frame('dtype')
    missing_count = data_frame.isna().sum().to_frame('missing_count')
    unique_count = data_frame.nunique(dropna=False).to_frame('unique_count')
    col_overview = pd.concat([dtypes, missing_count, unique_count], axis=1)
    return col_overview


def codificar_variables_texto(df: pd.DataFrame, mapeo_ordinal: dict = None) -> pd.DataFrame:
    """
    Identifica las columnas de tipo string/object y las convierte a enteros en el mismo lugar.

    Permite mapear variables ordinales (con orden lógico) mediante un diccionario.
    Para las variables nominales (sin orden), aplica LabelEncoder automáticamente.

    Args:
        df (pd.DataFrame): DataFrame original con tipos de datos mixtos.
        mapeo_ordinal (dict, opcional): Diccionario con el orden específico para ciertas columnas.
            Ej: {'calidad': {'malo': 0, 'regular': 1, 'bueno': 2}}

    Returns:
        pd.DataFrame: DataFrame con las columnas de texto transformadas a numéricas (int).
    """
    # Trabajamos sobre una copia para no alterar el DataFrame original inadvertidamente
    df_procesado = df.copy()
    mapeo_ordinal = mapeo_ordinal or {}

    # Identificar columnas que son de tipo object o string
    columnas_texto = df_procesado.select_dtypes(include=['object', 'string']).columns

    for col in columnas_texto:
        # 1. Si la columna tiene un orden definido por el usuario (Ordinal)
        if col in mapeo_ordinal:
            df_procesado[col] = df_procesado[col].map(mapeo_ordinal[col])

            # Validar si quedaron valores nulos por categorías no mapeadas
            if df_procesado[col].isnull().any():
                raise ValueError(
                    f"Se encontraron categorías en la columna '{col}' que no están "
                    f"definidas en el diccionario de mapeo."
                )

        # 2. Si la columna no tiene orden (Nominal), codificar alfabéticamente
        else:
            encoder = LabelEncoder()
            # Convertimos a string primero por si hay valores mixtos/nulos
            df_procesado[col] = encoder.fit_transform(df_procesado[col].astype(str))

    # Forzar la conversión final a tipo entero
    for col in columnas_texto:
        df_procesado[col] = df_procesado[col].astype(int)

    return df_procesado


#%%
# ---------------------
# Lectura de telemetría
# ---------------------

logger.info("Leyendo archivo desde: %s", file_path)
df = pd.read_csv(file_path)
logger.info("Archivo cargado exitosamente con dimensiones: %s", df.shape)

df_tel = cargar_y_preparar(path_telemetria=None, tel=df)

column_overview(df_tel)

#%%
# ----------------------------------------
# División datos de entrenamiento y prueba
# ----------------------------------------

X = df_tel[FEATURES]
y = df_tel['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# ------------------------------
# Modelo: RandomForestClassifier
# ------------------------------

# Clasificador base
# -----------------

rf = RandomForestClassifier(
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42
)

#%%
# Optimización de hiperparámetros usando GridSearchCV
# -----------------

# Cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': np.arange(10, 200, 20),       # Cantidad de árboles en el bosque
    'max_depth': [10, 20],          # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5],          # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2],           # Mínimo de muestras que debe haber en una "hoja" final
    'criterion': ['gini','entropy']
}

# Configurar GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,               # El modelo que vamos a usar
    param_grid=param_grid,      # El diccionario con los parámetros a probar
    cv=5,                       # Validación cruzada de 5 pliegues (divide X_train en 5 partes)
    n_jobs=-1,                  # Usa todos los núcleos de tu procesador para ir más rápido
    scoring='accuracy'          # Métrica a optimizar (exactitud)
)

logger.info("Buscando los mejores parámetros... Esto puede tomar unos segundos.")
start_time = time.time()
grid_search.fit(X_train, y_train)
time_fit = time.time() - start_time

logger.info("\n¡Búsqueda completada!")
logger.info("Tiempo de entrenamiento usando búsqueda de grilla: %.2f (s)", time_fit)
logger.info("Mejores hiperparámetros encontrados: \n%s", grid_search.best_params_)
logger.info("Mejor exactitud durante la validación cruzada: %.2f%%", grid_search.best_score_ * 100)

# Guardar el mejor modelo listo para usarse
best_rf_model = grid_search.best_estimator_

# Usar el modelo para predecir las etiquetas de los datos de prueba
y_pred = best_rf_model.predict(X_test)

# Calcular la exactitud (accuracy) general
precision_final = metrics.accuracy_score(y_test, y_pred)
logger.info("Precisión exacta en el conjunto de prueba: %.2f%%", precision_final * 100)

# Generar un reporte detallado para ver cómo le fue con cada número
logger.info("Reporte de Clasificación Detallado:")
report = metrics.classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# redondear y ordenar
df_report = df_report.round(3)
print(df_report)

historial_metricas = []

historial_metricas.append({
    'Modelo': 'Random Forest (GridSearch)',
    'Accuracy': precision_final,
})

# Calcular la matriz de confusión con los datos reales (y_test) y las predicciones (y_pred)
cm = confusion_matrix(y_test, y_pred)

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crear el mapa de calor (heatmap)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(3), yticklabels=range(3),
    linewidths=.5, square=True
)

# Añadir títulos y etiquetas para que sea fácil de leer
plt.title('Matriz de Confusión - Random Forest - GridSearchCV', size=15)
plt.ylabel('Número Real', size=12)
plt.xlabel('Predicción del Modelo', size=12)


file_figures = Path(config.paths.reports.figures) / "rf_GridSearchCV.png"
if not file_figures.exists():
    from src.config import PROJECT_PATH
    file_figures = PROJECT_PATH / config.paths.reports.figures / "rf_GridSearchCV.png"
plt.savefig(file_figures, dpi=150, bbox_inches="tight")
# 5. Mostrar el gráfico
plt.show()

#%%
# Optimización de hiperparámetros usando RandomizedSearchCV
# -----------------

# Cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': np.arange(10, 200, 20),       # Cantidad de árboles en el bosque
    'max_depth': [10, 20],          # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5],          # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2],           # Mínimo de muestras que debe haber en una "hoja" final
    'criterion': ['gini','entropy']
}

# Configurar RandomizedSearchCV
rnd_search = RandomizedSearchCV(
    estimator=rf,                    # El modelo que vamos a usar
    param_distributions=param_grid,  # El diccionario con los parámetros a probar
    cv=5,                            # Validación cruzada de 5 pliegues (divide X_train en 5 partes)
    n_jobs=-1,                       # Usa todos los núcleos de tu procesador para ir más rápido
    scoring='accuracy',              # Métrica a optimizar (exactitud)
    n_iter=15
)

logger.info("Buscando los mejores parámetros... Esto puede tomar unos segundos.")
start_time = time.time()
rnd_search.fit(X_train, y_train)
time_fit = time.time() - start_time

logger.info("\n¡Búsqueda completada!")
logger.info("Tiempo de entrenamiento usando búsqueda de grilla: %.2f (s)", time_fit)
logger.info("Mejores hiperparámetros encontrados: \n%s", rnd_search.best_params_)
logger.info("Mejor exactitud durante la validación cruzada: %.2f%%", rnd_search.best_score_ * 100)

# Guardar el mejor modelo listo para usarse
rnd_best_rf_model = rnd_search.best_estimator_

# Usar el modelo para predecir las etiquetas de los datos de prueba
y_pred_rnd = rnd_best_rf_model.predict(X_test)

# Calcular la exactitud (accuracy) general
rnd_precision_final = metrics.accuracy_score(y_test, y_pred_rnd)
logger.info("Precisión exacta en el conjunto de prueba: %.2f%%", rnd_precision_final * 100)

# Generar un reporte detallado para ver cómo le fue con cada número
logger.info("Reporte de Clasificación Detallado:")
report_rnd = metrics.classification_report(y_test, y_pred_rnd, output_dict=True)
df_report_rnd = pd.DataFrame(report_rnd).transpose()

# redondear y ordenar
df_report_rnd = df_report_rnd.round(3)
print(df_report_rnd)

historial_metricas.append({
    'Modelo': 'Random Forest (RandomizedSearchCV)',
    'Accuracy': rnd_precision_final,
})

# Calcular la matriz de confusión con los datos reales (y_test) y las predicciones (y_pred)
cm = confusion_matrix(y_test, y_pred_rnd)

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crear el mapa de calor (heatmap)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(3), yticklabels=range(3),
    linewidths=.5, square=True
)

# Añadir títulos y etiquetas para que sea fácil de leer
plt.title('Matriz de Confusión - Random Forest - RandomizedSearchCV', size=15)
plt.ylabel('Número Real', size=12)
plt.xlabel('Predicción del Modelo', size=12)


file_figures = Path(config.paths.reports.figures) / "rf_RandomizedSearchCV.png"
if not file_figures.exists():
    from src.config import PROJECT_PATH
    file_figures = PROJECT_PATH / config.paths.reports.figures / "rf_RandomizedSearchCV.png"
plt.savefig(file_figures, dpi=150, bbox_inches="tight")
# 5. Mostrar el gráfico
plt.show()

#%%
# Optimización de hiperparámetros usando BayesSearchCV
# ----------------------------------------------------

# Cuadrícula de hiperparámetros
param_grid = {
    'n_estimators': np.arange(10, 200, 20),       # Cantidad de árboles en el bosque
    'max_depth': [10, 20],          # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5],          # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2],           # Mínimo de muestras que debe haber en una "hoja" final
    'criterion': ['gini','entropy']
}

# Configurar BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=rf,                    # El modelo que vamos a usar
    search_spaces=param_grid,  # El diccionario con los parámetros a probar
    cv=5,                            # Validación cruzada de 5 pliegues (divide X_train en 5 partes)
    n_jobs=-1,                       # Usa todos los núcleos de tu procesador para ir más rápido
    scoring='accuracy',              # Métrica a optimizar (exactitud)
    n_iter=50,
    random_state= 42
)

logger.info("Buscando los mejores parámetros... Esto puede tomar unos segundos.")
start_time = time.time()
bayes_search.fit(X_train, y_train)
time_fit = time.time() - start_time

logger.info("\n¡Búsqueda completada!")
logger.info("Tiempo de entrenamiento usando búsqueda de grilla: %.2f (s)", time_fit)
logger.info("Mejores hiperparámetros encontrados: \n%s", bayes_search.best_params_)
logger.info("Mejor exactitud durante la validación cruzada: %.2f%%", bayes_search.best_score_ * 100)

# Guardar el mejor modelo listo para usarse
bayes_best_rf_model = bayes_search.best_estimator_

# Usar el modelo para predecir las etiquetas de los datos de prueba
y_pred_bayes = bayes_best_rf_model.predict(X_test)

# Calcular la exactitud (accuracy) general
bayes_precision_final = metrics.accuracy_score(y_test, y_pred_bayes)
logger.info("Precisión exacta en el conjunto de prueba: %.2f%%", bayes_precision_final * 100)

# Generar un reporte detallado para ver cómo le fue con cada número
logger.info("Reporte de Clasificación Detallado:")
report_bayes = metrics.classification_report(y_test, y_pred_bayes, output_dict=True)
df_report_bayes = pd.DataFrame(report_bayes).transpose()

# redondear y ordenar
df_report_bayes = df_report_bayes.round(3)
print(df_report_bayes)

historial_metricas.append({
    'Modelo': 'Random Forest (BayesSearchCV)',
    'Accuracy': bayes_precision_final,
})

# Calcular la matriz de confusión con los datos reales (y_test) y las predicciones (y_pred)
cm = confusion_matrix(y_test, y_pred_bayes)

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crear el mapa de calor (heatmap)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=range(3), yticklabels=range(3),
    linewidths=.5, square=True
)

# Añadir títulos y etiquetas para que sea fácil de leer
plt.title('Matriz de Confusión - Random Forest - BayesSearchCV', size=15)
plt.ylabel('Número Real', size=12)
plt.xlabel('Predicción del Modelo', size=12)


file_figures = Path(config.paths.reports.figures) / "rf_BayesSearchCV.png"
if not file_figures.exists():
    from src.config import PROJECT_PATH
    file_figures = PROJECT_PATH / config.paths.reports.figures / "rf_BayesSearchCV.png"
plt.savefig(file_figures, dpi=150, bbox_inches="tight")
# 5. Mostrar el gráfico
plt.show()

#%%
# Comparaciones
# -------------
df_comparacion = pd.DataFrame(historial_metricas)
df_comparacion = df_comparacion.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(df_comparacion)
print(f"Mejores hiperparámetros encontrados: \n{grid_search.best_params_}")
print(f"Mejor exactitud durante la validación cruzada: {grid_search.best_score_ * 100}:.2f%")
print(df_report)
print(f"Mejores hiperparámetros encontrados: \n{rnd_search.best_params_}")
print(f"Mejor exactitud durante la validación cruzada: {rnd_search.best_score_ * 100}:.2f%")
print(df_report_rnd)
print(f"Mejores hiperparámetros encontrados: \n{bayes_search.best_params_}")
print(f"Mejor exactitud durante la validación cruzada: {bayes_search.best_score_ * 100}:.2f%")
print(df_report_bayes)