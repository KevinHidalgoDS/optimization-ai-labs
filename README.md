# 🧠 Optimización e Inteligencia Artificial - 3011103

[![Python Version](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Flake8 & SonarQube](https://img.shields.io/badge/linter-flake8_|_sonarqube-brightgreen.svg)](https://flake8.pycqa.org/)
[![Data Versioning: DVC](https://img.shields.io/badge/data-DVC-orange.svg)](https://dvc.org/)

## 📖 Descripción del Proyecto

Este repositorio establece la estructura estándar y las mejores prácticas para los proyectos,
prácticas y talleres del curso **Optimización e Inteligencia Artificial** del Departamento de
Ciencias de la Computación y la Decisión de la **Universidad Nacional de Colombia, Sede Medellín**.

El propósito de este repositorio es facilitar la implementación, experimentación y evaluación de
métodos de optimización clásica y metaheurística. Abarca desde algoritmos basados en gradiente
(gradiente descendente estocástico, Newton, QuasiNewton) hasta enfoques metaheurísticos (Algoritmos
Genéticos, PSO, Colonia de Hormigas). Estos métodos se aplican directamente a la sintonización de
hiperparámetros en modelos de Machine Learning, incluyendo Random Forest, Redes Neuronales, Deep
Learning y enfoques no supervisados.

---

## 📂 Estructura de Directorios

Para mantener un ciclo de vida analítico organizado, reproducible y listo para integrarse en
pipelines de datos, se recomienda la siguiente jerarquía de archivos:

```text
├── config/              # Parámetros del proyecto (rutas, hiperparámetros, credenciales de ejemplo).
├── data/
│   ├── raw/             # Datos originales e inmutables.
│   ├── interim/         # Datos en transformación, resultados parciales de limpieza.
│   ├── processed/       # Datos limpios y listos para modelado.
│   └── external/        # Datos de terceros o fuentes externas.
├── docs/                # Documentación del proyecto (Sphinx/MkDocs), guías y referencias.
├── models/              # Modelos entrenados y serializados (ej. .pkl, .h5).
│   ├── trained/         # Modelos finales listos para inferencia.
│   ├── checkpoints/     # Puntos de guardado durante entrenamiento.
├── notebooks/           # Jupyter Notebooks para exploración y visualización (nombrados secuencialmente).
├── src/                 # Código fuente principal del proyecto.
│   ├── __init__.py
│   ├── data/            # Scripts para ingesta y transformación de datos.
│   ├── features/        # Scripts de feature engineering.
│   ├── models/          # Scripts para entrenamiento, optimización y predicción.
│   └── visualization/   # Generación de gráficos y pósteres digitales.
├── tests/               # Pruebas unitarias y de integración.
├── dashboards/          # Aplicaciones interactivas.
├── reports/             # Resultados y comunicación.
│   ├── figures/         # imágenes generadas por scripts/notebooks (versionadas o regenerables).
├── scripts/             # Automatización.
├── .gitignore           # Archivos ignorados por Git.
├── dvc.yaml             # Pipeline de versionado de datos y modelos.
├── environment.yml      # Dependencias para Conda.
├── requirements.txt     # Dependencias para pip.
└── README.md            # Este archivo.
```

---

## ⚙️ Requisitos y Dependencias

Para las sesiones prácticas y el desarrollo general, se recomienda el uso de **Anaconda** para la
gestión de entornos virtuales o **Google Colab** para la ejecución en la nube.

El proyecto gestiona sus dependencias de la siguiente forma:

| Herramienta | Archivo            | Propósito principal                                                                              |
| :---------- | :----------------- | :----------------------------------------------------------------------------------------------- |
| **Conda**   | `environment.yml`  | Ideal para resolver dependencias complejas en ciencia de datos (ej. librerías con binarios C++). |
| **Pip**     | `requirements.txt` | Instalación estándar de paquetes de Python en contenedores o entornos ligeros.                   |

**Librerías principales requeridas:**

- `numpy`, `pandas`, `scipy` (Manipulación matemática y de datos).
- `scikit-learn`, `tensorflow` / `pytorch` (Implementación de Random Forest y Redes Neuronales).
- `matplotlib`, `seaborn` (Visualización de convergencia).

---

## 🚀 Instalación y Configuración

Sigue estos pasos para replicar el entorno de desarrollo de manera consistente. Es imperativo tener
el entorno debidamente configurado antes de iniciar las ejecuciones prácticas.

**1. Clonar el repositorio:**

```bash
git clone https://github.com/KevinHidalgoDS/optimizacion-ai-labs.git
cd optimizacion-ai-labs
```

**2. Crear y activar el entorno virtual (Recomendado: Anaconda):**

```bash
conda env create -f environment.yml
conda activate optia_env
```

**3. (Alternativa) Instalación vía pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💻 Uso y Ejecución

El código está modularizado para separar las pruebas interactivas de la ejecución en lotes.

- **Exploración:** Utiliza la carpeta `notebooks/` para experimentación inicial o al ejecutar
  prácticas sincrónicas en **Google Colab**.
- **Entrenamiento:** Para correr optimizaciones completas y simulaciones robustas (como
  simulaciones de Monte Carlo o búsquedas exhaustivas en grilla), ejecuta los módulos directamente
  desde la terminal:

```bash
# Ejemplo: Optimización de Random Forest (Práctica 1)
python src/models/train_rf_optimization.py --method grid_search

# Ejemplo: Optimización de Redes Neuronales (Práctica 2)
python src/models/train_nn_optimization.py --method gradient_descent
```

---

## 📏 Estándares de Código

La legibilidad y el "clean-code" son fundamentales, especialmente al realizar tareas de refactoring
algorítmico o al orquestar modelos analíticos para entornos productivos.

1. **Estilo base:** Todo el código en Python debe seguir el estándar **PEP 8**.
2. **Formateadores:** Se debe utilizar **Black** (longitud de línea de 88 caracteres) para unificar
   el estilo de forma automática.
3. **Linters:** Se emplea **Flake8** para identificar violaciones de estilo. En integraciones más
   avanzadas y despliegues, es altamente recomendado integrar **SonarQube** en el flujo local para
   detectar vulnerabilidades, asegurar una correcta parametrización del linter y evitar errores de
   _backtracking_ al optimizar expresiones regulares complejas dentro de funciones de parseo.
4. **Docstrings:** Documentar clases y funciones utilizando el formato de _NumPy_ o _Google_.

**Ejemplo de formato de función:**

```python
def optimize_hyperparameters(model, param_grid: dict) -> dict:
    """
    Optimiza los hiperparámetros del modelo usando búsqueda en grilla.

    Args:
        model: Estimador base de machine learning.
        param_grid (dict): Diccionario con los parámetros a evaluar.

    Returns:
        dict: Mejores hiperparámetros encontrados.
    """
    pass
```

---

## 📦 Versionado de Datos y Modelos

Nunca incluyas datasets grandes (`.csv`, `.parquet`) ni artefactos de modelos (`.h5`, `.pkl`)
directamente en Git.

- **DVC (Data Version Control):** Utiliza DVC para rastrear cambios en los datos. Los archivos
  `.dvc` se añaden a Git, mientras que los datos reales se almacenan en un _remote storage_ (como
  un blob storage de nube empresarial, ej. Azure Blob Storage administrado con identidades locales
  para desarrollo).
- **Git LFS (Large File Storage):** Alternativa para versionar binarios grandes directamente
  vinculados al repositorio.

---

## 🧪 Testing y Validación

La validación es crítica para asegurar que las funciones matemáticas (ej. derivadas del gradiente)
converjan correctamente.

- Estructura las pruebas dentro de la carpeta `tests/`.
- Utiliza **Pytest** para la ejecución de pruebas unitarias.

```bash
# Ejecutar todas las pruebas
pytest tests/
```

---

## 📚 Documentación

- Mantén el `README.md` actualizado frente a nuevos requerimientos del proyecto.
- Para el **Proyecto de Clase**, la entrega requiere documentar la implementación en Python,
  generar un póster digital y preparar una presentación final. [cite: 1] Los recursos visuales para
  esto deben almacenarse en la carpeta `docs/` o `src/visualization/`.
- Comenta de manera justificada las decisiones complejas en el código (ej. por qué se eligió un
  hiperparámetro o se modificó el "learning rate" en un método quasi-Newton).

---

## 🤝 Contribuciones

Para mantener la integridad del código al trabajar en equipo:

1. Crea una rama para tu feature: `git checkout -b feature/algoritmo-genetico`
2. Realiza commits descriptivos y atómicos.
3. Asegúrate de que el código pase el linter localmente antes de subir los cambios
   (`black . && flake8 .`).
4. Abre un Pull Request (PR) y solicita revisión de al menos un compañero antes del merge.

---

## 📄 Licencia y Atribuciones

Este material está estructurado para fines académicos de la **Universidad Nacional de Colombia,
Sede Medellín**.

**Licencia sugerida para el código:** MIT License (permite uso, modificación y distribución). Por
favor, asegúrate de referenciar adecuadamente la literatura o los fragmentos de código reutilizados
en tus módulos u optimizadores.

