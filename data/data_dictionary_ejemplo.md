# Diccionario de Datos y Estructura del Directorio `data`

Este documento describe la organización de la carpeta `data` del proyecto, así como un diccionario
de datos de ejemplo para los conjuntos de datos procesados. La estructura sigue las mejores
prácticas de proyectos de ciencia de datos (como *Cookiecutter Data Science*).

---

## 📂 Estructura de Carpetas

El directorio `data` está dividido en las siguientes subcarpetas según el nivel de procesamiento de
la información:


|              Directorio              |                                                             Descripción                                                             |                                   Ejemplo                                    |
|:------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|      `data/raw/` (Datos Crudos)      |            Los datos originales, inmutables.  **Nunca**  deben editarse manualmente ni   sobrescribirse mediante código.            |   ` ventas_2026_crudo.csv `  ( Exportación directa de la base de datos ).    |
|  `data/external/` (Datos Externos)   |                      Datos de fuentes de terceros o APIs públicas que complementan nuestros datos   internos.                       |        ` inflacion_mensual_banrep.xlsx ` ,  ` datos_clima_api.json `.        |
| `data/interim/` (Datos Intermedios)  |    Datos que han sufrido alguna transformación, limpieza inicial o cruce, pero que   aún no están listos para el modelado final.    | ` ventas_sin_nulos.parquet `  ( Eliminación de duplicados y valores nulos ). |
| `data/processed/` (Datos Procesados) | Conjuntos de datos finales y canónicos listos para el entrenamiento de modelos de   Machine Learning o visualización en dashboards. |                     ` features_clientes_modelo_v1.csv `.                     |
|        ` data_dictionary.md `        |                   Documento que describe los conjuntos de datos, archivos y variables utilizados en el proyecto.                    |                                                                              |

---

## ➡️ Flujo general

```text
external/
    │
    ▼
  raw/
    │
    ▼
 interim/
    │
    ▼
processed/
```

---

## 📖 Diccionario de Datos (Ejemplo Dummy)

A continuación, se detalla la estructura del archivo final procesado que se utiliza para entrenar el
modelo de predicción de abandono (Churn).

**Archivo:** `data/processed/clientes_churn_preparado.csv`  
**Descripción:** Contiene las características de los clientes y la variable objetivo indicando si
abandonaron el servicio en los últimos 30 días.
**Formato:** CSV

| Nombre de la Columna | Tipo de Dato | Descripción | Valores Permitidos / Rango |
| :--- | :---: | :--- | :--- |
| `id_cliente` | `Cadena (String)` | Identificador único del cliente. | Ej: `CUS-100234` |
| `edad` | `Entero (Int)` | Edad del cliente en años. | `18` a `100` |
| `genero` | `Categoría` | Género reportado por el cliente. | `M`, `F`, `Otro` |
| `ingreso_mensual` | `Flotante (Float)` | Ingresos mensuales estimados en USD. | `0.0` a `50000.0` |
| `meses_antiguedad` | `Entero (Int)` | Tiempo en meses desde el registro inicial. | `0` a `120` |
| `plan_suscripcion` | `Categoría` | Tipo de plan actual del cliente. | `Basico`, `Estandar`, `Premium` |
| `gasto_total` | `Flotante (Float)` | Dinero total gastado históricamente en la plataforma. | `>= 0.0` |
| `flag_churn` (Target)| `Booleano (Int)` | Variable objetivo: 1 si el cliente se dio de baja, 0 si continúa activo. | `0`, `1` |

---

## 📜 Reglas de calidad

Los datos procesados deben cumplir, como mínimo, las siguientes reglas:

| Regla | Descripción |
|---|---|
| Unicidad | `id_garantia` debe ser único. |
| NIT | Debe cumplir el formato definido para NIT. |
| Fechas | `fecha_inicio` debe ser menor o igual a `fecha_fin`. |
| Valor | `valor_garantia` debe ser mayor que cero. |
| Campos obligatorios | Los campos marcados como obligatorios no deben contener valores nulos. |
| Estado | `estado` debe pertenecer al catálogo permitido. |

---

## 📋 Convenciones

### Fechas

Las fechas deben almacenarse utilizando el formato:

```text
YYYY-MM-DD
```

Ejemplo:

```text
2026-08-27
```

### Valores monetarios

Los valores monetarios deben almacenarse como valores numéricos, sin símbolos de moneda ni separadores de miles.

Ejemplo:

```text
150000000.00
```

### Codificación

Los archivos de texto deben utilizar:

```text
UTF-8
```

---

## 🔗 Trazabilidad

| Dataset origen | Dataset destino | Proceso |
|---|---|---|
| `raw/garantias_bancarias.csv` | `interim/garantias_normalizadas.csv` | Limpieza y normalización |
| `interim/garantias_normalizadas.csv` | `processed/garantias_procesadas.parquet` | Transformación y validación |
| `external/catalogo_entidades.csv` | `processed/garantias_procesadas.parquet` | Homologación de entidades |

---

## 👤 Responsable y actualización

| Propiedad | Valor |
|---|---|
| Responsable | Equipo de Datos |
| Frecuencia de actualización | Semanal |
| Última actualización | 2026-08-27 |
| Versión | 1.0 |

---

*Nota: Este documento debe actualizarse cada vez que se agreguen nuevas variables al conjunto de
datos procesados o se integren nuevas fuentes externas.*
