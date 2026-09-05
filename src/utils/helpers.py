#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config
Configuraciones generales del laboratorio.

Proyecto: optimizacion-ia

Tema: Configuraciones variables generales

Programa: config.py

Soporte: kfhidalgoh@unal.edu.co

version: 1.0.0

lenguaje: Python 3.14.5

CD: 20260902

LUD: 20260902

Comentarios:
    - 2026-09-02 Kevin Hidalgo -> creación.
"""

__authors__ = ["Kevin Hidalgo"]
__contact__ = "kfhidalgoh@unal.edu.co"
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Kevin Hidalgo"]
__email__ = "kfhidalgoh@unal.edu.co"
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-09-02"

import random
from datetime import datetime, timedelta


def fecha_actual_a_entero():
    """
    Captura la fecha y hora actual y la convierte a un número entero en formato yyyymmddHHMMSS.
    Ejemplo: 2025-08-12 17:05:30 -> 20250812170530

    Returns:
        int: Fecha y hora actual convertida a número entero
    """
    # Obtener fecha y hora actual
    ahora = datetime.now()

    # Formatear como string sin separadores
    fecha_str = ahora.strftime("%Y%m%d%H%M%S")

    # Convertir a entero
    return int(fecha_str)

def generar_cadena_fecha_hora_aleatoria(
    fecha_inicio: str = "1970-01-01", fecha_fin: str = "2037-12-31"
) -> str:
    """
    Genera una cadena de fecha y hora aleatoria con precisión de microsegundos dentro de un rango
    especificado.

    Esta función crea una fecha y hora aleatoria entre las fechas de inicio y fin proporcionadas,
    formateada como una cadena con precisión de microsegundos. Si no se proporcionan fechas, utiliza
    un rango predeterminado razonable desde 1970-01-01 hasta 2037-12-31.

    Args:
        fecha_inicio: Cadena opcional que representa la fecha de inicio en formato 'YYYY-MM-DD'.
                      Por defecto es '1970-01-01'.
        fecha_fin: Cadena opcional que representa la fecha de fin en formato 'YYYY-MM-DD'.
                    Por defecto es '2037-12-31'.

    Returns:
        str: Cadena de fecha y hora formateada en formato '%Y-%m-%d %H:%M:%S.%f'.

    Raises:
        ValueError: Si fecha_inicio o fecha_fin no están en formato 'YYYY-MM-DD',
                    o si fecha_inicio es posterior a fecha_fin.

    Ejemplo:
        >>> generar_cadena_fecha_hora_aleatoria()
        '2023-07-15 14:32:45.123456'
        >>> generar_cadena_fecha_hora_aleatoria('2020-01-01', '2020-12-31')
        '2020-06-15 09:45:22.654321'
    """
    try:
        # Convertir cadenas de fecha a objetos datetime
        fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        fecha_fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            "Formato de fecha inválido. Por favor use el formato 'YYYY-MM-DD' para fecha_inicio y "
            "fecha_fin."
        ) from e

    # Validar rango de fechas
    if fecha_inicio_dt > fecha_fin_dt:
        raise ValueError("fecha_inicio debe ser anterior o igual a fecha_fin.")

    # Calcular timestamp aleatorio entre fechas de inicio y fin
    tiempo_entre_fechas = fecha_fin_dt - fecha_inicio_dt
    dias_entre_fechas = tiempo_entre_fechas.days
    numero_aleatorio_de_dias = random.randrange(dias_entre_fechas + 1)  # +1 para incluir fecha fin
    fecha_aleatoria = fecha_inicio_dt + timedelta(days=numero_aleatorio_de_dias)

    # Añadir componente de tiempo aleatorio con precisión de microsegundos
    segundos_aleatorios = random.randint(0, 86399)  # 0 a 86399 segundos en un día
    microsegundos_aleatorios = random.randint(0, 999999)  # Rango completo de microsegundos
    tiempo_aleatorio = timedelta(
        seconds=segundos_aleatorios, microseconds=microsegundos_aleatorios
    )

    # Combinar componentes de fecha y tiempo
    fecha_hora_aleatoria = fecha_aleatoria + tiempo_aleatorio

    # Formatear y retornar la cadena de fecha y hora
    return fecha_hora_aleatoria.strftime("%Y-%m-%d %H:%M:%S.%f")
