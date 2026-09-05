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

import logging
import re
import sys
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path

from src import config

VAR_PAD = "=" * 40
RESET = '\033[00m'
CONFIG = config.CONFIG

class StripAnsiFormatter(logging.Formatter):
    """Formatter que elimina secuencias ANSI antes de escribir al log."""

    _ansi_escape = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )

    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro eliminando los códigos ANSI.

        Args:
            record: Registro de logging.

        Returns:
            Mensaje formateado sin secuencias ANSI.
        """
        formatted = super().format(record)
        return self._ansi_escape.sub("", formatted)

# Asegurar carpeta
Path(config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Formatter para consola (mantiene colores ANSI)
# -------------------------------------------------------------------------
console_formatter = logging.Formatter(
    fmt=config.LOG_FORMAT,
    datefmt=config.DATE_FORMAT,
)

# -------------------------------------------------------------------------
# Formatter para archivo (elimina colores ANSI)
# -------------------------------------------------------------------------
file_formatter = StripAnsiFormatter(
    fmt=config.LOG_FORMAT,
    datefmt=config.DATE_FORMAT,
)

# -------------------------------------------------------------------------
# Console Handler
# -------------------------------------------------------------------------
console_handler = logging.StreamHandler()
console_handler.setLevel(config.LOG_LEVEL.upper())
console_handler.setFormatter(console_formatter)

# -------------------------------------------------------------------------
# File Handler
# -------------------------------------------------------------------------
file_handler = logging.FileHandler(
    config.LOG_FILE,
    encoding="utf-8"
)
file_handler.setLevel(config.LOG_LEVEL.upper())
file_handler.setFormatter(file_formatter)

# Logger root
LOGGER = logging.getLogger(config.main_name)
LOGGER.setLevel(config.LOG_LEVEL.upper())
LOGGER.handlers.clear()  # limpia duplicados
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)

# -------------------------------------------------------------------------
# Control de tiempo
# -------------------------------------------------------------------------

def print_header(
    msg: str,
    use_logging: bool | None = None,
    separator: str = '=',
    color: bool = True
) -> None:
    """
    Imprime un encabezado formateado de manera inteligente.

    Args:
        msg (str): Mensaje a mostrar en el encabezado
        use_logging (bool, optional): Si True usa logging.info,
            si False usa print. Si None, detecta automáticamente
        separator (str): Carácter usado como separador
        color (bool): Si True usa colores ANSI (solo en terminales compatibles)

    Returns:
        None
    """
    # Detectar automáticamente si usar logging o print
    if use_logging is None:
        use_logging = hasattr(logging, 'info') and logging.root.handlers

    # Verificar si el terminal soporta colores
    supports_color = color and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    # Construir mensaje
    if supports_color:
        # Usar colores solo si el terminal lo soporta
        color_code = '\033[96m'  # Cyan
        reset_code = '\033[00m'
        formatted_msg = f"{color_code}{msg}{reset_code}"
    else:
        formatted_msg = msg

    separator_line = separator * 80
    full_message = f"{formatted_msg}\n{separator_line}"

    # Imprimir usando el método apropiado
    if use_logging:
        logging.info(full_message)
    else:
        print(full_message)

class OutputMode(Enum):
    """Define los modos de salida disponibles para los logs de tiempo."""
    PRINT = 0
    LOG = 1
    PAD = 2
    COLOR = 3


def log_process_time(event_type: str, output_mode: OutputMode = OutputMode.LOG) -> datetime:
    """Captura el tiempo actual y lo imprime según el modo seleccionado.

    Args:
        event_type (str): Tipo de evento (ej. "inicio", "fin").
        output_mode (OutputMode, optional): Modo de impresión. Defaults to OutputMode.LOG.

    Returns:
        datetime: Tiempo actual en formato datetime.
    """
    now = datetime.now()
    formatted_time = now.strftime(config.DATE_FORMAT)
    base_message = f"Hora {event_type} proceso -> {formatted_time}"

    if output_mode == OutputMode.LOG:
        LOGGER.info(base_message)
    elif output_mode == OutputMode.PAD:
        print(VAR_PAD)
        print(f"## {base_message} ########")
        print(VAR_PAD)
    elif output_mode == OutputMode.COLOR:
        print(f"\033[92m{formatted_time} [INFO]:\033[00m {base_message}")
    else:
        raise ValueError(f"Modo de impresión no soportado: {output_mode}")

    return now


def calculate_runtime(start_time: datetime, end_time: datetime, output_mode: OutputMode = OutputMode.LOG) -> None:
    """Calcula y muestra el tiempo total de ejecución del proceso."""
    # Eliminar microsegundos para una visualización más limpia
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)

    total_time = str(end_time - start_time)
    base_message = f"El tiempo total de ejecución fue: {total_time}"

    if output_mode == OutputMode.LOG:
        LOGGER.info(base_message)
    elif output_mode == OutputMode.PAD:
        print(VAR_PAD)
        print(f"## {base_message} #########")
        print(VAR_PAD)
    elif output_mode == OutputMode.COLOR:
        formatted_now = datetime.now().strftime(config.DATE_FORMAT)
        print(f"\033[92m{formatted_now} [INFO]:\033[00m {base_message}")
    else:
        raise ValueError(f"Modo de impresión no soportado: {output_mode}")


def medir_tiempo(step_name: str, output_mode: OutputMode = OutputMode.LOG):
    """Decorador para medir el tiempo de ejecución de una función.

    Args:
        step_name (str): Nombre de la etapa del proceso.
        output_mode (OutputMode, optional): Modo de impresión. Defaults to OutputMode.LOG.
    """
    def decorator(funcion):
        @wraps(funcion)
        def wrapper(*args, **kwargs):

            print_header(step_name)

            start_time = log_process_time("inicio", output_mode)
            result = funcion(*args, **kwargs)
            end_time = log_process_time("fin", output_mode)

            calculate_runtime(start_time, end_time, output_mode)

            return result
        return wrapper
    return decorator


def bomper(file_path: str, mode: OutputMode = OutputMode.LOG) -> None:  # pragma no cover
    """Imprime o registra un encabezado (banner) centrado.

    Args:
        file_path (str): Ruta del archivo (ej. __file__) o nombre a mostrar.
        mode (OutputMode, optional): Modo de salida (0: Print, 1: Log). Defaults to LOG.

    Returns:
        None
    """
    # Extrae el nombre final de la ruta (funciona igual si ya es solo un nombre)
    base_name = Path(file_path).name

    # Agrega un espacio a los lados para que se vea mejor: " mi_script.py "
    header_text = f" {base_name} "

    # Magia de Python: centra el texto a 80 caracteres rellenando con '='
    var_bomper = header_text.center(80, '=')

    separator = "=" * 80

    if mode == OutputMode.PRINT:
        print(separator)
        print(var_bomper)
        print(separator)
    elif mode == OutputMode.LOG:
        LOGGER.info(separator)
        LOGGER.info(var_bomper)
        LOGGER.info(separator)

