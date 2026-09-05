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

import inspect
import os
import sys
from pathlib import Path
from time import localtime, strftime

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class PathsData(BaseModel):
    raw: str
    interim: str
    processed: str
    external: str

class PathsModels(BaseModel):
    trained: str

class PathsReports(BaseModel):
    figures: str

class Paths(BaseModel):
    data: PathsData
    models: PathsModels
    reports: PathsReports

class VarsEnvironment(BaseModel):
    nombre: str
    entorno: str
    ide: str

class Config(BaseModel):
    project: dict
    paths: Paths
    environment: VarsEnvironment


def load_config(path: str = "config/config.yaml") -> Config:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        ruta_config = Path(__file__).resolve().parent / "config.yaml"
        with ruta_config.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return Config(**raw)

CONFIG = load_config()

# -----------------------------------------------------------------------------
# DIRECTORIOS LOCALES
# -----------------------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent.parent
LOG_DIRECTORY = (
    PROJECT_PATH / "logs"
)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
)

LOG_FORMAT = (
    "%(asctime)s "
    "[%(levelname)s] "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# -----------------------------------------------------------------------------
# ARCHIVOS
# -----------------------------------------------------------------------------

def get_main_name() -> str:
    """
    Obtiene el nombre del archivo o notebook principal en ejecución.

    La función determina el contexto de ejecución (terminal, Jupyter, IPython)
    y devuelve el nombre base del archivo o notebook.

    Returns:
        str: Nombre del archivo o notebook principal sin extensión.
    """
    main_file = _get_main_file_path()

    if main_file:
        return _extract_name_from_file(main_file)

    return _get_fallback_name()


def _get_main_file_path() -> Path | None:
    """
    Obtiene la ruta del archivo principal en ejecución.

    Returns:
        Optional[Path]: Ruta del archivo principal o None si no se puede determinar.
    """
    main_module = sys.modules.get('__main__')
    if not main_module:
        return None

    main_file = getattr(main_module, '__file__', None)
    if not main_file:
        return None

    return Path(main_file)


def _extract_name_from_file(file_path: Path) -> str:
    """
    Extrae el nombre base del archivo según el contexto de ejecución.

    Args:
        file_path: Ruta del archivo principal.

    Returns:
        str: Nombre base del archivo o notebook.
    """
    file_str = str(file_path)

    if _is_terminal_execution(file_str):
        return file_path.stem

    notebook_name = _get_notebook_name()
    if notebook_name:
        return notebook_name

    return file_path.stem


def _is_terminal_execution(file_path_str: str) -> bool:
    """
    Determina si la ejecución es desde terminal (no Jupyter/IPython).

    Args:
        file_path_str: Ruta del archivo como cadena.

    Returns:
        bool: True si es ejecución desde terminal, False en caso contrario.
    """
    return 'ipykernel' not in file_path_str and 'ipython' not in file_path_str


def _get_notebook_name() -> str | None:
    """
    Intenta obtener el nombre del notebook en entorno Jupyter/IPython.

    Returns:
        Optional[str]: Nombre del notebook o None si no se puede obtener.
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if not ipython or not hasattr(ipython, 'config'):
            return None

        connection_file = ipython.config.get('IPKernelApp', {}).get(
            'connection_file', ''
        )
        if not connection_file:
            return None

        return _clean_notebook_name(connection_file)

    except (ImportError, AttributeError):
        return None


def _clean_notebook_name(connection_file_path: str) -> str:
    """
    Limpia y formatea el nombre del notebook desde la ruta de conexión.

    Args:
        connection_file_path: Ruta del archivo de conexión del kernel.

    Returns:
        str: Nombre del notebook limpio y formateado.
    """
    raw_name = Path(connection_file_path).stem
    # Eliminar prefijo 'kernel-' y reemplazar guiones por guiones bajos
    clean_name = raw_name.replace('kernel-', '').replace('-', '_')
    return clean_name


def _get_fallback_name() -> str:
    """
    Obtiene un nombre alternativo cuando no se puede determinar el principal.

    Intenta obtener información del entorno interactivo o usa el nombre
    del script actual como último recurso.

    Returns:
        str: Nombre alternativo para el contexto actual.
    """
    fallback_name = _get_interactive_kernel_name()
    if fallback_name:
        return fallback_name

    return Path(__file__).stem


def _get_interactive_kernel_name() -> str | None:
    """
    Obtiene el nombre del kernel en entorno interactivo (Jupyter/IPython).

    Returns:
        Optional[str]: Nombre del kernel o None si no es un entorno interactivo.
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if not ipython:
            return None

        connection_file = ipython.config.get('IPKernelApp', {}).get(
            'connection_file', ''
        )
        if not connection_file:
            return None

        # Formatear nombre del notebook desde kernel
        raw_name = Path(connection_file).stem.replace('kernel-', '')
        return f"notebook_{raw_name}"

    except (ImportError, AttributeError):
        return None

def get_main_name_with_pycharm_detection() -> str:
    """
    Obtiene el nombre del archivo principal con detección mejorada para PyCharm.

    Esta función especializada detecta si el código se ejecuta en PyCharm
    (incluyendo modo debug) y devuelve el nombre del script o un nombre
    alternativo cuando no se puede determinar.

    Returns:
        str: Nombre del archivo principal o un identificador alternativo.
    """
    pycharm_context = _detect_pycharm_context()

    if pycharm_context.is_pycharm:
        _log_pycharm_detection(pycharm_context)

    script_name = _get_script_name_from_sys_argv()
    if script_name:
        return script_name

    script_name = _get_script_name_from_main_module()
    if script_name:
        return script_name

    script_name = _get_script_name_from_stack_trace(pycharm_context.is_debug)
    if script_name:
        return script_name

    return _generate_fallback_name()


class _PyCharmContext:
    """
    Contexto de ejecución de PyCharm.

    Attributes:
        is_pycharm (bool): Indica si el entorno es PyCharm.
        is_debug (bool): Indica si está en modo debug.
        detected_var (Optional[str]): Variable de entorno que detectó PyCharm.
    """

    def __init__(self, is_pycharm: bool, is_debug: bool,
                 detected_var: str | None = None):
        self.is_pycharm = is_pycharm
        self.is_debug = is_debug
        self.detected_var = detected_var


def _detect_pycharm_context() -> _PyCharmContext:
    """
    Detecta el contexto de ejecución de PyCharm.

    Returns:
        _PyCharmContext: Contexto con información de detección.
    """
    detected_var = _detect_pycharm_environment_variable()
    is_pycharm = detected_var is not None
    is_debug = _is_debug_mode_active()

    return _PyCharmContext(is_pycharm, is_debug, detected_var)


def _detect_pycharm_environment_variable() -> str | None:
    """
    Detecta la presencia de variables de entorno de PyCharm.

    Returns:
        Optional[str]: Nombre de la variable de entorno detectada o None.
    """
    pycharm_vars = [
        'PYCHARM_HOSTED',
        'PYCHARM_HELPERS',
        'PYCHARM_DISPLAY_PORT',
        'PYCHARM_DEBUG',
        'PYDEVD_IPYTHON_COMPATIBLE'
    ]

    for env_var in pycharm_vars:
        if env_var in os.environ:
            return env_var

    return None


def _is_debug_mode_active() -> bool:
    """
    Determina si el modo debug está activo.

    Returns:
        bool: True si el debug está activo, False en caso contrario.
    """
    has_pydevd = 'pydevd' in sys.modules
    has_trace = getattr(sys, 'gettrace', lambda: None)() is not None

    return has_pydevd or has_trace


def _log_pycharm_detection(context: _PyCharmContext) -> None:
    """
    Registra la detección de PyCharm en la salida estándar.

    Args:
        context: Contexto de PyCharm detectado.
    """
    if context.detected_var:
        print(f"PyCharm detectado vía variable: {context.detected_var}")

    if context.is_debug:
        print("Modo debug de PyCharm detectado")


def _get_script_name_from_sys_argv() -> str | None:
    """
    Obtiene el nombre del script desde sys.argv (método preferido para PyCharm).

    Returns:
        Optional[str]: Nombre del script o None si no se puede obtener.
    """
    try:
        if not sys.argv or len(sys.argv) == 0:
            return None

        script_path = sys.argv[0]
        if not script_path:
            return None

        # Verificar si la ruta existe
        path_obj = Path(script_path)
        if path_obj.exists():
            return path_obj.stem

        # Extraer nombre sin ruta
        script_name = path_obj.stem
        if script_name and script_name != 'pycharm':
            return script_name

        return None

    except Exception:
        return None


def _get_script_name_from_main_module() -> str | None:
    """
    Obtiene el nombre del script desde el módulo __main__.

    Returns:
        Optional[str]: Nombre del script o None si no se puede obtener.
    """
    try:
        main_module = sys.modules.get('__main__')
        if not main_module or not hasattr(main_module, '__file__'):
            return None

        filepath = main_module.__file__
        if not filepath:
            return None

        # En PyCharm, a veces __file__ es el script actual
        if not filepath.endswith('pydevd.py'):
            return Path(filepath).stem

        return None

    except Exception:
        return None


def _get_script_name_from_stack_trace(is_debug: bool) -> str | None:
    """
    Obtiene el nombre del script desde el stack trace (útil en modo debug).

    Args:
        is_debug: Indica si el modo debug está activo.

    Returns:
        Optional[str]: Nombre del script o None si no se puede obtener.
    """
    if not is_debug:
        return None

    try:
        stack = inspect.stack()
        for frame in stack:
            filename = frame.filename
            if not filename:
                continue

            # Filtrar archivos de PyCharm y debug
            if _is_valid_source_file(filename):
                return Path(filename).stem

        return None

    except Exception:
        return None


def _is_valid_source_file(filename: str) -> bool:
    """
    Verifica si un archivo es una fuente válida (no de PyCharm o debug).

    Args:
        filename: Ruta del archivo a verificar.

    Returns:
        bool: True si es una fuente válida, False en caso contrario.
    """
    exclude_patterns = ['pydevd', 'pycharm', 'debug']
    filename_lower = filename.lower()

    for pattern in exclude_patterns:
        if pattern in filename_lower:
            return False

    return True


def _generate_fallback_name() -> str:
    """
    Genera un nombre alternativo cuando no se puede determinar el script.

    Returns:
        str: Nombre alternativo basado en usuario y timestamp.
    """
    try:
        import getpass
        user = getpass.getuser()
        timestamp = strftime('%Y%m%d%H%M%S', localtime())
        return f"pycharm_{user}_{timestamp}"

    except Exception:
        return "pycharm_script"


main_name = get_main_name_with_pycharm_detection() if CONFIG.environment.ide == "PyCharm" else get_main_name()
LOG_FILE = LOG_DIRECTORY / (
    f"{strftime('%Y%m%d%H%M%S', localtime())}_{main_name}.log"
)
