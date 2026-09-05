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

import os
import time

from src.utils import logger as log

logger = log.LOGGER

def limpieza_carpeta(path):
    logger.info("Eliminando archivos con más de 15 dias en %s", path)
    var_now = time.time()
    for f in os.listdir(path):
        if os.stat(os.path.join(path, f)).st_mtime < var_now - 15 * 86400 and os.path.isfile(
            os.path.join(path, f)
        ):
            os.remove(os.path.join(path, f))
