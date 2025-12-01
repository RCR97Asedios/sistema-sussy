from __future__ import annotations

import logging
import time
from typing import Optional, Union

import cv2

Logger = logging.Logger
Source = Union[str, int]


def normalizar_source(texto: Optional[str]) -> Optional[Source]:
    """
    Convierte los argumentos CLI en un tipo aceptable por OpenCV:
      - "0" → 0 (webcam)
      - resto → se devuelve tal cual
    """
    if texto is None:
        return None
    texto = texto.strip()
    if not texto:
        return None
    if texto.isdigit():
        return int(texto)
    return texto


def abrir_fuente_video(
    source: Source,
    reintentos: int = 1,
    delay_segundos: float = 1.0,
    logger: Optional[Logger] = None,
) -> cv2.VideoCapture:
    """
    Intenta abrir la fuente solicitada con varios reintentos controlados.
    Siempre devuelve un objeto VideoCapture; el caller debe comprobar isOpened().
    """
    logger = logger or logging.getLogger("sussy.fuentes")
    intentos = max(1, reintentos)

    for intento in range(1, intentos + 1):
        logger.info("Abriendo fuente (%s/%s): %s", intento, intentos, source)
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            logger.info("Fuente abierta correctamente.")
            return cap

        logger.warning("No se pudo abrir la fuente en el intento %s.", intento)
        cap.release()

        if intento < intentos:
            time.sleep(max(0.0, delay_segundos))

    # Como fallback devolvemos el último VideoCapture aunque esté cerrado;
    # el caller decidirá si aborta o si quiere reintentar manualmente.
    return cv2.VideoCapture(source)

