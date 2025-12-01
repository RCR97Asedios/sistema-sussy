import logging
import os
from typing import Optional

from sussy.config import Config

_ENTORNO_INICIALIZADO = False


def configurar_entorno_privado(extra_logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Configura variables de entorno y logging globales para ejecutar Sussy
    en modo "privado" (sin telemetría externa). Esta función es idempotente
    para que los distintos entry-points puedan invocarla sin miedo.
    """
    global _ENTORNO_INICIALIZADO

    if Config.DESACTIVAR_TELEMETRIA:
        for clave, valor in Config.TELEMETRIA_VARS.items():
            os.environ.setdefault(clave, valor)

    # Configuración de logging unificada
    nivel = getattr(logging, Config.NIVEL_LOG.upper(), logging.INFO)
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    _ENTORNO_INICIALIZADO = True

    logger = extra_logger or logging.getLogger("sussy")
    if Config.DESACTIVAR_TELEMETRIA and extra_logger is None:
        logger.debug("Telemetría externa deshabilitada para esta sesión.")
    return logger


def entorno_inicializado() -> bool:
    """Permite conocer si ya se ejecutó la configuración básica."""
    return _ENTORNO_INICIALIZADO

