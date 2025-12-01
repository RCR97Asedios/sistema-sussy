from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EstadoActual:
    """Snapshot liviano para depurar o registrar el estado de la sesión."""

    activo: bool
    frames_consecutivos: int
    ultimo_objetivo: float
    inicio_alerta: Optional[float]


class GestorEstadoDeteccion:
    """
    Pequeño autómata que decide cuándo una racha de detecciones merece
    abrir/cerrar una "sesión" de interés (por ejemplo grabar o notificar).
    Mantiene su propio reloj para facilitar los tests.
    """

    def __init__(
        self,
        frames_necesarios: int,
        tiempo_rearme: float,
        duracion_max: float,
        reloj: Callable[[], float] | None = None,
    ) -> None:
        self.frames_necesarios = max(1, frames_necesarios)
        self.tiempo_rearme = max(0.1, tiempo_rearme)
        self.duracion_max = max(1.0, duracion_max)
        self._reloj = reloj or time.time

        self._frames_detectados = 0
        self._ultimo_objetivo = 0.0
        self._inicio_alerta: Optional[float] = None
        self._estado_activo = False

    def actualizar(self, hay_objetivo: bool) -> Optional[str]:
        """
        Actualiza el estado con la observación del frame actual.
        Devuelve "inicio" cuando se activa una sesión, "fin" cuando termina
        y None para el resto de casos.
        """
        ahora = self._reloj()
        evento: Optional[str] = None

        if hay_objetivo:
            self._frames_detectados += 1
            self._ultimo_objetivo = ahora

            if not self._estado_activo and self._frames_detectados >= self.frames_necesarios:
                self._estado_activo = True
                self._inicio_alerta = ahora
                evento = "inicio"
        else:
            # Si llevamos cierto tiempo sin ver objetivos, reiniciamos el contador
            if (ahora - self._ultimo_objetivo) > self.tiempo_rearme:
                self._frames_detectados = 0

        if self._estado_activo:
            # Fin por inactividad
            if not hay_objetivo and (ahora - self._ultimo_objetivo) > self.tiempo_rearme:
                evento = "fin"
                self._cerrar_alerta()
            # Fin por duración máxima
            elif self._inicio_alerta and (ahora - self._inicio_alerta) > self.duracion_max:
                evento = "fin"
                self._cerrar_alerta()

        return evento

    def _cerrar_alerta(self) -> None:
        self._estado_activo = False
        self._inicio_alerta = None
        self._frames_detectados = 0

    def snapshot(self) -> EstadoActual:
        """Devuelve un dataclass con la info relevante para logging."""
        return EstadoActual(
            activo=self._estado_activo,
            frames_consecutivos=self._frames_detectados,
            ultimo_objetivo=self._ultimo_objetivo,
            inicio_alerta=self._inicio_alerta,
        )

    @property
    def activo(self) -> bool:
        return self._estado_activo

