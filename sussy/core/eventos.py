import json
import logging
from typing import Any, Dict, Optional
from urllib import error, request


class GestorEventos:
    """
    Responsable de avisar al exterior cuando arranca o termina una sesión
    de interés. Por ahora soporta:
      - REST simple vía HTTP POST.
      - Logging local (para entornos sin backend).
    """

    def __init__(
        self,
        backend_url: Optional[str],
        timeout: float = 3.0,
        simular_en_log: bool = True,
    ) -> None:
        self.backend_url = backend_url
        self.timeout = timeout
        self.simular_en_log = simular_en_log
        self.logger = logging.getLogger("sussy.eventos")

    def notificar_inicio(self, contexto: Dict[str, Any]) -> None:
        self._emitir_evento("inicio", contexto)

    def notificar_fin(self, contexto: Dict[str, Any]) -> None:
        self._emitir_evento("fin", contexto)

    def _emitir_evento(self, tipo: str, contexto: Dict[str, Any]) -> None:
        payload = {
            "tipo": tipo,
            "contexto": contexto,
        }

        if self.backend_url:
            data = json.dumps(payload).encode("utf-8")
            peticion = request.Request(
                self.backend_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(peticion, timeout=self.timeout) as respuesta:
                    self.logger.debug("Evento %s enviado (%s)", tipo, respuesta.status)
                return
            except error.URLError as exc:
                self.logger.warning("Fallo al notificar evento %s: %s", tipo, exc)

        if self.simular_en_log:
            self.logger.info("Evento %s: %s", tipo, json.dumps(payload))

