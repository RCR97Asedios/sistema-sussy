from __future__ import annotations

import logging
from typing import List

from sussy.config import Config
from sussy.core.deteccion import Detection


class EvaluadorRelevancia:
    """
    Evalúa los objetos obtenidos del detector de movimiento y decide si
    merecen pasar al pipeline principal. Su enfoque es heurístico para
    evitar falsos positivos típicos como ramas movidas por el viento.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("sussy.relevancia")

    def filtrar(self, detecciones: List[Detection], frame_shape) -> List[Detection]:
        if not detecciones:
            return []

        alto, ancho = frame_shape[:2]
        area_total = max(1, ancho * alto)
        filtradas: List[Detection] = []

        for det in detecciones:
            if det.get("clase") != "movimiento":
                det.setdefault("relevante", True)
                filtradas.append(det)
                continue

            evaluada = self._evaluar_movimiento(det, ancho, alto, area_total)
            if evaluada is None:
                continue  # Demasiado ruido, no lo propagamos
            filtradas.append(evaluada)

        return filtradas

    def _evaluar_movimiento(
        self,
        det: Detection,
        ancho_frame: int,
        alto_frame: int,
        area_total: int,
    ) -> Detection | None:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        area_px = det.get("area_px", w * h)
        area_rel = area_px / area_total
        velocidad = float(det.get("velocidad_px", 0.0))
        frames_vivos = int(det.get("frames_vivos", 0))

        margen_x = ancho_frame * Config.RELEVANCIA_BORDE_PCT
        margen_y = alto_frame * Config.RELEVANCIA_BORDE_PCT
        toca_borde = (
            x1 <= margen_x
            or y1 <= margen_y
            or x2 >= (ancho_frame - margen_x)
            or y2 >= (alto_frame - margen_y)
        )

        aspecto = max(w, h) / max(1, min(w, h))

        # Regla 1: descartar oscilaciones grandes pero lentas (ramas, paredes)
        if area_rel >= Config.RELEVANCIA_AREA_RAMA_MIN and velocidad < Config.RELEVANCIA_VEL_MIN:
            return None

        # Regla 2: si está pegado al borde y apenas se desplaza, es ruido
        if toca_borde and velocidad < (Config.RELEVANCIA_VEL_MIN * 1.5):
            return None

        # Regla 3: formas extremadamente alargadas que no avanzan lo suficiente
        if (
            aspecto >= Config.RELEVANCIA_ASPECTO_RAMAS
            and velocidad < (Config.RELEVANCIA_VEL_MIN * 1.2)
            and area_rel > Config.RELEVANCIA_AREA_DRON_MAX
        ):
            return None

        # Regla 4: mini-objetos en movimiento sostenido → posible dron
        if (
            area_rel <= Config.RELEVANCIA_AREA_DRON_MAX
            and velocidad >= Config.RELEVANCIA_VEL_MIN
            and frames_vivos >= Config.MOVIMIENTO_MIN_FRAMES
            and not toca_borde
        ):
            det["clase"] = "posible_dron"
            det["relevante"] = True
            det["score"] = max(det.get("score", 0.0), min(0.9, 0.35 + velocidad * 0.05))
            det["descripcion"] = "Movimiento compacto con patrón compatible con dron"
            return det

        # Por defecto lo marcamos como movimiento relevante para dejar rastro
        det["clase"] = det.get("clase") or "movimiento"
        det["relevante"] = True
        return det

