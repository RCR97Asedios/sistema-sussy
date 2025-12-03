from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DiagnosticoEstabilidad:
    desplazamiento_medio: float = 0.0
    ratio_perdida: float = 0.0
    puntos_validos: int = 0
    total_puntos: int = 0
    factor_escala: float = 1.0
    ratio_pixeles: float = 0.0


class MonitorEstabilidadCamara:
    """
    Vigila el movimiento global del frame a partir de flujo óptico de esquinas.
    Si la mediana de los desplazamientos supera un umbral (o se pierden
    demasiados puntos) asumimos que es la cámara la que se mueve y no los
    objetos de interés. En ese caso se devuelve True para permitir que el
    pipeline pause el detector de movimiento.
    """

    def __init__(
        self,
        escala: float = 0.5,
        max_desplazamiento_px: float = 6.0,
        min_puntos: int = 50,
        max_ratio_perdidos: float = 0.45,
        frames_inestables_para_disparar: int = 3,
        frames_estables_para_recuperar: int = 6,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 7,
        activar_inmediato: bool = False,
        max_cambio_escala: float = 0.05,
        max_ratio_diferencia: float = 0.35,
    ) -> None:
        self.escala = max(0.1, min(1.0, escala))
        self.max_desplazamiento_px = max_desplazamiento_px
        self.min_puntos = min_puntos
        self.max_ratio_perdidos = max_ratio_perdidos
        self.frames_inestables_para_disparar = frames_inestables_para_disparar
        self.frames_estables_para_recuperar = frames_estables_para_recuperar
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.activar_inmediato = activar_inmediato
        self.max_cambio_escala = max(0.0, max_cambio_escala)
        self.max_ratio_diferencia = max(0.0, min(1.0, max_ratio_diferencia))
        self._umbral_diff_zoom = 20

        self._prev_gray: np.ndarray | None = None
        self._frames_inestables = 0
        self._frames_estables = 0
        self._en_movimiento = False
        self._diagnostico = DiagnosticoEstabilidad()
        self._ultima_motivo = "estable"

    @property
    def en_movimiento(self) -> bool:
        return self._en_movimiento

    @property
    def diagnostico(self) -> DiagnosticoEstabilidad:
        return self._diagnostico

    @property
    def motivo(self) -> str:
        return self._ultima_motivo

    def resetear(self) -> None:
        self._prev_gray = None
        self._frames_inestables = 0
        self._frames_estables = 0
        self._en_movimiento = False
        self._diagnostico = DiagnosticoEstabilidad()
        self._ultima_motivo = "reset"

    def _preprocesar(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.escala != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.escala,
                fy=self.escala,
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def _disparar_inmediato(
        self,
        gray: np.ndarray,
        motivo: str,
        desplazamiento_mediano: float,
        ratio_perdida: float,
        puntos_validos: int,
        total_puntos: int,
        factor_escala: float,
        ratio_pixeles: float,
    ) -> bool:
        self._frames_inestables = self.frames_inestables_para_disparar
        self._frames_estables = 0
        self._en_movimiento = True
        self._diagnostico = DiagnosticoEstabilidad(
            desplazamiento_medio=desplazamiento_mediano,
            ratio_perdida=ratio_perdida,
            puntos_validos=puntos_validos,
            total_puntos=total_puntos,
            factor_escala=factor_escala,
            ratio_pixeles=ratio_pixeles,
        )
        self._ultima_motivo = motivo
        self._prev_gray = gray
        return True

    def actualizar(self, frame: np.ndarray) -> bool:
        gray = self._preprocesar(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._diagnostico = DiagnosticoEstabilidad()
            self._ultima_motivo = "inicial"
            return False

        puntos_prev = cv2.goodFeaturesToTrack(
            self._prev_gray,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        movimiento_global = False
        desplazamiento_mediano = 0.0
        ratio_perdida = 1.0
        puntos_validos = 0
        total_puntos = int(puntos_prev.shape[0]) if puntos_prev is not None else 0
        factor_escala = 1.0
        motivo_actual = "traslacion"
        ratio_pixeles = 0.0

        diff = cv2.absdiff(self._prev_gray, gray)
        if diff.size > 0:
            ratio_pixeles = float(
                np.count_nonzero(diff > self._umbral_diff_zoom)
            ) / float(diff.size)
            if ratio_pixeles > self.max_ratio_diferencia:
                return self._disparar_inmediato(
                    gray=gray,
                    motivo="zoom_diff",
                    desplazamiento_mediano=0.0,
                    ratio_perdida=ratio_perdida,
                    puntos_validos=0,
                    total_puntos=0,
                    factor_escala=1.0,
                    ratio_pixeles=ratio_pixeles,
                )

        if puntos_prev is None or total_puntos < self.min_puntos:
            return self._disparar_inmediato(
                gray=gray,
                motivo="sin_pts",
                desplazamiento_mediano=0.0,
                ratio_perdida=ratio_perdida,
                puntos_validos=0,
                total_puntos=total_puntos,
                factor_escala=1.0,
                ratio_pixeles=ratio_pixeles,
            )
        else:
            puntos_act, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                puntos_prev,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.03,
                ),
            )

            if puntos_act is None or status is None:
                movimiento_global = True
            else:
                good_new = puntos_act[status.reshape(-1) == 1]
                good_old = puntos_prev[status.reshape(-1) == 1]

                if good_new.size == 0:
                    movimiento_global = True
                else:
                    good_new = good_new.reshape(-1, 2)
                    good_old = good_old.reshape(-1, 2)
                    puntos_validos = good_new.shape[0]
                    ratio_perdida = 1.0 - (
                        puntos_validos / max(1, total_puntos)
                    )

                    desplazamientos = np.linalg.norm(
                        good_new - good_old, axis=1
                    )
                    if desplazamientos.size > 0:
                        desplazamiento_mediano = float(
                            np.median(desplazamientos)
                        )

                    if good_new.shape[0] >= 5:
                        try:
                            M, _ = cv2.estimateAffinePartial2D(
                                good_old,
                                good_new,
                                method=cv2.RANSAC,
                                ransacReprojThreshold=3.0,
                                maxIters=2000,
                                confidence=0.99,
                            )
                            if M is not None:
                                a, b = M[0, 0], M[0, 1]
                                c, d = M[1, 0], M[1, 1]
                                scale_x = float((a * a + b * b) ** 0.5)
                                scale_y = float((c * c + d * d) ** 0.5)
                                factor_escala = max(1e-3, (scale_x + scale_y) / 2.0)
                                if abs(factor_escala - 1.0) > self.max_cambio_escala:
                                    return self._disparar_inmediato(
                                        gray=gray,
                                        motivo="zoom",
                                        desplazamiento_mediano=desplazamiento_mediano,
                                        ratio_perdida=ratio_perdida,
                                        puntos_validos=puntos_validos,
                                        total_puntos=total_puntos,
                                        factor_escala=factor_escala,
                                        ratio_pixeles=ratio_pixeles,
                                    )
                        except cv2.error:
                            pass

                    if not movimiento_global:
                        if puntos_validos < self.min_puntos:
                            return self._disparar_inmediato(
                                gray=gray,
                                motivo="sin_pts",
                                desplazamiento_mediano=desplazamiento_mediano,
                                ratio_perdida=ratio_perdida,
                                puntos_validos=puntos_validos,
                                total_puntos=total_puntos,
                                factor_escala=factor_escala,
                                ratio_pixeles=ratio_pixeles,
                            )
                        if ratio_perdida > self.max_ratio_perdidos:
                            return self._disparar_inmediato(
                                gray=gray,
                                motivo="perdida",
                                desplazamiento_mediano=desplazamiento_mediano,
                                ratio_perdida=ratio_perdida,
                                puntos_validos=puntos_validos,
                                total_puntos=total_puntos,
                                factor_escala=factor_escala,
                                ratio_pixeles=ratio_pixeles,
                            )
                        if desplazamiento_mediano > self.max_desplazamiento_px:
                            return self._disparar_inmediato(
                                gray=gray,
                                motivo="traslacion",
                                desplazamiento_mediano=desplazamiento_mediano,
                                ratio_perdida=ratio_perdida,
                                puntos_validos=puntos_validos,
                                total_puntos=total_puntos,
                                factor_escala=factor_escala,
                                ratio_pixeles=ratio_pixeles,
                            )

        # Si llegamos aquí es que no se detectó un zoom ni un cambio global severo
        self._frames_estables += 1
        self._frames_inestables = 0
        self._ultima_motivo = "estable"
        self._en_movimiento = False
        self._diagnostico = DiagnosticoEstabilidad(
            desplazamiento_medio=desplazamiento_mediano,
            ratio_perdida=ratio_perdida,
            puntos_validos=puntos_validos,
            total_puntos=total_puntos,
            factor_escala=factor_escala,
            ratio_pixeles=ratio_pixeles,
        )
        self._prev_gray = gray
        return False

