from __future__ import annotations

import logging
import os
import platform
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None
    _PIL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

_COMMON_FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

_FONT_PATH_CACHE: dict[str, Optional[str]] = {}
_WARNED_NO_FONT = False


def _needs_unicode(texto: str) -> bool:
    return any(ord(char) > 127 for char in texto)


def _normalize_font_key(font_path: Optional[str]) -> str:
    return font_path or "__auto__"


def _discover_font_path(font_path: Optional[str]) -> Optional[str]:
    key = _normalize_font_key(font_path)
    if key in _FONT_PATH_CACHE:
        return _FONT_PATH_CACHE[key]

    candidatos = []
    if font_path:
        candidatos.append(font_path)
    candidatos.extend(_COMMON_FONT_PATHS)

    # Añadir rutas específicas por plataforma
    sistema = platform.system().lower()
    if sistema == "windows":
        candidatos.extend(
            [
                r"C:\Windows\Fonts\Verdana.ttf",
                r"C:\Windows\Fonts\tahoma.ttf",
            ]
        )
    elif sistema == "darwin":
        candidatos.extend(
            [
                "/System/Library/Fonts/SFNSDisplay.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            ]
        )
    else:  # Linux
        candidatos.extend(
            [
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            ]
        )

    ruta_resuelta = None
    for ruta in candidatos:
        if ruta and os.path.exists(ruta):
            ruta_resuelta = ruta
            break

    _FONT_PATH_CACHE[key] = ruta_resuelta
    return ruta_resuelta


@lru_cache(maxsize=16)
def _obtener_fuente(font_path_key: str, font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    if not _PIL_AVAILABLE:
        return None

    ruta = _discover_font_path(None if font_path_key == "__auto__" else font_path_key)
    if not ruta:
        return None

    try:
        return ImageFont.truetype(ruta, font_size)
    except OSError:  # Fuente corrupta o no válida
        return None


def _to_ascii(texto: str) -> str:
    normalized = unicodedata.normalize("NFKD", texto)
    return normalized.encode("ascii", "ignore").decode("ascii") or texto


def _font_size_from_scale(font_scale: float) -> int:
    return max(12, int(round(font_scale * 36)))


def dibujar_texto(
    frame: np.ndarray,
    texto: str,
    posicion: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 2,
    font_path: Optional[str] = None,
) -> None:
    """
    Dibuja texto soportando caracteres acentuados cuando Pillow y una fuente
    TrueType están disponibles. Si no lo están, cae a ASCII puro para evitar
    los signos de interrogación dobles típicos de cv2.putText.
    """
    if not texto:
        return

    font_scale = max(0.1, float(font_scale))
    thickness = max(1, int(thickness))
    texto = str(texto)

    necesita_unicode = _needs_unicode(texto)
    if not necesita_unicode:
        cv2.putText(
            frame,
            texto,
            posicion,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return

    fuente = None
    if _PIL_AVAILABLE:
        font_key = _normalize_font_key(font_path)
        fuente = _obtener_fuente(font_key, _font_size_from_scale(font_scale))

    if fuente is None:
        global _WARNED_NO_FONT  # pragma: no cover
        if not _PIL_AVAILABLE and not _WARNED_NO_FONT:
            LOGGER.warning(
                "Pillow no está instalado; los textos se dibujarán sin acentos."
            )
            _WARNED_NO_FONT = True
        elif not _WARNED_NO_FONT:
            LOGGER.warning(
                "No se encontró una fuente TTF válida; los textos se dibujarán sin acentos."
            )
            _WARNED_NO_FONT = True

        texto_ascii = _to_ascii(texto)
        cv2.putText(
            frame,
            texto_ascii,
            posicion,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return

    rgb = (int(color[2]), int(color[1]), int(color[0]))
    x, y = posicion

    # Convertir a RGB para Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(imagen)

    ascent, descent = fuente.getmetrics()
    y_top = max(0, y - ascent)

    draw.text(
        (x, y_top),
        texto,
        font=fuente,
        fill=rgb,
        stroke_width=max(0, thickness - 1),
        stroke_fill=rgb,
    )

    frame_bgr = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    np.copyto(frame, frame_bgr)

