import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import time

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import torch  # type: ignore
except ImportError:
    torch = None

from sussy.core.deteccion import precargar_modelo, backend_activo
from sussy.config import Config
from sussy.core.entorno import configurar_entorno_privado
from sussy.core.pipeline import (
    FrameResult,
    SussyPipeline,
    presets_disponibles,
    presets_rendimiento_disponibles,
)
from sussy.core.texto import dibujar_texto

LOGGER = configurar_entorno_privado(logging.getLogger("sussy.main"))

PRESETS_CAMARA_DISPONIBLES = presets_disponibles()
PRESETS_RENDIMIENTO_DISPONIBLES = presets_rendimiento_disponibles()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema Sussy – Vista previa con pipeline básico (detección + tracking)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Fuente de vídeo (ruta, RTSP/HTTP o índice de webcam). Si no se indica se usa la de Config.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Alias legacy de --source para no romper scripts antiguos.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help=(
            "Procesar solo 1 de cada N fotogramas. "
            "Si no se indica, se usa Config.SKIP_FRAMES_DEFECTO (ajustable por preset)."
        ),
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Ruta de archivo CSV para registrar los tracks (opcional).",
    )
    if PRESETS_CAMARA_DISPONIBLES:
        parser.add_argument(
            "--cam-preset",
            type=str,
            choices=PRESETS_CAMARA_DISPONIBLES,
            help=(
                "Preset rápido de cámara (fija/orientable/movil/movil_plus). "
                "Sobrescribe Config antes de iniciar el pipeline."
            ),
        )
    if PRESETS_RENDIMIENTO_DISPONIBLES:
        parser.add_argument(
            "--perf-preset",
            type=str,
            choices=PRESETS_RENDIMIENTO_DISPONIBLES,
            help=(
                "Preset de rendimiento (minimo/equilibrado/maximo) para ajustar coste "
                "computacional: modelo, skip de frames y filtros IA."
            ),
        )
    return parser.parse_args()


def dibujar_ui(frame, pausado, frame_actual, total_frames):
    if not Config.MOSTRAR_UI:
        return {}

    alto, ancho = frame.shape[:2]
    
    # Configuración UI
    alto_barra = 60
    y_barra = alto - alto_barra
    
    # 1. Fondo semitransparente
    capa = frame.copy()
    cv2.rectangle(capa, (0, y_barra), (ancho, alto), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(capa, alpha, frame, 1 - alpha, 0, frame)
    
    # 2. Botón Play/Pause (Izquierda)
    btn_x, btn_y = 20, y_barra + 10
    btn_tam = 40
    
    # Dibujar icono
    color_icono = (255, 255, 255)
    if pausado:
        # Icono Play (Triángulo)
        pts = np.array([
            [btn_x + 10, btn_y + 5],
            [btn_x + 10, btn_y + 35],
            [btn_x + 35, btn_y + 20]
        ], np.int32)
        cv2.fillPoly(frame, [pts], color_icono)
    else:
        # Icono Pause (Dos barras)
        cv2.rectangle(frame, (btn_x + 10, btn_y + 5), (btn_x + 18, btn_y + 35), color_icono, -1)
        cv2.rectangle(frame, (btn_x + 22, btn_y + 5), (btn_x + 30, btn_y + 35), color_icono, -1)
        
    # 3. Línea de tiempo
    timeline_x = btn_x + btn_tam + 20
    timeline_y = y_barra + 30
    timeline_w = ancho - timeline_x - 30
    
    # Barra fondo
    cv2.rectangle(frame, (timeline_x, timeline_y - 2), (timeline_x + timeline_w, timeline_y + 2), (100, 100, 100), -1)
    
    # Barra progreso
    if total_frames > 0:
        ancho_progreso = int((frame_actual / total_frames) * timeline_w)
        cv2.rectangle(frame, (timeline_x, timeline_y - 2), (timeline_x + ancho_progreso, timeline_y + 2), (0, 255, 0), -1)
        
        # Círculo indicador
        cv2.circle(frame, (timeline_x + ancho_progreso, timeline_y), 8, (255, 255, 255), -1)

    return {
        "btn": (btn_x, btn_y, btn_tam, btn_tam),
        "timeline": (timeline_x, y_barra, timeline_w, alto_barra)
    }


def muestrear_rendimiento(cache: dict, intervalo: float = 1.0) -> dict:
    """
    Devuelve métricas básicas de rendimiento con cacheo ligero para no
    penalizar el loop de visualización.
    """
    ahora = time.monotonic()
    if cache and (ahora - cache.get("ts", 0) < intervalo):
        return cache

    cpu = psutil.cpu_percent(interval=None) if psutil else None
    mem = psutil.virtual_memory().percent if psutil else None

    gpu_txt = None
    if torch and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            mem_total = props.total_memory
            mem_usada = torch.cuda.memory_allocated(idx)
            gpu_txt = f"GPU {props.name}: {mem_usada/1e9:.1f}/{mem_total/1e9:.1f} GB"
        except Exception:
            gpu_txt = "GPU disponible"

    cache = {"ts": ahora, "cpu": cpu, "mem": mem, "gpu": gpu_txt}
    return cache


def dibujar_rendimiento(frame, ancho, alto, cache_rend):
    cache_rend = muestrear_rendimiento(cache_rend)

    partes = []
    if cache_rend.get("cpu") is not None:
        partes.append(f"CPU: {cache_rend['cpu']:.0f}%")
    if cache_rend.get("mem") is not None:
        partes.append(f"RAM: {cache_rend['mem']:.0f}%")
    if cache_rend.get("gpu"):
        partes.append(cache_rend["gpu"])

    if not partes:
        return cache_rend

    texto = " | ".join(partes)
    dibujar_texto(
        frame,
        texto,
        (20, alto - 20),
        color=(0, 255, 0),
        font_scale=0.6,
        thickness=2,
        font_path=Config.UI_FONT_PATH,
    )

    return cache_rend


def main() -> None:
    args = parse_args()

    print("Sistema Sussy – pipeline básico (INGESTA → DETECCIÓN → TRACKING → VISUALIZACIÓN)")
    try:
        precargar_modelo(Config.YOLO_MODELO)
        bk = backend_activo()
        if bk:
            LOGGER.info("Backend de inferencia activo: %s (%s)", bk.nombre, bk.descripcion)
            print(f"Backend de IA: {bk.nombre} – {bk.descripcion}")
    except Exception as exc:
        LOGGER.warning("No se pudo precargar el modelo: %s", exc)
    fuente_cli = args.source or args.video
    if args.video and not args.source:
        LOGGER.warning("El argumento --video quedará obsoleto; usa --source en su lugar.")

    pipeline = SussyPipeline(annotate=True)
    try:
        pipeline.start(
            fuente_cli,
            cam_preset=getattr(args, "cam_preset", None) or Config.CAMARA_PRESET_POR_DEFECTO,
            perf_preset=getattr(args, "perf_preset", None) or Config.RENDIMIENTO_PRESET_POR_DEFECTO,
            skip_frames=args.skip,
            log_csv=args.log_csv,
        )
    except Exception as exc:  # pragma: no cover - error inicial
        LOGGER.error("No se pudo iniciar el pipeline: %s", exc)
        print(f"Error al iniciar el pipeline: {exc}")
        return

    # Dimensiones
    cap = pipeline.cap
    assert cap is not None
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = pipeline.total_frames
    fps = pipeline.fps
    
    if total_frames <= 0:
        LOGGER.info("Fuente sin conteo total (probablemente streaming). Timeline relativo.")
    print(f"Resolución: {ancho}x{alto}, Frames: {total_frames if total_frames > 0 else 'desconocido'}, FPS: {fps:.2f}")

    print("Controles: [Espacio] Pausa/Play, [Click Botón] Pausa/Play, [Timeline] Buscar")
    print("Pulsa 'q' para salir.")

    ventana = "Sussy - Vista previa"
    if Config.MOSTRAR_UI or Config.MOSTRAR_TRACKS:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    
    pausado = False
    areas_ui = {}
    peticion_busqueda = -1
    resultado_actual: Optional[FrameResult] = None
    cache_rend = {}
    
    def callback_raton(event, x, y, flags, param):
        nonlocal pausado, peticion_busqueda
        if not Config.MOSTRAR_UI:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Chequear botón Play/Pause
            bx, by, bw, bh = areas_ui.get("btn", (0,0,0,0))
            if bx <= x <= bx + bw and by <= y <= by + bh:
                pausado = not pausado
                return

            # Chequear Timeline
            tx, ty, tw, th = areas_ui.get("timeline", (0,0,0,0))
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                # Calcular porcentaje
                rel_x = x - tx
                pct = max(0.0, min(1.0, rel_x / tw))
                peticion_busqueda = int(pct * total_frames)

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            # Arrastrar en timeline
            tx, ty, tw, th = areas_ui.get("timeline", (0,0,0,0))
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                rel_x = x - tx
                pct = max(0.0, min(1.0, rel_x / tw))
                peticion_busqueda = int(pct * total_frames)

    if Config.MOSTRAR_UI:
        cv2.setMouseCallback(ventana, callback_raton)

    # Prefetch inicial para mostrar algo aunque esté en pausa
    try:
        resultado_actual = pipeline.process_next()
    except Exception as exc:
        LOGGER.error("Error procesando el primer frame: %s", exc)
        return

    while True:
        # Gestionar seek solicitado desde la UI
        if peticion_busqueda >= 0 and total_frames > 0:
            pipeline.seek(peticion_busqueda)
            resultado_actual = pipeline.process_next()
            peticion_busqueda = -1
            pausado = True  # tras un seek solemos pausar para inspeccionar

        # Procesado normal si no está en pausa
        if not pausado:
            resultado_actual = pipeline.process_next()
            if resultado_actual.estado.get("finished"):
                print(f"Fin del vídeo. Frames procesados: {resultado_actual.estado.get('frame_idx', 0)}")
                break
                    
        # Frame a mostrar (si estamos en pausa reutilizamos el último)
        if resultado_actual is None:
            frame = np.zeros((alto, ancho, 3), dtype=np.uint8)
            tracks_actuales = []
            indice_frame_actual = 0
        else:
            frame = resultado_actual.frame.copy()
            tracks_actuales = resultado_actual.tracks
            indice_frame_actual = resultado_actual.estado.get("frame_idx", 0)
            
        # Overlay informativo
        if Config.MOSTRAR_UI:
            texto = f"Frame {indice_frame_actual}/{total_frames} - objs: {len(tracks_actuales)}"
            dibujar_texto(
                frame,
                texto,
                (ancho - 350, 40),
                color=(0, 255, 0) if not pausado else (0, 255, 255),
                font_scale=0.7,
                thickness=2,
                font_path=Config.UI_FONT_PATH,
            )

        # Dibujar UI propia (timeline + play/pause)
        areas_ui = dibujar_ui(frame, pausado, indice_frame_actual, total_frames)

        # Métricas de rendimiento (esquina inferior izquierda)
        cache_rend = dibujar_rendimiento(frame, ancho, alto, cache_rend)

        cv2.imshow(ventana, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            pausado = not pausado

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

