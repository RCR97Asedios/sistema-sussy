import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("sussy.backends")


@dataclass
class BackendInfo:
    """
    Describe un backend de inferencia disponible.
    nombre: etiqueta corta (cuda/mps/onnx/cpu)
    tipo: pila principal (torch u onnx)
    dispositivo: identificador para mover el modelo (p.ej. cuda:0, cpu)
    modelo_path: ruta del modelo a cargar con ese backend
    descripcion: texto informativo para logs/UI
    """

    nombre: str
    tipo: str
    dispositivo: str
    modelo_path: str
    descripcion: str


def _inferir_onnx_path(modelo_pt: str, modelo_onnx_cfg: Optional[str]) -> Optional[str]:
    """Usa la ruta configurada o, si no existe, intenta <modelo>.onnx en el mismo directorio."""
    if modelo_onnx_cfg and os.path.isfile(modelo_onnx_cfg):
        return modelo_onnx_cfg

    if modelo_pt:
        base, _ = os.path.splitext(modelo_pt)
        cand = f"{base}.onnx"
        if os.path.isfile(cand):
            return cand

    return None


def detectar_backends(
    preferencias: List[str],
    modelo_pt: str,
    modelo_onnx_cfg: Optional[str] = None,
) -> Tuple[Optional[BackendInfo], List[BackendInfo]]:
    """
    Construye una lista de backends disponibles y devuelve el primero
    que coincida con las preferencias (o el primero disponible).
    """
    candidatos: List[BackendInfo] = []

    torch_mod = None
    try:
        import torch as _torch  # type: ignore

        torch_mod = _torch
    except Exception:
        torch_mod = None

    # Backends Torch (CUDA/MPS/CPU)
    if torch_mod:
        if torch_mod.cuda.is_available():
            try:
                props = torch_mod.cuda.get_device_properties(0)
                desc = f"CUDA {props.name} (cc {props.major}.{props.minor})"
            except Exception:
                desc = "CUDA disponible"
            candidatos.append(
                BackendInfo(
                    nombre="cuda",
                    tipo="torch",
                    dispositivo="cuda:0",
                    modelo_path=modelo_pt,
                    descripcion=desc,
                )
            )

        if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():  # type: ignore
            candidatos.append(
                BackendInfo(
                    nombre="mps",
                    tipo="torch",
                    dispositivo="mps",
                    modelo_path=modelo_pt,
                    descripcion="Apple MPS",
                )
            )

        candidatos.append(
            BackendInfo(
                nombre="cpu",
                tipo="torch",
                dispositivo="cpu",
                modelo_path=modelo_pt,
                descripcion="CPU (PyTorch)",
            )
        )

    # Backend ONNX (requiere onnxruntime y modelo .onnx existente)
    onnx_path = _inferir_onnx_path(modelo_pt, modelo_onnx_cfg)
    if onnx_path:
        try:
            import onnxruntime as ort  # type: ignore

            providers = ort.get_available_providers()
            desc = f"ONNX Runtime ({', '.join(providers)})" if providers else "ONNX Runtime"
            candidatos.append(
                BackendInfo(
                    nombre="onnx",
                    tipo="onnx",
                    dispositivo="onnxruntime",
                    modelo_path=onnx_path,
                    descripcion=desc,
                )
            )
        except Exception:
            LOGGER.debug("onnxruntime no disponible; se omite backend ONNX.")

    seleccionado: Optional[BackendInfo] = None
    prefs = preferencias or []
    prefs = prefs if isinstance(prefs, list) else [prefs]

    for pref in prefs:
        for cand in candidatos:
            if cand.nombre == pref:
                seleccionado = cand
                break
        if seleccionado:
            break

    if not seleccionado and candidatos:
        seleccionado = candidatos[0]

    return seleccionado, candidatos


def describir_backends(disponibles: List[BackendInfo]) -> str:
    """Texto legible con los backends detectados."""
    if not disponibles:
        return "sin backends disponibles"
    return "; ".join(f"{b.nombre} -> {b.descripcion}" for b in disponibles)


