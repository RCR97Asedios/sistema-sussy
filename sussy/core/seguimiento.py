from typing import List, Dict, Any
from sussy.core.utilidades_iou import calcular_iou

class TrackerSimple:
    """
    Tracker mejorado que asocia detecciones por IoU (prioridad) y distancia (fallback).
    Mantiene tracks perdidos durante 'max_frames_lost' frames para evitar parpadeos.
    """
    def __init__(self, max_dist: int = 100, max_frames_lost: int = 10, iou_threshold: float = 0.3):
        self.max_dist = max_dist
        self.max_frames_lost = max_frames_lost
        self.iou_threshold = iou_threshold
        self.tracks = {}  # id -> {box, frames_lost, clase, score}
        self.next_id = 1

    def actualizar(self, detecciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 1. Predicción simple: asumimos que están donde estaban
        # (Aquí se podría añadir Kalman Filter para mejor predicción de movimiento)
        
        # 2. Asociación
        used_det_indices = set()
        used_track_ids = set()
        
        # --- PASO 1: Asociación por IoU (Objetos grandes/lentos) ---
        # Ordenamos detecciones por score para priorizar las mejores
        sorted_dets = sorted(enumerate(detecciones), key=lambda x: x[1]['score'], reverse=True)
        
        for idx, det in sorted_dets:
            best_tid = None
            best_iou = -1.0
            
            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                
                # Calcular IoU entre detección y último box del track
                # track['box'] es [x1, y1, x2, y2]
                # det es dict con x1, y1, x2, y2
                iou = calcular_iou(track['box'], det)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                # Match encontrado por IoU
                self._actualizar_track(best_tid, det)
                used_track_ids.add(best_tid)
                used_det_indices.add(idx)

        # --- PASO 2: Asociación por Distancia (Objetos pequeños/rápidos o sin solapamiento) ---
        # Solo para detecciones y tracks no usados
        for idx, det in sorted_dets:
            if idx in used_det_indices:
                continue
                
            cx = (det['x1'] + det['x2']) / 2
            cy = (det['y1'] + det['y2']) / 2
            
            best_tid = None
            best_dist = float('inf')

            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                
                tcx = (track['box'][0] + track['box'][2]) / 2
                tcy = (track['box'][1] + track['box'][3]) / 2
                
                dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                
                if dist < self.max_dist and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                # Match encontrado por Distancia
                self._actualizar_track(best_tid, det)
                used_track_ids.add(best_tid)
                used_det_indices.add(idx)
            else:
                # --- PASO 3: Nueva Detección ---
                self._crear_track(det)

        # --- PASO 4: Gestión de Tracks Perdidos ---
        final_tracks = []
        to_delete = []
        
        for tid, track in self.tracks.items():
            if tid in used_track_ids:
                # Track actualizado en este frame
                final_tracks.append({
                    "id": tid,
                    "box": track['box'],
                    "clase": track['clase'],
                    "score": track['score'],
                    "perdido": False
                })
            else:
                # Track perdido en este frame
                track['frames_lost'] += 1
                if track['frames_lost'] <= self.max_frames_lost:
                    # Aún tenemos paciencia, lo devolvemos para visualización (anti-parpadeo)
                    final_tracks.append({
                        "id": tid,
                        "box": track['box'], # Devolvemos la última posición conocida
                        "clase": track['clase'],
                        "score": track['score'],
                        "perdido": True # Flag por si queremos pintarlo diferente
                    })
                else:
                    # Se acabó la paciencia
                    to_delete.append(tid)
        
        for tid in to_delete:
            del self.tracks[tid]

        return final_tracks

    def _actualizar_track(self, tid, det):
        self.tracks[tid]['box'] = [det['x1'], det['y1'], det['x2'], det['y2']]
        self.tracks[tid]['frames_lost'] = 0
        self.tracks[tid]['clase'] = det['clase']
        self.tracks[tid]['score'] = det['score']

    def _crear_track(self, det):
        new_id = self.next_id
        self.next_id += 1
        self.tracks[new_id] = {
            "box": [det['x1'], det['y1'], det['x2'], det['y2']],
            "frames_lost": 0,
            "clase": det['clase'],
            "score": det['score']
        }
