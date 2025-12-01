def calcular_iou(boxA, boxB):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos cajas.
    Soporta input como diccionario {'x1':..., 'y1':...} o lista/tupla [x1, y1, x2, y2].
    """
    def _get_coords(box):
        if isinstance(box, dict):
            return box["x1"], box["y1"], box["x2"], box["y2"]
        return box[0], box[1], box[2], box[3]

    xA1, yA1, xA2, yA2 = _get_coords(boxA)
    xB1, yB1, xB2, yB2 = _get_coords(boxB)

    xA = max(xA1, xB1)
    yA = max(yA1, yB1)
    xB = min(xA2, xB2)
    yB = min(yA2, yB2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
