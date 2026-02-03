"""
Fix para manejar respuestas VLM del modelo basketball-jersey-numbers-ocr/7
Detecta el tipo de respuesta y extrae predicciones correctamente
"""

def detectar_numeros_fixed(self, imagen, confianza_min=0.4):
    """
    Version corregida que maneja tanto YOLO como VLM responses
    """
    if self.model is None:
        raise RuntimeError("Modelo no inicializado")

    # Inferencia local
    resultado = self.model.infer(imagen, confidence=confianza_min)

    # Manejar diferentes tipos de respuesta
    if isinstance(resultado, list):
        resultado = resultado[0]

    # Verificar tipo de respuesta
    tipo_respuesta = type(resultado).__name__
    print(f"[DEBUG] Tipo de respuesta: {tipo_respuesta}")

    detecciones = []

    # CASO 1: Respuesta VLM/LMM
    if hasattr(resultado, 'response'):
        # Para modelos VLM que retornan texto
        texto_respuesta = getattr(resultado, 'response', '')
        print(f"[INFO] Respuesta VLM: {texto_respuesta}")

        # Extraer numero del texto
        import re
        numeros = re.findall(r'\d+', texto_respuesta)

        if numeros:
            # Crear deteccion mock con el numero extraido
            detecciones.append({
                'numero': numeros[0],
                'confianza': 0.95,  # Confianza alta por defecto para VLM
                'bbox': {
                    'x': imagen.shape[1] // 2,
                    'y': imagen.shape[0] // 2,
                    'width': imagen.shape[1] // 2,
                    'height': imagen.shape[0] // 2
                }
            })

    # CASO 2: Respuesta YOLO estandar
    elif hasattr(resultado, 'predictions'):
        for pred in resultado.predictions:
            detecciones.append({
                'numero': pred.class_name,
                'confianza': round(pred.confidence, 3),
                'bbox': {
                    'x': int(pred.x),
                    'y': int(pred.y),
                    'width': int(pred.width),
                    'height': int(pred.height)
                }
            })

    # CASO 3: Otro tipo de respuesta
    else:
        print(f"[ERROR] Tipo de respuesta desconocido: {tipo_respuesta}")
        print(f"[DEBUG] Atributos disponibles: {dir(resultado)}")

        # Intentar extraer cualquier prediccion
        if hasattr(resultado, 'dict'):
            print(f"[DEBUG] Contenido: {resultado.dict()}")

    # Visualizar (simplificado sin supervision para VLM)
    if detecciones:
        imagen_anotada = self._visualizar_simple(imagen, detecciones)
    else:
        imagen_anotada = imagen.copy()

    # Guardar en log
    self._guardar_en_log(detecciones)

    return imagen_anotada, detecciones


def visualizar_simple(self, imagen, detecciones):
    """
    Visualizacion simple con OpenCV cuando supervision falla
    """
    import cv2
    import numpy as np

    img = imagen.copy()

    for det in detecciones:
        bbox = det['bbox']
        numero = det['numero']
        conf = det['confianza']

        # Calcular coordenadas
        x1 = int(bbox['x'] - bbox['width'] / 2)
        y1 = int(bbox['y'] - bbox['height'] / 2)
        x2 = int(bbox['x'] + bbox['width'] / 2)
        y2 = int(bbox['y'] + bbox['height'] / 2)

        # Dibujar bounding box verde
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Dibujar etiqueta
        label = f"{numero} ({conf:.2f})"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    return img


# Instrucciones de uso:
# 1. Copiar las funciones detectar_numeros_fixed y visualizar_simple
# 2. En la Celda 4 del notebook, reemplazar el metodo detectar_numeros
# 3. Agregar el metodo _visualizar_simple a la clase
