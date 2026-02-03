================================================================================
IMAGENES DE MUESTRA - Basketball Jersey Numbers OCR
================================================================================

UBICACION: ./sample_images/

CONTENIDO:
- 5 imagenes de camisetas de baloncesto con numeros
- Numeros incluidos: 00, 02, 04, 05, 28
- Formato: JPG
- Tamaño aproximado: 4-7 KB cada una

================================================================================
COMO USAR ESTAS IMAGENES
================================================================================

1. OPCION A: Interfaz Gradio (desde Colab o PyCharm)

   - Abrir la URL publica de Gradio
   - Click en "Image de entrada"
   - Upload > Seleccionar imagen de esta carpeta
   - Ajustar slider de confianza (0.4 recomendado)
   - Click "Analizar"

2. OPCION B: Arrastrar y soltar

   - Abrir interfaz Gradio
   - Arrastrar imagen directamente al area de entrada
   - Click "Analizar"

================================================================================
RESULTADOS ESPERADOS
================================================================================

Cada imagen deberia mostrar:
- Bounding box verde alrededor del numero
- Etiqueta con numero detectado y confianza
- Ejemplo: "28 (0.95)"

Estadisticas tipicas:
- Confianza promedio: 0.85-0.95
- Total detectado: 1 numero por imagen

================================================================================
DESCARGAR MAS IMAGENES
================================================================================

Para descargar mas imagenes del dataset, ejecutar:

python download_sample_images.py

Este script descargara 10 imagenes adicionales automaticamente.

================================================================================
ORIGEN DE LAS IMAGENES
================================================================================

Fuente: Roboflow Universe
Dataset: basketball-jersey-numbers-ocr/7
URL: https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr

Nota: Estas imagenes son ejemplos del conjunto de entrenamiento
y se usan exclusivamente para propositos de prueba.

================================================================================
