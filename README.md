# Basketball Jersey Numbers OCR

Deteccion de numeros en camisetas de baloncesto usando YOLOv8 con inferencia local en GPU.

## Especificaciones Tecnicas

- **Modelo**: basketball-jersey-numbers-ocr/7 (Roboflow)
- **Arquitectura**: YOLOv8 (deteccion de objetos)
- **GPU**: NVIDIA T4 (Google Colab)
- **Plataforma**: Google Colab (Jupyter Notebook)
- **Costo**: Cero creditos de API (inferencia 100% local)

## Instalacion y Uso

### Metodo Recomendado: Google Colab Notebook

**Archivo principal**: `test-colab.ipynb`

1. Subir `test-colab.ipynb` a Google Colab
2. Configurar runtime con GPU T4
3. Ejecutar celdas en orden (1 a 6)
4. Ingresar API key de Roboflow en Celda 6
5. Usar interfaz Gradio generada

Ver `COLAB_NOTEBOOK_GUIDE.txt` para instrucciones detalladas.

### Metodo Alternativo: Script Python Local

**Archivo**: `basketball_jersey_analyzer.py`

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar script
python basketball_jersey_analyzer.py
```

Nota: Requiere GPU NVIDIA con CUDA. Para desarrollo sin GPU, usar Colab.

### Obtener API Key de Roboflow

1. Registrarse en [Roboflow](https://app.roboflow.com)
2. Settings > API Keys
3. Copiar Private API Key

## Flujo de Trabajo

1. Subir imagen de camiseta de baloncesto
2. Ajustar umbral de confianza (slider 0.1-0.9)
3. Click en "Analizar"
4. Ver resultados con bounding boxes y estadisticas
5. Exportar detecciones a CSV si es necesario

## Estructura del Proyecto

```
basketball-jersey-numbers/
├── basketball_jersey_analyzer.py   # Script principal
├── requirements.txt                # Dependencias Python
├── .gitignore                      # Archivos ignorados por git
├── README.md                       # Documentacion
├── outputs/                        # Imagenes procesadas (auto-creado)
│   └── detections/
└── jersey_log.csv                  # Historial de detecciones (auto-creado)
```

## Funcionalidades

### Inferencia Local
- Usa libreria `inference` con GPU T4
- No consume creditos de Roboflow API
- API key solo para descarga inicial del modelo

### Visualizacion
- Bounding boxes con libreria `supervision`
- Etiquetas con numero y confianza
- Colores configurables

### Interfaz Gradio
- Dashboard profesional con `gr.Blocks`
- Entrada: imagen (upload o webcam)
- Salida: imagen anotada + estadisticas + tabla
- Controles: slider de confianza, botones de analisis y limpieza

### Exportacion
- Log CSV automatico con timestamp
- Exportacion manual de detecciones actuales
- Guardado local en carpeta `./outputs/`

### Estadisticas
- Total de numeros detectados
- Confianza promedio/maxima/minima
- Tabla de detecciones con clase y score

## Configuracion de Git

Este proyecto esta configurado con:

```bash
git config user.name "DavidHospinal"
git config user.email "u202021214@upc.edu.pe"
```

Todas las contribuciones se registran bajo el usuario DavidHospinal.

## Referencias

- [Roboflow Universe - Basketball Jersey Numbers](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr/dataset/7)
- [Roboflow Inference Docs](https://inference.roboflow.com/start/getting-started/)
- [Supervision Library](https://supervision.roboflow.com/)

## Licencia

Proyecto academico - UPC 2025
