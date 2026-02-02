# Basketball Jersey Numbers OCR

Deteccion de numeros en camisetas de baloncesto usando YOLOv8 con inferencia local en GPU.

## Especificaciones Tecnicas

- **Modelo**: basketball-jersey-numbers-ocr/7 (Roboflow)
- **Arquitectura**: YOLOv8 (deteccion de objetos)
- **GPU**: NVIDIA T4 (Google Colab)
- **IDE**: PyCharm Professional + Colab Remote Runtime
- **Costo**: Cero creditos de API (inferencia 100% local)

## Instalacion

### 1. Clonar repositorio

```bash
git clone <repository-url>
cd basketball-jersey-numbers
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar PyCharm con Colab

1. Abrir PyCharm Professional
2. Configurar Python Interpreter:
   - Settings > Project > Python Interpreter
   - Add Interpreter > On SSH
   - Conectar al runtime de Colab siguiendo la extension oficial

### 4. Obtener API Key de Roboflow

1. Registrarse en [Roboflow](https://roboflow.com)
2. Ir a Settings > API Keys
3. Copiar tu API key privada

## Uso

### Ejecucion del Script

```bash
python basketball_jersey_analyzer.py
```

El script solicitara tu API key al inicio.

### Flujo de Trabajo

1. Script verifica disponibilidad de GPU T4
2. Instala dependencias faltantes
3. Carga modelo localmente en GPU
4. Abre interfaz Gradio con URL publica
5. Subir imagen o usar camara web
6. Ajustar umbral de confianza (slider 0.1-0.9)
7. Hacer clic en "Analizar"
8. Ver resultados con bounding boxes y estadisticas
9. Exportar detecciones a CSV

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
