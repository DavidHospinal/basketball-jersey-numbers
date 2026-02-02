"""
Basketball Jersey Numbers OCR - Inferencia Local
Modelo: basketball-jersey-numbers-ocr/7
Entorno: PyCharm + Google Colab GPU T4
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import gradio as gr
import numpy as np
from PIL import Image


def verificar_gpu():
    """Verifica disponibilidad y especificaciones de GPU"""
    print("=" * 60)
    print("VERIFICACION DE GPU")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memoria asignada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memoria en cache: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print("=" * 60)
        return True
    else:
        print("ERROR: No se detecto GPU CUDA")
        print("Asegurate de estar conectado al runtime de Colab con GPU T4")
        print("=" * 60)
        return False


def instalar_dependencias():
    """Instala dependencias necesarias si no están presentes"""
    paquetes = {
        'inference': 'inference[gpu]',
        'supervision': 'supervision',
        'gradio': 'gradio',
        'roboflow': 'roboflow'
    }

    for modulo, paquete in paquetes.items():
        try:
            __import__(modulo)
            print(f"[OK] {modulo} ya instalado")
        except ImportError:
            print(f"[INSTALANDO] {paquete}...")
            os.system(f"{sys.executable} -m pip install -q {paquete}")
            print(f"[OK] {paquete} instalado")


class JerseyAnalyzer:
    """Analizador de numeros de camisetas de baloncesto con inferencia local"""

    def __init__(self, api_key: str, model_id: str = "basketball-jersey-numbers-ocr/7"):
        """
        Inicializa el analizador con inferencia local en GPU

        Args:
            api_key: Roboflow API key (solo para descargar modelo)
            model_id: ID del modelo en formato workspace/project/version
        """
        self.api_key = api_key
        self.model_id = model_id
        self.model = None
        self.output_dir = Path("./outputs/detections")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_log = Path("./jersey_log.csv")

        # Inicializar CSV si no existe
        if not self.csv_log.exists():
            with open(self.csv_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Numero Detectado', 'Confianza', 'Archivo'])

        self._cargar_modelo()

    def _cargar_modelo(self):
        """Carga el modelo para inferencia local en GPU"""
        try:
            print(f"\nCargando modelo {self.model_id} para inferencia local...")
            from inference import get_model

            # Inicializar modelo con API key (solo descarga, no consume creditos)
            self.model = get_model(
                model_id=self.model_id,
                api_key=self.api_key
            )
            print(f"[OK] Modelo cargado en GPU local")

        except Exception as e:
            print(f"[ERROR] Error al cargar modelo: {e}")
            raise

    def detectar_numeros(
        self,
        imagen: np.ndarray,
        confianza_min: float = 0.4
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detecta numeros en camiseta con inferencia local

        Args:
            imagen: Imagen en formato numpy array (RGB)
            confianza_min: Umbral minimo de confianza (0.0-1.0)

        Returns:
            Tupla de (imagen_anotada, lista_detecciones)
        """
        if self.model is None:
            raise RuntimeError("Modelo no inicializado")

        # Inferencia local (NO consume creditos de API)
        resultados = self.model.infer(imagen, confidence=confianza_min)[0]

        # Procesar detecciones
        detecciones = []
        for pred in resultados.predictions:
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

        # Visualizar con supervision
        imagen_anotada = self._visualizar_detecciones(imagen, resultados)

        # Guardar en log CSV
        self._guardar_en_log(detecciones)

        return imagen_anotada, detecciones

    def _visualizar_detecciones(self, imagen: np.ndarray, resultados) -> np.ndarray:
        """Dibuja bounding boxes y etiquetas con supervision"""
        try:
            import supervision as sv
            from inference import InferenceResponseObject

            # Convertir resultados a formato supervision
            detections = sv.Detections.from_inference(resultados)

            # Configurar anotadores
            box_annotator = sv.BoxAnnotator(
                thickness=3,
                color=sv.Color.from_hex("#00FF00")
            )

            label_annotator = sv.LabelAnnotator(
                text_scale=1.2,
                text_thickness=2,
                text_color=sv.Color.WHITE,
                color=sv.Color.from_hex("#00FF00")
            )

            # Generar etiquetas con clase y confianza
            labels = [
                f"{pred.class_name} ({pred.confidence:.2f})"
                for pred in resultados.predictions
            ]

            # Anotar imagen
            imagen_anotada = box_annotator.annotate(
                scene=imagen.copy(),
                detections=detections
            )
            imagen_anotada = label_annotator.annotate(
                scene=imagen_anotada,
                detections=detections,
                labels=labels
            )

            return imagen_anotada

        except Exception as e:
            print(f"[ADVERTENCIA] Error en visualizacion: {e}")
            return imagen

    def _guardar_en_log(self, detecciones: List[Dict]):
        """Añade detecciones al archivo CSV de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.csv_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for det in detecciones:
                writer.writerow([
                    timestamp,
                    det['numero'],
                    det['confianza'],
                    'gradio_upload'
                ])

    def calcular_estadisticas(self, detecciones: List[Dict]) -> Dict:
        """Calcula estadisticas de las detecciones"""
        if not detecciones:
            return {
                'total': 0,
                'confianza_promedio': 0.0,
                'confianza_max': 0.0,
                'confianza_min': 0.0
            }

        confianzas = [d['confianza'] for d in detecciones]

        return {
            'total': len(detecciones),
            'confianza_promedio': round(np.mean(confianzas), 3),
            'confianza_max': round(max(confianzas), 3),
            'confianza_min': round(min(confianzas), 3)
        }

    def exportar_csv(self, detecciones: List[Dict], filename: str = None) -> str:
        """Exporta detecciones actuales a CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detecciones_{timestamp}.csv"

        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['numero', 'confianza', 'bbox'])
            writer.writeheader()
            writer.writerows(detecciones)

        return str(filepath)


def crear_interfaz_gradio(analyzer: JerseyAnalyzer):
    """Crea interfaz Gradio profesional con todas las funcionalidades"""

    def analizar_imagen(imagen, confianza_min):
        """Procesa imagen y retorna resultados"""
        if imagen is None:
            return None, "No se cargo ninguna imagen", None

        # Convertir a numpy array RGB
        if isinstance(imagen, Image.Image):
            imagen = np.array(imagen)

        # Detectar numeros
        imagen_anotada, detecciones = analyzer.detectar_numeros(
            imagen,
            confianza_min=confianza_min
        )

        # Calcular estadisticas
        stats = analyzer.calcular_estadisticas(detecciones)

        # Formatear resultados
        texto_stats = f"""
ESTADISTICAS DE DETECCION:
- Total de numeros detectados: {stats['total']}
- Confianza promedio: {stats['confianza_promedio']:.3f}
- Confianza maxima: {stats['confianza_max']:.3f}
- Confianza minima: {stats['confianza_min']:.3f}
        """

        # Formatear tabla de detecciones
        tabla_detecciones = [
            [d['numero'], f"{d['confianza']:.3f}"]
            for d in detecciones
        ]

        return imagen_anotada, texto_stats, tabla_detecciones

    def limpiar_todo():
        """Limpia todos los campos"""
        return None, "", None

    def exportar_resultados_csv(detecciones_tabla):
        """Exporta tabla actual a CSV"""
        if not detecciones_tabla:
            return "No hay detecciones para exportar"

        detecciones = [
            {
                'numero': row[0],
                'confianza': float(row[1]),
                'bbox': {}
            }
            for row in detecciones_tabla
        ]

        filepath = analyzer.exportar_csv(detecciones)
        return f"Exportado a: {filepath}"

    # Crear interfaz con Blocks
    with gr.Blocks(
        title="Basketball Jersey Numbers OCR",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("# Basketball Jersey Numbers OCR")
        gr.Markdown("Deteccion de numeros en camisetas de baloncesto - Inferencia Local GPU")

        with gr.Row():
            with gr.Column(scale=1):
                imagen_entrada = gr.Image(
                    label="Imagen de entrada",
                    type="numpy",
                    sources=["upload", "webcam"]
                )

                confianza_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.4,
                    step=0.05,
                    label="Confianza minima"
                )

                with gr.Row():
                    btn_analizar = gr.Button("Analizar", variant="primary")
                    btn_limpiar = gr.Button("Limpiar")

            with gr.Column(scale=1):
                imagen_salida = gr.Image(
                    label="Detecciones",
                    type="numpy"
                )

                texto_stats = gr.Textbox(
                    label="Estadisticas",
                    lines=6,
                    interactive=False
                )

        gr.Markdown("### Historial de Detecciones")

        tabla_detecciones = gr.Dataframe(
            headers=["Numero", "Confianza"],
            label="Resultados",
            interactive=False
        )

        with gr.Row():
            btn_exportar = gr.Button("Exportar a CSV")
            texto_exportar = gr.Textbox(
                label="Estado de exportacion",
                interactive=False
            )

        # Conectar eventos
        btn_analizar.click(
            fn=analizar_imagen,
            inputs=[imagen_entrada, confianza_slider],
            outputs=[imagen_salida, texto_stats, tabla_detecciones]
        )

        btn_limpiar.click(
            fn=limpiar_todo,
            outputs=[imagen_entrada, texto_stats, tabla_detecciones]
        )

        btn_exportar.click(
            fn=exportar_resultados_csv,
            inputs=[tabla_detecciones],
            outputs=[texto_exportar]
        )

    return demo


def main():
    """Funcion principal de ejecucion"""
    print("\n" + "=" * 60)
    print("BASKETBALL JERSEY NUMBERS OCR - INFERENCIA LOCAL")
    print("=" * 60 + "\n")

    # 1. Verificar GPU
    if not verificar_gpu():
        print("\nAbortando: GPU no disponible")
        return

    # 2. Instalar dependencias
    print("\nVerificando dependencias...")
    instalar_dependencias()

    # 3. Solicitar API key
    api_key = input("\nIngresa tu Roboflow API key: ").strip()
    if not api_key:
        print("ERROR: API key requerida")
        return

    # 4. Inicializar analizador
    print("\nInicializando analizador...")
    analyzer = JerseyAnalyzer(api_key=api_key)

    # 5. Crear y lanzar interfaz Gradio
    print("\nLanzando interfaz Gradio...")
    demo = crear_interfaz_gradio(analyzer)

    demo.launch(
        share=True,  # Genera URL publica para acceso desde navegador local
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
