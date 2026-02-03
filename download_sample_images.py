"""
Script para descargar imagenes de muestra del dataset
Extrae URLs del archivo JSONL y descarga 10 imagenes para prueba
"""

import json
import requests
from pathlib import Path
from typing import List


def extraer_urls_imagenes(jsonl_path: str, max_imagenes: int = 10) -> List[tuple]:
    """
    Extrae URLs de imagenes del archivo JSONL

    Args:
        jsonl_path: Ruta al archivo JSONL
        max_imagenes: Numero maximo de imagenes a extraer

    Returns:
        Lista de tuplas (url, numero_etiqueta)
    """
    urls = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_imagenes:
                break

            data = json.loads(line)

            # Extraer URL de imagen
            for msg in data.get('messages', []):
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image_url':
                            url = item['image_url']['url']

                            # Extraer numero de la respuesta del asistente
                            numero = None
                            for msg2 in data['messages']:
                                if msg2.get('role') == 'assistant':
                                    numero = msg2.get('content', 'unknown')
                                    break

                            urls.append((url, numero))

    return urls


def descargar_imagen(url: str, output_path: Path) -> bool:
    """
    Descarga una imagen desde URL

    Args:
        url: URL de la imagen
        output_path: Ruta donde guardar la imagen

    Returns:
        True si se descargo correctamente
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return True

    except Exception as e:
        print(f"[ERROR] No se pudo descargar {url}: {e}")
        return False


def main():
    """Funcion principal"""
    print("=" * 70)
    print("DESCARGA DE IMAGENES DE MUESTRA")
    print("=" * 70)

    # Rutas
    dataset_dir = Path("../basketball-jersey-numbers-ocr.v7i.openai")
    jsonl_file = dataset_dir / "_annotations.train.jsonl"

    output_dir = Path("./sample_images")
    output_dir.mkdir(exist_ok=True)

    # Verificar que existe el archivo
    if not jsonl_file.exists():
        print(f"\n[ERROR] No se encontro: {jsonl_file}")
        print("Asegurate de haber descargado el dataset de Roboflow")
        return

    print(f"\nExtrayendo URLs de: {jsonl_file}")
    urls = extraer_urls_imagenes(str(jsonl_file), max_imagenes=10)

    print(f"[OK] Se encontraron {len(urls)} imagenes")
    print(f"\nDescargando imagenes a: {output_dir.absolute()}")
    print("-" * 70)

    # Descargar imagenes
    exitosas = 0

    for i, (url, numero) in enumerate(urls, 1):
        filename = f"jersey_{numero}_{i:02d}.jpg"
        output_path = output_dir / filename

        print(f"[{i}/{len(urls)}] Descargando: {filename} (Numero: {numero})")

        if descargar_imagen(url, output_path):
            exitosas += 1
            print(f"         [OK] Guardado en {output_path}")

    # Resumen
    print("\n" + "=" * 70)
    print(f"DESCARGA COMPLETADA")
    print("=" * 70)
    print(f"Imagenes descargadas: {exitosas}/{len(urls)}")
    print(f"Ubicacion: {output_dir.absolute()}")
    print("\nAhora puedes usar estas imagenes para probar la interfaz Gradio")
    print("=" * 70)


if __name__ == "__main__":
    main()
