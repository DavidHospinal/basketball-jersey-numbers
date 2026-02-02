"""
Script de instalacion de dependencias para runtime de Colab
Ejecutar este archivo PRIMERO desde PyCharm conectado a Colab
para asegurar que todas las librerias esten instaladas
"""

import subprocess
import sys


def ejecutar_comando(comando):
    """Ejecuta comando y muestra output en tiempo real"""
    print(f"\n[EJECUTANDO] {comando}")
    print("-" * 70)

    proceso = subprocess.Popen(
        comando,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    for linea in proceso.stdout:
        print(linea, end='')

    proceso.wait()
    return proceso.returncode


def main():
    print("=" * 70)
    print("INSTALACION DE DEPENDENCIAS PARA BASKETBALL JERSEY OCR")
    print("=" * 70)

    # Lista de paquetes a instalar
    paquetes = [
        "torch",
        "inference[gpu]",
        "supervision",
        "gradio",
        "roboflow",
        "pillow",
        "numpy"
    ]

    print(f"\nSe instalaran {len(paquetes)} paquetes:")
    for paquete in paquetes:
        print(f"  - {paquete}")

    print("\n" + "=" * 70)
    input("Presiona ENTER para continuar con la instalacion...")

    # Actualizar pip primero
    print("\n[PASO 1/2] Actualizando pip...")
    ejecutar_comando(f"{sys.executable} -m pip install --upgrade pip")

    # Instalar paquetes
    print("\n[PASO 2/2] Instalando dependencias...")
    for i, paquete in enumerate(paquetes, 1):
        print(f"\n[{i}/{len(paquetes)}] Instalando {paquete}...")
        resultado = ejecutar_comando(f"{sys.executable} -m pip install {paquete}")

        if resultado != 0:
            print(f"\n[ERROR] Fallo la instalacion de {paquete}")
            respuesta = input("Continuar con el siguiente paquete? (s/n): ")
            if respuesta.lower() != 's':
                print("Instalacion cancelada")
                return

    # Verificar instalacion
    print("\n" + "=" * 70)
    print("VERIFICACION DE INSTALACION")
    print("=" * 70)

    modulos_verificar = {
        'torch': 'PyTorch',
        'inference': 'Roboflow Inference',
        'supervision': 'Supervision',
        'gradio': 'Gradio',
        'roboflow': 'Roboflow',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }

    todos_ok = True

    for modulo, nombre in modulos_verificar.items():
        try:
            __import__(modulo)
            print(f"[OK] {nombre}")
        except ImportError:
            print(f"[FALTA] {nombre}")
            todos_ok = False

    # Verificar GPU
    print("\n" + "=" * 70)
    print("VERIFICACION DE GPU")
    print("=" * 70)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[ADVERTENCIA] GPU no disponible")
            print("Verifica la configuracion del runtime en Colab:")
            print("  Runtime > Change runtime type > GPU (T4)")
    except Exception as e:
        print(f"[ERROR] No se pudo verificar GPU: {e}")

    # Resultado final
    print("\n" + "=" * 70)
    if todos_ok:
        print("[EXITO] Todas las dependencias instaladas correctamente")
        print("\nProximos pasos:")
        print("1. Ejecutar verify_environment.py para confirmar configuracion")
        print("2. Ejecutar basketball_jersey_analyzer.py para iniciar la aplicacion")
    else:
        print("[ADVERTENCIA] Algunas dependencias no se instalaron correctamente")
        print("Revisa los errores arriba e intenta instalar manualmente:")
        print(f"  {sys.executable} -m pip install <paquete>")

    print("=" * 70)


if __name__ == "__main__":
    main()
