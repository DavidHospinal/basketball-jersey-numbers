"""
Script de verificacion de entorno
Ejecutar este archivo para confirmar que PyCharm esta usando el runtime remoto de Colab
"""

import sys
import platform

print("=" * 70)
print("VERIFICACION DE ENTORNO")
print("=" * 70)

# Verificar ubicacion del interprete
print(f"\nInterprete Python: {sys.executable}")
print(f"Version Python: {sys.version}")
print(f"Plataforma: {platform.platform()}")
print(f"Sistema operativo: {platform.system()}")

# Verificar si estamos en Colab
es_colab = 'google.colab' in str(sys.modules.keys())
print(f"\nEjecutando en Colab: {'SI' if es_colab else 'NO'}")

# Verificar PyTorch y CUDA
print("\n" + "-" * 70)
print("VERIFICACION DE PYTORCH Y GPU")
print("-" * 70)

try:
    import torch
    print(f"PyTorch instalado: SI")
    print(f"Version PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {'SI' if torch.cuda.is_available() else 'NO'}")

    if torch.cuda.is_available():
        print(f"Version CUDA: {torch.version.cuda}")
        print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Numero de GPUs: {torch.cuda.device_count()}")
except ImportError:
    print("PyTorch: NO INSTALADO")
    print("Instalar con: pip install torch")

# Verificar dependencias del proyecto
print("\n" + "-" * 70)
print("VERIFICACION DE DEPENDENCIAS DEL PROYECTO")
print("-" * 70)

dependencias = {
    'inference': 'inference[gpu]',
    'supervision': 'supervision',
    'gradio': 'gradio',
    'roboflow': 'roboflow',
    'PIL': 'pillow',
    'numpy': 'numpy'
}

modulos_faltantes = []

for modulo, paquete in dependencias.items():
    try:
        __import__(modulo)
        print(f"[OK] {modulo}")
    except ImportError:
        print(f"[FALTA] {modulo}")
        modulos_faltantes.append(paquete)

# Diagnostico final
print("\n" + "=" * 70)
print("DIAGNOSTICO FINAL")
print("=" * 70)

if sys.executable.startswith("C:"):
    print("\nADVERTENCIA: Estas ejecutando Python desde Windows local")
    print(f"Ruta: {sys.executable}")
    print("\nDEBES cambiar el interprete a Colab remoto en PyCharm:")
    print("1. File > Settings > Project > Python Interpreter")
    print("2. Seleccionar el interprete remoto de Colab")
    print("3. Ver archivo PYCHARM_INTERPRETER_FIX.txt para instrucciones detalladas")
elif es_colab or '/usr/bin/python' in sys.executable or '/opt/python' in sys.executable:
    print("\n[OK] Estas usando el interprete remoto correctamente")

    if torch.cuda.is_available():
        print("[OK] GPU NVIDIA detectada y funcional")
        print(f"[OK] Modelo: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[ADVERTENCIA] GPU no detectada")
        print("Verifica en Colab: Runtime > Change runtime type > GPU")

    if modulos_faltantes:
        print(f"\n[ADVERTENCIA] Faltan {len(modulos_faltantes)} modulos:")
        for paquete in modulos_faltantes:
            print(f"  pip install {paquete}")
    else:
        print("[OK] Todas las dependencias instaladas")
        print("\n[LISTO] El entorno esta configurado correctamente")
        print("Puedes ejecutar: basketball_jersey_analyzer.py")
else:
    print(f"\n[INFO] Interprete: {sys.executable}")
    print("Verifica que sea el interprete remoto de Colab")

print("=" * 70)
