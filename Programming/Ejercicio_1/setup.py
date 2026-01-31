#Codigo obtenido de Chat para descarga de depndencias que se ocupen para el proyecto#
import os
import subprocess
import sys

def run(command):
    print(f"> {command}")
    subprocess.check_call(command, shell=True)

def main():
    print("🚀 Iniciando instalación del proyecto")

    # 1. Crear entorno virtual
    if not os.path.exists("venv"):
        print("📦 Creando entorno virtual...")
        run(f"{sys.executable} -m venv venv")
    else:
        print("✔ El entorno virtual ya existe")

    # 2. Definir activación según SO
    if os.name == "nt":  # Windows
        activate_cmd = r"venv\Scripts\activate"
    else:  # Linux / macOS
        activate_cmd = "source venv/bin/activate"

    # 3. Instalar dependencias
    print("📥 Instalando dependencias...")
    run(f"{activate_cmd} && pip install --upgrade pip")
    run(f"{activate_cmd} && pip install -r requirements.txt")

    print("\n✅ Instalación completada con éxito")
    print("👉 Para activar el entorno:")
    if os.name == "nt":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")

if __name__ == "__main__":
    main()
