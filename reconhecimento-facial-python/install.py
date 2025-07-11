import subprocess
import sys
import os


def install_packages():
    print("⏳ Configurando ambiente...")

    # 1. First upgrade pip itself
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ pip atualizado com sucesso")
    except subprocess.CalledProcessError:
        print("⚠️ Não foi possível atualizar o pip, continuando...")

    # 2. Install packages one by one with error handling
    packages = [
        "numpy==1.23.5",
        "opencv-python==4.7.0.72",
        "mysql-connector-python==8.0.33",
        "scikit-learn==1.2.2",
        "tensorflow==2.10.1",
        "deepface==0.0.79",
    ]

    success_count = 0
    for pkg in packages:
        try:
            print(f"⏳ Instalando {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"✅ {pkg} instalado")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Falha ao instalar {pkg}: {e}")
            if "tensorflow" in pkg:
                print("Tentando instalar tensorflow-cpu...")
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "tensorflow-cpu==2.10.1",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print("✅ tensorflow-cpu instalado como fallback")
                    success_count += 1
                except subprocess.CalledProcessError:
                    print("⚠️ Falha ao instalar tensorflow-cpu")

    # 3. Verify critical packages
    critical_pkgs = ["numpy", "opencv-python", "tensorflow", "deepface"]
    all_ok = True
    for pkg in critical_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"❌ {pkg} não está disponível após instalação!")
            all_ok = False

    print(f"\n📊 Resultado: {success_count}/{len(packages)} pacotes instalados")
    if all_ok:
        print("✅ Ambiente configurado com sucesso!")
    else:
        print("⚠️ Alguns pacotes não foram instalados corretamente")


if __name__ == "__main__":
    install_packages()
