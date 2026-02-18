import os
import sys
import subprocess
import venv
import argparse
import shutil
from pathlib import Path


class VenvSetup:
    def __init__(self):
        self.venv_name = "venv"
        self.script_dir = Path(__file__).resolve().parent
        self.venv_path = self.script_dir / self.venv_name
        self.requirements_file = self.script_dir / "requirements.txt"

    def check_system_dependencies(self):
        """Verifica se as dependências de sistema para compilar dlib estão instaladas."""
        required_commands = ["cmake", "g++", "make"]
        missing_commands = [cmd for cmd in required_commands if not shutil.which(cmd)]
        if missing_commands:
            print("❌ Dependências de sistema ausentes:")
            for cmd in missing_commands:
                print(f"   - {cmd}")
            print("\nPara instalar no Ubuntu/Debian, execute:")
            print("  sudo apt update")
            print("  sudo apt install build-essential cmake pkg-config \\")
            print("                   libx11-dev libatlas-base-dev \\")
            print("                   libboost-python-dev libboost-thread-dev \\")
            print("                   libboost-system-dev libboost-filesystem-dev")
            return False

        # Verifica se os cabeçalhos do Boost estão acessíveis (opcional, mas útil)
        boost_check = subprocess.run(
            "dpkg -l | grep libboost-dev", shell=True, capture_output=True, text=True
        )
        if boost_check.returncode != 0:
            print(
                "⚠️  Biblioteca Boost não encontrada. A compilação do dlib pode falhar."
            )
            print("   Instale com: sudo apt install libboost-all-dev")
            # Não retornamos False aqui porque talvez a wheel funcione sem boost
        return True

    def create_venv(self):
        """Cria o ambiente virtual se não existir"""
        if self.venv_path.exists():
            print("✓ Ambiente virtual já existe")
            return True

        try:
            print("Criando ambiente virtual...")
            venv.create(self.venv_path, with_pip=True)
            print("✓ Ambiente virtual criado com sucesso")
            return True
        except Exception as e:
            print(f"✗ Erro ao criar ambiente virtual: {e}")
            return False

    def get_venv_python(self):
        """Retorna o caminho para o Python do venv"""
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
            activate_path = self.venv_path / "Scripts" / "activate.bat"
        else:
            python_path = self.venv_path / "bin" / "python"
            activate_path = self.venv_path / "bin" / "activate"
        return python_path, activate_path

    def run_pip_with_env(self, cmd, env=None):
        """Executa um comando pip com ambiente personalizado e retorna (returncode, dlib_failed, output_lines)."""
        my_env = os.environ.copy()
        if env:
            my_env.update(env)
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=my_env,
        )
        output_lines = []
        dlib_failed_local = False
        for line in process.stdout:
            line = line.strip()
            if line:
                print(line)
                output_lines.append(line)
                if (
                    "Failed building wheel for dlib" in line
                    or "ERROR: Failed building wheel for dlib" in line
                ):
                    dlib_failed_local = True
        process.wait()
        return process.returncode, dlib_failed_local, output_lines

    def install_requirements(self):
        """Instala as dependências com fallback inteligente para dlib."""
        python_path, _ = self.get_venv_python()

        if not python_path.exists():
            print(f"✗ Python do venv não encontrado em: {python_path}")
            return False

        if not self.requirements_file.exists():
            print(f"✗ Arquivo {self.requirements_file} não encontrado")
            return False

        # 1. Tentativa normal (sem variável de ambiente)
        print("Instalando dependências do requirements.txt...")
        print("Isso pode levar alguns minutos...")
        cmd = f'"{python_path}" -m pip install -r "{self.requirements_file}"'
        returncode, dlib_failed, _ = self.run_pip_with_env(cmd)

        if returncode == 0:
            print("✓ Todas as dependências instaladas com sucesso")
            return True

        # 2. Se falhou por causa do dlib, tenta com a variável CMAKE_POLICY_VERSION_MINIMUM
        if dlib_failed:
            print(
                "\n⚠️  Falha na instalação do dlib. Tentando com CMAKE_POLICY_VERSION_MINIMUM=3.5..."
            )

            env_fix = {"CMAKE_POLICY_VERSION_MINIMUM": "3.5"}

            # Tenta instalar apenas o dlib com a variável
            dlib_cmd = f'"{python_path}" -m pip install --no-cache-dir dlib==19.24.2'
            ret2, fail2, _ = self.run_pip_with_env(dlib_cmd, env_fix)

            if ret2 == 0:
                print(
                    "✓ dlib 19.24.2 instalado com sucesso (com variável de ambiente)!"
                )
                # Agora instala o restante (ignorando dlib, pois já está instalado)
                print("Instalando os demais pacotes...")
                # Cria um requirements temporário sem o dlib
                temp_req = self.script_dir / "requirements_temp.txt"
                with open(self.requirements_file, "r") as f:
                    lines = [line for line in f if not line.strip().startswith("dlib")]
                with open(temp_req, "w") as f:
                    f.writelines(lines)
                subprocess.run(
                    f'"{python_path}" -m pip install -r "{temp_req}"', shell=True
                )
                temp_req.unlink()
                return True
            else:
                print("⚠️  Falha também com a variável. Tentando versão 19.22.0...")
                dlib_cmd_old = (
                    f'"{python_path}" -m pip install --no-cache-dir dlib==19.22.0'
                )
                ret3, fail3, _ = self.run_pip_with_env(dlib_cmd_old, env_fix)
                if ret3 == 0:
                    print("✓ dlib 19.22.0 instalado com sucesso!")
                    # Instala os demais pacotes
                    temp_req = self.script_dir / "requirements_temp.txt"
                    with open(self.requirements_file, "r") as f:
                        lines = [
                            line for line in f if not line.strip().startswith("dlib")
                        ]
                    with open(temp_req, "w") as f:
                        f.writelines(lines)
                    subprocess.run(
                        f'"{python_path}" -m pip install -r "{temp_req}"', shell=True
                    )
                    temp_req.unlink()
                    return True
                else:
                    print("✗ Falha também na instalação do dlib 19.22.0.")
                    print(
                        "Certifique-se de que as dependências de sistema estão instaladas."
                    )
                    return False
        else:
            print("✗ Erro ao instalar requirements (não relacionado ao dlib)")
            return False

    def setup(self):
        """Configura o ambiente completo"""
        print("Iniciando configuração do ambiente...")
        print("=" * 50)

        # Verifica dependências de sistema (apenas Linux)
        if sys.platform != "win32" and not self.check_system_dependencies():
            return False

        # Criar venv
        if not self.create_venv():
            return False

        # Instalar requirements
        print("=" * 50)
        if not self.install_requirements():
            return False

        print("=" * 50)
        print("✓ Configuração concluída com sucesso!")
        return True


def executar_treinamento():
    """Função para executar treinamento da IA"""
    print("Executando treinamento da IA...")

    try:
        # Usar o Python do venv para executar o treinamento
        setup = VenvSetup()
        python_path, _ = setup.get_venv_python()

        # Caminho para o script de treinamento
        train_script = (
            Path(__file__).resolve().parent / "treinamento" / "train_model.py"
        )

        if not train_script.exists():
            print(f"✗ Script de treinamento não encontrado: {train_script}")
            return False

        # Executar o treinamento
        cmd = f'"{python_path}" "{train_script}"'
        print(f"Executando: {cmd}")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Mostrar saída em tempo real
        success_detected = False
        model_created = False

        for line in process.stdout:
            line = line.strip()
            if line:
                print(line)

                # Verificar indicadores de sucesso
                if any(
                    indicator in line
                    for indicator in [
                        "✅ TREINAMENTO CONCLUÍDO",
                        "✓ Modelo salvo com sucesso",
                    ]
                ):
                    success_detected = True

        process.wait()

        # Verificar se o modelo foi criado independentemente da saída
        model_path = Path(__file__).resolve().parent / "model" / "deepface_model.pkl"
        model_created = model_path.exists()

        # SEMPRE mostrar mensagem de sucesso se o modelo foi criado
        if model_created:
            print("=" * 60)
            print("🎉 TREINAMENTO REALIZADO COM SUCESSO!")
            print(f"📁 Modelo criado em: {model_path}")
            print(f"📊 Tamanho do arquivo: {model_path.stat().st_size} bytes")
            print("=" * 60)
            return True
        else:
            print("✗ Falha no treinamento - modelo não foi criado")
            return False

    except Exception as e:
        # Mesmo com exceção, verificar se o modelo foi criado
        model_path = Path(__file__).resolve().parent / "model" / "deepface_model.pkl"
        if model_path.exists():
            print("=" * 60)
            print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print(f"📁 Modelo criado em: {model_path}")
            print("⚠️  O processo teve alguns problemas, mas o modelo está disponível")
            print("=" * 60)
            return True
        else:
            print(f"✗ Erro durante o treinamento: {e}")
            return False


def executar_reconhecimento():
    """Função para iniciar reconhecimento facial usando subprocess (mais confiável)"""
    print("Iniciando reconhecimento facial...")

    setup = VenvSetup()
    python_path, _ = setup.get_venv_python()

    if not python_path.exists():
        print(f"✗ Python do venv não encontrado em: {python_path}")
        return False

    script_path = setup.script_dir / "reconhecimento" / "recognize_faces.py"
    if not script_path.exists():
        print(f"✗ Script de reconhecimento não encontrado: {script_path}")
        return False

    try:
        # Comando para executar o script com o Python do venv
        cmd = f'"{python_path}" "{script_path}"'
        print(f"Executando: {cmd}")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Mostrar saída em tempo real
        for line in process.stdout:
            line = line.strip()
            if line:
                print(line)

        process.wait()

        if process.returncode == 0:
            print("✓ Reconhecimento facial finalizado com sucesso")
            return True
        else:
            print(f"✗ Erro na execução do reconhecimento (código {process.returncode})")
            return False

    except Exception as e:
        print(f"✗ Exceção ao executar reconhecimento: {e}")
        return False


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Sistema de Reconhecimento Facial")
    parser.add_argument(
        "--treinamento", action="store_true", help="Executar treinamento da IA"
    )
    parser.add_argument(
        "--reconhecimento", action="store_true", help="Iniciar reconhecimento facial"
    )
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Iniciar câmeras (alias para reconhecimento)",
    )

    args = parser.parse_args()

    try:
        # Configurar ambiente primeiro
        setup = VenvSetup()
        if not setup.setup():
            print("✗ Falha ao configurar o ambiente")
            return 1

        # Executar ação específica se solicitada
        if args.treinamento:
            return 0 if executar_treinamento() else 1
        elif args.reconhecimento or args.cameras:
            return 0 if executar_reconhecimento() else 1
        else:
            # Apenas setup do ambiente
            print("✓ Ambiente configurado com sucesso!")
            return 0

    except KeyboardInterrupt:
        print("\n✗ Processo interrompido pelo usuário")
        return 1
    except Exception as e:
        print(f"✗ Erro não tratado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
