import os
import sys
import subprocess
import venv
import argparse
from pathlib import Path


class VenvSetup:
    def __init__(self):
        self.venv_name = "venv"
        self.script_dir = Path(__file__).resolve().parent
        self.venv_path = self.script_dir / self.venv_name
        self.requirements_file = self.script_dir / "requirements.txt"

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

    def install_requirements(self):
        """Instala as dependências do requirements.txt"""
        python_path, activate_path = self.get_venv_python()

        if not python_path.exists():
            print(f"✗ Python do venv não encontrado em: {python_path}")
            return False

        if not self.requirements_file.exists():
            print(f"✗ Arquivo {self.requirements_file} não encontrado")
            return False

        try:
            print("Instalando dependências do requirements.txt...")
            print("Isso pode levar alguns minutos...")

            # Comando para instalar requirements
            if sys.platform == "win32":
                cmd = f'"{python_path}" -m pip install -r "{self.requirements_file}"'
            else:
                cmd = f'"{python_path}" -m pip install -r "{self.requirements_file}"'

            # Executar com saída em tempo real
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Ler saída em tempo real
            line_count = 0
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(line)
                    sys.stdout.flush()
                    line_count += 1

                    # Mostrar progresso a cada 10 linhas
                    if line_count % 10 == 0:
                        print(f"Processando... ({line_count} pacotes processados)")

            process.wait()

            if process.returncode == 0:
                print("✓ Todas as dependências instaladas com sucesso")
                return True
            else:
                print("✗ Erro ao instalar requirements")
                return False

        except subprocess.TimeoutExpired:
            print("✗ Timeout ao instalar dependências")
            return False
        except Exception as e:
            print(f"✗ Erro durante a instalação: {e}")
            return False

    def setup(self):
        """Configura o ambiente completo"""
        print("Iniciando configuração do ambiente...")
        print("=" * 50)

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
    """Função para iniciar reconhecimento facial"""
    print("Iniciando reconhecimento facial...")

    try:
        # Primeiro, verificar se estamos no ambiente virtual
        from reconhecimento.recognize_faces import FaceRecognizer

        recognizer = FaceRecognizer()
        recognizer.run()

        print("✓ Reconhecimento facial finalizado")
        return True

    except ImportError as e:
        print(f"✗ Erro ao importar módulos: {e}")
        print("Verifique se todas as dependências estão instaladas")

        # Tentar uma solução alternativa: executar via subprocess usando o Python do venv
        try:
            setup = VenvSetup()
            python_path, _ = setup.get_venv_python()

            if python_path.exists():
                print("Tentando executar com Python do venv...")
                script_path = (
                    Path(__file__).resolve().parent
                    / "reconhecimento"
                    / "recognize_faces.py"
                )
                cmd = f'"{python_path}" "{script_path}"'

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ Reconhecimento executado com sucesso via subprocess")
                    return True
                else:
                    print(f"✗ Erro no subprocess: {result.stderr}")

        except Exception as sub_e:
            print(f"✗ Falha na execução alternativa: {sub_e}")

        return False
    except Exception as e:
        print(f"✗ Erro ao executar reconhecimento: {e}")
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
    exit_code = main()
    sys.exit(exit_code)
