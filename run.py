import os
import sys
import subprocess
import webbrowser
from time import sleep
from config import Config


def install_requirements():
    """Instala as dependências de forma robusta com feedback claro"""
    print("\n=== INSTALAÇÃO DE DEPENDÊNCIAS ===")

    # Verifica se o ambiente virtual existe
    venv_path = "venv" if sys.platform != "win32" else "venv\\Scripts"
    if not os.path.exists(venv_path):
        print("Criando ambiente virtual...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("✅ Ambiente virtual criado com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao criar ambiente virtual: {e}")
            return False

    # Determina o pip correto para o SO
    pip_cmd = (
        os.path.join("venv", "bin", "pip")
        if sys.platform != "win32"
        else os.path.join("venv", "Scripts", "pip.exe")
    )

    print("\nInstalando dependências...")
    try:
        # Instala as dependências básicas primeiro
        subprocess.check_call(
            [pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"]
        )

        # Instala os requirements em duas etapas (primeiro pacotes leves)
        light_deps = [
            "flask==2.3.2",
            "mysql-connector-python==8.0.33",
            "python-dotenv==1.0.0",
            "werkzeug==2.3.6",
            "numpy==1.24.3",
        ]
        subprocess.check_call([pip_cmd, "install"] + light_deps)

        # Depois instala os pacotes mais pesados
        heavy_deps = ["opencv-python==4.7.0.72", "face-recognition-models==0.3.0"]
        subprocess.check_call([pip_cmd, "install"] + heavy_deps)

        print("\n✅ Todas as dependências instaladas com sucesso!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Falha na instalação: {e}")
        return False


def initialize_database():
    """Inicializa o banco de dados com tratamento de erros"""
    print("\n=== CONFIGURAÇÃO DO BANCO DE DADOS ===")
    try:
        # Primeiro garante que o banco de dados existe
        if not Config.ensure_database_exists():
            print("❌ Falha ao verificar/criar o banco de dados")
            return False

        # Depois cria as tabelas
        from app.database import Database

        db = Database()
        db.initialize()  # Cria as tabelas no banco que já existe
        print("✅ Tabelas configuradas com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro no banco de dados: {e}")
        return False


def start_server():
    """Inicia o servidor Flask"""
    print("\n=== INICIANDO SERVIDOR ===")
    try:
        from server import app

        print("Servidor pronto em http://localhost:5000")
        webbrowser.open("http://localhost:5000")
        app.run(debug=True)
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")


def setup_directories():
    """Cria os diretórios necessários"""
    os.makedirs("uploads/faces", exist_ok=True)
    os.makedirs("uploads/temp", exist_ok=True)


def create_env_file():
    """Cria o arquivo .env se não existir"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("DB_HOST=localhost\n")
            f.write("DB_USER=root\n")
            f.write("DB_PASSWORD=\n")
            f.write("DB_NAME=reconhecimento_facial\n")
            f.write("SECRET_KEY=segredo-muito-seguro\n")
        print("ℹ️ Arquivo .env criado com configurações padrão")


def main():
    """Fluxo principal de execução"""
    try:
        # Configuração inicial
        create_env_file()
        setup_directories()

        # Instala dependências
        if not install_requirements():
            sys.exit(1)

        # Inicializa banco de dados
        if not initialize_database():
            sys.exit(1)

        # Inicia servidor
        start_server()
    except KeyboardInterrupt:
        print("\n⏹️ Aplicação encerrada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)


def setup_system():
    """Função alternativa para ser chamada pelo main.py"""
    main()


if __name__ == "__main__":
    main()
