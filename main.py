#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para criar ambiente virtual, instalar dependências e executar ações.
"""

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

        requirements_path = self.script_dir / self.requirements_file

        if not requirements_path.exists():
            print(f"✗ Arquivo {self.requirements_file} não encontrado")
            return False

        try:
            print("Instalando dependências do requirements.txt...")

            # Comando para instalar requirements
            if sys.platform == "win32":
                cmd = f'"{python_path}" -m pip install -r "{requirements_path}"'
            else:
                cmd = f'"{python_path}" -m pip install -r "{requirements_path}"'

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=600
            )

            if result.returncode == 0:
                print("✓ Todas as dependências instaladas com sucesso")
                return True
            else:
                print("✗ Erro ao instalar requirements:")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("✗ Timeout ao instalar dependências")
            return False
        except Exception as e:
            print(f"✗ Erro durante a instalação: {e}")
            return False

    def setup_environment(self):
        """Configura o ambiente completo"""
        print("Iniciando configuração do ambiente...")

        # Criar venv
        if not self.create_venv():
            return False

        # Instalar requirements
        if not self.install_requirements():
            return False

        print("✓ Configuração concluída com sucesso!")
        return True


def executar_treinamento():
    """Função para executar treinamento da IA"""
    print("Executando treinamento da IA...")
    # Aqui você adiciona o código do treinamento
    print("✓ Treinamento concluído com sucesso!")
    return True


def executar_cameras():
    """Função para iniciar câmeras"""
    print("Iniciando câmeras...")
    # Aqui você adiciona o código das câmeras
    print("✓ Câmeras iniciadas com sucesso!")
    return True


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Sistema de Reconhecimento Facial")
    parser.add_argument(
        "--treinamento", action="store_true", help="Executar treinamento da IA"
    )
    parser.add_argument("--cameras", action="store_true", help="Iniciar câmeras")

    args = parser.parse_args()

    try:
        # Configurar ambiente primeiro
        setup = VenvSetup()
        if not setup.setup_environment():
            print("✗ Falha ao configurar o ambiente")
            return 1

        # Executar ação específica se solicitada
        if args.treinamento:
            return 0 if executar_treinamento() else 1
        elif args.cameras:
            return 0 if executar_cameras() else 1
        else:
            # Apenas setup do ambiente
            print("✓ Ambiente configurado com sucesso!")
            return 0

    except Exception as e:
        print(f"✗ Erro não tratado: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
