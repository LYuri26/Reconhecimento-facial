import sys
from pathlib import Path
from run import setup_system


def configurar_ambiente():
    """Configura o ambiente Python"""
    projeto_dir = str(Path(__file__).parent)
    if projeto_dir not in sys.path:
        sys.path.insert(0, projeto_dir)


if __name__ == "__main__":
    configurar_ambiente()
    setup_system()
