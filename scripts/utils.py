"""Utilitários partilhados pelos scripts de Time Series Forecasting.

Este módulo centraliza funções e caminhos comuns aos scripts em ``scripts/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset" if (ROOT_DIR / "dataset").exists() else ROOT_DIR / "Dataset"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
PNG_DIR = IMAGES_DIR / "png"
SVG_DIR = IMAGES_DIR / "svg"
TABLES_DIR = OUTPUT_DIR / "tables"

# Estilo visual académico partilhado por todos os scripts.
RC_PARAMS: dict[RcKeyType, object] = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
}


def guardar_figura(fig: Figure, nome_base: str) -> tuple[Path, Path]:
    """Guarda a figura simultaneamente em PNG (alta resolução, 300 DPI) e SVG (vetorial)."""
    caminho_png = PNG_DIR / f"{nome_base}.png"
    caminho_svg = SVG_DIR / f"{nome_base}.svg"
    fig.savefig(caminho_png)
    fig.savefig(caminho_svg)
    plt.close(fig)
    print(f"    -> Gráficos gravados em:\n       [PNG] {caminho_png}\n       [SVG] {caminho_svg}")
    return caminho_png, caminho_svg
