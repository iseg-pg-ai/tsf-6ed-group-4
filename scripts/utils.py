"""Utilitários partilhados pelos scripts de Time Series Forecasting.

Este módulo centraliza funções e caminhos comuns aos scripts em ``scripts/``.
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType
from numpy.typing import ArrayLike

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


def iniciar_script(numero: str, titulo: str) -> None:
    """Imprime o cabeçalho padrão de arranque de um script."""
    print("=" * 80)
    print(f"INÍCIO DO SCRIPT {numero}: {titulo}")
    print("=" * 80)


def finalizar_script(numero: str) -> None:
    """Imprime o rodapé padrão de conclusão de um script."""
    print("=" * 80)
    print(f"[OK] SCRIPT {numero} CONCLUÍDO COM SUCESSO!")
    print(f"    Ficheiros gerados em: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


def carregar_serie_temporal(nome_ficheiro: str, frequencia: str) -> pd.DataFrame:
    """Carrega uma série de ``DATASET_DIR`` e aplica a higienização operacional padrão.

    Descarta a última observação incompleta e fixa explicitamente a frequência do
    ``DatetimeIndex``, conforme documentado em AGENTS.md.
    """
    caminho = DATASET_DIR / nome_ficheiro
    if not caminho.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {caminho}")

    df = pd.read_csv(
        caminho,
        parse_dates=["committer_date"],
        index_col="committer_date",
    ).iloc[:-1]
    df.index.freq = frequencia
    return df


def resumo_serie(df: pd.DataFrame, rotulo: str, unidade: str) -> None:
    """Imprime o número de observações e o intervalo temporal de uma série."""
    inicio = cast(pd.Timestamp, df.index.min())
    fim = cast(pd.Timestamp, df.index.max())
    print(
        f"    -> {rotulo}: {len(df)} {unidade} "
        f"({inicio:%Y-%m-%d} a {fim:%Y-%m-%d})"
    )


def dividir_treino_teste(df: pd.DataFrame, unidade: str, proporcao: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide a série cronologicamente em treino/teste e imprime o resumo dos períodos."""
    n_treino = int(proporcao * len(df))
    train = df.iloc[:n_treino].copy()
    test = df.iloc[n_treino:].copy()

    resumo_serie(df, "Amostra total", unidade)
    resumo_serie(train, f"Amostra de treino ({proporcao:.0%})", unidade)
    resumo_serie(test, f"Amostra de teste  ({1 - proporcao:.0%})", unidade)
    return train, test


def calcular_metricas(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calcula as quatro métricas obrigatórias da disciplina (Slide 13):

    REQM (RMSE), EAM (MAE), EPAM (MAPE) e EPAMs (sMAPE).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    reqm = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    eam = np.mean(np.abs(y_true_arr - y_pred_arr))
    epam = np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100
    epams = np.mean(200.0 * np.abs(y_pred_arr - y_true_arr) / (np.abs(y_true_arr) + np.abs(y_pred_arr)))

    return {
        "REQM": round(float(reqm), 2),
        "EAM": round(float(eam), 2),
        "EPAM": round(float(epam), 2),
        "EPAMs": round(float(epams), 2),
    }


def guardar_figura(fig: Figure, nome_base: str) -> None:
    """Guarda a figura simultaneamente em PNG (alta resolução, 300 DPI) e SVG (vetorial)."""
    caminho_png = PNG_DIR / f"{nome_base}.png"
    caminho_svg = SVG_DIR / f"{nome_base}.svg"
    fig.savefig(caminho_png)
    fig.savefig(caminho_svg)
    plt.close(fig)
    print(f"    -> Gráficos gravados em:\n       [PNG] {caminho_png}\n       [SVG] {caminho_svg}")


def guardar_tabela(df: pd.DataFrame, nome_base: str, descricao: str = "Tabela") -> None:
    """Guarda um DataFrame em CSV na pasta de tabelas."""
    caminho = TABLES_DIR / f"{nome_base}.csv"
    df.to_csv(caminho, index=False)
    print(f"    -> {descricao} gravada em: {caminho}")


def guardar_markdown(conteudo: str, nome_ficheiro: str, descricao: str = "Resumo") -> None:
    """Grava conteúdo Markdown na pasta de saída."""
    caminho = OUTPUT_DIR / nome_ficheiro
    caminho.write_text(conteudo, encoding="utf-8")
    print(f"    -> {descricao} gravado em: {caminho}")
