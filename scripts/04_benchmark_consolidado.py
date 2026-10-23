#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
# ]
# ///

"""
================================================================================
Script 04: Consolidação do Benchmark Comparativo (Econometria vs. Deep Learning)
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning
Docente: Prof. Jorge Caiado e Rubens Dias
Grupo 4 - Caso de Estudo: Linux Kernel Commit Velocity (2005-2026)

Objetivo:
- Consolidar os resultados dos scripts 02 e 03 numa única Tabela Mestre (tab_08_master_benchmark_metricas.csv)
  e numa única Figura Mestre (fig_08_master_benchmark.[png|svg]).
- Eliminar a confusão de nomenclatura entre os artefactos gerados individualmente.
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import (
    PNG_DIR,
    RC_PARAMS,
    SVG_DIR,
    TABLES_DIR,
    finalizar_script,
    guardar_figura,
    guardar_tabela,
    iniciar_script,
)

# Configuração de Caminhos e Ambiente
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual académica (definida em utils.py)
plt.rcParams.update(RC_PARAMS)


# Consolidação da Tabela Mestre (tab_08_master_benchmark_metricas.csv)
def consolidar_benchmark() -> pd.DataFrame:
    """Lê os resultados dos scripts 02 e 03 e gera a Tabela Mestre consolidada."""
    print("[*] A consolidar a Tabela Mestre de Benchmark (tab_08_master_benchmark_metricas.csv)...")

    # 1. Ler resultados dos modelos econométricos (Script 02)
    df_econ = pd.read_csv(TABLES_DIR / "tab_05_comparacao_econometrica.csv")

    # 2. Ler resultados dos modelos de Deep Learning (Script 03)
    df_dl = pd.read_csv(TABLES_DIR / "tab_07_dl_benchmark_metricas.csv")

    # 3. Formatar para uma estrutura unificada
    registos = []

    # Adicionar modelos econométricos
    for _, row in df_econ.iterrows():
        registos.append({
            "Modelo": row["Modelo"],
            "Família": "Econometria Clássica",
            "Frequência do Alvo": "Semanal",
            "Regime": row["Família"],
            "REQM (RMSE)": row["REQM"],
            "EAM (MAE)": row["EAM"],
            "EPAM (MAPE)": row["EPAM (%)"],
            "EPAMs (sMAPE)": row["EPAMs (%)"],
        })

    # Adicionar modelos de Deep Learning
    for _, row in df_dl.iterrows():
        registos.append({
            "Modelo": row["Modelo"],
            "Família": "Deep Learning",
            "Frequência do Alvo": "Diária",
            "Regime": row["Regime"],
            "REQM (RMSE)": row["REQM"],
            "EAM (MAE)": row["EAM"],
            "EPAM (MAPE)": row["EPAM"],
            "EPAMs (sMAPE)": row["EPAMs"],
        })

    df_master = pd.DataFrame(registos).sort_values(by="REQM (RMSE)", ascending=True)
    guardar_tabela(df_master, "tab_08_master_benchmark_metricas", "Tabela Mestre")

    return df_master


# Geração da Figura Mestre (fig_08_master_benchmark.[png|svg])
def gerar_figura_master(df_master: pd.DataFrame) -> None:
    """Gera a Figura Mestre em dois painéis, um por família de modelos.

    As duas famílias não partilham unidade nem horizonte de avaliação: a Deep Learning é
    avaliada um passo à frente numa série diária (commits/dia) e a econometria é avaliada
    em 224 passos dinâmicos numa série semanal (commits/semana). Barras lado a lado na
    mesma escala sugeriam que a DL é quatro vezes melhor, quando o que muda é a unidade.
    Cada família tem o seu painel, com eixo de x independente, e a leitura entre famílias
    faz-se pelo EPAM e pelo EPAMs, que são adimensionais.
    """
    print("[*] A gerar Figura Mestre de Benchmark (fig_08_master_benchmark)...")

    familias = [
        (
            "Deep Learning",
            "#2ca02c",
            "(a) Deep Learning — Série Diária",
            "1.563 dias de teste · previsão estática a 1 passo",
            "REQM (RMSE) em commits/dia — quanto mais baixo, melhor",
        ),
        (
            "Econometria Clássica",
            "#1f77b4",
            "(b) Econometria Clássica — Série Semanal",
            "224 semanas de teste · previsão dinâmica",
            "REQM (RMSE) em commits/semana — quanto mais baixo, melhor",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.suptitle(
        "Tabela Mestre de Benchmark: Econometria Clássica vs. Deep Learning\n"
        "Avaliação Out-of-Sample (REQM / RMSE, por família e na sua própria unidade)\n"
        "ISEG — Time Series Forecasting — Aulas #1 a #5",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    for ax, (familia, cor, titulo, subtitulo, rotulo_x) in zip(axes, familias):
        df_plot = df_master[df_master["Família"] == familia].sort_values(by="REQM (RMSE)", ascending=True)

        y_pos = np.arange(len(df_plot))
        barras = ax.barh(
            y_pos,
            df_plot["REQM (RMSE)"],
            color=cor,
            alpha=0.85,
            height=0.72,
        )

        # Os rótulos ficam dentro dos limites do eixo: o eixo de x é esticado para lhes
        # dar espaço (antes, os valores dos modelos de maior REQM saíam da figura).
        reqm_max = float(df_plot["REQM (RMSE)"].max())
        for barra, reqm, epam in zip(barras, df_plot["REQM (RMSE)"], df_plot["EPAM (MAPE)"]):
            ax.text(
                barra.get_width() + reqm_max * 0.015,
                barra.get_y() + barra.get_height() / 2,
                f"{reqm:.1f} · EPAM {epam:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot["Modelo"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, reqm_max * 1.45)
        ax.set_xlabel(rotulo_x)
        ax.set_title(f"{titulo}\n{subtitulo}", loc="left", fontweight="bold", fontsize=10.5)
        ax.grid(axis="x", alpha=0.4)

    fig.text(
        0.5,
        0.015,
        "O REQM não é comparável entre os dois painéis: mudam a unidade (commits/dia vs. commits/semana) e o\n"
        "horizonte de avaliação (1 passo estático vs. 224 passos dinâmicos). A comparação entre famílias faz-se\n"
        "pelo EPAM e pelo EPAMs, que são adimensionais.",
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.075, 1, 0.94))
    guardar_figura(fig, "fig_08_master_benchmark")


def main():
    iniciar_script("04", "CONSOLIDAÇÃO DO BENCHMARK COMPARATIVO")
    df_master = consolidar_benchmark()
    gerar_figura_master(df_master)
    print("\n[*] Resumo da Tabela Mestre:")
    print(df_master.to_string(index=False))
    finalizar_script("04")


if __name__ == "__main__":
    main()
