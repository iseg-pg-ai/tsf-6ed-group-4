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
    """Gera a Figura Mestre com a comparação de todos os modelos."""
    print("[*] A gerar Figura Mestre de Benchmark (fig_08_master_benchmark)...")

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        "Tabela Mestre de Benchmark: Econometria Clássica vs. Deep Learning\n"
        "Avaliação Out-of-Sample (REQM / RMSE)\n"
        "ISEG — Time Series Forecasting — Aulas #1 a #5",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    # Paleta de cores por família
    cores_familia = {
        "Econometria Clássica": "#1f77b4",
        "Deep Learning": "#2ca02c",
    }

    # Ordenar por REQM
    df_plot = df_master.sort_values(by="REQM (RMSE)", ascending=True)

    # Criar gráfico de barras horizontais
    y_pos = np.arange(len(df_plot))
    bars = ax.barh(
        y_pos,
        df_plot["REQM (RMSE)"],
        color=[cores_familia[f] for f in df_plot["Família"]],
        alpha=0.85,
    )

    # Adicionar rótulos nas barras
    for bar, reqm, epam in zip(bars, df_plot["REQM (RMSE)"], df_plot["EPAM (MAPE)"]):
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"REQM={reqm:.1f} | EPAM={epam:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Modelo"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("REQM (RMSE) — Quanto mais baixo, melhor")
    ax.set_title(
        "Comparação de Todos os Modelos Avaliados (224 Semanas de Teste para Econometria, 1.563 Dias para DL)",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )

    # Legenda
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=cores_familia["Econometria Clássica"], alpha=0.85, label="Econometria Clássica"),
        Patch(facecolor=cores_familia["Deep Learning"], alpha=0.85, label="Deep Learning"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
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
