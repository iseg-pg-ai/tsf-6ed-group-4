#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "statsmodels",
#   "scipy",
#   "matplotlib",
# ]
# ///

"""
================================================================================
Experimento: as variáveis do ciclo de release ajudam a prever a atividade?
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning

Este script é o registo reproduzível do experimento que justifica a inclusão da
fase do ciclo de release na especificação ARIMAX vencedora do script 02. Não
produz artefactos do relatório: imprime a tabela comparativa e grava-a em
`resultados_experimento.csv`, nesta mesma pasta.

A implementação que conta para o relatório é a do script 02. Este ficheiro existe
para que qualquer pessoa (colega ou docente) possa replicar a decisão e ver as
alternativas que foram testadas e rejeitadas.

Execução (a partir da raiz do repositório):

    uv run scripts/analise_covariaveis_release/experimento_covariaveis_release.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# O `utils.py` vive na pasta `scripts/`, um nível acima: sem isto, `from utils import`
# falha, porque o sys.path[0] de um script é a sua própria pasta.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import calcular_metricas, carregar_serie_temporal, dividir_treino_teste

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
ORDEM = (0, 1, 2)
LAGS_LJUNGBOX = (15, 20)
MAXITER = 100

COVARIAVEIS = ["unique_authors", "log_churn"]
FASE_2_DUMMIES = ["fase_rc_inicial", "fase_rc_final"]
FASE_3_DUMMIES = ["fase_merge", "fase_rc_inicial", "fase_rc_final"]
INTERACAO_ERA = ["fase_rc_inicial_x_era", "fase_rc_final_x_era"]


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega a série semanal e reconstrói exatamente a divisão e as variáveis do script 02.

    Devolve `(df, train_amostra, test)`. A amostra de treino é a interseção da divisão
    80:20 com as semanas em que a fase de release é conhecida (as 10 primeiras semanas
    de 2005 são anteriores ao primeiro tag e ficam de fora).
    """
    df = carregar_serie_temporal("kernel_weekly_ts.csv", "W-MON")

    df["log_churn"] = np.log1p(df["total_churn"])

    # A contagem de semanas é usada em nível nos candidatos e como base das dummies.
    semanas = df["weeks_since_release"].astype("float64")
    conhecido = semanas.notna()

    # Dummies de fase. `where(conhecido)` preserva o NaN das semanas sem tag: um
    # "desconhecido" nunca pode ser codificado como a fase de merge window.
    df["fase_merge"] = (semanas <= 1).where(conhecido).astype("float64")
    df["fase_rc_inicial"] = ((semanas >= 2) & (semanas <= 5)).where(conhecido).astype("float64")
    df["fase_rc_final"] = (semanas >= 6).where(conhecido).astype("float64")

    # Interação com a era: o sinal do efeito da fase inverte-se em 2015 (ver tab_02_outliers.csv).
    # O NaN das semanas sem tag propaga-se no produto, portanto não há aqui zeros falsos.
    era_moderna = (df.index.year >= 2015).astype("float64")
    df["fase_rc_inicial_x_era"] = df["fase_rc_inicial"] * era_moderna
    df["fase_rc_final_x_era"] = df["fase_rc_final"] * era_moderna

    train, test = dividir_treino_teste(df, "semanas")
    train_amostra = train.dropna(subset=["weeks_since_release"])

    return df, train_amostra, test


def construir_candidatos() -> list[tuple[str, list[str]]]:
    """Lista `(nome, colunas exógenas)` do modelo de controlo e de cada alternativa testada."""
    return [
        ("Covariáveis (controlo)", list(COVARIAVEIS)),
        ("+ is_merge_window (dummy)", COVARIAVEIS + ["is_merge_window"]),
        ("+ weeks_since_release (contagem)", COVARIAVEIS + ["weeks_since_release"]),
        ("+ rc_week (contagem)", COVARIAVEIS + ["rc_week"]),
        ("Só fase (sem covariáveis)", FASE_2_DUMMIES),
        ("+ Fase: 2 dummies (merge = referência)", COVARIAVEIS + FASE_2_DUMMIES),
        ("+ Fase: 3 dummies (sem referência)", COVARIAVEIS + FASE_3_DUMMIES),
        ("+ Fase x Era (interação 2015+)", COVARIAVEIS + FASE_2_DUMMIES + INTERACAO_ERA),
    ]


def estimar(nome: str, colunas: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Estima um SARIMAX(0,1,2) e devolve as métricas fora da amostra e o diagnóstico dos resíduos."""
    y_train = train["total_commits"]
    y_test = test["total_commits"]

    modelo = SARIMAX(
        y_train,
        exog=train[colunas],
        order=ORDEM,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    ajustado = modelo.fit(disp=False, maxiter=MAXITER)

    previsao = ajustado.forecast(steps=len(test), exog=test[colunas])
    metricas = calcular_metricas(y_test, previsao)

    ljungbox = acorr_ljungbox(ajustado.resid, lags=list(LAGS_LJUNGBOX), return_df=True)

    return {
        "Especificação": nome,
        "Exógenas": len(colunas),
        "REQM": metricas["REQM"],
        "EPAM (%)": metricas["EPAM"],
        "AIC": round(float(ajustado.aic), 2),
        "Ljung-Box 15 (p)": round(float(ljungbox.loc[15, "lb_pvalue"]), 4),
        "Ljung-Box 20 (p)": round(float(ljungbox.loc[20, "lb_pvalue"]), 4),
    }


def main() -> None:
    print("[*] A carregar a série e a reconstruir a divisão do script 02...")
    _, train, test = preparar_dados()
    print(f"    Treino: {len(train)} semanas | Teste: {len(test)} semanas\n")

    registos = []
    for nome, colunas in construir_candidatos():
        print(f"[*] A estimar: {nome}...")
        registos.append(estimar(nome, colunas, train, test))

    tabela = pd.DataFrame(registos)

    print("\n" + "=" * 110)
    print("COMPARAÇÃO DE ALTERNATIVAS (SARIMAX(0,1,2), mesma amostra de treino e de teste)")
    print("=" * 110)
    print(tabela.to_string(index=False))

    controlo = float(tabela.loc[tabela["Especificação"] == "Covariáveis (controlo)", "REQM"].iloc[0])
    print(f"\nREQM do controlo (sem nenhuma variável de release): {controlo:.2f}")

    destino = PASTA_EXPERIMENTO / "resultados_experimento.csv"
    tabela.to_csv(destino, index=False)
    print(f"Tabela gravada em: {destino}")


if __name__ == "__main__":
    main()
