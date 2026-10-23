#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "torch",
#   "scikit-learn",
#   "matplotlib",
# ]
# ///

"""
================================================================================
Experimento: a fase do ciclo de release também ajuda as redes neuronais?
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning

Registo reproduzível da escolha das features de release no script 03. Testa quatro
conjuntos de entrada no LSTM (a arquitetura que vence o benchmark de DL), com os
hiperparâmetros do pipeline (W=28, 15 épocas, batch 64, lr=0.001) e **cinco seeds**,
porque uma única corrida não distingue efeito de sorteio da inicialização.

Se `torch` não estiver disponível na máquina, este script é o único desta pasta que
não corre; a implementação que conta para o relatório está no script 03.

Execução (a partir da raiz do repositório):

    uv run scripts/analise_covariaveis_release/experimento_release_dl.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# O `utils.py` e o pacote `arquitecturas_neuronais` vivem um nível acima.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arquitecturas_neuronais import LSTMModel
from utils import calcular_metricas, carregar_serie_temporal, dividir_treino_teste

PASTA_EXPERIMENTO = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
WINDOW = 28
EPOCHS = 15
BATCH = 64
LR = 0.001

BASE = ["total_commits", "unique_authors", "log_churn", "files_changed", "sin_7_1", "cos_7_1"]


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstrói as features e a divisão do script 03, incluindo as duas dummies de fase."""
    df = carregar_serie_temporal("kernel_daily_ts.csv", "D")
    df["log_churn"] = np.log1p(df["total_churn"])

    t = np.arange(len(df))
    df["sin_7_1"] = np.sin(2 * np.pi * 1 * t / 7.0)
    df["cos_7_1"] = np.cos(2 * np.pi * 1 * t / 7.0)

    semanas = df["weeks_since_release"].astype("float64")
    conhecido = semanas.notna()
    df["fase_rc_inicial"] = ((semanas >= 2) & (semanas <= 5)).where(conhecido).astype("float64")
    df["fase_rc_final"] = (semanas >= 6).where(conhecido).astype("float64")

    train, test = dividir_treino_teste(df, "dias")

    # Mesma amostra de treino para todas as variantes (ver README, secção 9).
    return train.dropna(subset=["fase_rc_inicial"]), test


def criar_janelas(dados: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes de W dias com o alvo na coluna 0 (total_commits)."""
    X, y = [], []
    for i in range(len(dados) - WINDOW):
        X.append(dados[i : i + WINDOW, :])
        y.append(dados[i + WINDOW, 0])
    return np.array(X), np.array(y)


def treinar(colunas: list[str], train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, float]:
    """Treina um LSTM e devolve as quatro métricas fora da amostra."""
    torch.manual_seed(seed)

    scaler_X = MinMaxScaler().fit(train[colunas])
    scaler_y = MinMaxScaler().fit(train[["total_commits"]])

    X_train, y_train = criar_janelas(scaler_X.transform(train[colunas]))
    X_test, y_test = criar_janelas(scaler_X.transform(test[colunas]))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Validação cronológica nos últimos 10% do treino, como no script 03
    n_val = int(len(X_train_t) * 0.1)
    X_val, y_val = X_train_t[-n_val:], y_train_t[-n_val:]
    X_fit, y_fit = X_train_t[:-n_val], y_train_t[:-n_val]

    modelo = LSTMModel(n_features=len(colunas), hidden_size=50)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    perda = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_fit, y_fit), batch_size=BATCH, shuffle=False
    )

    for _ in range(EPOCHS):
        modelo.train()
        for x_batch, y_batch in loader:
            otimizador.zero_grad()
            perda(modelo(x_batch), y_batch).backward()
            otimizador.step()

    modelo.eval()
    with torch.no_grad():
        previsao = modelo(X_test_t).numpy()

    return calcular_metricas(
        scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        scaler_y.inverse_transform(previsao).flatten(),
    )


def main() -> None:
    train, test = preparar_dados()
    print(f"[*] Treino: {len(train)} dias | Teste: {len(test)} dias | seeds: {SEEDS}\n")

    variantes = [
        ("F=6 sem release", BASE),
        ("F=7 so weeks_since_release", BASE + ["weeks_since_release"]),
        ("F=8 weeks + is_merge_window", BASE + ["weeks_since_release", "is_merge_window"]),
        ("F=8 duas dummies (adotado)", BASE + ["fase_rc_inicial", "fase_rc_final"]),
    ]

    registos = []

    for nome, colunas in variantes:
        corridas = [treinar(colunas, train, test, seed) for seed in SEEDS]
        resumo: dict[str, object] = {"Features": nome, "F": len(colunas)}
        texto = f"{nome:30s} F={len(colunas)}"

        for metrica in ("REQM", "EAM", "EPAM", "EPAMs"):
            valores = np.array([c[metrica] for c in corridas])
            resumo[f"{metrica} media"] = round(float(valores.mean()), 2)
            resumo[f"{metrica} dp"] = round(float(valores.std()), 2)
            texto += f"  {metrica}={valores.mean():7.2f}+-{valores.std():5.2f}"

        print(texto)
        registos.append(resumo)

    print("\nMetricas: media +- desvio-padrao em 5 seeds (variabilidade da inicializacao)")

    destino = PASTA_EXPERIMENTO / "resultados_experimento_dl.csv"
    pd.DataFrame(registos).to_csv(destino, index=False)
    print(f"Tabela gravada em: {destino}")


if __name__ == "__main__":
    main()
