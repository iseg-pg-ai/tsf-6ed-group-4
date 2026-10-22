#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "torch",
#   "scikit-learn",
#   "matplotlib",
#   "seaborn",
# ]
# ///

"""
================================================================================
Script 03: Modelação com Aprendizagem Profunda (Deep Learning para Séries Temporais)
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning
Docente: Prof. Jorge Caiado e Rubens Dias
Grupo 4 - Caso de Estudo: Linux Kernel Commit Velocity (2005-2026)

Conformidade com os Slides da Aula #5 (Slides 79 a 111):
- Formulação por janelas deslizantes (Sliding Windows, Slide 85)
- 1D-CNN (Convolução Unidimensional, Slides 81-84, 87 e 92-94)
- SimpleRNN (Rede Recorrente Simples, Slides 95-96, 101)
- LSTM (Long Short-Term Memory, Slides 97-98, 102)
- GRU (Gated Recurrent Unit, Slides 99, 103)
- Regime Univariado vs. Multivariado (Slides 106-108)
- Avaliação out-of-sample (REQM, EAM, EPAM e EPAMs) (Slide 13, 108)
- Cuidado com sobreajustamento em séries temporais (Slide 94 e 111)
================================================================================
"""

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from arquitecturas_neuronais import CNN1D, GRUModel, LSTMModel, SimpleRNNModel
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from utils import (
    PNG_DIR,
    RC_PARAMS,
    SVG_DIR,
    TABLES_DIR,
    calcular_metricas,
    carregar_serie_temporal,
    dividir_treino_teste,
    finalizar_script,
    guardar_figura,
    guardar_markdown_template,
    guardar_tabela,
    iniciar_script,
)

warnings.filterwarnings("ignore")

# Configuração de Caminhos e Ambiente
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual académica (definida em utils.py)
plt.rcParams.update(RC_PARAMS)

# Configuração de dispositivo (CPU devido a incompatibilidade da placa gráfica de alguns colegas)
device = torch.device("cpu")


# Ingestão de Dados e Divisão Cronológica (80:20)
def carregar_dados_e_preparar() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Carrega a série diária, constrói variáveis exógenas e executa a divisão cronológica."""
    print("[*] A carregar série diária e a aplicar divisão treino/teste (80:20)...")

    df = carregar_serie_temporal("kernel_daily_ts.csv", "D")

    # Transformação logarítmica do code churn para estabilização de variância
    df["log_churn"] = np.log1p(df["total_churn"])

    # Codificação harmónica do dia da semana (s=7)
    t = np.arange(len(df))
    df["sin_7_1"] = np.sin(2 * np.pi * 1 * t / 7.0)
    df["cos_7_1"] = np.cos(2 * np.pi * 1 * t / 7.0)

    # Colunas de características para o regime multivariado
    features_mult = [
        "total_commits",
        "unique_authors",
        "log_churn",
        "files_changed",
        "sin_7_1",
        "cos_7_1",
    ]

    # Divisão estritamente cronológica 80:20 (sem baralhar)
    train, test = dividir_treino_teste(df, "dias")

    return train, test, features_mult


# Formulação por Janelas Deslizantes (Sliding Windows, Slide 85)
def criar_janelas_deslizantes(
    dados_scaled: np.ndarray, window_size: int, target_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Converte a série temporal numa matriz de pares (X, y) com janela fixa.

    X[i] = dados[i : i + window_size, :]
    y[i] = dados[i + window_size, target_idx]
    """
    X, y = [], []
    for i in range(len(dados_scaled) - window_size):
        X.append(dados_scaled[i : i + window_size, :])
        y.append(dados_scaled[i + window_size, target_idx])
    return np.array(X), np.array(y)


def preparar_janelas_deslizantes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features_mult: list[str],
    window_size: int = 28,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    MinMaxScaler,
]:
    """Aplica normalização e constrói janelas deslizantes univariadas e multivariadas.

    O ajuste do MinMaxScaler é feito exclusivamente no treino para prevenir data leakage.
    """
    print(f"[*] A construir janelas deslizantes (W={window_size} dias)...")

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler_X.fit_transform(train[features_mult])
    test_scaled = scaler_X.transform(test[features_mult])

    # Normalizar a variável alvo (total_commits) para o regime univariado
    train_scaled_y = scaler_y.fit_transform(train[["total_commits"]])
    test_scaled_y = scaler_y.transform(test[["total_commits"]])

    # Regime Univariado (F=1)
    X_train_uni, y_train_uni = criar_janelas_deslizantes(train_scaled_y, window_size)
    X_test_uni, y_test_uni = criar_janelas_deslizantes(test_scaled_y, window_size)

    # Regime Multivariado (F=6)
    X_train_multi, y_train_multi = criar_janelas_deslizantes(train_scaled, window_size)
    X_test_multi, y_test_multi = criar_janelas_deslizantes(test_scaled, window_size)

    print(f"    -> Regime Univariado: X_train={X_train_uni.shape}, X_test={X_test_uni.shape}")
    print(f"    -> Regime Multivariado: X_train={X_train_multi.shape}, X_test={X_test_multi.shape}")

    dados_uni = (X_train_uni, y_train_uni, X_test_uni, y_test_uni)
    dados_multi = (X_train_multi, y_train_multi, X_test_multi, y_test_multi)

    return dados_uni, dados_multi, scaler_y


# Treino e Avaliação de Modelos Neuronais (Slide 80, 108)
def treinar_e_avaliar_modelo(
    nome_modelo: str,
    modelo: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: MinMaxScaler,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
) -> dict:
    """Treina o modelo PyTorch e avalia fora da amostra (Slide 80, 108).

    Retorna métricas e previsões desnormalizadas.
    """
    print(f"    -> A treinar {nome_modelo} (epochs={epochs}, batch={batch_size}, lr={lr})...")

    modelo = modelo.to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Tensores
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Data Loader
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Histórico de perda
    train_losses = []
    val_losses = []

    # Divisão treino/validação (90:10 do treino para monitorização de sobreajustamento)
    n_val = int(len(X_train) * 0.1)
    X_val_t = X_tr_t[-n_val:]
    y_val_t = y_tr_t[-n_val:]
    X_tr_t = X_tr_t[:-n_val]
    y_tr_t = y_tr_t[:-n_val]

    for epoch in range(epochs):
        modelo.train()
        batch_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = modelo(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Validação
        modelo.eval()
        with torch.no_grad():
            val_pred = modelo(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

    # Previsão no conjunto de teste
    modelo.eval()
    with torch.no_grad():
        pred_scaled = modelo(X_te_t).cpu().numpy()

    # Desnormalização
    pred = scaler_y.inverse_transform(pred_scaled).flatten()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metricas = calcular_metricas(y_test_real, pred)

    df_historico = pd.DataFrame({
        "Modelo": nome_modelo,
        "Regime": "Univariado (F=1)" if "Univariado" in nome_modelo else "Multivariado (F=6)",
        "Época": list(range(1, epochs + 1)),
        "Train_Loss": np.round(train_losses, 6),
        "Val_Loss": np.round(val_losses, 6),
    })

    return {
        "previsoes": pred,
        "y_real": y_test_real,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "historico": df_historico,
        "metricas": metricas,
    }


# Treino de modelos
def treinar_modelos(
    dados_uni: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dados_multi: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scaler_y: MinMaxScaler,
    window_size: int = 28,
    epochs: int = 15,
) -> tuple[dict, dict, dict]:
    """Instancia, treina e avalia as 4 arquiteturas neuronais nos regimes univariado e multivariado."""
    print("\n[*] A treinar modelos de Aprendizagem Profunda...")

    X_train_uni, y_train_uni, X_test_uni, y_test_uni = dados_uni
    X_train_multi, y_train_multi, X_test_multi, y_test_multi = dados_multi

    resultados = {}
    hist_loss = {}
    previsoes_teste = {}
    historicos = []

    # --- Regime Univariado ---
    print("\n--- Regime Univariado (F=1: apenas total_commits) ---")

    modelos_uni = {
        "1D-CNN Univariado": CNN1D(n_features=1, window_size=window_size),
        "SimpleRNN Univariado": SimpleRNNModel(n_features=1, hidden_size=50),
        "LSTM Univariado": LSTMModel(n_features=1, hidden_size=50),
        "GRU Univariado": GRUModel(n_features=1, hidden_size=50),
    }

    for nome, modelo in modelos_uni.items():
        res = treinar_e_avaliar_modelo(
            nome,
            modelo,
            X_train_uni,
            y_train_uni,
            X_test_uni,
            y_test_uni,
            scaler_y,
            epochs=epochs,
            batch_size=64,
            lr=0.001,
        )
        resultados[nome] = res["metricas"]
        hist_loss[nome] = {"train": res["train_losses"], "val": res["val_losses"]}
        previsoes_teste[nome] = {"pred": res["previsoes"], "real": res["y_real"]}
        historicos.append(res["historico"])

    # --- Regime Multivariado ---
    print("\n--- Regime Multivariado (F=6: total_commits + exógenas + harmónicas) ---")

    modelos_multi = {
        "1D-CNN Multivariado": CNN1D(n_features=6, window_size=window_size),
        "SimpleRNN Multivariado": SimpleRNNModel(n_features=6, hidden_size=50),
        "LSTM Multivariado": LSTMModel(n_features=6, hidden_size=50),
        "GRU Multivariado": GRUModel(n_features=6, hidden_size=50),
    }

    for nome, modelo in modelos_multi.items():
        res = treinar_e_avaliar_modelo(
            nome,
            modelo,
            X_train_multi,
            y_train_multi,
            X_test_multi,
            y_test_multi,
            scaler_y,
            epochs=epochs,
            batch_size=64,
            lr=0.001,
        )
        resultados[nome] = res["metricas"]
        hist_loss[nome] = {"train": res["train_losses"], "val": res["val_losses"]}
        previsoes_teste[nome] = {"pred": res["previsoes"], "real": res["y_real"]}
        historicos.append(res["historico"])

    registos = []
    for nome in modelos_uni:
        m = resultados[nome]
        registos.append({
            "Modelo": nome,
            "Regime": "Univariado (F=1)",
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })
    for nome in modelos_multi:
        m = resultados[nome]
        registos.append({
            "Modelo": nome,
            "Regime": "Multivariado (F=6)",
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })

    df_dl = pd.DataFrame(registos)
    guardar_tabela(df_dl, "deep_learning", "Tabela de Deep Learning")

    df_perdas = pd.concat(historicos, ignore_index=True)
    guardar_tabela(df_perdas, "curvas_aprendizagem", "Tabela de perdas por época")

    return resultados, hist_loss, previsoes_teste


# Geração de Tabelas (tab_06 e tab_07)
def gerar_tabelas_dl(resultados: dict, window_size: int = 28, epochs: int = 15) -> None:
    """Gera e grava a tabela de hiperparâmetros (tab_06) e a tabela de métricas (tab_07)."""
    print("\n[*] A gravar tabela de hiperparâmetros (tab_06_arquiteturas_dl.csv)...")
    tabela_hp = pd.DataFrame([
        {
            "Modelo": "1D-CNN",
            "Regime": "Univariado (F=1)",
            "Camadas": "Conv1D(32, k=3) + MaxPool + Conv1D(64, k=3) + MaxPool + Dense(64) + Dropout(0.2) + Dense(1)",
            "Neurónios": "32 + 64 + 64",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "1D-CNN",
            "Regime": "Multivariado (F=6)",
            "Camadas": "Conv1D(32, k=3) + MaxPool + Conv1D(64, k=3) + MaxPool + Dense(64) + Dropout(0.2) + Dense(1)",
            "Neurónios": "32 + 64 + 64",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "SimpleRNN",
            "Regime": "Univariado (F=1)",
            "Camadas": "RNN(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "SimpleRNN",
            "Regime": "Multivariado (F=6)",
            "Camadas": "RNN(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "LSTM",
            "Regime": "Univariado (F=1)",
            "Camadas": "LSTM(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "LSTM",
            "Regime": "Multivariado (F=6)",
            "Camadas": "LSTM(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "GRU",
            "Regime": "Univariado (F=1)",
            "Camadas": "GRU(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
        {
            "Modelo": "GRU",
            "Regime": "Multivariado (F=6)",
            "Camadas": "GRU(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": window_size,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": epochs,
        },
    ])
    guardar_tabela(tabela_hp, "tab_06_arquiteturas_dl")

    print("[*] A gravar tabela de benchmark (tab_07_dl_benchmark_metricas.csv)...")
    tabela_benchmark = pd.DataFrame(
        [
            {"Modelo": nome, "Regime": "Univariado (F=1)", **resultados[nome]}
            for nome in ["1D-CNN Univariado", "SimpleRNN Univariado", "LSTM Univariado", "GRU Univariado"]
        ]
        + [
            {"Modelo": nome, "Regime": "Multivariado (F=6)", **resultados[nome]}
            for nome in ["1D-CNN Multivariado", "SimpleRNN Multivariado", "LSTM Multivariado", "GRU Multivariado"]
        ]
    )
    guardar_tabela(tabela_benchmark, "tab_07_dl_benchmark_metricas")


# Geração de Figura (fig_06)
def criar_figura_curvas_aprendizagem(hist_loss: dict) -> None:
    """Gera o painel de curvas de aprendizagem (Loss Treino vs. Validação)."""
    print("[*] A gerar curvas de aprendizagem (fig_06_curvas_aprendizagem)...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Curvas de Aprendizagem (Loss de Treino vs. Validação) — Aula #5 (Slide 80 e 94)\n"
        "ISEG — Time Series Forecasting — 1D-CNN, SimpleRNN, LSTM e GRU",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for idx, nome in enumerate(["1D-CNN Univariado", "SimpleRNN Univariado", "LSTM Univariado", "GRU Univariado"]):
        ax = axes[0, idx]
        ax.plot(hist_loss[nome]["train"], color="#1f77b4", lw=1.2, label="Treino (MSE)")
        ax.plot(hist_loss[nome]["val"], color="#d62728", lw=1.2, linestyle="--", label="Validação (MSE)")
        ax.set_title(nome.replace(" Univariado", ""), loc="left", fontweight="bold", fontsize=10)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Época")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.5)

    for idx, nome in enumerate([
        "1D-CNN Multivariado",
        "SimpleRNN Multivariado",
        "LSTM Multivariado",
        "GRU Multivariado",
    ]):
        ax = axes[1, idx]
        ax.plot(hist_loss[nome]["train"], color="#1f77b4", lw=1.2, label="Treino (MSE)")
        ax.plot(hist_loss[nome]["val"], color="#d62728", lw=1.2, linestyle="--", label="Validação (MSE)")
        ax.set_title(nome.replace(" Multivariado", ""), loc="left", fontweight="bold", fontsize=10)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Época")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.5)

    plt.tight_layout()
    guardar_figura(fig, "fig_06_curvas_aprendizagem")


# Geração de previsões Univariadas vs Multivariadas (fig_07)
def criar_figura_previsoes_teste(
    test: pd.DataFrame,
    previsoes_teste: dict,
    resultados: dict,
    window_size: int = 28,
) -> None:
    """Gera o painel duplo de previsões out-of-sample: Univariado vs. Multivariado."""
    print("[*] A gerar previsões no período de teste (fig_07_previsoes_dl_teste)...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "Previsões Out-of-Sample (Daily) — Deep Learning: Univariado vs. Multivariado\n"
        "ISEG — Time Series Forecasting — Aula #5 (Slides 106-108)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # Painel 1: Univariado
    ax1 = axes[0]
    y_real = previsoes_teste["GRU Univariado"]["real"]
    ax1.plot(
        test.index[window_size:],
        y_real,
        color="black",
        lw=1.8,
        label=r"Observado Real ($Y_{d,t}$)",
        zorder=5,
    )
    for nome, cor, estilo in [
        ("1D-CNN Univariado", "#ff7f0e", ":"),
        ("SimpleRNN Univariado", "#9467bd", "-."),
        ("LSTM Univariado", "#1f77b4", "--"),
        ("GRU Univariado", "#2ca02c", "-"),
    ]:
        p = previsoes_teste[nome]["pred"]
        epam = resultados[nome]["EPAM"]
        ax1.plot(
            test.index[window_size:],
            p,
            color=cor,
            linestyle=estilo,
            lw=1.3,
            label=f"{nome.replace(' Univariado', '')} (EPAM: {epam:.1f}%)",
        )
    ax1.set_title(
        "Regime Univariado (F=1): Apenas Histórico de Commits Diários",
        loc="left",
        fontweight="bold",
    )
    ax1.set_ylabel("Commits diários")
    ax1.set_xlabel("Data")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Painel 2: Multivariado
    ax2 = axes[1]
    ax2.plot(
        test.index[window_size:],
        y_real,
        color="black",
        lw=1.8,
        label=r"Observado Real ($Y_{d,t}$)",
        zorder=5,
    )
    for nome, cor, estilo in [
        ("1D-CNN Multivariado", "#ff7f0e", ":"),
        ("SimpleRNN Multivariado", "#9467bd", "-."),
        ("LSTM Multivariado", "#1f77b4", "--"),
        ("GRU Multivariado", "#2ca02c", "-"),
    ]:
        p = previsoes_teste[nome]["pred"]
        epam = resultados[nome]["EPAM"]
        ax2.plot(
            test.index[window_size:],
            p,
            color=cor,
            linestyle=estilo,
            lw=1.3,
            label=f"{nome.replace(' Multivariado', '')} (EPAM: {epam:.1f}%)",
        )
    ax2.set_title(
        "Regime Multivariado (F=6): Commits + Contribuidores + Code Churn + Files + Harmónicas ($s=7$)",
        loc="left",
        fontweight="bold",
    )
    ax2.set_ylabel("Commits diários")
    ax2.set_xlabel("Data")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    guardar_figura(fig, "fig_07_previsoes_dl_teste")


# Síntese Interpretativa em Markdown (DEEP_LEARNING_SUMMARY.md)
def exportar_resumo_dl(resultados: dict) -> None:
    """Exporta o documento Markdown de síntese de resultados para a equipa."""
    print("[*] A exportar síntese de Deep Learning em Markdown (output/DEEP_LEARNING_SUMMARY.md)...")

    # Identificar o melhor modelo por REQM
    melhor_modelo = min(resultados, key=lambda k: resultados[k]["REQM"])
    melhor_metricas = resultados[melhor_modelo]

    # Comparação Univariado vs Multivariado para a mesma arquitetura
    comparacao_uni_multi = ""
    for arch in ["1D-CNN", "SimpleRNN", "LSTM", "GRU"]:
        uni_reqm = resultados[f"{arch} Univariado"]["REQM"]
        multi_reqm = resultados[f"{arch} Multivariado"]["REQM"]
        uni_epam = resultados[f"{arch} Univariado"]["EPAM"]
        multi_epam = resultados[f"{arch} Multivariado"]["EPAM"]
        ganho = ((uni_reqm - multi_reqm) / uni_reqm) * 100
        comparacao_uni_multi += (
            f"  * **{arch}:** Univariado REQM={uni_reqm:.2f} (EPAM={uni_epam:.2f}%) "
            f"vs. Multivariado REQM={multi_reqm:.2f} (EPAM={multi_epam:.2f}%) "
            f"-> Ganho de {ganho:+.1f}%\n"
        )

    contexto: dict[str, object] = {
        "comparacao_uni_multi": comparacao_uni_multi.rstrip("\n"),
        "melhor_modelo": melhor_modelo,
        "melhor_reqm": f"{melhor_metricas['REQM']:.2f}",
        "melhor_epam": f"{melhor_metricas['EPAM']:.2f}",
    }

    guardar_markdown_template("deep_learning_summary.md", "DEEP_LEARNING_SUMMARY.md", contexto)


# 8. Execução Principal
def main():

    WINDOW_SIZE = 28
    EPOCHS = 15

    iniciar_script("03", "APRENDIZAGEM PROFUNDA (DEEP LEARNING)")
    train, test, features_mult = carregar_dados_e_preparar()
    dados_uni, dados_multi, scaler_y = preparar_janelas_deslizantes(train, test, features_mult, window_size=WINDOW_SIZE)
    resultados, hist_loss, previsoes_teste = treinar_modelos(
        dados_uni, dados_multi, scaler_y, window_size=WINDOW_SIZE, epochs=EPOCHS
    )
    gerar_tabelas_dl(resultados, window_size=WINDOW_SIZE, epochs=EPOCHS)
    criar_figura_curvas_aprendizagem(hist_loss)
    criar_figura_previsoes_teste(test, previsoes_teste, resultados, window_size=WINDOW_SIZE)
    exportar_resumo_dl(resultados)
    finalizar_script("03")


if __name__ == "__main__":
    main()
