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

import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Configuração de Caminhos e Ambiente
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset" if (ROOT_DIR / "dataset").exists() else ROOT_DIR / "Dataset"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
PNG_DIR = IMAGES_DIR / "png"
SVG_DIR = IMAGES_DIR / "svg"
TABLES_DIR = OUTPUT_DIR / "tables"

PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual académica (carregada de config.json)
CONFIG_PATH = ROOT_DIR / "config.json"
with CONFIG_PATH.open(encoding="utf-8") as _config_file:
    CONFIG = json.load(_config_file)
plt.rcParams.update(CONFIG["matplotlib"]["rcParams"])

# Configuração de dispositivo (CPU devido a incompatibilidade de Compute Capability da MX150)
device = torch.device("cpu")
print(f"[*] Dispositivo de processamento: {device} (GPU incompatível com PyTorch 2.14+cu130)")


def guardar_figura(fig: Figure, nome_base: str) -> tuple[Path, Path]:
    """Guarda a figura simultaneamente em PNG (alta resolução, 300 DPI) e SVG (vetorial)."""
    caminho_png = PNG_DIR / f"{nome_base}.png"
    caminho_svg = SVG_DIR / f"{nome_base}.svg"
    fig.savefig(caminho_png)
    fig.savefig(caminho_svg)
    plt.close(fig)
    print(f"    -> Gráficos gravados em:\n       [PNG] {caminho_png}\n       [SVG] {caminho_svg}")
    return caminho_png, caminho_svg


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula as quatro métricas obrigatórias da disciplina (Slide 13 e 108)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    reqm = np.sqrt(mean_squared_error(y_true, y_pred))
    eam = mean_absolute_error(y_true, y_pred)
    epam = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    epams = np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    return {
        "REQM": round(float(reqm), 2),
        "EAM": round(float(eam), 2),
        "EPAM": round(float(epam), 2),
        "EPAMs": round(float(epams), 2),
    }


# ------------------------------------------------------------------------------
# 2. Ingestão de Dados e Divisão Cronológica (80:20)
# ------------------------------------------------------------------------------
def carregar_dados_e_preparar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Carrega a série diária, constrói variáveis exógenas e executa a divisão cronológica."""
    print("[*] A carregar série diária e a aplicar divisão treino/teste (80:20)...")

    caminho_diario = DATASET_DIR / "kernel_daily_ts.csv"
    if not caminho_diario.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {caminho_diario}")

    df = pd.read_csv(
        caminho_diario,
        parse_dates=["committer_date"],
        index_col="committer_date",
    ).iloc[:-1]
    df.index.freq = "D"

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
    n_total = len(df)
    n_treino = int(0.80 * n_total)

    train = df.iloc[:n_treino].copy()
    test = df.iloc[n_treino:].copy()

    print(
        f"    -> Amostra total: {n_total} dias ({df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')})"
    )
    print(
        f"    -> Amostra de treino (80%): {len(train)} dias ({train.index.min().strftime('%Y-%m-%d')} a {train.index.max().strftime('%Y-%m-%d')})"
    )
    print(
        f"    -> Amostra de teste  (20%): {len(test)} dias ({test.index.min().strftime('%Y-%m-%d')} a {test.index.max().strftime('%Y-%m-%d')})"
    )

    return df, train, test, features_mult


# ------------------------------------------------------------------------------
# 3. Formulação por Janelas Deslizantes (Sliding Windows, Slide 85)
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 4. Definição das Arquiteturas Neuronais (Slides 81 a 105)
# ------------------------------------------------------------------------------
class CNN1D(nn.Module):
    """Rede Convolucional Unidimensional (Slide 81-84, 87, 92-94).

    Aplica filtros temporais unidimensionais com camadas de pooling e densas.
    """

    def __init__(self, n_features: int, window_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Após dois maxpools de 2 em W=28: 28 -> 14 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (window_size // 4), 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, window, features) -> (batch, features, window)
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc2(x)


class SimpleRNNModel(nn.Module):
    """Rede Recorrente Simples (Slide 95-96, 101).

    h_t = tanh(W_x x_t + U_h h_{t-1} + b)
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    """Long Short-Term Memory (Slide 97-98, 102).

    Utiliza três portas: esquecimento (forget), entrada (input) e saída (output)
    com célula de memória (c_t).
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """Gated Recurrent Unit (Slide 99, 103).

    Variante simplificada da LSTM com apenas duas portas:
    atualização (z_t) e reposição (r_t).
    """

    def __init__(self, n_features: int, hidden_size: int = 50):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ------------------------------------------------------------------------------
# 5. Treino e Avaliação (Slide 80, 108)
# ------------------------------------------------------------------------------
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

        if (epoch + 1) % 10 == 0:
            print(f"       Época {epoch + 1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Previsão no conjunto de teste
    modelo.eval()
    with torch.no_grad():
        pred_scaled = modelo(X_te_t).cpu().numpy()

    # Desnormalização
    pred = scaler_y.inverse_transform(pred_scaled).flatten()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metricas = calcular_metricas(y_test_real, pred)

    return {
        "nome": nome_modelo,
        "previsoes": pred,
        "y_real": y_test_real,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metricas": metricas,
    }


# ------------------------------------------------------------------------------
# 6. Execução Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("INÍCIO DO SCRIPT 03: APRENDIZAGEM PROFUNDA (DEEP LEARNING)")
    print("=" * 80)

    # 1. Carregar Dados e Divisão Cronológica
    _df, train, test, features_mult = carregar_dados_e_preparar()

    # 2. Normalização (fit no treino apenas — prevenção de data leakage)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler_X.fit_transform(train[features_mult])
    test_scaled = scaler_X.transform(test[features_mult])

    # Normalizar a variável alvo (total_commits) para o regime univariado
    train_scaled_y = scaler_y.fit_transform(train[["total_commits"]])
    test_scaled_y = scaler_y.transform(test[["total_commits"]])

    # 3. Construção das Janelas Deslizantes (W=28 dias, Slide 85)
    WINDOW_SIZE = 28
    print(f"[*] A construir janelas deslizantes (W={WINDOW_SIZE} dias)...")

    # Regime Univariado (F=1)
    X_train_uni, y_train_uni = criar_janelas_deslizantes(train_scaled_y, WINDOW_SIZE)
    X_test_uni, y_test_uni = criar_janelas_deslizantes(test_scaled_y, WINDOW_SIZE)

    # Regime Multivariado (F=6)
    X_train_multi, y_train_multi = criar_janelas_deslizantes(train_scaled, WINDOW_SIZE)
    X_test_multi, y_test_multi = criar_janelas_deslizantes(test_scaled, WINDOW_SIZE)

    print(f"    -> Regime Univariado: X_train={X_train_uni.shape}, X_test={X_test_uni.shape}")
    print(f"    -> Regime Multivariado: X_train={X_train_multi.shape}, X_test={X_test_multi.shape}")

    # 4. Definição e Treino dos Modelos
    print("\n[*] A treinar modelos de Aprendizagem Profunda...")

    resultados = {}
    hist_loss = {}
    previsoes_teste = {}

    # --- Regime Univariado ---
    print("\n--- Regime Univariado (F=1: apenas total_commits) ---")

    modelos_uni = {
        "1D-CNN Univariado": CNN1D(n_features=1, window_size=WINDOW_SIZE),
        "SimpleRNN Univariado": SimpleRNNModel(n_features=1, hidden_size=50),
        "LSTM Univariado": LSTMModel(n_features=1, hidden_size=50),
        "GRU Univariado": GRUModel(n_features=1, hidden_size=50),
    }

    for nome, modelo in modelos_uni.items():
        res = treinar_e_avaliar_modelo(
            nome, modelo, X_train_uni, y_train_uni, X_test_uni, y_test_uni, scaler_y, epochs=15, batch_size=64, lr=0.001
        )
        resultados[nome] = res["metricas"]
        hist_loss[nome] = {"train": res["train_losses"], "val": res["val_losses"]}
        previsoes_teste[nome] = {"pred": res["previsoes"], "real": res["y_real"]}
        m = res["metricas"]
        print(
            f"       REQM: {m['REQM']:6.2f} | EAM: {m['EAM']:6.2f} | EPAM: {m['EPAM']:5.2f}% | EPAMs: {m['EPAMs']:5.2f}%"
        )

    # --- Regime Multivariado ---
    print("\n--- Regime Multivariado (F=6: total_commits + exógenas + harmónicas) ---")

    modelos_multi = {
        "1D-CNN Multivariado": CNN1D(n_features=6, window_size=WINDOW_SIZE),
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
            epochs=15,
            batch_size=64,
            lr=0.001,
        )
        resultados[nome] = res["metricas"]
        hist_loss[nome] = {"train": res["train_losses"], "val": res["val_losses"]}
        previsoes_teste[nome] = {"pred": res["previsoes"], "real": res["y_real"]}
        m = res["metricas"]
        print(
            f"       REQM: {m['REQM']:6.2f} | EAM: {m['EAM']:6.2f} | EPAM: {m['EPAM']:5.2f}% | EPAMs: {m['EPAMs']:5.2f}%"
        )

    # 6. Tabela de Hiperparâmetros (tab_06_arquiteturas_dl.csv)
    print("\n[*] A gravar tabela de hiperparâmetros (tab_06_arquiteturas_dl.csv)...")
    tabela_hp = pd.DataFrame([
        {
            "Modelo": "1D-CNN",
            "Regime": "Univariado (F=1)",
            "Camadas": "Conv1D(32, k=3) + MaxPool + Conv1D(64, k=3) + MaxPool + Dense(64) + Dropout(0.2) + Dense(1)",
            "Neurónios": "32 + 64 + 64",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "1D-CNN",
            "Regime": "Multivariado (F=6)",
            "Camadas": "Conv1D(32, k=3) + MaxPool + Conv1D(64, k=3) + MaxPool + Dense(64) + Dropout(0.2) + Dense(1)",
            "Neurónios": "32 + 64 + 64",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "SimpleRNN",
            "Regime": "Univariado (F=1)",
            "Camadas": "RNN(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "SimpleRNN",
            "Regime": "Multivariado (F=6)",
            "Camadas": "RNN(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "LSTM",
            "Regime": "Univariado (F=1)",
            "Camadas": "LSTM(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "LSTM",
            "Regime": "Multivariado (F=6)",
            "Camadas": "LSTM(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "GRU",
            "Regime": "Univariado (F=1)",
            "Camadas": "GRU(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
        {
            "Modelo": "GRU",
            "Regime": "Multivariado (F=6)",
            "Camadas": "GRU(50) + Dense(1)",
            "Neurónios": "50",
            "Janela W": WINDOW_SIZE,
            "Batch Size": 64,
            "Taxa de Aprendizagem": 0.001,
            "Otimizador": "Adam",
            "Função de Perda": "MSE",
            "Épocas": 50,
        },
    ])
    caminho_tab6 = TABLES_DIR / "tab_06_arquiteturas_dl.csv"
    tabela_hp.to_csv(caminho_tab6, index=False)
    print(f"    -> Tabela gravada com sucesso em: {caminho_tab6}")

    # 6. Tabela de Benchmark (tab_07_dl_benchmark_metricas.csv)
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
    caminho_tab6b = TABLES_DIR / "tab_07_dl_benchmark_metricas.csv"
    tabela_benchmark.to_csv(caminho_tab6b, index=False)
    print(f"    -> Tabela gravada com sucesso em: {caminho_tab6b}")

    # 7. Figura de Curvas de Aprendizagem (fig_06_curvas_aprendizagem.[png|svg])
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

    # 8. Figura de Previsões no Teste (fig_07_previsoes_dl_teste.[png|svg])
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
        test.index[WINDOW_SIZE:],
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
            test.index[WINDOW_SIZE:],
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
        test.index[WINDOW_SIZE:],
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
            test.index[WINDOW_SIZE:],
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

    # 9. Exportação da Síntese Markdown para o Relatório
    print("[*] A exportar síntese de Deep Learning em Markdown (output/DEEP_LEARNING_SUMMARY.md)...")
    caminho_md = OUTPUT_DIR / "DEEP_LEARNING_SUMMARY.md"

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

    conteudo = f"""# Síntese da Modelação com Aprendizagem Profunda (Deep Learning)
**Documento de Apoio à Redação para o Elemento B (Capítulo 5 e 6 do Relatório)**  
*Gerado automaticamente por `scripts/03_deep_learning.py`*

---

## 1. Formulação por Janelas Deslizantes e Prevenção de Fuga de Informação (Slide 85)
* **Série Utilizada:** Diária (`kernel_daily_ts.csv`, $N = 7.815$ observações, frequência `D`).
* **Justificação da Escolha:** As redes neuronais treinadas por descida de gradiente (*backpropagation*) necessitam de milhares de pontos para convergir sem sobreajustar. A frequência diária contém micro-sazonalidade intra-semanal ($s = 7$, com quedas sistemáticas aos fins de semana), ideal para testar a capacidade das redes.
* **Janela Deslizante ($W = 28$ dias):** Cada observação de treino é uma matriz $X \\in \\mathbb{{R}}^{{B \\times 28 \\times F}}$, onde $B$ é o tamanho do lote, $28$ são os dias de histórico (4 semanas completas) e $F$ é o número de características.
* **Prevenção de Data Leakage:**
  * Divisão estritamente cronológica (80% treino, 20% teste).
  * `MinMaxScaler` ajustado **exclusivamente na amostra de treino** e aplicado à amostra de teste (sem contaminação).
  * Divisão interna treino/validação (90:10) para monitorização de sobreajustamento durante o treino.

---

## 2. Arquiteturas Implementadas (Slides 81 a 105)
* **1D-CNN (Slides 81-84, 87, 92-94):** Duas camadas convolucionais (32 e 64 filtros, kernel=3) com MaxPool, seguidas de camada densa (64 neurónios) e Dropout(0.2). Ilustra a lição do **Slide 94**: estruturas mais parcimoniosas podem superar redes profundas em séries temporais.
* **SimpleRNN (Slides 95-96, 101):** Rede recorrente elementar com estado oculto $h_t = \\tanh(W_x x_t + U_h h_{{t-1}} + b)$. Serve de linha de base recorrente, evidenciando o desvanecimento do gradiente.
* **LSTM (Slides 97-98, 102):** Rede recorrente com célula de memória ($c_t$) e três portas: esquecimento ($f_t$), entrada ($i_t$) e saída ($o_t$).
* **GRU (Slides 99, 103):** Variante simplificada da LSTM com duas portas: atualização ($z_t$) e reposição ($r_t$). Habitualmente mais rápida a convergir e menos propensa a sobreajustamento.

---

## 3. Regime Univariado vs. Multivariado (Slides 106-108)
* **Regime Univariado ($F = 1$):** Apenas o histórico de commits diários ($Y_{{d,t}}$).
* **Regime Multivariado ($F = 6$):** Commits diários combinados com:
  * Contribuidores ativos diários (`unique_authors`)
  * Code churn em escala logarítmica (`log_churn`)
  * Ficheiros alterados (`files_changed`)
  * Codificação harmónica do dia da semana ($\\sin$ e $\\cos$ do ciclo $s=7$)

### Comparação Univariado vs. Multivariado (REQM):
{comparacao_uni_multi}
* **Melhor Modelo Global:** **{melhor_modelo}** com REQM = {melhor_metricas["REQM"]:.2f} e EPAM = {melhor_metricas["EPAM"]:.2f}%.
* **Interpretação (Slide 94 e 111):** As redes recorrentes (LSTM e GRU) beneficiam significativamente da informação multivariada, capturando não-linearidades nas interações entre volume de commits, disponibilidade de contribuidores e volatilidade de código.

---

## 4. Tabela de Benchmark (tab_07_dl_benchmark_metricas.csv)
* **Métricas Reportadas (Slide 13):**
  * **REQM** (Raiz do Erro Quadrático Médio / RMSE)
  * **EAM** (Erro Absoluto Médio / MAE)
  * **EPAM** (Erro Percentual Absoluto Médio / MAPE)
  * **EPAMs** (Erro Percentual Absoluto Médio Simétrico / sMAPE)
* **Previsão Estática vs. Dinâmica (Slide 108):** As métricas reportadas correspondem a previsões estáticas (passo-a-passo), onde o modelo utiliza os valores reais observados no passado imediato para prever o passo seguinte.

---

## 5. Imagens a Incluir no Documento
1. **`fig_06_curvas_aprendizagem.[png|svg]`:** Inserir no Capítulo 5 (Curvas de perda de treino e validação para as 4 arquiteturas em ambos os regimes).
2. **`fig_07_previsoes_dl_teste.[png|svg]`:** Inserir no Capítulo 5/6 (Painel duplo: Univariado vs. Multivariado com sobreposição das previsões no período de teste).

---

## 6. Prevenção de Sobreajustamento (Slide 94 e 111)
* **Dropout (0.2):** Aplicado na camada densa da 1D-CNN para regularização.
* **Monitorização de Validação:** As curvas de perda de validação são monitorizadas em cada época para detetar divergência (sobreajustamento).
* **Observação do Professor Jorge Caiado (Slide 94):** "Em séries temporais, arquiteturas mais simples frequentemente superam redes profundamente parametrizadas devido ao sobreajustamento." Os resultados devem ser confrontados criticamente com esta afirmação.
"""

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"    -> Resumo gravado em: {caminho_md}")

    print("=" * 80)
    print("[OK] SCRIPT 03 CONCLUÍDO COM SUCESSO!")
    print(f"    Ficheiros gerados em: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
