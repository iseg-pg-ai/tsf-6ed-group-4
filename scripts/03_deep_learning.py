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
- Terceiro regime: multivariado + fase do ciclo de release (justificação em
  scripts/analise_covariaveis_release/README.md)
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

# Seed fixa: sem ela, duas execuções do mesmo script davam métricas diferentes (a
# inicialização dos pesos e a ordem dos lotes são aleatórias). A seed é reposta no
# início de cada regime, para que a mesma arquitetura arranque dos mesmos pesos nos
# três regimes e a diferença de métricas meça o efeito das features, não o sorteio.
SEED = 42

# Rótulos dos regimes neuronais. O número de features não é escrito à mão: é lido da
# forma dos tensores, para nunca dessincronizar da lista de colunas em carregar_dados_e_preparar.
REGIMES = ("Univariado", "Multivariado", "Multivariado + Release")

# (X, y) de treino e de teste de um regime
DadosJanela = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# Ingestão de Dados e Divisão Cronológica (80:20)
def carregar_dados_e_preparar() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Carrega a série diária, constrói variáveis exógenas e executa a divisão cronológica."""
    print("[*] A carregar série diária e a aplicar divisão treino/teste (80:20)...")

    df = carregar_serie_temporal("kernel_daily_ts.csv", "D")

    # Transformação logarítmica do code churn para estabilização de variância
    df["log_churn"] = np.log1p(df["total_churn"])

    # Codificação harmónica do dia da semana (s=7)
    t = np.arange(len(df))
    df["sin_7_1"] = np.sin(2 * np.pi * 1 * t / 7.0)
    df["cos_7_1"] = np.cos(2 * np.pi * 1 * t / 7.0)

    # Fase do ciclo de release como duas dummies, com a merge window como referência:
    # exatamente a codificação da especificação ARIMAX vencedora do script 02, para que
    # as duas famílias de modelos leiam a mesma variável da mesma maneira.
    # `where(conhecido)` preserva o NaN dos 62 dias anteriores ao primeiro tag: um valor
    # desconhecido nunca pode ser codificado como merge window.
    semanas_release = df["weeks_since_release"].astype("float64")
    release_conhecido = semanas_release.notna()
    df["fase_rc_inicial"] = ((semanas_release >= 2) & (semanas_release <= 5)).where(release_conhecido).astype("float64")
    df["fase_rc_final"] = (semanas_release >= 6).where(release_conhecido).astype("float64")

    # Colunas de características para o regime multivariado
    features_mult = [
        "total_commits",
        "unique_authors",
        "log_churn",
        "files_changed",
        "sin_7_1",
        "cos_7_1",
    ]

    # Terceiro regime: as mesmas seis colunas mais a fase do ciclo de release (F=8)
    features_release = features_mult + ["fase_rc_inicial", "fase_rc_final"]

    # Divisão estritamente cronológica 80:20 (sem baralhar)
    train, test = dividir_treino_teste(df, "dias")

    # Os 62 dias anteriores ao primeiro tag não têm fase conhecida. São retirados da
    # amostra de treino para todos os regimes: se só o regime F=8 os perdesse, os outros
    # treinariam com mais 62 janelas e a comparação entre regimes deixaria de ser justa.
    # O conjunto de teste (2022-2026) não tem valores em falta, pelo que não é afetado.
    train = train.dropna(subset=["fase_rc_inicial"])
    print(f"    -> Treino após remover os dias sem fase de release conhecida: {len(train)} dias")

    return train, test, features_mult, features_release


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
    features_release: list[str],
    window_size: int = 28,
) -> tuple[dict[str, DadosJanela], MinMaxScaler]:
    """Aplica normalização e constrói as janelas deslizantes dos três regimes.

    O ajuste de cada MinMaxScaler é feito exclusivamente no treino para prevenir data
    leakage. Há um scaler por regime porque os regimes têm conjuntos de colunas diferentes.
    """
    print(f"[*] A construir janelas deslizantes (W={window_size} dias)...")

    # Alvo normalizado, partilhado pelos três regimes
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_scaled_y = scaler_y.fit_transform(train[["total_commits"]])
    test_scaled_y = scaler_y.transform(test[["total_commits"]])

    # Regime Univariado (F=1): apenas o histórico de commits
    X_train_uni, y_train_uni = criar_janelas_deslizantes(train_scaled_y, window_size)
    X_test_uni, y_test_uni = criar_janelas_deslizantes(test_scaled_y, window_size)
    dados: dict[str, DadosJanela] = {
        "Univariado": (X_train_uni, y_train_uni, X_test_uni, y_test_uni),
    }
    print(f"    -> Regime Univariado: X_train={X_train_uni.shape}, X_test={X_test_uni.shape}")

    # Regimes multivariados: mudam apenas as colunas de entrada
    colunas_por_regime = {
        "Multivariado": features_mult,
        "Multivariado + Release": features_release,
    }

    for rotulo, colunas in colunas_por_regime.items():
        scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(train[colunas])
        X_train_multi, y_train_multi = criar_janelas_deslizantes(scaler_X.transform(train[colunas]), window_size)
        X_test_multi, y_test_multi = criar_janelas_deslizantes(scaler_X.transform(test[colunas]), window_size)
        dados[rotulo] = (X_train_multi, y_train_multi, X_test_multi, y_test_multi)
        print(
            f"    -> Regime {rotulo} (F={X_train_multi.shape[2]}): "
            f"X_train={X_train_multi.shape}, X_test={X_test_multi.shape}"
        )

    return dados, scaler_y


# Treino e Avaliação de Modelos Neuronais (Slide 80, 108)
def treinar_e_avaliar_modelo(
    nome_modelo: str,
    regime: str,
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

    # Histórico de perda
    train_losses = []
    val_losses = []

    # Divisão treino/validação (90:10 do treino para monitorização de sobreajustamento).
    # A parte final do treino é reservada para validação e não entra nos lotes.
    n_val = int(len(X_train) * 0.1)
    X_val_t = X_tr_t[-n_val:]
    y_val_t = y_tr_t[-n_val:]
    X_tr_t = X_tr_t[:-n_val]
    y_tr_t = y_tr_t[:-n_val]

    # O DataLoader é construído DEPOIS do corte. Construído antes (como estava), os
    # últimos 10% do treino entravam nos lotes de treino e a curva de validação media
    # dados que o modelo já tinha visto: a validação deixava de detetar sobreajustamento,
    # que é precisamente o que os Slides 94 e 111 pedem para vigiar.
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Parâmetros treináveis: vai para a tabela de hiperparâmetros (tab_06), para que a
    # capacidade de cada arquitetura fique documentada e não estimada a partir do nome.
    n_parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

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
        "Regime": regime,
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
        "parametros": n_parametros,
    }


# Treino de modelos
def treinar_modelos(
    dados_por_regime: dict[str, DadosJanela],
    scaler_y: MinMaxScaler,
    window_size: int = 28,
    epochs: int = 15,
) -> tuple[dict, dict, dict, dict[str, int]]:
    """Instancia, treina e avalia as 4 arquiteturas nos três regimes de features."""
    print("\n[*] A treinar modelos de Aprendizagem Profunda...")

    construtores = {
        "1D-CNN": lambda n: CNN1D(n_features=n, window_size=window_size),
        "SimpleRNN": lambda n: SimpleRNNModel(n_features=n, hidden_size=50),
        "LSTM": lambda n: LSTMModel(n_features=n, hidden_size=50),
        "GRU": lambda n: GRUModel(n_features=n, hidden_size=50),
    }

    resultados = {}
    hist_loss = {}
    previsoes_teste = {}
    parametros: dict[str, int] = {}
    historicos = []

    for rotulo in REGIMES:
        X_train, y_train, X_test, y_test = dados_por_regime[rotulo]
        n_features = X_train.shape[2]
        rotulo_regime = f"{rotulo} (F={n_features})"

        print(f"\n--- Regime {rotulo_regime} ---")

        # Repor a seed no início de cada regime: como as arquiteturas são construídas na
        # mesma ordem, cada uma arranca dos mesmos pesos nos três regimes e a diferença
        # de métricas mede o efeito das features, não o sorteio da inicialização.
        torch.manual_seed(SEED)

        for nome_arquitetura, construtor in construtores.items():
            nome = f"{nome_arquitetura} {rotulo}"
            res = treinar_e_avaliar_modelo(
                nome,
                rotulo_regime,
                construtor(n_features),
                X_train,
                y_train,
                X_test,
                y_test,
                scaler_y,
                epochs=epochs,
                batch_size=64,
                lr=0.001,
            )
            resultados[nome] = res["metricas"]
            parametros[nome] = res["parametros"]
            hist_loss[nome] = {"train": res["train_losses"], "val": res["val_losses"]}
            previsoes_teste[nome] = {"pred": res["previsoes"], "real": res["y_real"]}
            historicos.append(res["historico"])

    registos = []
    for rotulo in REGIMES:
        rotulo_regime = f"{rotulo} (F={dados_por_regime[rotulo][0].shape[2]})"
        for nome_arquitetura in construtores:
            m = resultados[f"{nome_arquitetura} {rotulo}"]
            registos.append({
                "Modelo": f"{nome_arquitetura} {rotulo}",
                "Regime": rotulo_regime,
                "REQM": m["REQM"],
                "EAM": m["EAM"],
                "EPAM (%)": m["EPAM"],
                "EPAMs (%)": m["EPAMs"],
            })

    df_dl = pd.DataFrame(registos)
    guardar_tabela(df_dl, "deep_learning", "Tabela de Deep Learning")

    df_perdas = pd.concat(historicos, ignore_index=True)
    guardar_tabela(df_perdas, "curvas_aprendizagem", "Tabela de perdas por época")

    return resultados, hist_loss, previsoes_teste, parametros


# Geração de Tabelas (tab_06 e tab_07)
def gerar_tabelas_dl(
    resultados: dict,
    parametros: dict[str, int],
    dados_por_regime: dict[str, DadosJanela],
    window_size: int = 28,
    epochs: int = 15,
) -> None:
    """Gera e grava a tabela de hiperparâmetros (tab_06) e a tabela de métricas (tab_07)."""
    print("\n[*] A gravar tabela de hiperparâmetros (tab_06_arquiteturas_dl.csv)...")

    # Especificação fixa de cada arquitetura. Cada linha é replicada pelos três regimes,
    # porque o que muda entre regimes é o número de features de entrada, não a rede.
    especificacoes = {
        "1D-CNN": (
            "Conv1D(32, k=3) + MaxPool + Conv1D(64, k=3) + MaxPool + Dense(64) + Dropout(0.2) + Dense(1)",
            "32 + 64 + 64",
        ),
        "SimpleRNN": ("RNN(50) + Dense(1)", "50"),
        "LSTM": ("LSTM(50) + Dense(1)", "50"),
        "GRU": ("GRU(50) + Dense(1)", "50"),
    }

    linhas_hp = []
    linhas_benchmark = []

    for rotulo in REGIMES:
        n_features = dados_por_regime[rotulo][0].shape[2]
        rotulo_regime = f"{rotulo} (F={n_features})"

        for nome_arquitetura, (camadas, neuronios) in especificacoes.items():
            nome = f"{nome_arquitetura} {rotulo}"
            linhas_hp.append({
                "Modelo": nome_arquitetura,
                "Regime": rotulo_regime,
                "Features de Entrada (F)": n_features,
                "Camadas": camadas,
                "Neurónios": neuronios,
                "Parametros_Treinaveis": parametros[nome],
                "Janela W": window_size,
                "Batch Size": 64,
                "Taxa de Aprendizagem": 0.001,
                "Otimizador": "Adam",
                "Função de Perda": "MSE",
                "Épocas": epochs,
                "Seed": SEED,
            })
            linhas_benchmark.append({"Modelo": nome, "Regime": rotulo_regime, **resultados[nome]})

    guardar_tabela(pd.DataFrame(linhas_hp), "tab_06_arquiteturas_dl")

    print("[*] A gravar tabela de benchmark (tab_07_dl_benchmark_metricas.csv)...")
    guardar_tabela(pd.DataFrame(linhas_benchmark), "tab_07_dl_benchmark_metricas")


# Geração de Figura (fig_06)
def criar_figura_curvas_aprendizagem(hist_loss: dict) -> None:
    """Gera o painel de curvas de aprendizagem (Loss Treino vs. Validação)."""
    print("[*] A gerar curvas de aprendizagem (fig_06_curvas_aprendizagem)...")
    arquiteturas = ["1D-CNN", "SimpleRNN", "LSTM", "GRU"]

    fig, axes = plt.subplots(len(REGIMES), len(arquiteturas), figsize=(18, 3.6 * len(REGIMES)))
    fig.suptitle(
        "Curvas de Aprendizagem (Loss de Treino vs. Validação) — Aula #5 (Slide 80 e 94)\n"
        "ISEG — Time Series Forecasting — 1D-CNN, SimpleRNN, LSTM e GRU nos três regimes de features",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for linha, rotulo in enumerate(REGIMES):
        for coluna, arquitetura in enumerate(arquiteturas):
            ax = axes[linha, coluna]
            nome = f"{arquitetura} {rotulo}"
            ax.plot(range(1, len(hist_loss[nome]["train"]) + 1), hist_loss[nome]["train"], color="#1f77b4", lw=1.2,
                    label="Treino (MSE)")
            ax.plot(range(1, len(hist_loss[nome]["train"]) + 1), hist_loss[nome]["val"], color="#d62728", lw=1.2,
                    linestyle="--", label="Validação (MSE)")
            ax.set_title(f"{arquitetura} — {rotulo}", loc="left", fontweight="bold", fontsize=10)
            ax.set_ylabel("MSE")
            ax.set_xlabel("Época")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.5)

    plt.tight_layout()
    guardar_figura(fig, "fig_06_curvas_aprendizagem")


# Geração de previsões por regime (fig_07)
def criar_figura_previsoes_teste(
    test: pd.DataFrame,
    previsoes_teste: dict,
    resultados: dict,
    dimensoes: dict[str, int],
    window_size: int = 28,
) -> None:
    """Gera um painel por regime de features com as previsões out-of-sample das 4 arquiteturas."""
    print("[*] A gerar previsões no período de teste (fig_07_previsoes_dl_teste)...")
    arquiteturas = [
        ("1D-CNN", "#ff7f0e", ":"),
        ("SimpleRNN", "#9467bd", "-."),
        ("LSTM", "#1f77b4", "--"),
        ("GRU", "#2ca02c", "-"),
    ]

    # O alvo é o mesmo nos três regimes, por isso a série observada é plotada uma vez por painel
    y_real = previsoes_teste[f"GRU {REGIMES[0]}"]["real"]

    descricao_regime = {
        "Univariado": "Apenas Histórico de Commits Diários",
        "Multivariado": "Commits + Contribuidores + Code Churn + Files + Harmónicas ($s=7$)",
        "Multivariado + Release": "O anterior + Fase do Ciclo de Release",
    }

    fig, axes = plt.subplots(len(REGIMES), 1, figsize=(14, 4.6 * len(REGIMES)))
    fig.suptitle(
        "Previsões Out-of-Sample (Daily) — Deep Learning por Regime de Features\n"
        "ISEG — Time Series Forecasting — Aula #5 (Slides 106-108)",
        fontsize=13,
        fontweight="bold",
        y=1.005,
    )

    for ax, rotulo in zip(axes, REGIMES):
        # A série observada fica em cinza translúcido no fundo: desenhada a preto e a
        # traço grosso por cima das previsões, tapava-as e tornava o painel ilegível.
        ax.plot(
            test.index[window_size:],
            y_real,
            color="#555555",
            lw=0.9,
            alpha=0.55,
            label=r"Observado Real ($Y_{d,t}$)",
            zorder=2,
        )
        for arquitetura, cor, estilo in arquiteturas:
            nome = f"{arquitetura} {rotulo}"
            p = previsoes_teste[nome]["pred"]
            epam = resultados[nome]["EPAM"]
            ax.plot(
                test.index[window_size:],
                p,
                color=cor,
                linestyle=estilo,
                lw=1.2,
                label=f"{arquitetura} (EPAM: {epam:.1f}%)",
                zorder=3,
            )
        ax.set_title(
            f"Regime {rotulo} (F={dimensoes[rotulo]}): {descricao_regime[rotulo]}",
            loc="left",
            fontweight="bold",
        )
        ax.set_ylabel("Commits diários")
        ax.set_xlabel("Data")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    guardar_figura(fig, "fig_07_previsoes_dl_teste")


# Síntese Interpretativa em Markdown (DEEP_LEARNING_SUMMARY.md)
def exportar_resumo_dl(resultados: dict) -> None:
    """Exporta o documento Markdown de síntese de resultados para a equipa."""
    print("[*] A exportar síntese de Deep Learning em Markdown (output/DEEP_LEARNING_SUMMARY.md)...")

    # Identificar o melhor modelo por REQM
    melhor_modelo = min(resultados, key=lambda k: resultados[k]["REQM"])
    melhor_metricas = resultados[melhor_modelo]

    arquiteturas = ["1D-CNN", "SimpleRNN", "LSTM", "GRU"]

    # Comparação por arquitetura: univariado -> multivariado -> multivariado + release
    comparacao_uni_multi = ""
    comparacao_release = ""
    ganhos_release = []

    for arch in arquiteturas:
        uni = resultados[f"{arch} Univariado"]
        multi = resultados[f"{arch} Multivariado"]
        release = resultados[f"{arch} Multivariado + Release"]

        ganho_multi = ((uni["REQM"] - multi["REQM"]) / uni["REQM"]) * 100
        ganho_release = ((multi["REQM"] - release["REQM"]) / multi["REQM"]) * 100
        ganhos_release.append(ganho_release)

        comparacao_uni_multi += (
            f"  * **{arch}:** Univariado REQM={uni['REQM']:.2f} (EPAM={uni['EPAM']:.2f}%) "
            f"vs. Multivariado REQM={multi['REQM']:.2f} (EPAM={multi['EPAM']:.2f}%) "
            f"-> Ganho de {ganho_multi:+.1f}%\n"
        )
        comparacao_release += (
            f"  * **{arch}:** Multivariado REQM={multi['REQM']:.2f} "
            f"vs. Multivariado + Release REQM={release['REQM']:.2f} "
            f"-> Ganho de {ganho_release:+.1f}% (EPAM {multi['EPAM']:.2f}% -> {release['EPAM']:.2f}%)\n"
        )

    efeito_release = (
        f"a fase do ciclo de release melhora o REQM em média {np.mean(ganhos_release):.1f}% "
        f"(entre {np.min(ganhos_release):+.1f}% e {np.max(ganhos_release):+.1f}% nas quatro arquiteturas)"
    )

    contexto: dict[str, object] = {
        "comparacao_uni_multi": comparacao_uni_multi.rstrip("\n"),
        "comparacao_release": comparacao_release.rstrip("\n"),
        "efeito_release": efeito_release,
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
    train, test, features_mult, features_release = carregar_dados_e_preparar()
    dados_por_regime, scaler_y = preparar_janelas_deslizantes(
        train, test, features_mult, features_release, window_size=WINDOW_SIZE
    )
    resultados, hist_loss, previsoes_teste, parametros = treinar_modelos(
        dados_por_regime, scaler_y, window_size=WINDOW_SIZE, epochs=EPOCHS
    )
    dimensoes = {rotulo: dados_por_regime[rotulo][0].shape[2] for rotulo in REGIMES}
    gerar_tabelas_dl(resultados, parametros, dados_por_regime, window_size=WINDOW_SIZE, epochs=EPOCHS)
    criar_figura_curvas_aprendizagem(hist_loss)
    criar_figura_previsoes_teste(test, previsoes_teste, resultados, dimensoes, window_size=WINDOW_SIZE)
    exportar_resumo_dl(resultados)
    finalizar_script("03")


if __name__ == "__main__":
    main()
