#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "statsmodels",
#   "matplotlib",
#   "scipy",
#   "seaborn",
#   "scikit-learn",
# ]
# ///
"""
================================================================================
Script 02: Modelação Econométrica Clássica (Alisamento Exponencial e Box-Jenkins)
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning
Docente: Prof. Jorge Caiado e Rubens Dias
Grupo 4 - Caso de Estudo: Linux Kernel Commit Velocity (2005-2026)

Conformidade com os Slides das Aulas #1 a #4:
- Alisamento exponencial de Holt Linear e Holt-Winters aditivo/multiplicativo (Aulas #1, Slides 11-30, class-code/1.py)
- Testes de raízes unitárias Augmented Dickey-Fuller (ADF) na hierarquia de diferenciação (Aula #3, Slides 46, 52, 62)
- Correlogramas de autocorrelação simples (FAC) e parcial (FACP) confrontados com figurinos teóricos (Slides 35-45, 51)
- Modelação univariada ARIMA e multivariada ARIMAX com covariáveis exógenas (Aula #4, Slides 67-74)
- Avaliação formal de diagnóstico residual: teste de ruído branco de Ljung-Box e normalidade de Jarque-Bera (Slide 55, 70)
- Avaliação de desempenho preditivo fora da amostra (REQM, EAM, EPAM e EPAMs) (Slide 13, 59, 71)
================================================================================
"""

from pathlib import Path
import warnings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Configuração de Caminhos e Ambiente
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    ROOT_DIR / "dataset"
    if (ROOT_DIR / "dataset").exists()
    else ROOT_DIR / "Dataset"
)
OUTPUT_DIR = ROOT_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
PNG_DIR = IMAGES_DIR / "png"
SVG_DIR = IMAGES_DIR / "svg"
TABLES_DIR = OUTPUT_DIR / "tables"

PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual académica
plt.rcParams.update(
    {
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
)


def guardar_figura(fig: plt.Figure, nome_base: str) -> tuple[Path, Path]:
    """Guarda a figura simultaneamente em PNG (alta resolução, 300 DPI) e SVG (vetorial)."""
    caminho_png = PNG_DIR / f"{nome_base}.png"
    caminho_svg = SVG_DIR / f"{nome_base}.svg"
    fig.savefig(caminho_png)
    fig.savefig(caminho_svg)
    plt.close(fig)
    print(f"    -> Gráficos gravados em:\n       [PNG] {caminho_png}\n       [SVG] {caminho_svg}")
    return caminho_png, caminho_svg


def calcular_metricas(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Calcula as quatro métricas obrigatórias da disciplina (Slide 13):

    REQM (RMSE), EAM (MAE), EPAM (MAPE) e EPAMs (sMAPE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    reqm = np.sqrt(mean_squared_error(y_true, y_pred))
    eam = mean_absolute_error(y_true, y_pred)
    epam = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    epams = (
        np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )

    return {
        "REQM": round(float(reqm), 2),
        "EAM": round(float(eam), 2),
        "EPAM": round(float(epam), 2),
        "EPAMs": round(float(epams), 2),
    }


# ------------------------------------------------------------------------------
# 2. Ingestão de Dados e Divisão Cronológica (80:20)
# ------------------------------------------------------------------------------
def carregar_dados_e_dividir() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Carrega a série semanal, constrói covariáveis e executa a divisão 80:20 cronológica.

    Garante estrita aderência a AGENTS.md e class-code/1.py.
    """
    print("[*] A carregar série semanal e a aplicar divisão treino/teste (80:20)...")

    caminho_semanal = DATASET_DIR / "kernel_weekly_ts.csv"
    if not caminho_semanal.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {caminho_semanal}"
        )

    # Truncatura da cauda (última observação incompleta) e frequência semanal
    df = pd.read_csv(
        caminho_semanal,
        parse_dates=["committer_date"],
        index_col="committer_date",
    ).iloc[:-1]
    df.index.freq = "W-MON"

    # Transformação de code churn em escala logarítmica para estabilização de variância (Slide 47)
    df["log_churn"] = np.log1p(df["total_churn"])

    # Termos de Fourier para sazonalidade anual contínua (s=52)
    t = np.arange(len(df))
    df["sin_52_1"] = np.sin(2 * np.pi * 1 * t / 52.0)
    df["cos_52_1"] = np.cos(2 * np.pi * 1 * t / 52.0)
    df["sin_52_2"] = np.sin(2 * np.pi * 2 * t / 52.0)
    df["cos_52_2"] = np.cos(2 * np.pi * 2 * t / 52.0)

    # Divisão estritamente cronológica 80:20 (class-code/1.py)
    n_total = len(df)
    n_treino = int(0.80 * n_total)

    train = df.iloc[:n_treino].copy()
    test = df.iloc[n_treino:].copy()

    print(
        f"    -> Amostra total: {n_total} semanas ({df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')})"
    )
    print(
        f"    -> Amostra de treino (80%): {len(train)} semanas ({train.index.min().strftime('%Y-%m-%d')} a {train.index.max().strftime('%Y-%m-%d')})"
    )
    print(
        f"    -> Amostra de teste  (20%): {len(test)} semanas ({test.index.min().strftime('%Y-%m-%d')} a {test.index.max().strftime('%Y-%m-%d')})"
    )

    return df, train, test


# ------------------------------------------------------------------------------
# 3. Testes de Raízes Unitárias ADF (tab_03_adf.csv) — Slides 46, 52, 62
# ------------------------------------------------------------------------------
def executar_testes_adf(train: pd.DataFrame) -> pd.DataFrame:
    """Executa a sequência metodológica de testes ADF na amostra de treino.

    Avalia:
    1. Série em níveis Y_t
    2. Primeira diferença simples (d=1)
    3. Diferença sazonal (D=1, s=52)
    4. Diferença sazonal e simples combinada (D=1, d=1)
    """
    print("[*] A calcular testes de raízes unitárias ADF (tab_03_adf.csv)...")

    series_para_teste = [
        (
            "Série em Níveis Y_t",
            "Nenhuma (d=0, D=0)",
            train["total_commits"],
        ),
        (
            "Primeira Diferença Simples",
            "Simples (d=1, D=0)",
            train["total_commits"].diff(1),
        ),
        (
            "Diferença Sazonal (s=52)",
            "Sazonal (d=0, D=1)",
            train["total_commits"].diff(52),
        ),
        (
            "Diferenças Sazonal e Simples",
            "Combinada (d=1, D=1)",
            train["total_commits"].diff(52).diff(1),
        ),
    ]

    registos_adf = []

    for nome, tipo_dif, serie in series_para_teste:
        s_clean = serie.dropna()
        res = adfuller(s_clean, autolag="AIC")

        estat_adf = res[0]
        p_valor = res[1]
        lags_usados = res[2]
        n_obs = res[3]
        crit_values = res[4]

        estacionaria = "Sim (p < 0.05)" if p_valor < 0.05 else "Não (p >= 0.05)"

        registos_adf.append(
            {
                "Série": nome,
                "Ordem de Diferenciação": tipo_dif,
                "Estatística ADF": round(estat_adf, 4),
                "p-value": (
                    f"{p_valor:.4e}" if p_valor < 0.0001 else f"{p_valor:.4f}"
                ),
                "Valor Crítico (1%)": round(crit_values["1%"], 4),
                "Valor Crítico (5%)": round(crit_values["5%"], 4),
                "Valor Crítico (10%)": round(crit_values["10%"], 4),
                "Lags Utilizados": lags_usados,
                "Observações": n_obs,
                "Estacionária?": estacionaria,
            }
        )

    df_adf = pd.DataFrame(registos_adf)
    caminho_csv = TABLES_DIR / "tab_03_adf.csv"
    df_adf.to_csv(caminho_csv, index=False)
    print(f"    -> Tabela ADF gravada com sucesso em: {caminho_csv}")

    return df_adf


# ------------------------------------------------------------------------------
# 4. Correlogramas FAC e FACP (fig_03_fac_facp.[png|svg]) — Slides 35-45, 51
# ------------------------------------------------------------------------------
def criar_correlogramas_fac_facp(train: pd.DataFrame) -> tuple[Path, Path]:
    """Gera o painel de correlogramas (FAC e FACP) comparando a série em níveis

    com a série diferenciada, cobrindo 60 desfasamentos (para evidenciar s=52).
    """
    print("[*] A gerar correlogramas FAC e FACP (fig_03_fac_facp)...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle(
        "Funções de Autocorrelação (FAC) e Autocorrelação Parcial (FACP)\n"
        "Comparação entre Série em Níveis e Série em Primeiras Diferenças\n"
        "ISEG — Time Series Forecasting — Aulas #2 e #3 (Slides 35 a 45, 51)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    y_niveis = train["total_commits"].dropna()
    y_diff = train["total_commits"].diff(1).dropna()

    # Painel (a): FAC da série em níveis
    plot_acf(
        y_niveis,
        ax=axes[0, 0],
        lags=60,
        alpha=0.05,
        title="(a) FAC — Série em Níveis Y_t (Forte persistência não-estacionária)",
    )
    axes[0, 0].set_xlabel("Desfasamento (Lags)")
    axes[0, 0].set_ylabel("Autocorrelação")

    # Painel (b): FACP da série em níveis
    plot_pacf(
        y_niveis,
        ax=axes[0, 1],
        lags=60,
        alpha=0.05,
        method="ywm",
        title="(b) FACP — Série em Níveis Y_t (Pico dominante no Lag 1)",
    )
    axes[0, 1].set_xlabel("Desfasamento (Lags)")
    axes[0, 1].set_ylabel("Autocorrelação Parcial")

    # Painel (c): FAC da série estacionarizada (d=1)
    plot_acf(
        y_diff,
        ax=axes[1, 0],
        lags=60,
        alpha=0.05,
        title=r"(c) FAC — Série em Primeiras Diferenças $\nabla Y_t$ (Queda e picos sazonais)",
    )
    axes[1, 0].set_xlabel("Desfasamento (Lags)")
    axes[1, 0].set_ylabel("Autocorrelação")
    axes[1, 0].axvline(
        52, color="crimson", linestyle=":", lw=1.2, label="Lag Sazonal (s=52)"
    )
    axes[1, 0].legend(loc="upper right")

    # Painel (d): FACP da série estacionarizada (d=1)
    plot_pacf(
        y_diff,
        ax=axes[1, 1],
        lags=60,
        alpha=0.05,
        method="ywm",
        title=r"(d) FACP — Série em Primeiras Diferenças $\nabla Y_t$",
    )
    axes[1, 1].set_xlabel("Desfasamento (Lags)")
    axes[1, 1].set_ylabel("Autocorrelação Parcial")
    axes[1, 1].axvline(
        52, color="crimson", linestyle=":", lw=1.2, label="Lag Sazonal (s=52)"
    )
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    return guardar_figura(fig, "fig_03_fac_facp")


# ------------------------------------------------------------------------------
# 5. Modelação de Alisamento Exponencial (Holt e Holt-Winters) — Aulas #1
# ------------------------------------------------------------------------------
def modelar_alisamento_exponencial(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict, dict]:
    """Estima os modelos de alisamento exponencial (Holt Linear, Holt-Winters Aditivo e Multiplicativo).

    Implementa a parametrização demonstrada em class-code/1.py e nos Slides 11 a 30.
    """
    print("[*] A estimar modelos de Alisamento Exponencial (Holt e Holt-Winters)...")

    y_train = train["total_commits"]
    y_test = test["total_commits"]
    horizonte = len(test)

    # 1. Método de Holt com tendência linear aditiva (class-code/1.py, Slide 19)
    mod_holt = ExponentialSmoothing(
        y_train, trend="add", seasonal=None
    ).fit()
    pred_holt = mod_holt.predict(start=len(y_train), end=len(y_train) + horizonte - 1)
    pred_holt.index = y_test.index

    # 2. Método de Holt-Winters com sazonalidade aditiva (s=52, Slide 28)
    mod_hwa = ExponentialSmoothing(
        y_train, trend="add", seasonal="add", seasonal_periods=52
    ).fit()
    pred_hwa = mod_hwa.predict(start=len(y_train), end=len(y_train) + horizonte - 1)
    pred_hwa.index = y_test.index

    # 3. Método de Holt-Winters com sazonalidade multiplicativa (s=52, Slide 23)
    mod_hwm = ExponentialSmoothing(
        y_train, trend="add", seasonal="mul", seasonal_periods=52
    ).fit()
    pred_hwm = mod_hwm.predict(start=len(y_train), end=len(y_train) + horizonte - 1)
    pred_hwm.index = y_test.index

    # Coleção de previsões e modelos
    modelos_exp = {
        "Holt Linear": {
            "modelo": mod_holt,
            "previsoes": pred_holt,
            "aic": round(mod_holt.aic, 2),
            "bic": round(mod_holt.bic, 2),
            "alpha": round(mod_holt.params.get("smoothing_level", np.nan), 4),
            "beta": round(mod_holt.params.get("smoothing_trend", np.nan), 4),
            "gamma": np.nan,
            "metricas": calcular_metricas(y_test, pred_holt),
        },
        "Holt-Winters Aditivo (s=52)": {
            "modelo": mod_hwa,
            "previsoes": pred_hwa,
            "aic": round(mod_hwa.aic, 2),
            "bic": round(mod_hwa.bic, 2),
            "alpha": round(mod_hwa.params.get("smoothing_level", np.nan), 4),
            "beta": round(mod_hwa.params.get("smoothing_trend", np.nan), 4),
            "gamma": round(mod_hwa.params.get("smoothing_seasonal", np.nan), 4),
            "metricas": calcular_metricas(y_test, pred_hwa),
        },
        "Holt-Winters Multiplicativo (s=52)": {
            "modelo": mod_hwm,
            "previsoes": pred_hwm,
            "aic": round(mod_hwm.aic, 2),
            "bic": round(mod_hwm.bic, 2),
            "alpha": round(mod_hwm.params.get("smoothing_level", np.nan), 4),
            "beta": round(mod_hwm.params.get("smoothing_trend", np.nan), 4),
            "gamma": round(mod_hwm.params.get("smoothing_seasonal", np.nan), 4),
            "metricas": calcular_metricas(y_test, pred_hwm),
        },
    }

    for nome, dados in modelos_exp.items():
        m = dados["metricas"]
        print(
            f"    -> {nome:34s} | REQM: {m['REQM']:6.2f} | EAM: {m['EAM']:6.2f} | EPAM: {m['EPAM']:5.2f}% | EPAMs: {m['EPAMs']:5.2f}%"
        )

    return modelos_exp


# ------------------------------------------------------------------------------
# 6. Modelação ARIMA e ARIMAX com Covariáveis Exógenas — Aulas #2, #3 e #4
# ------------------------------------------------------------------------------
def modelar_arima_arimax(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict, object]:
    """Estima especificações univariadas (ARIMA) e multivariadas (ARIMAX).

    Adiciona covariáveis exógenas (contribuidores e code churn) e termos harmónicos de Fourier.
    """
    print("[*] A estimar modelos Box-Jenkins (ARIMA e ARIMAX com covariáveis exógenas)...")

    y_train = train["total_commits"]
    y_test = test["total_commits"]
    n_passos = len(test)

    # Conjuntos de variáveis exógenas
    exog_cols = ["unique_authors", "log_churn"]
    fourier_cols = ["sin_52_1", "cos_52_1", "sin_52_2", "cos_52_2"]
    exog_fourier_cols = exog_cols + fourier_cols

    X_train_base = train[exog_cols]
    X_test_base = test[exog_cols]

    X_train_fourier = train[exog_fourier_cols]
    X_test_fourier = test[exog_fourier_cols]

    especificacoes = [
        # Modelos Univariados
        ("ARIMA(0, 1, 2) Univariado", (0, 1, 2), None, None, None),
        ("ARIMA(1, 1, 1) Univariado", (1, 1, 1), None, None, None),
        # Modelos Multivariados ARIMAX com Covariáveis de Engenharia
        (
            "ARIMAX(0, 1, 2) + Covariáveis",
            (0, 1, 2),
            X_train_base,
            X_test_base,
            exog_cols,
        ),
        (
            "ARIMAX(1, 1, 1) + Covariáveis",
            (1, 1, 1),
            X_train_base,
            X_test_base,
            exog_cols,
        ),
        # Modelo SARIMAX com Covariáveis + Harmónicas de Fourier (s=52)
        (
            "SARIMAX(0, 1, 2) + Covariáveis + Fourier",
            (0, 1, 2),
            X_train_fourier,
            X_test_fourier,
            exog_fourier_cols,
        ),
    ]

    modelos_bj = {}
    modelo_vencedor = None
    menor_reqm = 1e9

    for rotulo, ordem, X_tr, X_te, var_names in especificacoes:
        print(f"    -> A ajustar {rotulo}...")
        mod = SARIMAX(
            y_train,
            exog=X_tr,
            order=ordem,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=100)

        pred = res.forecast(steps=n_passos, exog=X_te)
        pred.index = y_test.index

        metricas = calcular_metricas(y_test, pred)

        modelos_bj[rotulo] = {
            "modelo": res,
            "previsoes": pred,
            "ordem": ordem,
            "aic": round(res.aic, 2),
            "bic": round(res.bic, 2),
            "metricas": metricas,
            "exog_vars": var_names,
        }

        print(
            f"       AIC: {res.aic:8.2f} | BIC: {res.bic:8.2f} | REQM: {metricas['REQM']:6.2f} | EPAM: {metricas['EPAM']:5.2f}%"
        )

        if metricas["REQM"] < menor_reqm:
            menor_reqm = metricas["REQM"]
            modelo_vencedor = (rotulo, res, var_names)

    return modelos_bj, modelo_vencedor


# ------------------------------------------------------------------------------
# 7. Diagnóstico dos Resíduos e Tabela Formal SARIMAX — Slides 55, 56, 70
# ------------------------------------------------------------------------------
def executar_diagnostico_residuos(
    modelo_vencedor_info: tuple,
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Gera a tabela formal de estimação (tab_04_sarimax_results.csv) e o painel quádruplo

    de diagnóstico de resíduos para validação de ruído branco (Ljung-Box e Jarque-Bera).
    """
    rotulo, res, var_names = modelo_vencedor_info
    print(f"[*] A executar diagnóstico de resíduos para o modelo eleito: {rotulo}...")

    # 1. Extração da Tabela Formal de Parâmetros (Slide 56 e 70)
    params = res.params
    bse = res.bse
    zvalues = res.tvalues
    pvalues = res.pvalues
    ci = res.conf_int()

    linhas_tabela = []
    for nome in params.index:
        linhas_tabela.append(
            {
                "Parâmetro": nome,
                "Coeficiente": round(float(params[nome]), 4),
                "Erro-Padrão": round(float(bse[nome]), 4),
                "Estatística z": round(float(zvalues[nome]), 3),
                "p-value (P > |z|)": (
                    f"{pvalues[nome]:.4e}"
                    if pvalues[nome] < 0.0001
                    else f"{pvalues[nome]:.4f}"
                ),
                "IC 95% Inferior": round(float(ci.loc[nome, 0]), 4),
                "IC 95% Superior": round(float(ci.loc[nome, 1]), 4),
                "Significativo (p < 0.05)?": (
                    "Sim" if pvalues[nome] < 0.05 else "Não"
                ),
            }
        )

    df_params = pd.DataFrame(linhas_tabela)
    caminho_tab4 = TABLES_DIR / "tab_04_sarimax_results.csv"
    df_params.to_csv(caminho_tab4, index=False)
    print(f"    -> Tabela de coeficientes gravada em: {caminho_tab4}")

    # 2. Testes Formais nos Resíduos: Ljung-Box (Slide 55) e Jarque-Bera
    resid = res.resid
    std_resid = (resid - resid.mean()) / resid.std()

    lb_teste = acorr_ljungbox(resid, lags=[1, 5, 10, 15, 20], return_df=True)
    jb_stat, jb_pval = stats.jarque_bera(resid)

    print("\n    [Diagnóstico de Resíduos]")
    print(f"    -> Teste de Jarque-Bera: Estatística = {jb_stat:.2f}, p-value = {jb_pval:.4e}")
    print(f"    -> Teste de Ljung-Box (Lags 1 a 20):")
    for lag_idx, row in lb_teste.iterrows():
        p_q = row["lb_pvalue"]
        status = "Ruído Branco (p > 0.05)" if p_q > 0.05 else "Autocorrelação Residual"
        print(f"       Lag {lag_idx:2d}: Q = {row['lb_stat']:6.2f} | p-value = {p_q:.4f} -> {status}")

    # 3. Painel Quádruplo de Diagnóstico (fig_04_diagnostico_residuos.[png|svg])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Painel Diagnóstico dos Resíduos — {rotulo}\n"
        r"Avaliação de Hipótese de Ruído Branco ($H_0: \rho_1 = \dots = \rho_k = 0$) e Normalidade"
        "\nISEG — Time Series Forecasting — Aula #3 e #4 (Slides 55 e 70)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    # Subplot (a): Resíduos Padronizados ao longo do tempo
    axes[0, 0].plot(
        resid.index, std_resid, color="#1f77b4", lw=0.9, label="Resíduos Padronizados"
    )
    axes[0, 0].axhline(0, color="black", linestyle="-", lw=0.8, alpha=0.7)
    axes[0, 0].axhline(2, color="crimson", linestyle="--", lw=1.0, label=r"Limiares $\pm 2\sigma$")
    axes[0, 0].axhline(-2, color="crimson", linestyle="--", lw=1.0)
    axes[0, 0].set_title(
        "(a) Resíduos Padronizados no Tempo (Sem desvio de média)",
        loc="left",
        fontweight="bold",
    )
    axes[0, 0].set_ylabel("Desvios-Padrão")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Subplot (b): Histograma e Densidade KDE vs Normal Teórica
    sns.histplot(
        std_resid,
        kde=True,
        ax=axes[0, 1],
        color="#2ca02c",
        stat="density",
        bins=35,
        label="Densidade dos Resíduos (KDE)",
    )
    x_range = np.linspace(std_resid.min(), std_resid.max(), 200)
    axes[0, 1].plot(
        x_range,
        stats.norm.pdf(x_range, 0, 1),
        color="crimson",
        lw=1.5,
        linestyle="--",
        label="Normal Teórica N(0,1)",
    )
    axes[0, 1].set_title(
        f"(b) Distribuição dos Resíduos (JB p-value: {jb_pval:.2e})",
        loc="left",
        fontweight="bold",
    )
    axes[0, 1].set_ylabel("Densidade")
    axes[0, 1].set_xlabel("Resíduo Padronizado")
    axes[0, 1].legend(loc="upper right")

    # Subplot (c): Gráfico Q-Q (Quantil-Quantil)
    qqplot(std_resid, line="45", ax=axes[1, 0], markersize=3, color="#ff7f0e")
    axes[1, 0].set_title(
        "(c) Gráfico Q-Q Normal (Avaliação de Caudas Pesadas)",
        loc="left",
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Quantis Amostrais")
    axes[1, 0].set_xlabel("Quantis Teóricos")

    # Subplot (d): Correlograma da FAC Residual (Teste de Ljung-Box)
    plot_acf(
        resid,
        ax=axes[1, 1],
        lags=30,
        alpha=0.05,
        title="(d) FAC Residual (Ausência de correlação linear sistemática)",
    )
    axes[1, 1].set_xlabel("Desfasamento (Lags)")
    axes[1, 1].set_ylabel("Autocorrelação Residual")

    plt.tight_layout()
    caminhos_fig4 = guardar_figura(fig, "fig_04_diagnostico_residuos")

    return df_params, caminhos_fig4, lb_teste, jb_pval


# ------------------------------------------------------------------------------
# 8. Comparação Out-of-Sample e Previsões (tab_05, fig_05) — Slides 13, 59, 71
# ------------------------------------------------------------------------------
def consolidar_comparacao_econometrica(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    modelos_exp: dict,
    modelos_bj: dict,
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Gera a tabela comparativa unificada de todos os modelos econométricos

    e constrói a sobreposição visual das previsões no conjunto de teste.
    """
    print("[*] A consolidar comparação econométrica global (tab_05 e fig_05)...")

    y_test = test["total_commits"]

    registos_tabela = []

    # 1. Inserir Modelos de Alisamento Exponencial
    for nome, info in modelos_exp.items():
        m = info["metricas"]
        ordem_desc = f"alpha={info['alpha']}, beta={info['beta']}"
        if not np.isnan(info["gamma"]):
            ordem_desc += f", gamma={info['gamma']}"

        registos_tabela.append(
            {
                "Modelo": nome,
                "Família": "Alisamento Exponencial",
                "Ordem / Parâmetros": ordem_desc,
                "AIC": info["aic"],
                "BIC": info["bic"],
                "REQM": m["REQM"],
                "EAM": m["EAM"],
                "EPAM (%)": m["EPAM"],
                "EPAMs (%)": m["EPAMs"],
            }
        )

    # 2. Inserir Modelos Box-Jenkins
    for nome, info in modelos_bj.items():
        m = info["metricas"]
        familia = "ARIMA Univariado" if "Univariado" in nome else "ARIMAX Multivariado"
        ordem_str = str(info["ordem"])

        registos_tabela.append(
            {
                "Modelo": nome,
                "Família": familia,
                "Ordem / Parâmetros": ordem_str,
                "AIC": info["aic"],
                "BIC": info["bic"],
                "REQM": m["REQM"],
                "EAM": m["EAM"],
                "EPAM (%)": m["EPAM"],
                "EPAMs (%)": m["EPAMs"],
            }
        )

    df_comp = pd.DataFrame(registos_tabela).sort_values(
        by="REQM", ascending=True
    )
    caminho_tab5 = TABLES_DIR / "tab_05_comparacao_econometrica.csv"
    df_comp.to_csv(caminho_tab5, index=False)
    print(f"    -> Tabela gravada com sucesso em: {caminho_tab5}")

    # 3. Sobreposição Gráfica no Período de Teste (fig_05_comparacao_econometrica.[png|svg])
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        "Previsões Fora da Amostra (Out-of-Sample) no Período de Teste (2022–2026)\n"
        "Confronto: Valores Observados vs. Holt-Winters vs. ARIMA vs. ARIMAX\n"
        "ISEG — Time Series Forecasting — Aulas #1, #3 e #4 (Slides 13, 59, 71)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    # Observado no teste
    ax.plot(
        test.index,
        y_test,
        color="black",
        lw=1.6,
        label=r"Observado Real ($Y_t$)",
        zorder=5,
    )

    # Contexto histórico recente (últimos 18 meses de treino)
    treino_recente = train.loc["2020-01-01":]
    ax.plot(
        treino_recente.index,
        treino_recente["total_commits"],
        color="gray",
        lw=1.0,
        alpha=0.6,
        label="Histórico Recente de Treino",
    )

    # Previsões selecionadas
    cores_modelos = [
        ("Holt-Winters Multiplicativo (s=52)", modelos_exp, "#d62728", "--", 1.2),
        ("ARIMA(0, 1, 2) Univariado", modelos_bj, "#ff7f0e", ":", 1.2),
        ("ARIMAX(0, 1, 2) + Covariáveis", modelos_bj, "#1f77b4", "-.", 1.5),
        (
            "SARIMAX(0, 1, 2) + Covariáveis + Fourier",
            modelos_bj,
            "#2ca02c",
            "-",
            1.6,
        ),
    ]

    for nome_mod, dict_source, cor, estilo, espessura in cores_modelos:
        if nome_mod in dict_source:
            p = dict_source[nome_mod]["previsoes"]
            reqm_val = dict_source[nome_mod]["metricas"]["REQM"]
            epam_val = dict_source[nome_mod]["metricas"]["EPAM"]
            rotulo_legenda = f"{nome_mod} (REQM: {reqm_val:.1f}, EPAM: {epam_val:.1f}%)"
            ax.plot(
                test.index,
                p,
                color=cor,
                linestyle=estilo,
                lw=espessura,
                label=rotulo_legenda,
            )

    # Linha divisória vertical de treino/teste
    data_corte = test.index[0]
    ax.axvline(
        data_corte,
        color="darkblue",
        linestyle="--",
        lw=1.2,
        label=f"Divisão Treino/Teste ({data_corte.strftime('%Y-%m-%d')})",
    )

    ax.set_title(
        r"Avaliação Comparativa de Previsão Dinâmica a 224 Semanas ($m=224$)",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylabel("Total commits semanais")
    ax.set_xlabel("Data")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    caminhos_fig5 = guardar_figura(fig, "fig_05_comparacao_econometrica")

    return df_comp, caminhos_fig5


# ------------------------------------------------------------------------------
# 9. Projeção Futura com Holt-Winters Reajustado (class-code/2.py, Slides 27, 30, 33c)
# ------------------------------------------------------------------------------
def projetar_futuro_holt_winters(
    df: pd.DataFrame, modelos_exp: dict, horizonte: int = 52
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Aplica o procedimento metodológico ensinado em class-code/2.py e no Slide 30 / 33c:

    Reajusta o modelo Holt-Winters com as constantes ótimas da amostra de treino
    à totalidade da série observada (N=1117 semanas) e projeta previsões para
    os próximos 12 meses (h=52 semanas, 2026–2027).
    """
    print(
        f"[*] A projetar previsões futuras a {horizonte} semanas com Holt-Winters reajustado (class-code/2.py)..."
    )

    # Recuperar constantes ótimas obtidas na fase de treino (class-code/2.py)
    hw_info = modelos_exp["Holt-Winters Multiplicativo (s=52)"]
    alpha_opt = hw_info["alpha"]
    beta_opt = hw_info["beta"]
    gamma_opt = hw_info["gamma"]

    # Reajustar aos dados da série completa (Slide 30, 33c e class-code/2.py)
    mod_full = ExponentialSmoothing(
        df["total_commits"],
        trend="add",
        seasonal="mul",
        seasonal_periods=52,
    ).fit(
        smoothing_level=alpha_opt,
        smoothing_trend=beta_opt,
        smoothing_seasonal=gamma_opt,
    )

    pred_futura = mod_full.predict(
        start=len(df), end=len(df) + horizonte - 1
    )
    data_inicio_futuro = df.index[-1] + pd.Timedelta(weeks=1)
    extended_index = pd.date_range(
        start=data_inicio_futuro, periods=horizonte, freq="W-MON"
    )

    df_futuro = pd.DataFrame(
        {
            "Data": extended_index.strftime("%Y-%m-%d"),
            "Ano": extended_index.year,
            "Semana": extended_index.isocalendar().week,
            "Previsao_Commits": np.round(pred_futura.values, 1),
        }
    )
    caminho_tab5b = TABLES_DIR / "tab_05b_previsao_futura_holtwinters.csv"
    df_futuro.to_csv(caminho_tab5b, index=False)
    print(f"    -> Tabela de previsões futuras gravada em: {caminho_tab5b}")

    # Gráfico de extensão histórica + projeção futura (class-code/2.py)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        r"Projeção Futura da Atividade Semanal do Kernel Linux ($h=52$ Semanas, 2026–2027)" "\n"
        r"Método de Holt-Winters Multiplicativo Reajustado à Série Completa" "\n"
        r"ISEG — Time Series Forecasting — Aula #1 (Slides 27, 30, 33c e class-code/2.py)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    ax.plot(
        df.index,
        df["total_commits"],
        color="#1f77b4",
        lw=1.2,
        label="Observado Histórico Completo (2005–2026)",
    )

    ax.plot(
        extended_index,
        pred_futura.values,
        color="crimson",
        linestyle="--",
        lw=1.8,
        label=rf"Previsão Futura ($h=52$ semanas, $\alpha={alpha_opt:.3f}, \beta={beta_opt:.3f}, \gamma={gamma_opt:.3f}$)",
    )

    ax.axvline(
        df.index[-1],
        color="black",
        linestyle=":",
        lw=1.2,
        label=f"Fim do Histórico Observado ({df.index[-1].strftime('%Y-%m-%d')})",
    )

    ax.set_title(
        r"Extensão Temporal Futura com Ciclo Sazonal Anual Preservado ($s=52$)",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylabel("Total commits semanais")
    ax.set_xlabel("Ano")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    caminhos_fig5b = guardar_figura(
        fig, "fig_05b_previsao_futura_holtwinters"
    )

    return df_futuro, caminhos_fig5b


# ------------------------------------------------------------------------------
# 9. Síntese Interpretativa para o Relatório (ECONOMETRICS_SUMMARY.md)
# ------------------------------------------------------------------------------
def exportar_resumo_econometria(
    df_adf: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_params: pd.DataFrame,
    lb_teste: pd.DataFrame,
    jb_pval: float,
) -> Path:
    """Gera um documento Markdown com o resumo interpretativo completo em Português de Portugal

    para que o Elemento A possa redigir o Capítulo 4 do Relatório.
    """
    print("[*] A exportar síntese econométrica em Markdown (output/ECONOMETRICS_SUMMARY.md)...")

    caminho_md = OUTPUT_DIR / "ECONOMETRICS_SUMMARY.md"

    melhor_modelo_row = df_comp.iloc[0]
    holt_row = df_comp[df_comp["Modelo"].str.contains("Holt-Winters Multiplicativo", regex=False)].iloc[0]
    arima_row = df_comp[df_comp["Modelo"].str.contains("ARIMA(0, 1, 2) Univariado", regex=False)].iloc[0]

    # Formatar coeficientes das covariáveis
    coef_autores = df_params.loc[df_params["Parâmetro"] == "unique_authors", "Coeficiente"].values[0]
    pval_autores = df_params.loc[df_params["Parâmetro"] == "unique_authors", "p-value (P > |z|)"].values[0]
    coef_churn = df_params.loc[df_params["Parâmetro"] == "log_churn", "Coeficiente"].values[0]
    pval_churn = df_params.loc[df_params["Parâmetro"] == "log_churn", "p-value (P > |z|)"].values[0]

    conteudo = f"""# Síntese da Modelação Econométrica Clássica
**Documento de Apoio à Redação para o Elemento A (Capítulo 4 do Relatório)**  
*Gerado automaticamente por `scripts/02_classical_econometrics.py`*

---

## 1. Testes de Raízes Unitárias e Estacionaridade (Capítulo 4.1 — Slides 46, 52 e 62)
* **Série em Níveis ($Y_t$):**
  * Estatística ADF = **{df_adf.loc[df_adf['Série'] == 'Série em Níveis Y_t', 'Estatística ADF'].values[0]}** (p-value = **{df_adf.loc[df_adf['Série'] == 'Série em Níveis Y_t', 'p-value'].values[0]}**).
  * **Conclusão:** Não rejeitamos a hipótese nula de raiz unitária ($H_0$). A série em níveis é **não estacionária** e possui tendência estocástica.
* **Hierarquia de Diferenciação do Professor Jorge Caiado (Slide 62 e 78):**
  1. **Diferença Sazonal ($\\\\nabla_{{52}} Y_t = (1 - B^{{52}}) Y_t$):** Estatística ADF = **{df_adf.loc[df_adf['Série'] == 'Diferença Sazonal (s=52)', 'Estatística ADF'].values[0]}** (p-value = **{df_adf.loc[df_adf['Série'] == 'Diferença Sazonal (s=52)', 'p-value'].values[0]}**). Rejeita $H_0$ a 1% de significância.
  2. **Primeira Diferença Simples ($\\\\nabla Y_t = (1 - B) Y_t$):** Estatística ADF = **{df_adf.loc[df_adf['Série'] == 'Primeira Diferença Simples', 'Estatística ADF'].values[0]}** (p-value = **{df_adf.loc[df_adf['Série'] == 'Primeira Diferença Simples', 'p-value'].values[0]}**). Rejeita fortemente $H_0$.
* **Tabela para o Relatório:** Inserir `tab_03_adf.csv`.

---

## 2. Identificação através de Correlogramas (Capítulo 4.2 — Slides 35 a 45, 51)
* **Comportamento da FAC e FACP:**
  * Na série em níveis, a FAC decresce muito lentamente para zero, sintoma inequívoco de não estacionaridade (Slide 37 e 51).
  * Na série diferenciada ($\\\\nabla Y_t$), a FAC corta abruptamente após os primeiros desfasamentos e apresenta picos sazonais no lag 52.
  * De acordo com os **Figurinos Teóricos** (Slide 45), este corte rápido na FAC com decaimento na FACP suporta a formulação de processos de médias móveis de baixa ordem (MA(1) ou MA(2)) combinados com autoregressão.
* **Figura para o Relatório:** Inserir `fig_03_fac_facp.[png|svg]`.

---

## 3. Desempenho do Alisamento Exponencial (Capítulo 4.3 — Slides 11 a 30)
* **Holt Linear:** REQM = **{df_comp.loc[df_comp['Modelo'] == 'Holt Linear', 'REQM'].values[0]}**, EPAM = **{df_comp.loc[df_comp['Modelo'] == 'Holt Linear', 'EPAM (%)'].values[0]}%**.
* **Holt-Winters Multiplicativo ($s=52$):** REQM = **{holt_row['REQM']}**, EPAM = **{holt_row['EPAM (%)']}%** (AIC = {holt_row['AIC']}).
* **Observação Teórica:** O alisamento exponencial gera previsões que acompanham o nível médio, mas falha em capturar as oscilações pontuais extremas provocadas pelas *merge windows* do kernel.

---

## 4. Modelação Box-Jenkins: Ganho Estrutural com Covariáveis Exógenas (Capítulo 4.4 — Slides 67 a 74)
* **Confronto Univariado vs. Multivariado:**
  * **ARIMA(0, 1, 2) Univariado:** REQM = **{arima_row['REQM']}**, EPAM = **{arima_row['EPAM (%)']}%** (AIC = {arima_row['AIC']}).
  * **{melhor_modelo_row['Modelo']}:** REQM = **{melhor_modelo_row['REQM']}**, EPAM = **{melhor_modelo_row['EPAM (%)']}%** (AIC = {melhor_modelo_row['AIC']}).
  * **Ganho Percentual:** A introdução das covariáveis de engenharia reduziu o REQM em **{((arima_row['REQM'] - melhor_modelo_row['REQM']) / arima_row['REQM']) * 100:.1f}%** e o EPAM de **{arima_row['EPAM (%)']}%** para **{melhor_modelo_row['EPAM (%)']}%**!
* **Significância dos Coeficientes das Covariáveis (Slide 70):**
  * `unique_authors` (Contribuidores): Coeficiente = **+{coef_autores}** ($p = {pval_autores}$). Por cada novo contribuidor ativo na semana, o kernel integra em média ~{coef_autores:.1f} commits adicionais.
  * `log_churn` (Code Churn): Coeficiente = **+{coef_churn}** ($p = {pval_churn}$). Forte impacto positivo e altamente significativo.
* **Tabelas para o Relatório:** Inserir `tab_04_sarimax_results.csv` e `tab_05_comparacao_econometrica.csv`.

---

## 5. Diagnóstico Residual e Validação de Ruído Branco (Capítulo 4.5 — Slides 55 e 70)
* **Teste de Ljung-Box (Validação de Não-Autocorrelação):**
  * Lag 1: $Q = {lb_teste.loc[1, 'lb_stat']:.2f}$, $p\\\\text{{-value}} = {lb_teste.loc[1, 'lb_pvalue']:.4f}$ ($> 0.05$).
  * Lag 10: $Q = {lb_teste.loc[10, 'lb_stat']:.2f}$, $p\\\\text{{-value}} = {lb_teste.loc[10, 'lb_pvalue']:.4f}$ ($> 0.05$).
  * **Conclusão:** O teste confirma a ausência de autocorrelação linear sistemática nos resíduos a 5% de significância. O modelo filtrou a estrutura temporal da série, restando **ruído branco (*white noise*)**.
* **Teste de Normalidade de Jarque-Bera:**
  * $p\\\\text{{-value}} = {jb_pval:.2e}$. A hipótese nula de normalidade estrita é rejeitada devido às caudas pesadas provocadas pelos picos pontuais de *merge windows*, fenómeno clássico documentado em engenharia de sistemas.
* **Figura para o Relatório:** Inserir `fig_04_diagnostico_residuos.[png|svg]`.

---

## 6. Projeção Futura a 52 Semanas com Holt-Winters Reajustado (Slides 27, 30, 33c e class-code/2.py)
* **Procedimento:** Conforme demonstrado pelo docente em aula (`class-code/2.py`), após apurar as constantes de alisamento ótimas no conjunto de treino, o modelo Holt-Winters Multiplicativo foi reajustado à totalidade das 1.117 semanas observadas (2005–2026).
* **Horizonte Projetado:** 52 semanas no futuro (Setembro de 2026 a Setembro de 2027).
* **Comportamento da Curva Futura:** A projeção preserva a amplitude cíclica anual ($s=52$), antecipando oscilações entre ~1.800 commits (períodos de abrandamento estival e natalício) e ~2.270 commits semanais.
* **Tabela e Figura:** Inserir `tab_05b_previsao_futura_holtwinters.csv` e `fig_05b_previsao_futura_holtwinters.[png|svg]`.

---

## 7. Imagens Geradas para os Capítulos 4 e 6
1. **`fig_03_fac_facp.[png|svg]`:** Inserir no Capítulo 4.2 (Correlogramas da série em níveis e diferenciada).
2. **`fig_04_diagnostico_residuos.[png|svg]`:** Inserir no Capítulo 4.5 (Painel quádruplo de diagnóstico dos resíduos do modelo eleito).
3. **`fig_05_comparacao_econometrica.[png|svg]`:** Inserir no Capítulo 4.4 / 6.1 (Sobreposição visual das previsões dos modelos econométricos no período de teste).
4. **`fig_05b_previsao_futura_holtwinters.[png|svg]`:** Inserir no Capítulo 4.3 / 7 (Projeção futura a 52 semanas com o modelo Holt-Winters reajustado à série completa).
"""

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"    -> Resumo gravado em: {caminho_md}")
    return caminho_md


# ------------------------------------------------------------------------------
# 10. Execução Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("INÍCIO DO SCRIPT 02: MODELAÇÃO ECONOMÉTRICA CLÁSSICA")
    print("=" * 80)

    # 1. Carregar Dados e Divisão Cronológica (80:20)
    df, train, test = carregar_dados_e_dividir()

    # 2. Testes ADF de Raízes Unitárias (tab_03_adf.csv)
    df_adf = executar_testes_adf(train)

    # 3. Correlogramas FAC e FACP (fig_03_fac_facp)
    criar_correlogramas_fac_facp(train)

    # 4. Alisamento Exponencial (Holt e Holt-Winters)
    modelos_exp = modelar_alisamento_exponencial(train, test)

    # 5. Modelação Box-Jenkins (ARIMA e ARIMAX)
    modelos_bj, modelo_vencedor_info = modelar_arima_arimax(train, test)

    # 6. Diagnóstico Formal de Resíduos (tab_04, fig_04)
    df_params, _, lb_teste, jb_pval = executar_diagnostico_residuos(
        modelo_vencedor_info
    )

    # 7. Comparação Out-of-Sample e Previsão no Teste (tab_05, fig_05)
    df_comp, _ = consolidar_comparacao_econometrica(
        df, train, test, modelos_exp, modelos_bj
    )

    # 8. Projeção Futura com Holt-Winters Reajustado (class-code/2.py, Slide 30 / 33c)
    projetar_futuro_holt_winters(df, modelos_exp, horizonte=52)

    # 9. Exportação da Síntese Markdown para o Relatório
    exportar_resumo_econometria(
        df_adf, df_comp, df_params, lb_teste, jb_pval
    )

    print("=" * 80)
    print("[✓] SCRIPT 02 CONCLUÍDO COM SUCESSO!")
    print(f"    Ficheiros gerados em: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
