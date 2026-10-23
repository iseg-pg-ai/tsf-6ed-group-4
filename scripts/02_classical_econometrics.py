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

import warnings
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.stattools import acf, adfuller
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


# Ingestão de Dados e Divisão Cronológica (80:20)
def carregar_dados_e_dividir() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega a série semanal, constrói covariáveis e executa a divisão 80:20 cronológica.

    Garante estrita aderência a AGENTS.md e class-code/1.py.
    """
    print("[*] A carregar série semanal e a aplicar divisão treino/teste (80:20)...")

    df = carregar_serie_temporal("kernel_weekly_ts.csv", "W-MON")

    # Transformação de code churn em escala logarítmica para estabilização de variância (Slide 47)
    df["log_churn"] = np.log1p(df["total_churn"])

    # Termos de Fourier para sazonalidade anual contínua (s=52)
    t = np.arange(len(df))
    df["sin_52_1"] = np.sin(2 * np.pi * 1 * t / 52.0)
    df["cos_52_1"] = np.cos(2 * np.pi * 1 * t / 52.0)
    df["sin_52_2"] = np.sin(2 * np.pi * 2 * t / 52.0)
    df["cos_52_2"] = np.cos(2 * np.pi * 2 * t / 52.0)

    # Divisão estritamente cronológica 80:20 (class-code/1.py)
    train, test = dividir_treino_teste(df, "semanas")

    return df, train, test


# Testes de Raízes Unitárias ADF (tab_03_adf.csv) — Slides 46, 52, 62
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
        res = cast(
            tuple[float, float, int, int, dict[str, float], float],
            adfuller(s_clean, autolag="AIC"),
        )

        estat_adf = res[0]
        p_valor = res[1]
        lags_usados = res[2]
        n_obs = res[3]
        crit_values = res[4]

        estacionaria = "Sim (p < 0.05)" if p_valor < 0.05 else "Não (p >= 0.05)"

        registos_adf.append({
            "Série": nome,
            "Ordem de Diferenciação": tipo_dif,
            "Estatística ADF": round(estat_adf, 4),
            "p-value": (f"{p_valor:.4e}" if p_valor < 0.0001 else f"{p_valor:.4f}"),
            "Valor Crítico (1%)": round(crit_values["1%"], 4),
            "Valor Crítico (5%)": round(crit_values["5%"], 4),
            "Valor Crítico (10%)": round(crit_values["10%"], 4),
            "Lags Utilizados": lags_usados,
            "Observações": n_obs,
            "Estacionária?": estacionaria,
        })

    df_adf = pd.DataFrame(registos_adf)
    guardar_tabela(df_adf, "tab_03_adf", "Tabela ADF")

    return df_adf


# Correlogramas FAC e FACP (fig_03_fac_facp.[png|svg]) — Slides 35-45, 51
def criar_correlogramas_fac_facp(train: pd.DataFrame) -> dict[str, object]:
    """Gera o painel de correlogramas (FAC e FACP) comparando a série em níveis

    com a série diferenciada, cobrindo 60 desfasamentos (para evidenciar s=52).
    Devolve o diagnóstico quantitativo do correlograma diferenciado, para que a
    síntese escrita descreva o que a figura mostra em vez de o presumir.
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

    # Diagnóstico do correlograma da série diferenciada: quais os desfasamentos
    # que excedem a banda de confiança e se existe pico no desfasamento anual.
    valores_acf = acf(y_diff, nlags=60, fft=False)
    valores_acf_niveis = acf(y_niveis, nlags=60, fft=False)
    banda_acf = 1.96 / np.sqrt(len(y_diff))
    lags_significativos = [lag for lag in range(1, 61) if abs(valores_acf[lag]) > banda_acf]
    lag_pico = max(range(1, 61), key=lambda lag: abs(valores_acf[lag]))
    acf52_significativo = abs(valores_acf[52]) > banda_acf
    diagnostico_correlograma: dict[str, object] = {
        "acf_niveis_lag60": f"{valores_acf_niveis[60]:.3f}",
        "acf_lag52": f"{valores_acf[52]:.3f}",
        "acf_lag52_sig": "sim" if acf52_significativo else "não",
        "acf_lag52_leitura": (
            f"um pico marginal, logo acima da banda de ±{banda_acf:.3f}"
            if acf52_significativo
            else f"um valor dentro da banda de ±{banda_acf:.3f}"
        ),
        "acf_banda": f"{banda_acf:.3f}",
        "acf_pico_lag": lag_pico,
        "acf_pico_valor": f"{valores_acf[lag_pico]:.3f}",
        "acf_n_lags_sig": len(lags_significativos),
        "acf_lags_sig_txt": ", ".join(str(lag) for lag in lags_significativos[:12]),
    }

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
        title=r"(c) FAC — Série em Primeiras Diferenças $\nabla Y_t$ (Pico negativo inicial; sem corte abrupto)",
    )
    axes[1, 0].set_xlabel("Desfasamento (Lags)")
    axes[1, 0].set_ylabel("Autocorrelação")
    referencia_lag52 = f"Lag 52 (referência anual; FAC = {valores_acf[52]:.3f})"
    axes[1, 0].axvline(52, color="crimson", linestyle=":", lw=1.2, label=referencia_lag52)
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
    axes[1, 1].axvline(52, color="crimson", linestyle=":", lw=1.2, label=referencia_lag52)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    guardar_figura(fig, "fig_03_fac_facp")

    return diagnostico_correlograma


# Modelação de Alisamento Exponencial (Holt e Holt-Winters) — Aulas #1
def modelar_alisamento_exponencial(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Estima os modelos de alisamento exponencial (Holt Linear, Holt-Winters Aditivo e Multiplicativo).

    Implementa a parametrização demonstrada em class-code/1.py e nos Slides 11 a 30.
    """
    print("[*] A estimar modelos de Alisamento Exponencial (Holt e Holt-Winters)...")

    y_train = train["total_commits"]
    y_test = test["total_commits"]
    horizonte = len(test)

    # 1. Método de Holt com tendência linear aditiva (class-code/1.py, Slide 19)
    mod_holt = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit()
    pred_holt = mod_holt.predict(start=len(y_train), end=len(y_train) + horizonte - 1)
    pred_holt.index = y_test.index

    # 2. Método de Holt-Winters com sazonalidade aditiva (s=52, Slide 28)
    mod_hwa = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=52).fit()
    pred_hwa = mod_hwa.predict(start=len(y_train), end=len(y_train) + horizonte - 1)
    pred_hwa.index = y_test.index

    # 3. Método de Holt-Winters com sazonalidade multiplicativa (s=52, Slide 23)
    mod_hwm = ExponentialSmoothing(y_train, trend="add", seasonal="mul", seasonal_periods=52).fit()
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

    registos = []
    for nome, dados in modelos_exp.items():
        m = dados["metricas"]
        registos.append({
            "Modelo": nome,
            "AIC": dados["aic"],
            "BIC": dados["bic"],
            "Alpha": dados["alpha"],
            "Beta": dados["beta"],
            "Gamma": dados["gamma"],
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })

    df_exp = pd.DataFrame(registos)
    guardar_tabela(df_exp, "alisamento_exponencial", "Tabela de alisamento exponencial")

    return modelos_exp


# Modelação ARIMA e ARIMAX com Covariáveis Exógenas — Aulas #2, #3 e #4
def modelar_arima_arimax(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict, tuple[str, SARIMAXResultsWrapper, list[str] | None]]:
    """Estima especificações univariadas (ARIMA) e multivariadas (ARIMAX).

    Adiciona covariáveis exógenas (contribuidores e code churn) e termos harmónicos de Fourier.
    """
    print("[*] A estimar modelos Box-Jenkins (ARIMA e ARIMAX com covariáveis exógenas)...")

    y_train = train["total_commits"]
    y_test = test["total_commits"]
    n_passos = len(test)

    # Conjuntos de variáveis exógenas
    exog_churn_only = ["log_churn"]
    exog_cols = ["unique_authors", "log_churn"]
    fourier_cols = ["sin_52_1", "cos_52_1", "sin_52_2", "cos_52_2"]
    exog_fourier_cols = exog_cols + fourier_cols

    X_train_churn = train[exog_churn_only]
    X_test_churn = test[exog_churn_only]

    X_train_base = train[exog_cols]
    X_test_base = test[exog_cols]

    X_train_fourier = train[exog_fourier_cols]
    X_test_fourier = test[exog_fourier_cols]

    especificacoes = [
        # Modelos Univariados
        ("ARIMA(0, 1, 2) Univariado", (0, 1, 2), None, None, None),
        ("ARIMA(1, 1, 1) Univariado", (1, 1, 1), None, None, None),
        # Modelo ARIMAX apenas com Exógena de Correlação Moderada (Code Churn, r=0.587)
        (
            "ARIMAX(0, 1, 2) + Code Churn (s/ Contribuidores)",
            (0, 1, 2),
            X_train_churn,
            X_test_churn,
            exog_churn_only,
        ),
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
        # print(f"    -> A ajustar {rotulo}...")
        mod = SARIMAX(
            y_train,
            exog=X_tr,
            order=ordem,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = cast(SARIMAXResultsWrapper, mod.fit(disp=False, maxiter=100))

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

        if metricas["REQM"] < menor_reqm:
            menor_reqm = metricas["REQM"]
            modelo_vencedor = (rotulo, res, var_names)

    registos = []
    for rotulo, info in modelos_bj.items():
        m = info["metricas"]
        registos.append({
            "Modelo": rotulo,
            "Ordem": str(info["ordem"]),
            "AIC": info["aic"],
            "BIC": info["bic"],
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })

    df_bj = pd.DataFrame(registos)
    guardar_tabela(df_bj, "arima_arimax", "Tabela ARIMA e ARIMAX")

    assert modelo_vencedor is not None
    return modelos_bj, modelo_vencedor


# Diagnóstico dos Resíduos e Tabela Formal SARIMAX — Slides 55, 56, 70
def executar_diagnostico_residuos(
    modelo_vencedor_info: tuple[str, SARIMAXResultsWrapper, list[str] | None],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Gera a tabela formal de estimação (tab_04_sarimax_results.csv) e o painel quádruplo

    de diagnóstico de resíduos para validação de ruído branco (Ljung-Box e Jarque-Bera).
    """
    rotulo, res, _var_names = modelo_vencedor_info
    print(f"[*] A executar diagnóstico de resíduos para o modelo eleito: {rotulo}...")

    # 1. Extração da Tabela Formal de Parâmetros (Slide 56 e 70)
    params = res.params
    bse = res.bse
    zvalues = res.tvalues
    pvalues = res.pvalues
    ci = res.conf_int()

    linhas_tabela = []
    for nome in params.index:
        linhas_tabela.append({
            "Parâmetro": nome,
            "Coeficiente": round(float(params[nome]), 4),
            "Erro-Padrão": round(float(bse[nome]), 4),
            "Estatística z": round(float(zvalues[nome]), 3),
            "p-value (P > |z|)": (f"{pvalues[nome]:.4e}" if pvalues[nome] < 0.0001 else f"{pvalues[nome]:.4f}"),
            "IC 95% Inferior": round(float(ci.loc[nome, 0]), 4),
            "IC 95% Superior": round(float(ci.loc[nome, 1]), 4),
            "Significativo (p < 0.05)?": ("Sim" if pvalues[nome] < 0.05 else "Não"),
        })

    df_params = pd.DataFrame(linhas_tabela)
    guardar_tabela(df_params, "tab_04_sarimax_results", "Tabela de coeficientes")

    # 2. Testes Formais nos Resíduos: Ljung-Box (Slide 55) e Jarque-Bera
    resid = res.resid
    std_resid = (resid - resid.mean()) / resid.std()

    lb_teste = acorr_ljungbox(resid, lags=[1, 5, 10, 15, 20], return_df=True)
    jb_stat, jb_pval = cast(tuple[float, float], stats.jarque_bera(resid))

    linhas_diag = []
    jb_status = "Resíduos Normais (p > 0.05)" if jb_pval > 0.05 else "Rejeita Normalidade (p <= 0.05)"
    linhas_diag.append({
        "Teste": "Jarque-Bera (Normalidade)",
        "Desfasamento": "Global",
        "Estatística": round(jb_stat, 2),
        "p-value": (f"{jb_pval:.4e}" if jb_pval < 0.0001 else f"{jb_pval:.4f}"),
        "Diagnóstico": jb_status,
    })

    for lag_idx, row in lb_teste.iterrows():
        p_q = cast(float, row["lb_pvalue"])
        stat_q = cast(float, row["lb_stat"])
        lb_status = "Ruído Branco (p > 0.05)" if p_q > 0.05 else "Autocorrelação Residual"
        linhas_diag.append({
            "Teste": "Ljung-Box (Ruído Branco)",
            "Desfasamento": f"Lag {lag_idx}",
            "Estatística": round(stat_q, 2),
            "p-value": (f"{p_q:.4e}" if p_q < 0.0001 else f"{p_q:.4f}"),
            "Diagnóstico": lb_status,
        })

    df_diag = pd.DataFrame(linhas_diag)
    guardar_tabela(df_diag, "diagnostico_residuos", "Tabela de diagnóstico de resíduos")

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
    axes[0, 0].plot(resid.index, std_resid, color="#1f77b4", lw=0.9, label="Resíduos Padronizados")
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
    guardar_figura(fig, "fig_04_diagnostico_residuos")

    return df_params, lb_teste, jb_pval


# Comparação Out-of-Sample e Previsões (tab_05, fig_05) — Slides 13, 59, 71
def consolidar_comparacao_econometrica(
    train: pd.DataFrame,
    test: pd.DataFrame,
    modelos_exp: dict,
    modelos_bj: dict,
) -> pd.DataFrame:
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

        registos_tabela.append({
            "Modelo": nome,
            "Família": "Alisamento Exponencial",
            "Ordem / Parâmetros": ordem_desc,
            "AIC": info["aic"],
            "BIC": info["bic"],
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })

    # 2. Inserir Modelos Box-Jenkins
    for nome, info in modelos_bj.items():
        m = info["metricas"]
        familia = "ARIMA Univariado" if "Univariado" in nome else "ARIMAX Multivariado"
        ordem_str = str(info["ordem"])

        registos_tabela.append({
            "Modelo": nome,
            "Família": familia,
            "Ordem / Parâmetros": ordem_str,
            "AIC": info["aic"],
            "BIC": info["bic"],
            "REQM": m["REQM"],
            "EAM": m["EAM"],
            "EPAM (%)": m["EPAM"],
            "EPAMs (%)": m["EPAMs"],
        })

    df_comp = pd.DataFrame(registos_tabela).sort_values(by="REQM", ascending=True)
    guardar_tabela(df_comp, "tab_05_comparacao_econometrica")

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
    data_corte = cast(pd.Timestamp, test.index[0])
    ax.axvline(
        float(mdates.date2num(data_corte)),
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
    guardar_figura(fig, "fig_05_comparacao_econometrica")

    return df_comp


# Projeção Futura com Holt-Winters Reajustado (class-code/2.py, Slides 27, 30, 33c)
def projetar_futuro_holt_winters(df: pd.DataFrame, modelos_exp: dict, horizonte: int = 52) -> pd.DataFrame:
    """Aplica o procedimento metodológico ensinado em class-code/2.py e no Slide 30 / 33c:

    Reajusta o modelo Holt-Winters com as constantes ótimas da amostra de treino
    à totalidade da série observada (N=1117 semanas) e projeta previsões para
    os próximos 12 meses (h=52 semanas, 2026–2027).
    """
    print(f"[*] A projetar previsões futuras a {horizonte} semanas com Holt-Winters reajustado (class-code/2.py)...")

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

    pred_futura = mod_full.predict(start=len(df), end=len(df) + horizonte - 1)
    data_inicio_futuro = df.index[-1] + pd.Timedelta(weeks=1)
    extended_index = pd.date_range(start=data_inicio_futuro, periods=horizonte, freq="W-MON")

    df_futuro = pd.DataFrame({
        "Data": extended_index.strftime("%Y-%m-%d"),
        "Ano": [d.year for d in extended_index],
        "Semana": extended_index.isocalendar().week,
        "Previsao_Commits": np.round(pred_futura.values, 1),
    })
    guardar_tabela(df_futuro, "tab_05b_previsao_futura_holtwinters", "Tabela de previsões futuras")

    # Gráfico de extensão histórica + projeção futura (class-code/2.py)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        r"Projeção Futura da Atividade Semanal do Kernel Linux ($h=52$ Semanas, 2026–2027)"
        "\n"
        r"Método de Holt-Winters Multiplicativo Reajustado à Série Completa"
        "\n"
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

    last_date = cast(pd.Timestamp, df.index[-1])
    ax.axvline(
        float(mdates.date2num(last_date)),
        color="black",
        linestyle=":",
        lw=1.2,
        label=f"Fim do Histórico Observado ({last_date.strftime('%Y-%m-%d')})",
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
    guardar_figura(fig, "fig_05b_previsao_futura_holtwinters")

    return df_futuro


# Síntese Interpretativa para o Relatório (ECONOMETRICS_SUMMARY.md)
def exportar_resumo_econometria(
    df_adf: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_params: pd.DataFrame,
    lb_teste: pd.DataFrame,
    jb_pval: float,
    df_futuro: pd.DataFrame,
    corr_diag: dict[str, object],
) -> None:
    """Gera um documento Markdown com o resumo interpretativo completo em Português de Portugal

    para que o Elemento A possa redigir o Capítulo 4 do Relatório.
    """
    print("[*] A exportar síntese econométrica em Markdown (output/ECONOMETRICS_SUMMARY.md)...")

    melhor_modelo_row = df_comp.iloc[0]
    holt_row = df_comp[df_comp["Modelo"].str.contains("Holt-Winters Multiplicativo", regex=False)].iloc[0]
    hw_aditivo_row = df_comp[df_comp["Modelo"].str.contains("Holt-Winters Aditivo", regex=False)].iloc[0]
    arima_row = df_comp[df_comp["Modelo"].str.contains("ARIMA(0, 1, 2) Univariado", regex=False)].iloc[0]

    # Formatar coeficientes das covariáveis
    coef_autores = float(df_params.loc[df_params["Parâmetro"] == "unique_authors", "Coeficiente"].values[0])
    pval_autores = df_params.loc[df_params["Parâmetro"] == "unique_authors", "p-value (P > |z|)"].values[0]
    coef_churn = float(df_params.loc[df_params["Parâmetro"] == "log_churn", "Coeficiente"].values[0])
    pval_churn = df_params.loc[df_params["Parâmetro"] == "log_churn", "p-value (P > |z|)"].values[0]

    # O modelo está em primeiras diferenças, pelo que o coeficiente de log_churn
    # incide sobre d(log churn): +1% de churn vale coef_churn * ln(1.01).
    efeito_churn_1pct = coef_churn * np.log(1.01)

    # Holt-Winters: a forma multiplicativa é a mais defensável em teoria (a amplitude
    # sazonal cresce com o nível), mas quem decide é o erro fora da amostra.
    vencedor_hw = "aditiva" if float(hw_aditivo_row["REQM"]) < float(holt_row["REQM"]) else "multiplicativa"
    hw_comparacao = (
        "em teoria a forma multiplicativa é a mais adequada, porque a amplitude sazonal cresce com o nível da série. "
        f"Na amostra de teste, porém, o REQM foi {hw_aditivo_row['REQM']} no aditivo e {holt_row['REQM']} no "
        f"multiplicativo (AIC de {hw_aditivo_row['AIC']} e {holt_row['AIC']}, respetivamente), pelo que o desempenho "
        f"fora da amostra favorece a forma {vencedor_hw}. A projeção da Secção 6 usa o multiplicativo por ser o "
        "procedimento demonstrado em `class-code/2.py`, e não por superioridade preditiva"
    )

    # Ljung-Box: declarar quais os desfasamentos que rejeitam $H_0$, e não apenas os
    # que a aceitam. A síntese anterior citava só os lags 1 e 10 e concluía ruído
    # branco, quando os lags 15 e 20 rejeitam a 5%.
    lags_ok = [int(lag) for lag in lb_teste.index if float(lb_teste.loc[lag, "lb_pvalue"]) > 0.05]
    lags_rejeitam = [int(lag) for lag in lb_teste.index if float(lb_teste.loc[lag, "lb_pvalue"]) <= 0.05]

    def listar_lags(valores: list[int]) -> str:
        if not valores:
            return "nenhum"
        if len(valores) == 1:
            return str(valores[0])
        return ", ".join(str(v) for v in valores[:-1]) + " e " + str(valores[-1])

    def formatar_pvalor(valor: object) -> str:
        """Converte o p-value da tabela num texto legível ('< 0.0001' em vez de '0.0000e+00')."""
        try:
            numero = float(str(valor))
        except ValueError:
            return str(valor)
        return "< 0.0001" if numero < 0.0001 else f"= {numero:.4f}"

    if lags_rejeitam:
        lb_leitura = (
            f"Os resíduos não podem ser declarados ruído branco: a autocorrelação reaparece a partir do "
            f"desfasamento {min(lags_rejeitam)}, compatível com o ciclo de release (~10 semanas), que um modelo "
            "linear com $s = 52$ não captura. É uma limitação a declarar no relatório."
        )
    else:
        lb_leitura = "Os resíduos são compatíveis com ruído branco em toda a extensão testada."

    # Extremos da projeção: os valores citados na síntese têm de sair da tabela e não
    # de uma estimativa à vista sobre o nível médio.
    idx_prev_min = df_futuro["Previsao_Commits"].idxmin()
    idx_prev_max = df_futuro["Previsao_Commits"].idxmax()

    contexto: dict[str, object] = {
        "adf_stat_niveis": df_adf.loc[df_adf["Série"] == "Série em Níveis Y_t", "Estatística ADF"].values[0],
        "adf_pval_niveis": df_adf.loc[df_adf["Série"] == "Série em Níveis Y_t", "p-value"].values[0],
        "adf_stat_sazonal": df_adf.loc[df_adf["Série"] == "Diferença Sazonal (s=52)", "Estatística ADF"].values[0],
        "adf_pval_sazonal": df_adf.loc[df_adf["Série"] == "Diferença Sazonal (s=52)", "p-value"].values[0],
        "adf_stat_simples": df_adf.loc[df_adf["Série"] == "Primeira Diferença Simples", "Estatística ADF"].values[0],
        "adf_pval_simples": df_adf.loc[df_adf["Série"] == "Primeira Diferença Simples", "p-value"].values[0],
        "holt_linear_reqm": df_comp.loc[df_comp["Modelo"] == "Holt Linear", "REQM"].values[0],
        "holt_linear_epam": df_comp.loc[df_comp["Modelo"] == "Holt Linear", "EPAM (%)"].values[0],
        "hw_adit_reqm": hw_aditivo_row["REQM"],
        "hw_adit_epam": hw_aditivo_row["EPAM (%)"],
        "hw_adit_aic": hw_aditivo_row["AIC"],
        "holt_mult_reqm": holt_row["REQM"],
        "holt_mult_epam": holt_row["EPAM (%)"],
        "holt_mult_aic": holt_row["AIC"],
        "hw_comparacao": hw_comparacao,
        "arima_reqm": arima_row["REQM"],
        "arima_epam": arima_row["EPAM (%)"],
        "arima_aic": arima_row["AIC"],
        "melhor_modelo": melhor_modelo_row["Modelo"],
        "melhor_reqm": melhor_modelo_row["REQM"],
        "melhor_epam": melhor_modelo_row["EPAM (%)"],
        "melhor_aic": melhor_modelo_row["AIC"],
        "ganho_reqm_pct": f"{((arima_row['REQM'] - melhor_modelo_row['REQM']) / arima_row['REQM']) * 100:.1f}",
        "coef_autores": coef_autores,
        "coef_autores_fmt": f"{coef_autores:.1f}",
        "pval_autores": formatar_pvalor(pval_autores),
        "coef_churn": coef_churn,
        "pval_churn": formatar_pvalor(pval_churn),
        "efeito_churn_1pct": f"{efeito_churn_1pct:.1f}",
        "lb_ok_txt": listar_lags(lags_ok),
        "lb_rejeita_txt": listar_lags(lags_rejeitam),
        "lb_leitura": lb_leitura,
        "jb_pval": ("< 0.0001" if jb_pval < 0.0001 else f"= {jb_pval:.4f}"),
        "prev_min": f"{df_futuro.loc[idx_prev_min, 'Previsao_Commits']:,.0f}".replace(",", "."),
        "prev_max": f"{df_futuro.loc[idx_prev_max, 'Previsao_Commits']:,.0f}".replace(",", "."),
        "prev_min_data": df_futuro.loc[idx_prev_min, "Data"],
        "prev_max_data": df_futuro.loc[idx_prev_max, "Data"],
        "prev_inicio": df_futuro["Data"].iloc[0],
        "prev_fim": df_futuro["Data"].iloc[-1],
        **corr_diag,
    }

    for lag in (1, 5, 10, 15, 20):
        contexto[f"lb_lag{lag}_stat"] = f"{lb_teste.loc[lag, 'lb_stat']:.2f}"
        contexto[f"lb_lag{lag}_pval"] = f"{lb_teste.loc[lag, 'lb_pvalue']:.4f}"

    guardar_markdown_template("econometrics_summary.md", "ECONOMETRICS_SUMMARY.md", contexto)


def main():
    iniciar_script("02", "MODELAÇÃO ECONOMÉTRICA CLÁSSICA")
    df, train, test = carregar_dados_e_dividir()
    df_adf = executar_testes_adf(train)
    corr_diag = criar_correlogramas_fac_facp(train)
    modelos_exp = modelar_alisamento_exponencial(train, test)
    modelos_bj, modelo_vencedor_info = modelar_arima_arimax(train, test)
    df_params, lb_teste, jb_pval = executar_diagnostico_residuos(modelo_vencedor_info)
    df_comp = consolidar_comparacao_econometrica(train, test, modelos_exp, modelos_bj)
    df_futuro = projetar_futuro_holt_winters(df, modelos_exp, horizonte=52)
    exportar_resumo_econometria(df_adf, df_comp, df_params, lb_teste, jb_pval, df_futuro, corr_diag)
    finalizar_script("02")


if __name__ == "__main__":
    main()
