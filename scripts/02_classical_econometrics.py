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
import statsmodels.api as sm
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

    # Fase do ciclo de release em três níveis, derivada de `weeks_since_release`.
    # A merge window (semanas 0-1) é a categoria de referência das especificações
    # ARIMAX, pelo que só as outras duas entram no modelo como dummies. As 10 semanas
    # anteriores ao primeiro tag ficam NaN (desconhecido) e nunca 0.
    # Justificação e evidência: scripts/analise_covariaveis_release/README.md
    semanas_release = df["weeks_since_release"].astype("float64")
    release_conhecido = semanas_release.notna()
    df["fase_merge"] = (semanas_release <= 1).where(release_conhecido).astype("float64")
    df["fase_rc_inicial"] = ((semanas_release >= 2) & (semanas_release <= 5)).where(release_conhecido).astype("float64")
    df["fase_rc_final"] = (semanas_release >= 6).where(release_conhecido).astype("float64")

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
    # `acf` devolve um ndarray porque `qstat` e `alpha` ficam por omissão, mas os stubs
    # declaram também hipóteses de tuplo: converte-se explicitamente para ndarray para
    # que indexar um desfasamento não seja lido como acesso a um campo de tuplo.
    valores_acf = np.asarray(acf(y_diff, nlags=60, fft=False), dtype=float)
    valores_acf_niveis = np.asarray(acf(y_niveis, nlags=60, fft=False), dtype=float)
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

    Adiciona covariáveis exógenas (contribuidores, code churn), termos harmónicos de
    Fourier e a fase do ciclo de release. Todas as especificações são estimadas na
    mesma amostra de treino (as semanas com fase de release conhecida), para que a
    escolha do vencedor por REQM compare modelos estimados nas mesmas observações.
    """
    print("[*] A estimar modelos Box-Jenkins (ARIMA e ARIMAX com covariáveis exógenas)...")

    # As variáveis do ciclo de release não existem antes do primeiro tag (10 semanas
    # de 2005). Excluí-las da amostra de estimação é o que permite comparar modelos
    # com e sem a fase sem lhes dar amostras de tamanhos diferentes.
    colunas_fase = ["fase_rc_inicial", "fase_rc_final"]
    amostra = train.dropna(subset=colunas_fase)

    y_train = amostra["total_commits"]
    y_test = test["total_commits"]
    n_passos = len(test)

    # Conjuntos de variáveis exógenas
    exog_churn_only = ["log_churn"]
    exog_cols = ["unique_authors", "log_churn"]
    exog_fase_cols = exog_cols + colunas_fase
    fourier_cols = ["sin_52_1", "cos_52_1", "sin_52_2", "cos_52_2"]
    exog_fourier_cols = exog_cols + fourier_cols

    X_train_churn = amostra[exog_churn_only]
    X_test_churn = test[exog_churn_only]

    X_train_base = amostra[exog_cols]
    X_test_base = test[exog_cols]

    X_train_fase = amostra[exog_fase_cols]
    X_test_fase = test[exog_fase_cols]

    X_train_fourier = amostra[exog_fourier_cols]
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
        # A fase do ciclo de release entra como duas dummies (merge window = referência).
        # É a única covariável conhecida antes de a semana começar; ver
        # scripts/analise_covariaveis_release/README.md para a evidência da escolha.
        (
            "ARIMAX(0, 1, 2) + Covariáveis + Fase de Release",
            (0, 1, 2),
            X_train_fase,
            X_test_fase,
            exog_fase_cols,
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
def comparar_leitura_coeficientes(train: pd.DataFrame, var_names: list[str]) -> dict[str, str]:
    """Verifica empiricamente sobre que grandeza os coeficientes do ARIMAX são estimados.

    Numa regressão com erros ARIMA em primeiras diferenças, os coeficientes são
    identificados pela co-movimentação das variações. Se incidissem sobre o nível,
    coincidiriam com os de uma regressão OLS em níveis; incidindo sobre a variação,
    coincidem com os de uma regressão OLS em primeiras diferenças. A síntese cita as
    duas estimativas para sustentar a leitura de curto prazo em vez de a afirmar.
    """
    if not {"unique_authors", "log_churn"} <= set(var_names):
        return {
            "ols_niveis_autores": "n/d",
            "ols_dif_autores": "n/d",
            "ols_niveis_churn": "n/d",
            "ols_dif_churn": "n/d",
        }

    amostra = train.dropna(subset=["fase_rc_inicial"])
    y_niveis = amostra["total_commits"].astype(float)
    x_niveis = amostra[var_names].astype(float)

    ols_niveis = sm.OLS(y_niveis, sm.add_constant(x_niveis)).fit()

    dif = amostra[["total_commits", *var_names]].astype(float).diff().dropna()
    ols_dif = sm.OLS(dif["total_commits"], sm.add_constant(dif[var_names])).fit()

    def fmt(valor: float) -> str:
        return f"{valor:.2f}".replace(".", ",")

    return {
        "ols_niveis_autores": fmt(float(ols_niveis.params["unique_authors"])),
        "ols_dif_autores": fmt(float(ols_dif.params["unique_authors"])),
        "ols_niveis_churn": fmt(float(ols_niveis.params["log_churn"])),
        "ols_dif_churn": fmt(float(ols_dif.params["log_churn"])),
    }


def executar_diagnostico_residuos(
    modelo_vencedor_info: tuple[str, SARIMAXResultsWrapper, list[str] | None],
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Gera a tabela formal de estimação (tab_04_sarimax_results.csv) e o painel quádruplo

    de diagnóstico de resíduos para validação de ruído branco (Ljung-Box e Jarque-Bera).
    Devolve também as estatísticas descritivas dos resíduos, para que a síntese escrita
    cite a assimetria e a curtose em vez de qualificar as caudas à vista.
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

    # 1b. Verificação da leitura dos coeficientes: OLS em níveis vs. OLS em diferenças
    ols_diag = comparar_leitura_coeficientes(train, var_names or [])

    # 2. Testes Formais nos Resíduos: Ljung-Box (Slide 55) e Jarque-Bera
    # A primeira observação é descartada. O modelo é estimado na forma de espaço de
    # estados e o filtro de Kalman arranca com inicialização difusa, pelo que o primeiro
    # resíduo absorve a incerteza inicial (-1.291 commits, ~8 desvios-padrão) e não um
    # choque da série. Mantê-lo inflaciona o Jarque-Bera (5.709 contra 3.913) e não é
    # prática defensável num teste de normalidade.
    resid_bruto = res.resid
    resid = resid_bruto.iloc[1:]
    std_resid = (resid - resid.mean()) / resid.std()

    lb_teste = acorr_ljungbox(resid, lags=[1, 5, 10, 15, 20], return_df=True)
    jb_stat, jb_pval = cast(tuple[float, float], stats.jarque_bera(resid))
    jb_stat_bruto, jb_pval_bruto = cast(tuple[float, float], stats.jarque_bera(resid_bruto))
    assimetria = float(stats.skew(resid))
    curtose_excesso = float(stats.kurtosis(resid))

    linhas_diag = [
        {
            "Teste": "Jarque-Bera (Normalidade)",
            "Desfasamento": f"Global (n={len(resid)}, sem resíduo inicial)",
            "Estatística": round(jb_stat, 2),
            "p-value": (f"{jb_pval:.4e}" if jb_pval < 0.0001 else f"{jb_pval:.4f}"),
            "Diagnóstico": "Resíduos Normais (p > 0.05)" if jb_pval > 0.05 else "Rejeita Normalidade (p <= 0.05)",
        },
        {
            "Teste": "Jarque-Bera (Normalidade)",
            "Desfasamento": f"Global (n={len(resid_bruto)}, com resíduo inicial)",
            "Estatística": round(jb_stat_bruto, 2),
            "p-value": (f"{jb_pval_bruto:.4e}" if jb_pval_bruto < 0.0001 else f"{jb_pval_bruto:.4f}"),
            "Diagnóstico": "Rejeita Normalidade (inflacionado pela inicialização)",
        },
        {
            "Teste": "Assimetria (Skewness)",
            "Desfasamento": f"Global (n={len(resid)})",
            "Estatística": round(assimetria, 4),
            "p-value": "",
            "Diagnóstico": "Cauda direita" if assimetria > 0.5 else "Aproximadamente simétrica",
        },
        {
            "Teste": "Curtose em Excesso (Fisher)",
            "Desfasamento": f"Global (n={len(resid)})",
            "Estatística": round(curtose_excesso, 4),
            "p-value": "",
            "Diagnóstico": "Caudas pesadas" if curtose_excesso > 1.0 else "Curtose compatível com a normal",
        },
    ]

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
    data_max = cast(pd.Timestamp, std_resid.idxmax())
    axes[0, 0].plot(resid.index, std_resid, color="#1f77b4", lw=0.9, label="Resíduos Padronizados")
    axes[0, 0].axhline(0, color="black", linestyle="-", lw=0.8, alpha=0.7)
    axes[0, 0].axhline(2, color="crimson", linestyle="--", lw=1.0, label=r"Limiares $\pm 2\sigma$")
    axes[0, 0].axhline(-2, color="crimson", linestyle="--", lw=1.0)
    axes[0, 0].set_title(
        f"(a) Resíduos Padronizados no Tempo (máx. {std_resid.max():+.1f}$\\sigma$ em "
        f"{data_max.strftime('%Y-%m-%d')})",
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
        f"(b) Distribuição dos Resíduos (JB = {jb_stat:.0f}; assimetria {assimetria:+.2f}; "
        f"curtose {curtose_excesso:+.1f})",
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
    # O título tem de dizer o que se vê. A FAC fica dentro da banda nos primeiros
    # desfasamentos, mas o Ljung-Box rejeita a partir de um deles; escrever "ausência de
    # correlação linear sistemática" contradizia a tabela de diagnóstico.
    lags_resid_rejeitam = [int(lag) for lag in lb_teste.index if float(lb_teste.loc[lag, "lb_pvalue"]) <= 0.05]
    titulo_fac_resid = (
        f"(d) FAC Residual (Autocorrelação significativa a partir do lag {min(lags_resid_rejeitam)})"
        if lags_resid_rejeitam
        else "(d) FAC Residual (Sem autocorrelação significativa nos lags testados)"
    )
    plot_acf(
        resid,
        ax=axes[1, 1],
        lags=30,
        alpha=0.05,
        title=titulo_fac_resid,
    )
    # O lag 0 vale sempre 1 e esmagaria a escala; o eixo é limitado à zona onde as
    # autocorrelações residuais (e a banda de confiança) são visíveis.
    axes[1, 1].set_ylim(-0.25, 0.25)
    axes[1, 1].set_xlabel("Desfasamento (Lags)")
    axes[1, 1].set_ylabel("Autocorrelação Residual")

    plt.tight_layout()
    guardar_figura(fig, "fig_04_diagnostico_residuos")

    diag_resid: dict[str, object] = {
        "jb_stat": float(jb_stat),
        "jb_pval": float(jb_pval),
        "jb_stat_bruto": float(jb_stat_bruto),
        "jb_pval_bruto": float(jb_pval_bruto),
        "assimetria": assimetria,
        "curtose_excesso": curtose_excesso,
        "n_resid": int(len(resid)),
        "dp_resid": float(resid.std()),
        "max_resid": float(resid.max()),
        "max_resid_data": data_max.strftime("%Y-%m-%d"),
        **ols_diag,
    }

    return df_params, lb_teste, diag_resid


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
        "Confronto: Observado vs. Holt-Winters vs. ARIMA vs. ARIMAX (destaque: vencedor por REQM)\n"
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
    # O vencedor por REQM é destacado; as restantes especificações ficam em segundo
    # plano. Sem ele, a figura que sustenta o Capítulo 4.4 não mostra o modelo que a
    # síntese elege.
    nome_vencedor = min(modelos_bj, key=lambda chave: modelos_bj[chave]["metricas"]["REQM"])
    cores_modelos = [
        (nome_vencedor, modelos_bj, "#9467bd", "-", 2.4),
        ("Holt-Winters Multiplicativo (s=52)", modelos_exp, "#d62728", "--", 1.1),
        ("ARIMA(0, 1, 2) Univariado", modelos_bj, "#ff7f0e", ":", 1.1),
        ("ARIMAX(0, 1, 2) + Covariáveis", modelos_bj, "#1f77b4", "-.", 1.3),
        ("SARIMAX(0, 1, 2) + Covariáveis + Fourier", modelos_bj, "#2ca02c", "-", 1.1),
    ]

    for nome_mod, dict_source, cor, estilo, espessura in cores_modelos:
        if nome_mod in dict_source:
            p = dict_source[nome_mod]["previsoes"]
            reqm_val = dict_source[nome_mod]["metricas"]["REQM"]
            epam_val = dict_source[nome_mod]["metricas"]["EPAM"]
            etiqueta = "Vencedor: " if nome_mod == nome_vencedor else ""
            rotulo_legenda = f"{etiqueta}{nome_mod} (REQM: {reqm_val:.1f}, EPAM: {epam_val:.1f}%)"
            ax.plot(
                test.index,
                p,
                color=cor,
                linestyle=estilo,
                lw=espessura,
                label=rotulo_legenda,
                zorder=4 if nome_mod == nome_vencedor else 2,
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


# Projeção de Cenários com o ARIMAX Vencedor (Slides 75-76 e 109-110)
def projetar_cenarios_arimax(
    df: pd.DataFrame,
    modelos_bj: dict,
    modelo_vencedor_info: tuple[str, SARIMAXResultsWrapper, list[str] | None],
    horizonte: int = 52,
) -> dict[str, object]:
    """Projeta cenários a 52 semanas com o modelo ARIMAX vencedor, reajustado à amostra completa.

    O modelo é reestimado em todas as semanas com fase de release conhecida (n = 1107), e
    não na amostra de treino: uma projeção para o futuro usa toda a informação disponível.
    As métricas fora da amostra continuam a ser as da estimação em treino, e a síntese
    distingue explicitamente as duas coisas.

    Como o ARIMAX é uma regressão com erros ARIMA e não um modelo de séries temporais
    multivariado, o caminho futuro das covariáveis tem de ser fixado pelo analista. São
    produzidos três cenários:

    - **Cenário 1** congela as covariáveis no valor da última semana observada. É a leitura
      literal de "X futuro = último X", e serve de diagnóstico: a última semana é um vale
      (197 autores, 415 commits) e o erro do modelo é integrado, pelo que o desvio não é
      corrigido ao longo do horizonte.
    - **Cenário 1b** congela as covariáveis na média das últimas 52 semanas. É o cenário
      estático defensável: descreve a atividade de um ano "típico", sem inventar calendário
      de releases futuro.
    - **Cenário 2** repete o caminho observado nas últimas 52 semanas (setembro de 2025 a
      setembro de 2026), preservando a cadência real de merge windows e fases de release.
      É o cenário "homólogo", o mais realista dos três.
    """
    rotulo, _res_treino, var_names = modelo_vencedor_info
    exog_cols = var_names or []
    if not exog_cols:
        raise ValueError("o modelo vencedor não tem covariáveis exógenas; não há cenários a projetar")

    print(f"[*] A projetar cenários a {horizonte} semanas com o ARIMAX vencedor ({rotulo})...")

    amostra = df.dropna(subset=["fase_rc_inicial"])
    ordem = cast(tuple[int, int, int], modelos_bj[rotulo]["ordem"])

    # Reajuste à amostra completa (mesma especificação, mais informação)
    mod_full = SARIMAX(
        amostra["total_commits"],
        exog=amostra[exog_cols],
        order=ordem,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res_full = cast(SARIMAXResultsWrapper, mod_full.fit(disp=False, maxiter=200))

    # Índice futuro: semanas W-MON a seguir à última observação
    ultima_data = cast(pd.Timestamp, amostra.index[-1])
    indice_futuro = pd.date_range(
        start=ultima_data + pd.Timedelta(weeks=1), periods=horizonte, freq="W-MON"
    )

    ultima_semana = amostra[exog_cols].iloc[-1].to_numpy(dtype=float)
    media_52 = amostra[exog_cols].iloc[-horizonte:].mean().to_numpy(dtype=float)
    homologo = amostra[exog_cols].iloc[-horizonte:].to_numpy(dtype=float)

    cenarios: dict[str, np.ndarray] = {
        "Cenario_1_Estatico_Ultima_Semana": np.tile(ultima_semana, (horizonte, 1)),
        "Cenario_1b_Estatico_Media_52s": np.tile(media_52, (horizonte, 1)),
        "Cenario_2_Replay_Ultimas_52s": homologo,
    }

    previsoes = {
        nome: res_full.forecast(
            steps=horizonte,
            exog=pd.DataFrame(matriz, columns=exog_cols, index=indice_futuro),
        )
        for nome, matriz in cenarios.items()
    }

    df_cenarios = pd.DataFrame({
        "Data": indice_futuro.strftime("%Y-%m-%d"),
        **{nome: np.round(prev.to_numpy(), 1) for nome, prev in previsoes.items()},
    })
    guardar_tabela(df_cenarios, "tab_09_previsao_futura_arimax_cenarios", "Tabela de cenários ARIMAX")

    # Figura: história recente + os três cenários
    n_amostra_txt = f"{len(amostra):,}".replace(",", ".")
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.suptitle(
        "Cenários de Projeção a 52 Semanas com o ARIMAX Vencedor (2026-2027)\n"
        f"Reajuste de {rotulo} à Amostra Completa ({n_amostra_txt} semanas com fase de release conhecida)\n"
        "ISEG — Time Series Forecasting — Aula #4 (Slides 75-76) e Aula #5 (Slides 109-110)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    historico = df.loc["2019-01-01":]
    ax.plot(historico.index, historico["total_commits"], color="black", lw=1.3, label="Observado (2019-2026)")

    estilos = {
        "Cenario_1_Estatico_Ultima_Semana": (
            "#d62728",
            ":",
            r"Cenário 1 — covariáveis congeladas na última semana ($X_{T}$: 197 autores, um vale)",
        ),
        "Cenario_1b_Estatico_Media_52s": (
            "#2ca02c",
            "--",
            r"Cenário 1b — covariáveis congeladas na média das últimas 52 semanas",
        ),
        "Cenario_2_Replay_Ultimas_52s": (
            "#1f77b4",
            "-",
            "Cenário 2 — covariáveis homólogas (repetição das últimas 52 semanas)",
        ),
    }
    for nome, (cor, estilo, etiqueta) in estilos.items():
        ax.plot(indice_futuro, previsoes[nome].to_numpy(), color=cor, linestyle=estilo, lw=1.8, label=etiqueta)

    ax.axvline(
        float(mdates.date2num(ultima_data)),
        color="darkblue",
        linestyle="--",
        lw=1.2,
        label=f"Fim do histórico observado ({ultima_data.strftime('%Y-%m-%d')})",
    )
    ax.axvspan(
        float(mdates.date2num(ultima_data)),
        float(mdates.date2num(cast(pd.Timestamp, indice_futuro[-1]))),
        color="grey",
        alpha=0.08,
    )

    ax.set_title(
        r"Trajetória Condicional: o ARIMAX não projeta as covariáveis, pelo que cada cenário é uma hipótese sobre $X_{t+h}$",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylabel("Total commits semanais")
    ax.set_xlabel("Data")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, framealpha=0.9, fontsize=8.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    guardar_figura(fig, "fig_09_previsao_futura_arimax_cenarios")

    # Síntese numérica dos cenários e das âncoras, para a Secção 7 da síntese escrita
    def milhares(valor: float) -> str:
        return f"{valor:,.0f}".replace(",", ".")

    def resumo(nome: str) -> dict[str, str]:
        prev = previsoes[nome]
        return {
            "min": milhares(float(prev.min())),
            "max": milhares(float(prev.max())),
            "mediana": milhares(float(prev.median())),
            "primeira": milhares(float(prev.iloc[0])),
            "ultima": milhares(float(prev.iloc[-1])),
        }

    obs_52 = amostra["total_commits"].iloc[-horizonte:]

    return {
        "cenario_modelo": rotulo,
        "cenario_n": milhares(float(len(amostra))),
        "cenario_ultima_data": ultima_data.strftime("%Y-%m-%d"),
        "cenario_obs_inicio": cast(pd.Timestamp, amostra.index[-horizonte]).strftime("%Y-%m-%d"),
        "cenario_inicio": cast(pd.Timestamp, indice_futuro[0]).strftime("%Y-%m-%d"),
        "cenario_fim": cast(pd.Timestamp, indice_futuro[-1]).strftime("%Y-%m-%d"),
        "cenario_ultimo_observado": milhares(float(amostra["total_commits"].iloc[-1])),
        "cenario_obs_media": milhares(float(obs_52.mean())),
        "cenario_obs_mediana": milhares(float(obs_52.median())),
        "cenario_obs_min": milhares(float(obs_52.min())),
        "cenario_obs_max": milhares(float(obs_52.max())),
        "cenario_ancora_autores": milhares(float(ultima_semana[0])),
        "cenario_media_autores": milhares(float(media_52[0])),
        "cenario_ancora_churn_linhas": milhares(float(np.expm1(ultima_semana[1]))),
        "cenario_media_churn_linhas": milhares(float(np.expm1(media_52[1]))),
        "cenario_media_rc_inicial": f"{float(media_52[2]) * 100:.0f}",
        "cenario_media_rc_final": f"{float(media_52[3]) * 100:.0f}",
        "cenario1_min": resumo("Cenario_1_Estatico_Ultima_Semana")["min"],
        "cenario1_max": resumo("Cenario_1_Estatico_Ultima_Semana")["max"],
        "cenario1b_min": resumo("Cenario_1b_Estatico_Media_52s")["min"],
        "cenario1b_max": resumo("Cenario_1b_Estatico_Media_52s")["max"],
        "cenario1b_mediana": resumo("Cenario_1b_Estatico_Media_52s")["mediana"],
        "cenario2_min": resumo("Cenario_2_Replay_Ultimas_52s")["min"],
        "cenario2_max": resumo("Cenario_2_Replay_Ultimas_52s")["max"],
        "cenario2_mediana": resumo("Cenario_2_Replay_Ultimas_52s")["mediana"],
        "cenario2_primeira": resumo("Cenario_2_Replay_Ultimas_52s")["primeira"],
        "cenario2_ultima": resumo("Cenario_2_Replay_Ultimas_52s")["ultima"],
    }


# Síntese Interpretativa para o Relatório (ECONOMETRICS_SUMMARY.md)
def exportar_resumo_econometria(
    df_adf: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_params: pd.DataFrame,
    lb_teste: pd.DataFrame,
    diag_resid: dict[str, object],
    df_futuro: pd.DataFrame,
    corr_diag: dict[str, object],
    cenario_diag: dict[str, object],
    train: pd.DataFrame,
    test: pd.DataFrame,
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

    def valor_param(nome: str) -> object:
        """Lê um parâmetro do modelo vencedor, devolvendo 'n/d' se ele não o incluir."""
        linhas = df_params.loc[df_params["Parâmetro"] == nome, "Coeficiente"]
        return linhas.values[0] if not linhas.empty else "n/d"

    def pval_param(nome: str) -> object:
        linhas = df_params.loc[df_params["Parâmetro"] == nome, "p-value (P > |z|)"]
        return linhas.values[0] if not linhas.empty else "n/d"

    def num(chave: str) -> float:
        """Lê uma estatística do dicionário de diagnóstico como float."""
        return float(cast(float, diag_resid[chave]))

    # Contributo incremental da fase de release: é a única forma honesta de justificar
    # a variável, porque o ganho global da Secção 4 é dominado por autores e churn.
    linha_sem_fase = df_comp[df_comp["Modelo"] == "ARIMAX(0, 1, 2) + Covariáveis"]
    if "Fase de Release" in str(melhor_modelo_row["Modelo"]) and not linha_sem_fase.empty:
        reqm_sem_fase = float(linha_sem_fase.iloc[0]["REQM"])
        reqm_com_fase = float(melhor_modelo_row["REQM"])
        ganho_fase = (
            f"o modelo com a fase reduz o REQM em {((reqm_sem_fase - reqm_com_fase) / reqm_sem_fase) * 100:.1f}% "
            f"face à especificação sem ela ({reqm_com_fase:.2f} contra {reqm_sem_fase:.2f}), na mesma amostra "
            "de treino"
        )
    else:
        ganho_fase = "a especificação com a fase do ciclo de release não superou o modelo sem ela em REQM"

    # Tamanhos de amostra: a comparação entre especificações só é válida porque todas
    # são estimadas nas mesmas observações (ver modelar_arima_arimax).
    n_treino = len(train)
    # `.notna().sum()` devolve um escalar cujo tipo os stubs não reduzem a int (e
    # `.count()` resolve para `int | Series` conforme o overload), pelo que a contagem
    # é feita em numpy, que é explicitamente numérica.
    n_treino_bj = int(np.count_nonzero(train["fase_rc_inicial"].notna().to_numpy()))
    n_teste = len(test)

    if lags_rejeitam:
        if "Fase de Release" in str(melhor_modelo_row["Modelo"]):
            lb_leitura = (
                f"Os resíduos não podem ser declarados ruído branco: resta autocorrelação a partir do "
                f"desfasamento {min(lags_rejeitam)}, mesmo com a fase do ciclo de release no modelo. A dummy de "
                "fase capta o nível médio de cada fase, não a cadência irregular dos ciclos (entre 8 e 15 semanas), "
                "pelo que a estrutura de médio prazo fica por explicar. É uma limitação a declarar no relatório."
            )
        else:
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
        "coef_fase_rc_inicial": valor_param("fase_rc_inicial"),
        "pval_fase_rc_inicial": formatar_pvalor(pval_param("fase_rc_inicial")),
        "coef_fase_rc_final": valor_param("fase_rc_final"),
        "pval_fase_rc_final": formatar_pvalor(pval_param("fase_rc_final")),
        "ganho_fase": ganho_fase,
        "n_treino": n_treino,
        "n_treino_bj": n_treino_bj,
        "n_teste": n_teste,
        "lb_ok_txt": listar_lags(lags_ok),
        "lb_rejeita_txt": listar_lags(lags_rejeitam),
        "lb_leitura": lb_leitura,
        "jb_pval": ("< 0.0001" if num("jb_pval") < 0.0001 else f"= {num('jb_pval'):.4f}"),
        "jb_stat": f"{num('jb_stat'):,.0f}".replace(",", "."),
        "jb_pval_bruto": ("< 0.0001" if num("jb_pval_bruto") < 0.0001 else f"= {num('jb_pval_bruto'):.4f}"),
        "jb_stat_bruto": f"{num('jb_stat_bruto'):,.0f}".replace(",", "."),
        "n_resid": cast(int, diag_resid["n_resid"]),
        "assimetria": f"{num('assimetria'):+.2f}",
        "curtose_excesso": f"{num('curtose_excesso'):.1f}",
        "dp_resid": f"{num('dp_resid'):.1f}".replace(".", ","),
        "max_resid": f"{num('max_resid'):,.0f}".replace(",", "."),
        "max_resid_data": diag_resid["max_resid_data"],
        "max_resid_sigmas": f"{num('max_resid') / num('dp_resid'):.1f}".replace(".", ","),
        "ols_niveis_autores": diag_resid["ols_niveis_autores"],
        "ols_dif_autores": diag_resid["ols_dif_autores"],
        "ols_niveis_churn": diag_resid["ols_niveis_churn"],
        "ols_dif_churn": diag_resid["ols_dif_churn"],
        "prev_min": f"{df_futuro.loc[idx_prev_min, 'Previsao_Commits']:,.0f}".replace(",", "."),
        "prev_max": f"{df_futuro.loc[idx_prev_max, 'Previsao_Commits']:,.0f}".replace(",", "."),
        "prev_min_data": df_futuro.loc[idx_prev_min, "Data"],
        "prev_max_data": df_futuro.loc[idx_prev_max, "Data"],
        "prev_inicio": df_futuro["Data"].iloc[0],
        "prev_fim": df_futuro["Data"].iloc[-1],
        **corr_diag,
        **cenario_diag,
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
    df_params, lb_teste, diag_resid = executar_diagnostico_residuos(modelo_vencedor_info, train)
    df_comp = consolidar_comparacao_econometrica(train, test, modelos_exp, modelos_bj)
    df_futuro = projetar_futuro_holt_winters(df, modelos_exp, horizonte=52)
    cenario_diag = projetar_cenarios_arimax(df, modelos_bj, modelo_vencedor_info, horizonte=52)
    exportar_resumo_econometria(
        df_adf, df_comp, df_params, lb_teste, diag_resid, df_futuro, corr_diag, cenario_diag, train, test
    )
    finalizar_script("02")


if __name__ == "__main__":
    main()
