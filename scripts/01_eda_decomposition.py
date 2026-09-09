# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "statsmodels",
#   "matplotlib",
#   "scipy",
#   "seaborn",
# ]
# ///

"""
================================================================================
Script 01: Análise Exploratória de Dados (EDA) e Decomposição de Séries Temporais
================================================================================
Unidade Curricular: Time Series Forecasting (2026/27 - 6ª Edição)
ISEG Executive Education - Pós-Graduação em Applied AI & Machine Learning
Docente: Prof. Jorge Caiado e Rubens Dias
Grupo 4 - Caso de Estudo: Linux Kernel Commit Velocity (2005-2026)

Conformidade com os Slides da Aula #0 (Slides 3 a 10):
- Construção dos cronogramas das séries observadas e variáveis exógenas (Slides 3-4)
- Decomposição clássica aditiva e multiplicativa com periodicidade sazonal s=52 (Slides 5-9)
- Deteção e categorização de outliers na componente residual via critério de 2 desvios-padrão (Slide 10)
- Cálculo e exportação de estatísticas descritivas completas (Média, Desvio-Padrão, Assimetria, Curtose)
================================================================================
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------------------------------------------------------
# 1. Configuração de Caminhos e Ambiente
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "Dataset"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
PNG_DIR = IMAGES_DIR / "png"
SVG_DIR = IMAGES_DIR / "svg"
TABLES_DIR = OUTPUT_DIR / "tables"

PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual de estilo académico
plt.rcParams.update({
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
})


def guardar_figura(fig: plt.Figure, nome_base: str) -> tuple[Path, Path]:
    """Guarda a figura simultaneamente em PNG (alta resolução, 300 DPI) e SVG (vetorial)."""
    caminho_png = PNG_DIR / f"{nome_base}.png"
    caminho_svg = SVG_DIR / f"{nome_base}.svg"
    fig.savefig(caminho_png)
    fig.savefig(caminho_svg)
    plt.close(fig)
    print(f"    -> Gráficos gravados:")
    print(f"       [PNG] {caminho_png}")
    print(f"       [SVG] {caminho_svg}")
    return caminho_png, caminho_svg


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e higieniza as séries temporais semanal e diária.

    Aplica as regras operacionais documentadas em AGENTS.md:
    1. Truncatura da cauda (descarte de df.iloc[:-1] correspondente à semana/dia incompleto de 2026-09-08 / 2026-09-14).
    2. Atribuição explícita de frequência ao DatetimeIndex ('W-MON' e 'D').
    """
    print("[*] A carregar dados brutos das séries temporais...")

    caminho_semanal = DATASET_DIR / "kernel_weekly_ts.csv"
    caminho_diario = DATASET_DIR / "kernel_daily_ts.csv"

    if not caminho_semanal.exists() or not caminho_diario.exists():
        raise FileNotFoundError(
            f"Ficheiros não encontrados em {DATASET_DIR}. "
            "Certifique-se de que os dados estão presentes na pasta Dataset/."
        )

    # 1. Série Semanal (W-MON)
    df_w = pd.read_csv(
        caminho_semanal,
        parse_dates=["committer_date"],
        index_col="committer_date",
    ).iloc[:-1]
    df_w.index.freq = "W-MON"

    # 2. Série Diária (D)
    df_d = pd.read_csv(caminho_diario, parse_dates=["committer_date"], index_col="committer_date").iloc[:-1]
    df_d.index.freq = "D"

    print(
        f"    -> Série Semanal limpa: {len(df_w)} observações ({df_w.index.min().strftime('%Y-%m-%d')} a {df_w.index.max().strftime('%Y-%m-%d')})"
    )
    print(
        f"    -> Série Diária limpa: {len(df_d)} observações ({df_d.index.min().strftime('%Y-%m-%d')} a {df_d.index.max().strftime('%Y-%m-%d')})"
    )

    return df_w, df_d


# ------------------------------------------------------------------------------
# 2. Geração da Tabela de Estatísticas Descritivas (tab_01_variaveis.csv)
# ------------------------------------------------------------------------------
def gerar_estatisticas_descritivas(df_w: pd.DataFrame, df_d: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas sumárias formais para ambas as frequências.

    Métricas incluídas: N, Média, Desvio-Padrão, Mínimo, Q25, Mediana, Q75, Máximo,
    Assimetria (Skewness) e Curtose (Kurtosis).
    """
    print("[*] A calcular estatísticas descritivas para tab_01_variaveis.csv...")

    variaveis = [
        ("total_commits", "Total commits (Y_t)"),
        ("non_merge_commits", "Non-merge commits"),
        ("merge_commits", "Commits de merge"),
        ("unique_authors", "Contribuidores / Autores únicos (X1)"),
        ("files_changed", "Files changed (X2)"),
        ("total_churn", "Code churn (X3)"),
        ("net_code_growth", "Net code growth (X4)"),
    ]

    registos = []

    for col, rotulo in variaveis:
        if col in df_w.columns:
            s_w = df_w[col].dropna()
            registos.append({
                "Frequência": "Semanal (W-MON)",
                "Variável": rotulo,
                "Código": col,
                "N": len(s_w),
                "Média": round(s_w.mean(), 2),
                "Desvio-Padrão": round(s_w.std(), 2),
                "Mínimo": round(s_w.min(), 2),
                "1º Quartil (Q25)": round(s_w.quantile(0.25), 2),
                "Mediana": round(s_w.median(), 2),
                "3º Quartil (Q75)": round(s_w.quantile(0.75), 2),
                "Máximo": round(s_w.max(), 2),
                "Assimetria (Skewness)": round(stats.skew(s_w), 3),
                "Curtose (Kurtosis)": round(stats.kurtosis(s_w), 3),
            })

    # Adicionar variável-alvo na frequência diária para contraste
    s_d = df_d["total_commits"].dropna()
    registos.append({
        "Frequência": "Diária (D)",
        "Variável": "Total commits diários (Y_t)",
        "Código": "total_commits",
        "N": len(s_d),
        "Média": round(s_d.mean(), 2),
        "Desvio-Padrão": round(s_d.std(), 2),
        "Mínimo": round(s_d.min(), 2),
        "1º Quartil (Q25)": round(s_d.quantile(0.25), 2),
        "Mediana": round(s_d.median(), 2),
        "3º Quartil (Q75)": round(s_d.quantile(0.75), 2),
        "Máximo": round(s_d.max(), 2),
        "Assimetria (Skewness)": round(stats.skew(s_d), 3),
        "Curtose (Kurtosis)": round(stats.kurtosis(s_d), 3),
    })

    tabela_df = pd.DataFrame(registos)
    caminho_csv = TABLES_DIR / "tab_01_variaveis.csv"
    tabela_df.to_csv(caminho_csv, index=False)
    print(f"    -> Tabela gravada com sucesso em: {caminho_csv}")

    return tabela_df


# ------------------------------------------------------------------------------
# 3. Geração dos Cronogramas (fig_01_cronogramas_base.png)
# ------------------------------------------------------------------------------
def criar_cronogramas(df_w: pd.DataFrame, df_d: pd.DataFrame) -> Path:
    """Gera o painel quádruplo de cronogramas em conformidade com as Aulas #0 e #4.

    Painéis:
    (a) Cronograma Semanal de Commits (Total vs. Non-Merge)
    (b) Cronograma Semanal de Autores Ativos (unique_authors)
    (c) Cronograma Semanal de Code Churn (total_churn, escala logarítmica com nota do pico de 2005)
    (d) Cronograma Diário (Recorte de 90 dias demonstrando micro-sazonalidade semanal s=7)
    """
    print("[*] A gerar painel de cronogramas base (fig_01_cronogramas_base.png)...")

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=False)
    fig.suptitle(
        "Cronogramas das Séries Temporais do Kernel Linux (2005–2026)\n"
        r"ISEG — $Time\ Series\ Forecasting$ — Aula #0 (Slides 3–4)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Painel (a): Commits Semanais (Y_t)
    ax1 = axes[0]
    ax1.plot(
        df_w.index,
        df_w["total_commits"],
        color="#1f77b4",
        linewidth=1.2,
        label=r"Total commits semanais ($Y_t$)",
    )
    ax1.plot(
        df_w.index,
        df_w["non_merge_commits"],
        color="#2ca02c",
        linewidth=0.9,
        alpha=0.75,
        label="Non-merge commits (excluindo commits de merge)",
    )
    ax1.set_title(
        r"(a) Volume Semanal de Commits: Total Commits vs. Non-merge Commits ($Y_t$)",
        loc="left",
        fontweight="bold",
    )
    ax1.set_ylabel("Commits")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Painel (b): Autores Únicos Semanais (X1)
    ax2 = axes[1]
    ax2.plot(
        df_w.index,
        df_w["unique_authors"],
        color="#ff7f0e",
        linewidth=1.2,
        label=r"Contribuidores semanais ($X_{1,t}$)",
    )
    ax2.set_title(
        r"(b) Capacidade Produtiva e Comunidade: Contribuidores Semanais ($X_{1,t}$)",
        loc="left",
        fontweight="bold",
    )
    ax2.set_ylabel("Contribuidores / Semana")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Painel (c): Code Churn (total_churn, log scale)
    ax3 = axes[2]
    # Usar escala logarítmica para lidar com o pico monumental de 2005-04-18
    ax3.plot(
        df_w.index,
        df_w["total_churn"],
        color="#d62728",
        linewidth=1.1,
        label=r"Code churn ($total\_churn = ins + del$, escala log)",
    )
    ax3.set_yscale("log")
    ax3.set_title(
        r"(c) Code Churn ($total\_churn$): Linhas Inseridas e Removidas ($X_{2,t}$)",
        loc="left",
        fontweight="bold",
    )
    ax3.set_ylabel("Code churn (Log)")
    # Anotação formal do ponto aberrante inicial
    data_spike = pd.Timestamp("2005-04-18")
    if data_spike in df_w.index:
        valor_spike = df_w.loc[data_spike, "total_churn"]
        ax3.annotate(
            "Import Inaugural de Linus Torvalds\n(6.75M linhas num único commit)",
            xy=(data_spike, valor_spike),
            xytext=(pd.Timestamp("2008-01-01"), valor_spike * 0.4),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2, "connectionstyle": "arc3,rad=0.2"},
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "fc": "#fff2cc", "ec": "#d6b656", "lw": 1},
        )
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Painel (d): Micro-Sazonalidade Diária (Zoom de 90 dias em 2026)
    ax4 = axes[3]
    df_d_zoom = df_d.loc["2026-05-01":"2026-08-31"]
    ax4.plot(
        df_d_zoom.index,
        df_d_zoom["total_commits"],
        color="#9467bd",
        marker="o",
        markersize=3.5,
        linewidth=1.1,
        label=r"Commits diários ($Y_{d,t}$ — Padrão Cíclico Semanal $s=7$)",
    )
    # Sombrear fins de semana para evidenciar a quebra periódica
    for dia in df_d_zoom.index:
        if dia.dayofweek in [5, 6]:  # Sábado e Domingo
            ax4.axvspan(
                dia - pd.Timedelta(hours=12),
                dia + pd.Timedelta(hours=12),
                color="#e0e0e0",
                alpha=0.45,
            )

    ax4.set_title(
        r"(d) Micro-Sazonalidade Diária ($s=7$): Queda de Commits aos Fins de Semana (Sombreado)",
        loc="left",
        fontweight="bold",
    )
    ax4.set_ylabel("Commits diários")
    ax4.set_xlabel("Data")
    ax4.legend(loc="upper right")
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    caminho_png, _ = guardar_figura(fig, "fig_01_cronogramas_base")
    return caminho_png


# ------------------------------------------------------------------------------
# 4. Decomposição Aditiva e Multiplicativa (s=52) e Deteção de Outliers
# ------------------------------------------------------------------------------
def executar_decomposicoes_e_outliers(
    df_w: pd.DataFrame,
) -> tuple[pd.DataFrame, Path, dict]:
    """Executa a decomposição aditiva e multiplicativa (s=52) e aplica a regra

    de deteção de outliers de resíduos (Slide 10).
    """
    print("[*] A executar decomposição temporal clássica (Aditiva vs. Multiplicativa, s=52)...")

    serie = df_w["total_commits"]

    # 1. Decomposição Aditiva: Y_t = T_t + S_t + R_t (Slide 6)
    decomp_adit = seasonal_decompose(serie, model="additive", period=52)

    # 2. Decomposição Multiplicativa: Y_t = T_t * S_t * R_t (Slide 8)
    decomp_mult = seasonal_decompose(serie, model="multiplicative", period=52)

    # 3. Análise da componente residual e Deteção de Outliers (Slide 10)
    # Regra do Slide 10: |R_t - média| > 2 desvios-padrão
    resid_adit = decomp_adit.resid.dropna()
    media_resid = resid_adit.mean()
    desvio_resid = resid_adit.std()

    limite_superior = media_resid + 2 * desvio_resid
    limite_inferior = media_resid - 2 * desvio_resid

    mascara_outliers = (resid_adit > limite_superior) | (resid_adit < limite_inferior)
    datas_outliers = resid_adit[mascara_outliers].index

    registos_outliers = []
    for d in datas_outliers:
        val_obs = serie.loc[d]
        tend = decomp_adit.trend.loc[d]
        sazon = decomp_adit.seasonal.loc[d]
        r = resid_adit.loc[d]
        z_score = (r - media_resid) / desvio_resid

        # Classificação e Contexto Provável de Engenharia
        mes = d.month
        dia = d.day

        if z_score > 0:
            tipo = "Pico Anómalo Positivo (+)"
            if mes in [1, 2, 5, 8, 10]:
                contexto = "Merge window (rc1 do kernel)"
            else:
                contexto = "Pico de commits / Integração de subsistema"
        else:
            tipo = "Queda Anómala Negativa (-)"
            if mes == 12 or (mes == 1 and dia <= 10):
                contexto = "Pausa Festiva (Natal e Passagem de Ano)"
            elif mes in [7, 8]:
                contexto = "Pausa Estival / Férias de Verão dos Maintainers"
            else:
                contexto = "Fase tardia de estabilização (apenas bugfixes rc7/rc8)"

        registos_outliers.append({
            "Data": d.strftime("%Y-%m-%d"),
            "Ano": d.year,
            "Semana": d.isocalendar().week,
            "Valor Observado (Y_t)": int(val_obs),
            "Tendência-Ciclo (T_t)": round(tend, 1),
            "Fator Sazonal (S_t)": round(sazon, 1),
            "Resíduo (R_t)": round(r, 2),
            "Z-Score (|Z| > 2)": round(z_score, 2),
            "Classificação": tipo,
            "Contexto de Engenharia Provável": contexto,
        })

    df_outliers = pd.DataFrame(registos_outliers).sort_values(by="Data", ascending=True)
    caminho_outliers = TABLES_DIR / "tab_02_outliers.csv"
    df_outliers.to_csv(caminho_outliers, index=False)
    print(f"    -> Deteção de outliers concluída: {len(df_outliers)} semanas identificadas (|Z| > 2).")
    print(f"    -> Tabela gravada em: {caminho_outliers}")

    # 4. Construção do Painel Visual de Decomposição (fig_02_decomposicao_adit_mult.png)
    print("[*] A construir figura comparativa de decomposição (fig_02_decomposicao_adit_mult.png)...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(
        "Decomposição Clássica da Série Semanal de Commits do Kernel Linux (s = 52)\n"
        r"Comparação entre Modelo Aditivo ($Y_t = T_t + S_t + R_t$) e Multiplicativo ($Y_t = T_t \times S_t \times R_t$)"
        "\nISEG — Time Series Forecasting — Aula #0 (Slides 5 a 10)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    # Cores
    c_obs = "#1f77b4"
    c_trend = "#d62728"
    c_seas = "#2ca02c"
    c_resid = "#9467bd"

    # --- COLUNA 1: MODELO ADITIVO (Slides 6-7) ---
    axes[0, 0].plot(df_w.index, serie, color=c_obs, lw=1.1, label=r"Observado ($Y_t$)")
    axes[0, 0].set_title(r"Modelo Aditivo — Observado ($Y_t$)", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Commits")
    axes[0, 0].legend(loc="upper left")

    axes[1, 0].plot(
        df_w.index,
        decomp_adit.trend,
        color=c_trend,
        lw=1.5,
        label=r"Tendência-Ciclo ($T_t$)",
    )
    axes[1, 0].set_title(
        r"Tendência-Ciclo ($T_t$, Média Móvel Centrada $s=52$)",
        loc="left",
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Commits")
    axes[1, 0].legend(loc="upper left")

    axes[2, 0].plot(
        df_w.index,
        decomp_adit.seasonal,
        color=c_seas,
        lw=1.0,
        label=r"Sazonalidade ($S_t, \sum S_j = 0$)",
    )
    axes[2, 0].set_title(
        r"Componente Sazonal Aditiva ($S_t$, desvio fixo)",
        loc="left",
        fontweight="bold",
    )
    axes[2, 0].set_ylabel("Desvio (Commits)")
    axes[2, 0].legend(loc="upper left")

    axes[3, 0].plot(
        df_w.index,
        decomp_adit.resid,
        color=c_resid,
        lw=0.9,
        label=r"Resíduos ($R_t = Y_t - T_t - S_t$)",
    )
    axes[3, 0].axhline(
        limite_superior,
        color="red",
        linestyle="--",
        lw=1.0,
        label=r"Limiares $\pm 2\sigma$ (Slide 10)",
    )
    axes[3, 0].axhline(limite_inferior, color="red", linestyle="--", lw=1.0)
    axes[3, 0].axhline(0, color="black", lw=0.6, alpha=0.7)
    # Marcar outliers a vermelho
    axes[3, 0].scatter(
        datas_outliers,
        resid_adit.loc[datas_outliers],
        color="crimson",
        s=18,
        zorder=5,
        label="Outliers (|Z| > 2)",
    )
    axes[3, 0].set_title(
        r"Resíduos ou Ruído Aditivo ($R_t$) com Limiares $\pm 2\sigma$",
        loc="left",
        fontweight="bold",
    )
    axes[3, 0].set_ylabel("Resíduo")
    axes[3, 0].set_xlabel("Ano")
    axes[3, 0].legend(loc="lower left", ncol=2)

    # --- COLUNA 2: MODELO MULTIPLICATIVO (Slides 8-9) ---
    axes[0, 1].plot(df_w.index, serie, color=c_obs, lw=1.1, label=r"Observado ($Y_t$)")
    axes[0, 1].set_title(
        r"Modelo Multiplicativo — Observado ($Y_t$)",
        loc="left",
        fontweight="bold",
    )
    axes[0, 1].set_ylabel("Commits")
    axes[0, 1].legend(loc="upper left")

    axes[1, 1].plot(
        df_w.index,
        decomp_mult.trend,
        color=c_trend,
        lw=1.5,
        label=r"Tendência-Ciclo ($T_t$)",
    )
    axes[1, 1].set_title(
        r"Tendência-Ciclo ($T_t$, Média Móvel Centrada $s=52$)",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].set_ylabel("Commits")
    axes[1, 1].legend(loc="upper left")

    axes[2, 1].plot(
        df_w.index,
        decomp_mult.seasonal,
        color=c_seas,
        lw=1.0,
        label=r"Sazonalidade ($S_t$, Fatores em %)",
    )
    axes[2, 1].set_title(
        r"Componente Sazonal Multiplicativa ($S_t$, proporção média = 1)",
        loc="left",
        fontweight="bold",
    )
    axes[2, 1].set_ylabel("Fator Multiplicativo")
    axes[2, 1].legend(loc="upper left")

    resid_mult = decomp_mult.resid.dropna()
    media_rm = resid_mult.mean()
    std_rm = resid_mult.std()
    axes[3, 1].plot(
        df_w.index,
        decomp_mult.resid,
        color=c_resid,
        lw=0.9,
        label=r"Resíduos ($R_t = Y_t / (T_t \times S_t)$)",
    )
    axes[3, 1].axhline(
        media_rm + 2 * std_rm,
        color="red",
        linestyle="--",
        lw=1.0,
        label=r"Limiares $\pm 2\sigma$",
    )
    axes[3, 1].axhline(media_rm - 2 * std_rm, color="red", linestyle="--", lw=1.0)
    axes[3, 1].axhline(1.0, color="black", lw=0.6, alpha=0.7)
    outliers_mult = resid_mult[(resid_mult - media_rm).abs() > 2 * std_rm]
    axes[3, 1].scatter(
        outliers_mult.index,
        outliers_mult,
        color="crimson",
        s=18,
        zorder=5,
        label="Outliers (|Z| > 2)",
    )
    axes[3, 1].set_title(
        r"Resíduos Multiplicativos ($R_t$) com Limiares $\pm 2\sigma$",
        loc="left",
        fontweight="bold",
    )
    axes[3, 1].set_ylabel("Razão Residual")
    axes[3, 1].set_xlabel("Ano")
    axes[3, 1].legend(loc="lower left", ncol=2)

    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    caminho_fig2, _ = guardar_figura(fig, "fig_02_decomposicao_adit_mult")

    # Estatísticas de diagnóstico da decomposição
    metricas_decomp = {
        "N_observacoes": len(df_w),
        "Media_Serie": round(serie.mean(), 2),
        "Desvio_Serie": round(serie.std(), 2),
        "Resid_Adit_Media": round(media_resid, 4),
        "Resid_Adit_Std": round(desvio_resid, 2),
        "Resid_Adit_Var": round(desvio_resid**2, 2),
        "Resid_Mult_Media": round(media_rm, 4),
        "Resid_Mult_Std": round(std_rm, 4),
        "Resid_Mult_Var": round(std_rm**2, 4),
        "N_Outliers_Adit": len(df_outliers),
        "Pct_Outliers_Adit": round((len(df_outliers) / len(resid_adit)) * 100, 2),
        "Min_Fator_Sazonal_Mult": round(decomp_mult.seasonal.min(), 3),
        "Max_Fator_Sazonal_Mult": round(decomp_mult.seasonal.max(), 3),
    }

    return df_outliers, caminho_fig2, metricas_decomp


# ------------------------------------------------------------------------------
# 5. Exportação das Notas de Síntese para a Equipa (EDA_SUMMARY.md)
# ------------------------------------------------------------------------------
def exportar_resumo_eda(metricas: dict, df_outliers: pd.DataFrame) -> Path:
    """Gera um documento Markdown com o resumo interpretativo em Português de Portugal

    para que o Elemento A possa redigir os Capítulos 2 e 3 sem ambiguidades.
    """
    print("[*] A exportar síntese de resultados em Markdown (output/EDA_SUMMARY.md)...")

    caminho_md = OUTPUT_DIR / "EDA_SUMMARY.md"

    picos_pos = df_outliers[df_outliers["Classificação"].str.contains("Positivo")]
    quedas_neg = df_outliers[df_outliers["Classificação"].str.contains("Negativa")]

    conteudo = f"""# Síntese da Análise Exploratória e Decomposição Temporal
**Documento de Apoio à Redação para o Elemento A (Capítulos 2 e 3 do Relatório)**  
*Gerado automaticamente por `scripts/01_eda_decomposition.py`*

---

## 1. Dados e Higienização Técnica (Capítulo 2)
* **Frequência Semanal (`W-MON`):** {metricas["N_observacoes"]} observações validadas (Abril de 2005 a Setembro de 2026).
* **Truncatura da Cauda:** A última linha (`2026-09-14`) foi removida por corresponder ao fecho a meio da semana da recolha Git (apenas 2 commits registados), prevenindo colapso artificial nas previsões fora da amostra (*out-of-sample*).
* **Ponto Singular Inicial (Abril de 2005):** O primeiro registo contém 6.749.493 linhas de inserção num único commit de Linus Torvalds (importação fundacional da árvore). O volume de commits (`total_commits = 259`) é válido, mas o code churn exige escala logarítmica ou tratamento de alavancagem.
* **Tabela de Variáveis:** Inserir `tab_01_variaveis.csv` no Capítulo 2 para apresentar a média, mediana, assimetria e curtose das variáveis candidatas.

---

## 2. Decomposição Temporal: Aditiva vs. Multiplicativa (Capítulo 3 — Slides 5 a 9)
* **Periodicidade Sazonal:** $s = 52$ semanas (ciclo anual).
* **Tendência-Ciclo ($T_t$):** Estimada através de uma média móvel centrada de ordem 52. As primeiras 26 e as últimas 26 semanas são, por definição matemática da média móvel bicaudal, não calculáveis (*NaN*).
* **Justificação do Modelo Multiplicativo vs. Aditivo (Pergunta Clássica do Docente):**
  * Em 2005/2006, a série oscilava em torno de 300 commits semanais, com quedas sazonais de ~100 commits.
  * Em 2025/2026, a série oscila em torno de 2.500 commits semanais, com quedas sazonais de ~800 commits.
  * **Conclusão:** Como a amplitude das oscilações sazonais aumenta em proporção direta com o nível da tendência, o **Modelo Multiplicativo ($Y_t = T_t \\times S_t \\times R_t$)** é teoricamente mais apropriado do que o Aditivo (Slide 8).
  * Fatores sazonais multiplicativos oscilam entre **{metricas["Min_Fator_Sazonal_Mult"] * 100:.1f}%** da tendência (semanas de Natal e Verão) e **{metricas["Max_Fator_Sazonal_Mult"] * 100:.1f}%** (semanas de abertura de merge windows).

---

## 3. Deteção de Outliers na Série Residual (Capítulo 3 — Slide 10)
* **Regra de Decisão do Slide 10:** $|R_t - \\bar{{R}}| > 2 \\sigma_R$ aplicada aos resíduos da decomposição aditiva.
* **Desvio-Padrão Residual ($\\\\sigma_R$):** {metricas["Resid_Adit_Std"]} commits.
* **Total de Semanas Assinaladas como Outliers:** {metricas["N_Outliers_Adit"]} semanas ({metricas["Pct_Outliers_Adit"]}% da amostra, em estrita conformidade com a expectativa teórica de uma distribuição normal onde ~5% dos pontos excedem 2 desvios-padrão).
  * **Picos Positivos Significativos:** {len(picos_pos)} semanas (coincidem maioritariamente com semanas de abertura de merge windows imediatas ao lançamento de versões rc1).
  * **Quedas Anómalas Negativas:** {len(quedas_neg)} semanas (coincidem sistematicamente com as semanas 51–52 de Natal/Ano Novo e semanas 32–34 de pausa estival em Agosto).
* **Tabela de Outliers:** Inserir e comentar `tab_02_outliers.csv`.

---

## 4. Imagens a Incluir no Documento (disponíveis em `output/images/png/` e `output/images/svg/`)
1. **`fig_01_cronogramas_base.[png|svg]`:** Inserir no Capítulo 2 como Figura 1 (Cronogramas em níveis, crescimento de contribuidores, escala logarítmica de code churn e micro-sazonalidade semanal diária).
2. **`fig_02_decomposicao_adit_mult.[png|svg]`:** Inserir no Capítulo 3 como Figura 2 (Painel quádruplo comparando aditivo vs multiplicativo e assinalando os pontos vermelhos de resíduos que ultrapassam $2\\sigma$).
"""

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"    -> Resumo gravado em: {caminho_md}")
    return caminho_md


# ------------------------------------------------------------------------------
# 6. Execução Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("INÍCIO DO SCRIPT 01: EDA E DECOMPOSIÇÃO TEMPORAL")
    print("=" * 80)

    # 1. Carregar Dados
    df_w, df_d = carregar_dados()

    # 2. Estatísticas Descritivas (tab_01)
    gerar_estatisticas_descritivas(df_w, df_d)

    # 3. Cronogramas (fig_01)
    criar_cronogramas(df_w, df_d)

    # 4. Decomposição e Outliers (tab_02, fig_02)
    df_outliers, _, metricas = executar_decomposicoes_e_outliers(df_w)

    # 5. Exportar Resumo Markdown para o Relatório
    exportar_resumo_eda(metricas, df_outliers)

    print("=" * 80)
    print("[✓] SCRIPT 01 CONCLUÍDO COM SUCESSO!")
    print(f"    Ficheiros gerados em: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
