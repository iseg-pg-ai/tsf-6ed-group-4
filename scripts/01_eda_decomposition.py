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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from utils import (
    PNG_DIR,
    RC_PARAMS,
    SVG_DIR,
    TABLES_DIR,
    carregar_serie_temporal,
    finalizar_script,
    guardar_figura,
    guardar_markdown_template,
    guardar_tabela,
    iniciar_script,
    resumo_serie,
)

# Configuração de Caminhos e Ambiente
PNG_DIR.mkdir(parents=True, exist_ok=True)
SVG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuração visual académica (definida em utils.py)
plt.rcParams.update(RC_PARAMS)


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e higieniza as séries temporais semanal e diária.

    Aplica as regras operacionais documentadas em AGENTS.md:
    1. Truncatura da cauda (descarte de df.iloc[:-1] correspondente à semana/dia incompleto de 2026-09-08 / 2026-09-14).
    2. Atribuição explícita de frequência ao DatetimeIndex ('W-MON' e 'D').
    """
    print("[*] A carregar dados brutos das séries temporais...")

    df_w = carregar_serie_temporal("kernel_weekly_ts.csv", "W-MON")
    df_d = carregar_serie_temporal("kernel_daily_ts.csv", "D")

    resumo_serie(df_w, "Série Semanal limpa", "observações")
    resumo_serie(df_d, "Série Diária limpa", "observações")

    return df_w, df_d


# Matrizes de Correlação entre Variáveis (fig_00_* e tab_00_*)
def gerar_matriz_correlacao(df_w: pd.DataFrame) -> None:
    """Gera as Matrizes de Correlação (Pearson e Spearman) e as Tabelas 0.

    Abrange as nove variáveis de código documentadas em
    dataset/dataset_fields_description.md (total_commits, non_merge_commits,
    merge_commits, unique_authors, files_changed, insertions, deletions,
    total_churn, net_code_growth) e as três variáveis quantitativas do ciclo de
    release (weeks_since_release, is_merge_window, rc_week).

    `last_release_tag` e `last_release_date` ficam de fora por não serem medidas:
    um identificador de tag e uma data correlacionam-se com tudo o que tem
    tendência temporal, sem informar sobre a intensidade da atividade.

    Pearson e Spearman são gravados em ficheiros separados porque a matriz passou
    a 12x12 e dois painéis no mesmo ficheiro deixariam de ser legíveis impressos.
    A amostra de Pearson exclui a semana inaugural de 2005 (import de 6.75M
    linhas); a de Spearman usa a amostra completa, robusta a outliers.
    """
    print("[*] A gerar Matrizes de Correlação (fig_00_* e tab_00_*)...")

    colunas_codigo = [
        "total_commits",
        "non_merge_commits",
        "merge_commits",
        "unique_authors",
        "files_changed",
        "insertions",
        "deletions",
        "total_churn",
        "net_code_growth",
    ]
    colunas_release = ["weeks_since_release", "is_merge_window", "rc_week"]
    colunas = colunas_codigo + colunas_release

    rotulos_curtos = [
        "total_commits (Y_t)",
        "non_merge_commits",
        "merge_commits",
        "unique_authors",
        "files_changed",
        "insertions",
        "deletions",
        "total_churn",
        "net_code_growth",
        "weeks_since_release",
        "is_merge_window",
        "rc_week",
    ]

    # 1. Cálculos de correlação
    # As colunas do ciclo de release são Int64 nullable: converter para float para
    # que os 10 NaN (semanas anteriores ao primeiro tag) contem como omissos e não
    # como zero. O `corr` usa pares completos, pelo que as restantes linhas não se
    # perdem.
    df_cols = df_w[colunas].astype("float64")
    assert isinstance(df_cols, pd.DataFrame)
    corr_pearson_all = df_cols.corr(method="pearson")
    # Excluir a 1ª semana (import inicial de 6.75M linhas) para não distorcer linearmente o code churn
    corr_pearson_clean = df_cols.iloc[1:].corr(method="pearson")
    # Correlação em primeiras diferenças (d=1, sem o ponto inicial)
    corr_diff_clean = df_cols.iloc[1:].diff().dropna().corr(method="pearson")
    # Correlação de postos de Spearman (amostra total)
    corr_spearman = df_cols.corr(method="spearman")
    n_semanas = len(df_cols)
    n_com_release = int(df_cols[colunas_release].notna().sum(axis=1).eq(len(colunas_release)).sum())

    # Diagnóstico qualitativo de colinearidade e adequação como variável exógena.
    # Os coeficientes não se repetem no texto: estão nas colunas numéricas desta
    # mesma tabela, onde são calculados, e um texto com números fixos desatualiza-se
    # em silêncio (era o caso de files_changed, com d1=0.762 escrito à mão contra
    # 0.761 calculado).
    diagnosticos = {
        "total_commits": "Variavel Dependente / Alvo (Y_t)",
        "non_merge_commits": "Identidade Contabilistica com Y_t — Excluir das exogenas",
        "merge_commits": "Componente direta da soma de Y_t — Evitar por endogeneidade",
        "unique_authors": "Correlacao muito elevada com Y_t — Risco de simultaneidade",
        "files_changed": "Correlacao elevada com Y_t — Proxy de escopo arquitetural",
        "insertions": "Correlacao moderada com Y_t mas colinear com total_churn",
        "deletions": "Correlacao moderada com Y_t — Indicador de refatoracao/limpeza",
        "total_churn": "Correlacao moderada com Y_t — Excelente exogena ortogonal",
        "net_code_growth": "Correlacao baixa/moderada com Y_t — Informacao complementar",
        "weeks_since_release": (
            "Contador do ciclo (0-15 semanas): relacao nao linear e sinal que inverte em 2015. "
            "Nao usar como regressor linear unico; codificar por fase (ver tab_02_outliers.csv)"
        ),
        "is_merge_window": (
            "Dummy (semanas 0-1), identica a weeks_since_release < 2: a correlacao linear agrupada "
            "e nula porque o efeito muda de sinal em 2015"
        ),
        "rc_week": (
            "Redundante com weeks_since_release (rc_week = max(0, weeks-1) em todas as linhas): "
            "escolher uma das duas codificacoes, nunca ambas"
        ),
    }

    registos_pearson = []
    for col in colunas:
        linha = {
            "Variavel": col,
            "r_Pearson_com_Yt_Sem_Outlier2005": round(corr_pearson_clean.loc[col, "total_commits"], 3),
            "r_Pearson_com_Yt_Primeiras_Diferencas": round(corr_diff_clean.loc[col, "total_commits"], 3),
            "r_Pearson_com_Yt_Bruto_Com_2005": round(corr_pearson_all.loc[col, "total_commits"], 3),
            "Avaliacao_Como_Exogena": diagnosticos[col],
        }
        for col_dest in colunas:
            linha[f"corr_{col_dest}"] = round(corr_pearson_clean.loc[col, col_dest], 3)
        registos_pearson.append(linha)

    guardar_tabela(
        pd.DataFrame(registos_pearson),
        "tab_00_matriz_correlacao_pearson",
        "Tabela de correlações de Pearson",
    )

    registos_spearman = []
    for col in colunas:
        linha = {"Variavel": col, "rho_Spearman_com_Yt": round(corr_spearman.loc[col, "total_commits"], 3)}
        for col_dest in colunas:
            linha[f"rho_{col_dest}"] = round(corr_spearman.loc[col, col_dest], 3)
        registos_spearman.append(linha)

    guardar_tabela(
        pd.DataFrame(registos_spearman),
        "tab_00_matriz_correlacao_spearman",
        "Tabela de correlações de Spearman",
    )

    # 2. Construção das duas Figuras 0 (Pearson e Spearman em ficheiros separados)
    def desenhar_matriz(
        matriz: pd.DataFrame,
        titulo: str,
        subtitulo: str,
        rotulo_barra: str,
        nome_ficheiro: str,
    ) -> None:
        """Desenha uma matriz 12x12 anotada, com o bloco do ciclo de release delimitado."""
        fig, ax = plt.subplots(figsize=(13, 11))
        fig.suptitle(
            f"{titulo}\n"
            "Investigação de Colinearidade, Identidades Contabilísticas e Seleção de Variáveis Exógenas\n"
            "ISEG — Time Series Forecasting — Análise Preliminar de Variáveis (dataset_fields_description.md)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        sns.heatmap(
            matriz.values,
            ax=ax,
            cmap="RdBu_r",
            vmin=-0.2,
            vmax=1.0,
            center=0.4,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            xticklabels=rotulos_curtos,
            yticklabels=rotulos_curtos,
            cbar_kws={"label": rotulo_barra, "shrink": 0.8},
            linewidths=0.6,
            linecolor="white",
        )
        # Delimitar o bloco do ciclo de release (últimas três linhas e colunas)
        ax.axhline(len(colunas_codigo), color="black", lw=1.4)
        ax.axvline(len(colunas_codigo), color="black", lw=1.4)
        ax.set_title(subtitulo, loc="left", fontweight="bold", fontsize=10.5)
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        guardar_figura(fig, nome_ficheiro)

    desenhar_matriz(
        corr_pearson_clean,
        "Figura 0a: Matriz de Correlação Linear de Pearson (r)",
        f"Regime normal de operação, sem o outlier inaugural de 2005-04-18 (N={n_semanas - 1} semanas)",
        "Coeficiente de Correlação de Pearson (r)",
        "fig_00_matriz_correlacao_pearson",
    )
    desenhar_matriz(
        corr_spearman,
        "Figura 0b: Matriz de Correlação de Postos de Spearman",
        f"Amostra completa (N={n_semanas} semanas, {n_com_release} com release de referência), robusta a outliers",
        r"Coeficiente de Correlação de Spearman ($\rho$)",
        "fig_00_matriz_correlacao_spearman",
    )


# Geração da Tabela de Estatísticas Descritivas (tab_01_variaveis.csv)
def gerar_estatisticas_descritivas(df_w: pd.DataFrame, df_d: pd.DataFrame) -> None:
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
        ("insertions", "Insertions (+)"),
        ("deletions", "Deletions (-)"),
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
    guardar_tabela(tabela_df, "tab_01_variaveis")


# Geração dos Cronogramas (fig_01_cronogramas_base.png)
def criar_cronogramas(df_w: pd.DataFrame, df_d: pd.DataFrame) -> None:
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
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO.weekday))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    guardar_figura(fig, "fig_01_cronogramas_base")


# Decomposição Aditiva e Multiplicativa (s=52) e Deteção de Outliers
FASE_MERGE = "Merge window (semanas 0-1)"
FASE_RC_INICIAL = "Estabilização inicial (rc1-rc5)"
FASE_RC_FINAL = "Estabilização final (rc6+)"
FASE_SEM_RELEASE = "Sem release de referência"


def classificar_fase_release(semanas: float | None, positivo: bool) -> tuple[str, str]:
    """Traduz a posição no ciclo de release em fase e contexto de engenharia.

    A fase é lida do calendário de releases (``weeks_since_release``) e não do mês
    do calendário. O mês não determina nada: a mesma semana de maio pode estar na
    merge window de um ciclo e na estabilização final do ciclo seguinte, pelo que
    uma heurística baseada no mês classifica corretamente por acidente, quando o faz.
    """
    # `pd.isna` é escalar aqui, mas o tipo devolvido é `bool | NDArray | NDFrame`;
    # o `bool(...)` torna a checagem explícita para o verificador de tipos e mantém
    # a tolerância a `pd.NA` caso a coluna passe a ser Int64 nullable.
    if semanas is None or bool(pd.isna(semanas)):
        return FASE_SEM_RELEASE, "Sem release de referência (início da série)"

    if semanas <= 1:
        fase = FASE_MERGE
        pico = "Pico na merge window (integração de subsistemas)"
        queda = "Queda na merge window (mainline congelada para integração)"
    elif semanas <= 5:
        fase = FASE_RC_INICIAL
        pico = "Pico na fase inicial de estabilização (rc1-rc5)"
        queda = "Queda na fase inicial de estabilização (rc1-rc5)"
    else:
        fase = FASE_RC_FINAL
        pico = "Pico na fase final de estabilização (imediata ao release)"
        queda = "Queda na fase final de estabilização (apenas correções)"

    return fase, pico if positivo else queda


def executar_decomposicoes_e_outliers(
    df_w: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
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
        positivo = z_score > 0
        tipo = "Pico Anómalo Positivo (+)" if positivo else "Queda Anómala Negativa (-)"

        # A fase vem do calendário de releases; o mês do calendário não a determina.
        semanas = df_w.at[d, "weeks_since_release"] if "weeks_since_release" in df_w.columns else None
        fase, contexto = classificar_fase_release(semanas, positivo)

        registos_outliers.append({
            "Data": d.strftime("%Y-%m-%d"),
            "Ano": d.year,
            "Semana": d.isocalendar().week,
            "Valor Observado (Y_t)": int(val_obs),
            "Tendência-Ciclo (T_t)": round(tend, 1),
            "Fator Sazonal (S_t)": round(sazon, 1),
            "Resíduo (R_t)": round(r, 2),
            "Z-Score (|Z| > 2)": round(z_score, 2),
            "Semanas Desde Release": semanas,
            "Fase do Ciclo de Release": fase,
            "Classificação": tipo,
            "Contexto de Engenharia Provável": contexto,
        })

    df_outliers = pd.DataFrame(registos_outliers).sort_values(by="Data", ascending=True)
    df_outliers["Semanas Desde Release"] = df_outliers["Semanas Desde Release"].astype("Int64")
    print(f"    -> Deteção de outliers concluída: {len(df_outliers)} semanas identificadas (|Z| > 2).")
    guardar_tabela(df_outliers, "tab_02_outliers")

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
    guardar_figura(fig, "fig_02_decomposicao_adit_mult")

    # Estatísticas de diagnóstico da decomposição
    # Semana do calendário em que o fator sazonal atinge os extremos: o mês não
    # identifica a fase do ciclo, mas a sazonalidade anual é de facto indexada
    # pela semana do ano.
    sazonal = decomp_mult.seasonal.dropna()
    fator_por_semana = sazonal.groupby(sazonal.index.isocalendar().week).mean()

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
        "Semana_Min_Fator_Mult": int(fator_por_semana.idxmin()),
        "Semana_Max_Fator_Mult": int(fator_por_semana.idxmax()),
    }

    return df_outliers, metricas_decomp


# Exportação das Notas de Síntese para a Equipa (EDA_SUMMARY.md)
def exportar_resumo_eda(metricas: dict, df_outliers: pd.DataFrame) -> None:
    """Gera um documento Markdown com o resumo interpretativo em Português de Portugal

    para que o Elemento A possa redigir os Capítulos 2 e 3 sem ambiguidades.
    """
    print("[*] A exportar síntese de resultados em Markdown (output/EDA_SUMMARY.md)...")

    picos_pos = df_outliers[df_outliers["Classificação"].str.contains("Positivo")]
    quedas_neg = df_outliers[df_outliers["Classificação"].str.contains("Negativa")]

    # Distribuição dos extremos pelas fases do ciclo de release, separada por regime.
    # A merge window inverte o sinal em 2015: na amostra completa os picos positivos
    # parecem concentrar-se nela, mas todos esses casos são anteriores a 2015. Uma
    # contagem agregada esconde dois processos distintos e não pode ser citada sozinha.
    fase_pos = picos_pos["Fase do Ciclo de Release"]
    fase_neg = quedas_neg["Fase do Ciclo de Release"]
    pos_antigo = picos_pos["Ano"] < 2015
    neg_antigo = quedas_neg["Ano"] < 2015

    contexto: dict[str, object] = {
        "n_observacoes": metricas["N_observacoes"],
        "min_fator_sazonal_mult": f"{metricas['Min_Fator_Sazonal_Mult'] * 100:.1f}",
        "max_fator_sazonal_mult": f"{metricas['Max_Fator_Sazonal_Mult'] * 100:.1f}",
        "semana_min_fator": metricas["Semana_Min_Fator_Mult"],
        "semana_max_fator": metricas["Semana_Max_Fator_Mult"],
        "resid_adit_std": metricas["Resid_Adit_Std"],
        "n_outliers_adit": metricas["N_Outliers_Adit"],
        "pct_outliers_adit": metricas["Pct_Outliers_Adit"],
        "n_picos_pos": len(picos_pos),
        "n_quedas_neg": len(quedas_neg),
        "picos_pos_merge": int((fase_pos == FASE_MERGE).sum()),
        "picos_pos_inicial": int((fase_pos == FASE_RC_INICIAL).sum()),
        "picos_pos_final": int((fase_pos == FASE_RC_FINAL).sum()),
        "picos_pos_merge_ant": int(((fase_pos == FASE_MERGE) & pos_antigo).sum()),
        "picos_pos_merge_dep": int(((fase_pos == FASE_MERGE) & ~pos_antigo).sum()),
        "picos_pos_final_ant": int(((fase_pos == FASE_RC_FINAL) & pos_antigo).sum()),
        "picos_pos_final_dep": int(((fase_pos == FASE_RC_FINAL) & ~pos_antigo).sum()),
        "quedas_merge": int((fase_neg == FASE_MERGE).sum()),
        "quedas_inicial": int((fase_neg == FASE_RC_INICIAL).sum()),
        "quedas_final": int((fase_neg == FASE_RC_FINAL).sum()),
        "quedas_ant": int(neg_antigo.sum()),
        "quedas_dep": int((~neg_antigo).sum()),
    }

    guardar_markdown_template("eda_summary.md", "EDA_SUMMARY.md", contexto)


def main():
    iniciar_script("01", "EDA E DECOMPOSIÇÃO TEMPORAL")
    df_w, df_d = carregar_dados()
    gerar_matriz_correlacao(df_w)
    gerar_estatisticas_descritivas(df_w, df_d)
    criar_cronogramas(df_w, df_d)
    df_outliers, metricas = executar_decomposicoes_e_outliers(df_w)
    exportar_resumo_eda(metricas, df_outliers)
    finalizar_script("01")


if __name__ == "__main__":
    main()
