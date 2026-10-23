# Síntese da Análise Exploratória e Decomposição Temporal
**Documento de Apoio à Redação para o Elemento A (Capítulos 2 e 3 do Relatório)**  
*Gerado automaticamente por `scripts/01_eda_decomposition.py`*

---

## 1. Dados, Higienização e Matriz de Correlação (Capítulo 2)
* **Frequência Semanal (`W-MON`):** {{ n_observacoes }} observações validadas (Abril de 2005 a Setembro de 2026).
* **Truncatura da Cauda:** A última linha (`2026-09-14`) foi removida por corresponder ao fecho a meio da semana da recolha Git (apenas 2 commits registados), prevenindo colapso artificial nas previsões fora da amostra (*out-of-sample*).
* **Ponto Singular Inicial (Abril de 2005):** O primeiro registo contém 6.749.493 linhas de inserção num único commit de Linus Torvalds (importação fundacional da árvore). O volume de commits (`total_commits = 259`) é válido, mas o code churn exige escala logarítmica ou tratamento de alavancagem.
* **Análise da Matriz de Correlação (`fig_00_matriz_correlacao_pearson.[png|svg]`, `fig_00_matriz_correlacao_spearman.[png|svg]`, `tab_00_matriz_correlacao_pearson.csv` e `tab_00_matriz_correlacao_spearman.csv`):**
  * **Identidades Contabilísticas a Excluir das Exógenas:** `non_merge_commits` apresenta correlação de $r = 0.998$ com `total_commits` ($\text{total\_commits} = \text{non\_merge\_commits} + \text{merge\_commits}$). Do mesmo modo, `insertions` tem correlação de $r = 0.939$ com `total_churn` ($\text{ins} + \text{del}$). Incluir estas parcelas em simultâneo geraria multicolinearidade severa ou tautologia contabilística.
  * **Colinearidade vs. Ortogonalidade nas Candidatas Exógenas:** `unique_authors` apresenta uma correlação forte com `total_commits` ($r = 0.934$ em níveis; $r = 0.898$ em primeiras diferenças), refletindo que o volume de commits escala diretamente com o número de contribuidores ativos na semana. Em contrapartida, `total_churn` ($r = 0.587$), `insertions` ($r = 0.560$), `deletions` ($r = 0.431$) e `net_code_growth` ($r = 0.379$) apresentam correlações moderadas com `total_commits` e baixa colinearidade com `unique_authors` ($r = 0.452$), fornecendo informação estrutural genuinamente ortogonal sobre a dimensão e complexidade das alterações.
  * **Bloco do Ciclo de Release (últimas três linhas, delimitado a preto nas figuras):** `weeks_since_release`, `is_merge_window` e `rc_week` entram aqui pela primeira vez. As correlações agrupadas com `total_commits` são fracas e próximas de zero porque o efeito muda de sinal em 2015 (Secção 3 e `tab_02_outliers.csv`); a leitura correta é por era, não pelo coeficiente agrupado. `rc_week` é exatamente `max(0, weeks_since_release - 1)`, pelo que as duas são a mesma variável deslocada e nunca podem entrar em simultâneo num modelo. `last_release_tag` e `last_release_date` ficam de fora por serem identificadores e não medidas.
* **Tabelas de Variáveis:** Inserir `tab_00_matriz_correlacao_pearson.csv`, `tab_00_matriz_correlacao_spearman.csv` e `tab_01_variaveis.csv` no Capítulo 2.

---

## 2. Decomposição Temporal: Aditiva vs. Multiplicativa (Capítulo 3 — Slides 5 a 9)
* **Periodicidade Sazonal:** $s = 52$ semanas (ciclo anual).
* **Tendência-Ciclo ($T_t$):** Estimada através de uma média móvel centrada de ordem 52. As primeiras 26 e as últimas 26 semanas são, por definição matemática da média móvel bicaudal, não calculáveis (*NaN*).
* **Justificação do Modelo Multiplicativo vs. Aditivo (Pergunta Clássica do Docente):**
  * Em 2005/2006, a série oscilava em torno de 300 commits semanais, com quedas sazonais de ~100 commits.
  * Em 2025/2026, a série oscila em torno de 2.500 commits semanais, com quedas sazonais de ~800 commits.
  * **Conclusão:** Como a amplitude das oscilações sazonais aumenta em proporção direta com o nível da tendência, o **Modelo Multiplicativo ($Y_t = T_t \times S_t \times R_t$)** é teoricamente mais apropriado do que o Aditivo (Slide 8).
  * Fatores sazonais multiplicativos oscilam entre **{{ min_fator_sazonal_mult }}%** da tendência (mínimo na semana {{ semana_min_fator }} do calendário) e **{{ max_fator_sazonal_mult }}%** (máximo na semana {{ semana_max_fator }}): o vale coincide com o arranque do ano e com as férias de Verão, e o pico com o início do outono. Sendo indexado pela semana do ano, o componente com $s = 52$ não absorve o ciclo de release do kernel, cuja periodicidade é variável e ronda as dez semanas.

---

## 3. Deteção de Outliers na Série Residual (Capítulo 3 — Slide 10)
* **Regra de Decisão do Slide 10:** $|R_t - \bar{R}| > 2 \sigma_R$ aplicada aos resíduos da decomposição aditiva.
* **Desvio-Padrão Residual ($\sigma_R$):** {{ resid_adit_std }} commits.
* **Total de Semanas Assinaladas como Outliers:** {{ n_outliers_adit }} semanas ({{ pct_outliers_adit }}% da amostra, em estrita conformidade com a expectativa teórica de uma distribuição normal onde ~5% dos pontos excedem 2 desvios-padrão).
  * **Picos Positivos Significativos:** {{ n_picos_pos }} semanas — {{ picos_pos_merge }} na merge window, {{ picos_pos_inicial }} nas primeiras semanas de estabilização (rc1–rc5) e {{ picos_pos_final }} na fase final do ciclo (a partir da semana 6).
    * **Por regime:** até 2014 registam-se {{ picos_pos_merge_ant }} picos na merge window contra {{ picos_pos_final_ant }} na fase final; a partir de 2015, {{ picos_pos_merge_dep }} contra {{ picos_pos_final_dep }}. A merge window inverte de sinal entre os dois regimes, pelo que a contagem agregada não deve ser citada sem esta separação.
  * **Quedas Anómalas Negativas:** {{ n_quedas_neg }} semanas — {{ quedas_merge }} na merge window, {{ quedas_inicial }} nas primeiras semanas rc e {{ quedas_final }} na fase final do ciclo ({{ quedas_ant }} antes de 2015 e {{ quedas_dep }} a partir de 2015).
* **Tabela de Outliers:** Inserir e comentar `tab_02_outliers.csv`.

---

## 4. Imagens a Incluir no Documento (disponíveis em `output/images/png/` e `output/images/svg/`)
0. **`fig_00_matriz_correlacao_pearson.[png|svg]` e `fig_00_matriz_correlacao_spearman.[png|svg]`:** Inserir no Capítulo 2 como Figura 0 (matrizes de correlação de Pearson e de Spearman entre as 12 variáveis: as 9 de código e as 3 quantitativas do ciclo de release, delimitadas a preto; justificam a seleção de variáveis exógenas e a exclusão das variáveis tautológicas).
1. **`fig_01_cronogramas_base.[png|svg]`:** Inserir no Capítulo 2 como Figura 1 (Cronogramas em níveis, crescimento de contribuidores, escala logarítmica de code churn e micro-sazonalidade semanal diária).
2. **`fig_02_decomposicao_adit_mult.[png|svg]`:** Inserir no Capítulo 3 como Figura 2 (Painel quádruplo comparando aditivo vs multiplicativo e assinalando os pontos vermelhos de resíduos que ultrapassam $2\sigma$).
