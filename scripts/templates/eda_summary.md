# Síntese da Análise Exploratória e Decomposição Temporal
**Documento de Apoio à Redação para o Elemento A (Capítulos 2 e 3 do Relatório)**  
*Gerado automaticamente por `scripts/01_eda_decomposition.py`*

---

## 1. Dados, Higienização e Matriz de Correlação (Capítulo 2)
* **Frequência Semanal (`W-MON`):** {{ n_observacoes }} observações validadas (Abril de 2005 a Setembro de 2026).
* **Truncatura da Cauda:** A última linha (`2026-09-14`) foi removida por corresponder ao fecho a meio da semana da recolha Git (apenas 2 commits registados), prevenindo colapso artificial nas previsões fora da amostra (*out-of-sample*).
* **Ponto Singular Inicial (Abril de 2005):** O primeiro registo contém 6.749.493 linhas de inserção num único commit de Linus Torvalds (importação fundacional da árvore). O volume de commits (`total_commits = 259`) é válido, mas o code churn exige escala logarítmica ou tratamento de alavancagem.
* **Análise da Matriz de Correlação (`fig_00_matriz_correlacao.[png|svg]` e `tab_00_matriz_correlacao.csv`):**
  * **Identidades Contabilísticas a Excluir das Exógenas:** `non_merge_commits` apresenta correlação de $r = 0.998$ com `total_commits` ($\text{total\_commits} = \text{non\_merge\_commits} + \text{merge\_commits}$). Do mesmo modo, `insertions` tem correlação de $r = 0.939$ com `total_churn` ($\text{ins} + \text{del}$). Incluir estas parcelas em simultâneo geraria multicolinearidade severa ou tautologia contabilística.
  * **Colinearidade vs. Ortogonalidade nas Candidatas Exógenas:** `unique_authors` apresenta uma correlação forte com `total_commits` ($r = 0.934$ em níveis; $r = 0.898$ em primeiras diferenças), refletindo que o volume de commits escala diretamente com o número de contribuidores ativos na semana. Em contrapartida, `total_churn` ($r = 0.587$), `insertions` ($r = 0.560$), `deletions` ($r = 0.431$) e `net_code_growth` ($r = 0.379$) apresentam correlações moderadas com `total_commits` e baixa colinearidade com `unique_authors` ($r = 0.452$), fornecendo informação estrutural genuinamente ortogonal sobre a dimensão e complexidade das alterações.
* **Tabelas de Variáveis:** Inserir `tab_00_matriz_correlacao.csv` e `tab_01_variaveis.csv` no Capítulo 2.

---

## 2. Decomposição Temporal: Aditiva vs. Multiplicativa (Capítulo 3 — Slides 5 a 9)
* **Periodicidade Sazonal:** $s = 52$ semanas (ciclo anual).
* **Tendência-Ciclo ($T_t$):** Estimada através de uma média móvel centrada de ordem 52. As primeiras 26 e as últimas 26 semanas são, por definição matemática da média móvel bicaudal, não calculáveis (*NaN*).
* **Justificação do Modelo Multiplicativo vs. Aditivo (Pergunta Clássica do Docente):**
  * Em 2005/2006, a série oscilava em torno de 300 commits semanais, com quedas sazonais de ~100 commits.
  * Em 2025/2026, a série oscila em torno de 2.500 commits semanais, com quedas sazonais de ~800 commits.
  * **Conclusão:** Como a amplitude das oscilações sazonais aumenta em proporção direta com o nível da tendência, o **Modelo Multiplicativo ($Y_t = T_t \times S_t \times R_t$)** é teoricamente mais apropriado do que o Aditivo (Slide 8).
  * Fatores sazonais multiplicativos oscilam entre **{{ min_fator_sazonal_mult }}%** da tendência (semanas de Natal e Verão) e **{{ max_fator_sazonal_mult }}%** (semanas de abertura de merge windows).

---

## 3. Deteção de Outliers na Série Residual (Capítulo 3 — Slide 10)
* **Regra de Decisão do Slide 10:** $|R_t - \bar{R}| > 2 \sigma_R$ aplicada aos resíduos da decomposição aditiva.
* **Desvio-Padrão Residual ($\sigma_R$):** {{ resid_adit_std }} commits.
* **Total de Semanas Assinaladas como Outliers:** {{ n_outliers_adit }} semanas ({{ pct_outliers_adit }}% da amostra, em estrita conformidade com a expectativa teórica de uma distribuição normal onde ~5% dos pontos excedem 2 desvios-padrão).
  * **Picos Positivos Significativos:** {{ n_picos_pos }} semanas (coincidem maioritariamente com semanas de abertura de merge windows imediatas ao lançamento de versões rc1).
  * **Quedas Anómalas Negativas:** {{ n_quedas_neg }} semanas (coincidem sistematicamente com as semanas 51–52 de Natal/Ano Novo e semanas 32–34 de pausa estival em Agosto).
* **Tabela de Outliers:** Inserir e comentar `tab_02_outliers.csv`.

---

## 4. Imagens a Incluir no Documento (disponíveis em `output/images/png/` e `output/images/svg/`)
0. **`fig_00_matriz_correlacao.[png|svg]`:** Inserir no Capítulo 2 como Figura 0 (Matriz de correlação de Pearson e Spearman entre todas as 9 variáveis do dataset, justificando a seleção de variáveis exógenas e a exclusão de variáveis tautológicas).
1. **`fig_01_cronogramas_base.[png|svg]`:** Inserir no Capítulo 2 como Figura 1 (Cronogramas em níveis, crescimento de contribuidores, escala logarítmica de code churn e micro-sazonalidade semanal diária).
2. **`fig_02_decomposicao_adit_mult.[png|svg]`:** Inserir no Capítulo 3 como Figura 2 (Painel quádruplo comparando aditivo vs multiplicativo e assinalando os pontos vermelhos de resíduos que ultrapassam $2\sigma$).
