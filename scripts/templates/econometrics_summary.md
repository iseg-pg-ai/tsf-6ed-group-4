# Síntese da Modelação Econométrica Clássica
**Documento de Apoio à Redação para o Elemento A (Capítulo 4 do Relatório)**  
*Gerado automaticamente por `scripts/02_classical_econometrics.py`*

---

## 1. Testes de Raízes Unitárias e Estacionaridade (Capítulo 4.1 — Slides 46, 52 e 62)
* **Série em Níveis ($Y_t$):**
  * Estatística ADF = **{{ adf_stat_niveis }}** (p-value = **{{ adf_pval_niveis }}**).
  * **Conclusão:** Não rejeitamos a hipótese nula de raiz unitária ($H_0$). A série em níveis é **não estacionária** e possui tendência estocástica.
* **Hierarquia de Diferenciação do Professor Jorge Caiado (Slide 62 e 78):**
  1. **Diferença Sazonal ($\nabla_{52} Y_t = (1 - B^{52}) Y_t$):** Estatística ADF = **{{ adf_stat_sazonal }}** (p-value = **{{ adf_pval_sazonal }}**). Rejeita $H_0$ a 1% de significância.
  2. **Primeira Diferença Simples ($\nabla Y_t = (1 - B) Y_t$):** Estatística ADF = **{{ adf_stat_simples }}** (p-value = **{{ adf_pval_simples }}**). Rejeita fortemente $H_0$.
* **Tabela para o Relatório:** Inserir `tab_03_adf.csv`.

---

## 2. Identificação através de Correlogramas (Capítulo 4.2 — Slides 35 a 45, 51)
* **Comportamento da FAC e FACP:**
  * Na série em níveis, a FAC decresce muito lentamente para zero, sintoma inequívoco de não estacionaridade (Slide 37 e 51).
  * Na série diferenciada ($\nabla Y_t$), a FAC corta abruptamente após os primeiros desfasamentos e apresenta picos sazonais no lag 52.
  * De acordo com os **Figurinos Teóricos** (Slide 45), este corte rápido na FAC com decaimento na FACP suporta a formulação de processos de médias móveis de baixa ordem (MA(1) ou MA(2)) combinados com autoregressão.
* **Figura para o Relatório:** Inserir `fig_03_fac_facp.[png|svg]`.

---

## 3. Desempenho do Alisamento Exponencial (Capítulo 4.3 — Slides 11 a 30)
* **Holt Linear:** REQM = **{{ holt_linear_reqm }}**, EPAM = **{{ holt_linear_epam }}%**.
* **Holt-Winters Multiplicativo ($s=52$):** REQM = **{{ holt_mult_reqm }}**, EPAM = **{{ holt_mult_epam }}%** (AIC = {{ holt_mult_aic }}).
* **Observação Teórica:** O alisamento exponencial gera previsões que acompanham o nível médio, mas falha em capturar as oscilações pontuais extremas provocadas pelas *merge windows* do kernel.
* **Tabela Específica de Suavização (`alisamento_exponencial.csv`):**
  * *Conteúdo:* Constantes de alisamento ($\alpha$, $\beta$, $\gamma$), critérios de informação (AIC, BIC) e métricas out-of-sample (REQM, EAM, EPAM, EPAMs) para Holt Linear, Holt-Winters Aditivo e Holt-Winters Multiplicativo.
  * *Por que considerar:* Isola formalmente a família de suavização exponencial para análise detalhada no Capítulo 4.3, antes da comparação agregada com Box-Jenkins.
  * *Valor analítico:* Demonstra a sensibilidade dos parâmetros de tendência e sazonalidade e fundamenta a superioridade do modelo multiplicativo para a projeção futura.

---

## 4. Modelação Box-Jenkins: Ganho Estrutural com Covariáveis Exógenas (Capítulo 4.4 — Slides 67 a 74)
* **Confronto Univariado vs. Multivariado:**
  * **ARIMA(0, 1, 2) Univariado:** REQM = **{{ arima_reqm }}**, EPAM = **{{ arima_epam }}%** (AIC = {{ arima_aic }}).
  * **{{ melhor_modelo }}:** REQM = **{{ melhor_reqm }}**, EPAM = **{{ melhor_epam }}%** (AIC = {{ melhor_aic }}).
  * **Ganho Percentual:** A introdução das covariáveis de engenharia reduziu o REQM em **{{ ganho_reqm_pct }}%** e o EPAM de **{{ arima_epam }}%** para **{{ melhor_epam }}%**!
* **Significância dos Coeficientes das Covariáveis (Slide 70):**
  * `unique_authors` (Contribuidores): Coeficiente = **+{{ coef_autores }}** ($p = {{ pval_autores }}$). Por cada novo contribuidor ativo na semana, o kernel integra em média ~{{ coef_autores_fmt }} commits adicionais.
  * `log_churn` (Code Churn): Coeficiente = **+{{ coef_churn }}** ($p = {{ pval_churn }}$). Forte impacto positivo e altamente significativo.
* **Tabelas para o Relatório:** Inserir `tab_04_sarimax_results.csv` e `tab_05_comparacao_econometrica.csv`.
* **Tabela de Especificações Box-Jenkins (`arima_arimax.csv`):**
  * *Conteúdo:* Resumo comparativo de todas as especificações ARIMA univariadas e ARIMAX multivariadas ajustadas, com ordens $(p, d, q)$, AIC, BIC e métricas preditivas.
  * *Por que considerar:* Permite demonstrar a progressão metodológica da disciplina: desde os modelos univariados de referência até à incorporação sequencial de variáveis exógenas (code churn, autores e termos harmónicos de Fourier).
  * *Valor analítico:* Justifica formalmente porque certas combinações de covariáveis foram retidas ou descartadas com base no compromisso entre parcimónia (AIC/BIC) e precisão fora da amostra (REQM).

---

## 5. Diagnóstico Residual e Validação de Ruído Branco (Capítulo 4.5 — Slides 55 e 70)
* **Teste de Ljung-Box (Validação de Não-Autocorrelação):**
  * Lag 1: $Q = {{ lb_lag1_stat }}$, $p\text{-value} = {{ lb_lag1_pval }}$ ($> 0.05$).
  * Lag 10: $Q = {{ lb_lag10_stat }}$, $p\text{-value} = {{ lb_lag10_pval }}$ ($> 0.05$).
  * **Conclusão:** O teste confirma a ausência de autocorrelação linear sistemática nos resíduos a 5% de significância. O modelo filtrou a estrutura temporal da série, restando **ruído branco (*white noise*)**.
* **Teste de Normalidade de Jarque-Bera:**
  * $p\text{-value} = {{ jb_pval }}$. A hipótese nula de normalidade estrita é rejeitada devido às caudas pesadas provocadas pelos picos pontuais de *merge windows*, fenómeno clássico documentado em engenharia de sistemas.
* **Figura para o Relatório:** Inserir `fig_04_diagnostico_residuos.[png|svg]`.
* **Tabela de Validação Residual (`diagnostico_residuos.csv`):**
  * *Conteúdo:* Resultados numéricos completos do teste de normalidade de Jarque-Bera (global) e do teste de ruído branco de Ljung-Box para os desfasamentos 1, 5, 10, 15 e 20, com estatísticas, p-values e conclusões diagnósticas.
  * *Por que considerar:* Substitui os outputs textuais de consola por uma tabela limpa e auditável, adequada para inclusão direta no relatório ou em anexo metodológico.
  * *Valor analítico:* Constitui o comprovativo estatístico mandatório de que o modelo eleito filtrou com sucesso a dependência temporal dos dados, validando a ausência de autocorrelação residual ($p > 0.05$).

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

---

## 8. Guia de Tabelas Complementares Geradas (em `output/tables/`)
Tabelas geradas sem prefixo numérico para apoio e consulta detalhada da equipa de redação:
1. **`alisamento_exponencial.csv`:** Parâmetros e métricas de desempenho dos modelos de alisamento exponencial (Capítulo 4.3).
2. **`arima_arimax.csv`:** Comparativo de todas as ordens e especificações Box-Jenkins testadas (Capítulo 4.4).
3. **`diagnostico_residuos.csv`:** Testes formais de Jarque-Bera e Ljung-Box do modelo vencedor (Capítulo 4.5).
