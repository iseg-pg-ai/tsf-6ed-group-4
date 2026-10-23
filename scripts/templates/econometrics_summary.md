# Síntese da Modelação Econométrica Clássica
**Documento de Apoio à Redação para o Elemento A (Capítulo 4 do Relatório)**  
*Gerado automaticamente por `scripts/02_classical_econometrics.py`*

---

## 1. Testes de Raízes Unitárias e Estacionaridade (Capítulo 4.1 — Slides 46, 52 e 62)
* **Série em Níveis ($Y_t$):**
  * Estatística ADF = **{{ adf_stat_niveis }}** (p-value = **{{ adf_pval_niveis }}**).
  * **Conclusão:** Não rejeitamos a hipótese nula de raiz unitária ($H_0$). A série em níveis é **não estacionária** e comporta-se como um processo com tendência estocástica.
* **Hierarquia de Diferenciação do Professor Jorge Caiado (Slide 62 e 78):**
  1. **Diferença Sazonal ($\nabla_{52} Y_t = (1 - B^{52}) Y_t$):** Estatística ADF = **{{ adf_stat_sazonal }}** (p-value = **{{ adf_pval_sazonal }}**). Rejeita $H_0$ a 1% de significância.
  2. **Primeira Diferença Simples ($\nabla Y_t = (1 - B) Y_t$):** Estatística ADF = **{{ adf_stat_simples }}** (p-value = **{{ adf_pval_simples }}**). Rejeita fortemente $H_0$.
* **Porque se retém $d = 1$ e não $D = 1$?** A hierarquia confirma que ambas as transformações estacionarizam a série, mas o modelo final usa apenas a diferença simples. A série já é estacionária em primeira diferença, e acumular as duas diferenças ($d = 1$ e $D = 1$) sobre-diferencia, introduzindo uma raiz unitária sazonal não invertível. Acresce que o período relevante desta série não é o ano civil: é o ciclo de release de ~10 semanas, que a diferença de 52 semanas não remove (Secção 2). O modelo vencedor é, por isso, um ARIMA$(0, 1, 2)$ com $d = 1$ e $D = 0$.
* **Tabela para o Relatório:** Inserir `tab_03_adf.csv`.

---

## 2. Identificação através de Correlogramas (Capítulo 4.2 — Slides 35 a 45, 51)
* **Comportamento da FAC e FACP:**
  * Na série em níveis, a FAC decresce muito lentamente (ainda {{ acf_niveis_lag60 }} no desfasamento 60, muito acima da banda de ±{{ acf_banda }}), sintoma inequívoco de não estacionaridade (Slide 37 e 51).
  * Na série diferenciada ($\nabla Y_t$), a FAC apresenta valores negativos significativos nos primeiros desfasamentos mas **não se anula a seguir**: {{ acf_n_lags_sig }} desfasamentos entre 1 e 60 excedem a banda de ±{{ acf_banda }}, entre os quais {{ acf_lags_sig_txt }}.
  * **Pico anual:** a autocorrelação no desfasamento 52 vale {{ acf_lag52 }}, {{ acf_lag52_leitura }}; o maior valor absoluto do correlograma é {{ acf_pico_valor }}, no desfasamento {{ acf_pico_lag }}. O ciclo do kernel ronda as dez semanas, não o ano, pelo que o período $s = 52$ descreve a sazonalidade de calendário e não a cadência de releases.
  * **Consequência para a identificação:** o correlograma não mostra o corte abrupto típico de um MA de ordem baixa. A ordem $q = 2$ foi escolhida por critério de informação dentro de uma grelha parcimoniosa, e não por leitura direta da FAC; a estrutura cíclica remanescente reaparece no diagnóstico residual (Secção 5).
* **Figura para o Relatório:** Inserir `fig_03_fac_facp.[png|svg]`.

---

## 3. Desempenho do Alisamento Exponencial (Capítulo 4.3 — Slides 11 a 30)
* **Holt Linear:** REQM = **{{ holt_linear_reqm }}**, EPAM = **{{ holt_linear_epam }}%**.
* **Holt-Winters Aditivo ($s=52$):** REQM = **{{ hw_adit_reqm }}**, EPAM = **{{ hw_adit_epam }}%** (AIC = {{ hw_adit_aic }}).
* **Holt-Winters Multiplicativo ($s=52$):** REQM = **{{ holt_mult_reqm }}**, EPAM = **{{ holt_mult_epam }}%** (AIC = {{ holt_mult_aic }}).
* **Forma multiplicativa vs. aditiva:** {{ hw_comparacao }}.
* **Observação Teórica:** O alisamento exponencial gera previsões que acompanham o nível médio, mas falha em capturar as oscilações pontuais extremas impostas pelo ciclo de release do kernel.
* **Tabela Específica de Suavização (`alisamento_exponencial.csv`):**
  * *Conteúdo:* Constantes de alisamento ($\alpha$, $\beta$, $\gamma$), critérios de informação (AIC, BIC) e métricas out-of-sample (REQM, EAM, EPAM, EPAMs) para Holt Linear, Holt-Winters Aditivo e Holt-Winters Multiplicativo.
  * *Por que considerar:* Isola formalmente a família de suavização exponencial para análise detalhada no Capítulo 4.3, antes da comparação agregada com Box-Jenkins.
  * *Valor analítico:* Demonstra a sensibilidade dos parâmetros de tendência e sazonalidade e permite comparar as duas formas de sazonalidade com os mesmos critérios.

---

## 4. Modelação Box-Jenkins: Ganho Estrutural com Covariáveis Exógenas (Capítulo 4.4 — Slides 67 a 74)
* **Amostra de estimação:** divisão cronológica 80:20 com {{ n_treino }} semanas de treino e {{ n_teste }} de teste (2022 a 2026). As 10 semanas anteriores ao primeiro *tag* de release não têm fase conhecida, pelo que todas as especificações Box-Jenkins são estimadas nas mesmas **{{ n_treino_bj }} semanas**. É isso que torna a comparação como-para-como: sem esta restrição, o modelo com a fase teria uma amostra diferente do modelo sem ela.
* **Confronto Univariado vs. Multivariado:**
  * **ARIMA(0, 1, 2) Univariado:** REQM = **{{ arima_reqm }}**, EPAM = **{{ arima_epam }}%** (AIC = {{ arima_aic }}).
  * **{{ melhor_modelo }}:** REQM = **{{ melhor_reqm }}**, EPAM = **{{ melhor_epam }}%** (AIC = {{ melhor_aic }}).
  * **Ganho Percentual:** A introdução das covariáveis exógenas reduziu o REQM em **{{ ganho_reqm_pct }}%** e o EPAM de **{{ arima_epam }}%** para **{{ melhor_epam }}%**.
* **Fase do ciclo de release (merge window como referência):**
  * `fase_rc_inicial` (estabilização inicial, rc1-5): Coeficiente = **{{ coef_fase_rc_inicial }}** ($p$-value {{ pval_fase_rc_inicial }}).
  * `fase_rc_final` (estabilização final, rc6+): Coeficiente = **{{ coef_fase_rc_final }}** ($p$-value {{ pval_fase_rc_final }}).
  * **Contributo da variável:** {{ ganho_fase }}.
  * **Como ler:** estando o modelo em primeiras diferenças, cada coeficiente compara a **variação semanal** média dessa fase com a da merge window (categoria de referência), e não o nível de commits da fase. Uma semana que entra em rc1-5 sofre, em média, uma variação de **{{ coef_fase_rc_inicial }}** commits face a uma semana equivalente na merge window. São médias de 2005-2026 e o efeito inverte de sinal em 2015 (ver `tab_02_outliers.csv`), pelo que a média esconde a heterogeneidade entre eras. A fase é a única covariável conhecida antes de a semana começar; a justificação da escolha e a comparação com as alternativas estão em `scripts/analise_covariaveis_release/README.md`.
* **Significância dos Coeficientes das Covariáveis (Slide 70):**
  * `unique_authors` (Contribuidores): Coeficiente = **+{{ coef_autores }}** ($p$-value {{ pval_autores }}).
  * `log_churn` (Code Churn): Coeficiente = **+{{ coef_churn }}** ($p$-value {{ pval_churn }}).
  * **Como ler estes coeficientes:** o modelo é uma **regressão com erros ARIMA** estimada em primeiras diferenças ($d = 1$), pelo que os coeficientes são identificados pela co-movimentação das **variações** semanais, e não pelo nível. O teste é direto e reproduzível: uma regressão OLS de $\nabla Y_t$ sobre $\nabla X_t$ devolve praticamente os mesmos valores ({{ ols_dif_autores }} para `unique_authors` e {{ ols_dif_churn }} para `log_churn`), enquanto a mesma regressão em níveis dá {{ ols_niveis_autores }} e {{ ols_niveis_churn }}. A leitura correta é, pois, de curto prazo: **um contribuidor ativo adicional está associado a cerca de +{{ coef_autores_fmt }} commits na semana em que essa alteração ocorre**, e um aumento de 1% no churn a ~{{ efeito_churn_1pct }} commits nessa semana.
  * **Limitação a declarar no relatório:** o erro do modelo é integrado e não existe termo de correção do erro (o modelo não é de cointegração e a estacionaridade de $Y_t - X_t\beta$ não foi testada). Por isso, na projeção o coeficiente também desloca o nível de forma permanente. A leitura de nível — «cada contribuidor ativo acrescenta {{ coef_autores_fmt }} commits por semana ao total» — é uma implicação da especificação, não um efeito validado por um teste; a de curto prazo é a que os dados suportam. Um indício de que a leitura de nível está errada: ela daria às semanas de rc1-5 um nível ~83 commits *abaixo* da merge window, quando os dados mostram que, desde 2015, a merge window é o vale do ciclo e não o pico. A consequência prática desta limitação vê-se na Secção 7: congelar as covariáveis num valor atípico desloca toda a projeção.
* **Tabelas para o Relatório:** Inserir `tab_04_sarimax_results.csv` e `tab_05_comparacao_econometrica.csv`.
* **Tabela de Especificações Box-Jenkins (`arima_arimax.csv`):**
  * *Conteúdo:* Resumo comparativo de todas as especificações ARIMA univariadas e ARIMAX multivariadas ajustadas, com ordens $(p, d, q)$, AIC, BIC e métricas preditivas.
  * *Por que considerar:* Permite demonstrar a progressão metodológica da disciplina: desde os modelos univariados de referência até à incorporação sequencial de variáveis exógenas (code churn, autores e termos harmónicos de Fourier).
  * *Valor analítico:* Justifica formalmente porque certas combinações de covariáveis foram retidas ou descartadas com base no compromisso entre parcimónia (AIC/BIC) e precisão fora da amostra (REQM).

---

## 5. Diagnóstico Residual e Validação de Ruído Branco (Capítulo 4.5 — Slides 55 e 70)
* **Teste de Ljung-Box (Validação de Não-Autocorrelação):**
  * Lag 1: $Q = {{ lb_lag1_stat }}$, $p\text{-value} = {{ lb_lag1_pval }}$.
  * Lag 5: $Q = {{ lb_lag5_stat }}$, $p\text{-value} = {{ lb_lag5_pval }}$.
  * Lag 10: $Q = {{ lb_lag10_stat }}$, $p\text{-value} = {{ lb_lag10_pval }}$.
  * Lag 15: $Q = {{ lb_lag15_stat }}$, $p\text{-value} = {{ lb_lag15_pval }}$.
  * Lag 20: $Q = {{ lb_lag20_stat }}$, $p\text{-value} = {{ lb_lag20_pval }}$.
  * **Conclusão:** $H_0$ não é rejeitada nos desfasamentos {{ lb_ok_txt }}, mas é rejeitada em {{ lb_rejeita_txt }}. {{ lb_leitura }}
* **Nota sobre o primeiro resíduo:** a estimação é feita na forma de espaço de estados, com inicialização difusa do filtro de Kalman, pelo que a primeira observação absorve a incerteza inicial (-1.291 commits, ~8 desvios-padrão) e não um choque da série. É por isso excluída, e os testes correm sobre {{ n_resid }} resíduos.
* **Teste de Normalidade de Jarque-Bera:**
  * $p\text{-value} {{ jb_pval }}$ com $JB = {{ jb_stat }}$ (n = {{ n_resid }}). A hipótese nula de normalidade estrita é rejeitada.
  * **A rejeição não é um artefacto da inicialização:** mantendo o primeiro resíduo, $JB$ sobe para {{ jb_stat_bruto }} ($p$-value {{ jb_pval_bruto }}), mas já era enorme sem ele. O que domina é a forma da distribuição, não o valor inicial.
  * **Assimetria = {{ assimetria }} e curtose em excesso = {{ curtose_excesso }}** (coeficiente de Fisher; 0 = normal). A distribuição tem cauda direita: o resíduo positivo máximo é {{ max_resid }} commits ({{ max_resid_data }}), cerca de {{ max_resid_sigmas }} desvios-padrão, contra um desvio-padrão residual de {{ dp_resid }}. É da mesma natureza dos picos do ciclo de release identificados na análise exploratória (`tab_02_outliers.csv`): a especificação linear com dummies desloca o nível médio de cada fase, mas não reproduz a magnitude destes picos.
* **Figura para o Relatório:** Inserir `fig_04_diagnostico_residuos.[png|svg]`.
* **Tabela de Validação Residual (`diagnostico_residuos.csv`):**
  * *Conteúdo:* Resultados numéricos completos do teste de normalidade de Jarque-Bera (global) e do teste de ruído branco de Ljung-Box para os desfasamentos 1, 5, 10, 15 e 20, com estatísticas, p-values e conclusões diagnósticas.
  * *Por que considerar:* Substitui os outputs textuais de consola por uma tabela limpa e auditável, adequada para inclusão direta no relatório ou em anexo metodológico.
  * *Valor analítico:* É o comprovativo estatístico do diagnóstico: confirma a ausência de autocorrelação de curto prazo, mas **não** valida ruído branco em toda a extensão testada. Deve ser citado em conjunto com a ressalva acima, e não como prova de que o modelo esgotou a dependência temporal.

---

## 6. Projeção Futura a 52 Semanas com Holt-Winters Reajustado (Slides 27, 30, 33c e class-code/2.py)
* **Procedimento:** Conforme demonstrado pelo docente em aula (`class-code/2.py`), após apurar as constantes de alisamento ótimas no conjunto de treino, o modelo Holt-Winters Multiplicativo foi reajustado à totalidade das 1.117 semanas observadas (2005–2026).
* **Horizonte Projetado:** 52 semanas, de {{ prev_inicio }} a {{ prev_fim }}.
* **Comportamento da Curva Futura:** A projeção preserva a amplitude cíclica anual ($s=52$), antecipando oscilações entre ~{{ prev_min }} commits (semana de {{ prev_min_data }}) e ~{{ prev_max }} commits (semana de {{ prev_max_data }}).
* **Tabela e Figura:** Inserir `tab_05b_previsao_futura_holtwinters.csv` e `fig_05b_previsao_futura_holtwinters.[png|svg]`.

---

## 7. Cenários de Projeção a 52 Semanas com o ARIMAX Vencedor (Capítulo 7 — Slides 75-76 e 109-110)
* **Porque não há uma projeção única:** o {{ cenario_modelo }} é uma regressão com erros ARIMA, não um modelo multivariado. As covariáveis (contribuidores, churn e fase do ciclo) não são projetadas pelo modelo: têm de ser fixadas pelo analista. A projeção é, por isso, **condicional**, e o relatório deve apresentar cenários e não um número único.
* **Reajuste:** o modelo foi reestimado na amostra completa de {{ cenario_n }} semanas com fase de release conhecida (2005-06-27 a {{ cenario_ultima_data }}), e não na amostra de treino, porque uma projeção para o futuro usa toda a informação disponível. As métricas fora da amostra da Secção 4 continuam a ser as da estimação em treino: são coisas distintas e não devem ser misturadas.
* **Horizonte:** 52 semanas, de {{ cenario_inicio }} a {{ cenario_fim }}.
* **Cenário 1 — estático, ancorado na última semana.** Covariáveis congeladas em $X_T$ ({{ cenario_ancora_autores }} autores, churn de {{ cenario_ancora_churn_linhas }} linhas, semana 2 do ciclo). A projeção começa em ~{{ cenario1_max }} commits e estabiliza em ~{{ cenario1_min }}. **Não é uma previsão utilizável e não deve ser levada ao relatório como tal.** A última semana observada é um vale ({{ cenario_ultimo_observado }} commits) e o modelo, estimado em diferenças e sem termo de correção do erro, propaga esse desvio a todo o horizonte. Serve para mostrar, com números, o risco de ancorar um cenário numa única observação atípica.
* **Cenário 1b — estático, ancorado no último ano.** Covariáveis congeladas na média das últimas 52 semanas ({{ cenario_media_autores }} autores, churn de {{ cenario_media_churn_linhas }} linhas; {{ cenario_media_rc_inicial }}% das semanas em rc1-5 e {{ cenario_media_rc_final }}% em rc6+). A projeção estabiliza em ~{{ cenario1b_mediana }} commits, praticamente igual à média observada do último ano ({{ cenario_obs_media }}). É o cenário estático defensável: descreve um ano "típico" sem inventar calendário de releases.
* **Cenário 2 — homólogo.** Repete o caminho das covariáveis efetivamente observado nas últimas 52 semanas ({{ cenario_obs_inicio }} a {{ cenario_ultima_data }}). A projeção varia entre {{ cenario2_min }} e {{ cenario2_max }} commits (mediana {{ cenario2_mediana }}), começando em {{ cenario2_primeira }}. É o mais realista dos três, porque preserva a cadência real de merge windows e fases de release, e também o mais frágil conceptualmente: assume que o próximo ano repete o anterior. **Atenção:** o último valor ({{ cenario2_ultima }}) é o *replay* da semana de vale de 2026-09-07 e não um sinal de colapso da atividade.
* **Referência observada (últimas 52 semanas):** média {{ cenario_obs_media }}, mediana {{ cenario_obs_mediana }}, mínimo {{ cenario_obs_min }} e máximo {{ cenario_obs_max }} commits.
* **Como escrever isto no relatório:** apresentar os três cenários como análise de sensibilidade ao caminho das covariáveis, com o Cenário 2 como referência e o Cenário 1b como alternativa estática. O Cenário 1 cita-se apenas como aviso metodológico. A projeção da Secção 6 (Holt-Winters, univariada e com sazonalidade anual) responde a outra pergunta e não é concorrente desta: uma descreve o ciclo de calendário, esta descreve a trajetória condicional às covariáveis.
* **Tabela e Figura:** Inserir `tab_09_previsao_futura_arimax_cenarios.csv` e `fig_09_previsao_futura_arimax_cenarios.[png|svg]`.

---

## 8. Imagens Geradas para os Capítulos 4 e 6
1. **`fig_03_fac_facp.[png|svg]`:** Inserir no Capítulo 4.2 (Correlogramas da série em níveis e diferenciada).
2. **`fig_04_diagnostico_residuos.[png|svg]`:** Inserir no Capítulo 4.5 (Painel quádruplo de diagnóstico dos resíduos do modelo eleito).
3. **`fig_05_comparacao_econometrica.[png|svg]`:** Inserir no Capítulo 4.4 / 6.1 (Sobreposição visual das previsões dos modelos econométricos no período de teste).
4. **`fig_05b_previsao_futura_holtwinters.[png|svg]`:** Inserir no Capítulo 4.3 / 7 (Projeção futura a 52 semanas com o modelo Holt-Winters reajustado à série completa).
5. **`fig_09_previsao_futura_arimax_cenarios.[png|svg]`:** Inserir no Capítulo 7 (Cenários de projeção a 52 semanas com o ARIMAX vencedor).

---

## 9. Guia de Tabelas Complementares Geradas (em `output/tables/`)
Tabelas geradas sem prefixo numérico para apoio e consulta detalhada da equipa de redação:
1. **`alisamento_exponencial.csv`:** Parâmetros e métricas de desempenho dos modelos de alisamento exponencial (Capítulo 4.3).
2. **`arima_arimax.csv`:** Comparativo de todas as ordens e especificações Box-Jenkins testadas (Capítulo 4.4).
3. **`diagnostico_residuos.csv`:** Testes formais de Jarque-Bera e Ljung-Box, assimetria e curtose dos resíduos do modelo vencedor (Capítulo 4.5).
4. **`tab_09_previsao_futura_arimax_cenarios.csv`:** Projeção a 52 semanas do ARIMAX vencedor nos três cenários de covariáveis (Capítulo 7).
