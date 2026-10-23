# Síntese da Modelação Econométrica Clássica
**Documento de Apoio à Redação para o Elemento A (Capítulo 4 do Relatório)**  
*Gerado automaticamente por `scripts/02_classical_econometrics.py`*

---

## 1. Testes de Raízes Unitárias e Estacionaridade (Capítulo 4.1 — Slides 46, 52 e 62)
* **Série em Níveis ($Y_t$):**
  * Estatística ADF = **-1.9514** (p-value = **0.3083**).
  * **Conclusão:** Não rejeitamos a hipótese nula de raiz unitária ($H_0$). A série em níveis é **não estacionária** e comporta-se como um processo com tendência estocástica.
* **Hierarquia de Diferenciação do Professor Jorge Caiado (Slide 62 e 78):**
  1. **Diferença Sazonal ($\nabla_{52} Y_t = (1 - B^{52}) Y_t$):** Estatística ADF = **-4.0808** (p-value = **0.0010**). Rejeita $H_0$ a 1% de significância.
  2. **Primeira Diferença Simples ($\nabla Y_t = (1 - B) Y_t$):** Estatística ADF = **-15.083** (p-value = **8.4021e-28**). Rejeita fortemente $H_0$.
* **Tabela para o Relatório:** Inserir `tab_03_adf.csv`.

---

## 2. Identificação através de Correlogramas (Capítulo 4.2 — Slides 35 a 45, 51)
* **Comportamento da FAC e FACP:**
  * Na série em níveis, a FAC decresce muito lentamente (ainda 0.320 no desfasamento 60, muito acima da banda de ±0.066), sintoma inequívoco de não estacionaridade (Slide 37 e 51).
  * Na série diferenciada ($\nabla Y_t$), a FAC apresenta valores negativos significativos nos primeiros desfasamentos mas **não se anula a seguir**: 16 desfasamentos entre 1 e 60 excedem a banda de ±0.066, entre os quais 1, 2, 3, 9, 10, 16, 19, 21, 25, 28, 37, 38.
  * **Pico anual:** a autocorrelação no desfasamento 52 vale 0.070, um pico marginal, logo acima da banda de ±0.066; o maior valor absoluto do correlograma é -0.184, no desfasamento 2. O ciclo do kernel ronda as dez semanas, não o ano, pelo que o período $s = 52$ descreve a sazonalidade de calendário e não a cadência de releases.
  * **Consequência para a identificação:** o correlograma não mostra o corte abrupto típico de um MA de ordem baixa. A ordem $q = 2$ foi escolhida por critério de informação dentro de uma grelha parcimoniosa, e não por leitura direta da FAC; a estrutura cíclica remanescente reaparece no diagnóstico residual (Secção 5).
* **Figura para o Relatório:** Inserir `fig_03_fac_facp.[png|svg]`.

---

## 3. Desempenho do Alisamento Exponencial (Capítulo 4.3 — Slides 11 a 30)
* **Holt Linear:** REQM = **618.21**, EPAM = **38.86%**.
* **Holt-Winters Aditivo ($s=52$):** REQM = **643.53**, EPAM = **41.81%** (AIC = 11227.09).
* **Holt-Winters Multiplicativo ($s=52$):** REQM = **663.81**, EPAM = **42.84%** (AIC = 11266.0).
* **Forma multiplicativa vs. aditiva:** em teoria a forma multiplicativa é a mais adequada, porque a amplitude sazonal cresce com o nível da série. Na amostra de teste, porém, o REQM foi 643.53 no aditivo e 663.81 no multiplicativo (AIC de 11227.09 e 11266.0, respetivamente), pelo que o desempenho fora da amostra favorece a forma aditiva. A projeção da Secção 6 usa o multiplicativo por ser o procedimento demonstrado em `class-code/2.py`, e não por superioridade preditiva.
* **Observação Teórica:** O alisamento exponencial gera previsões que acompanham o nível médio, mas falha em capturar as oscilações pontuais extremas impostas pelo ciclo de release do kernel.
* **Tabela Específica de Suavização (`alisamento_exponencial.csv`):**
  * *Conteúdo:* Constantes de alisamento ($\alpha$, $\beta$, $\gamma$), critérios de informação (AIC, BIC) e métricas out-of-sample (REQM, EAM, EPAM, EPAMs) para Holt Linear, Holt-Winters Aditivo e Holt-Winters Multiplicativo.
  * *Por que considerar:* Isola formalmente a família de suavização exponencial para análise detalhada no Capítulo 4.3, antes da comparação agregada com Box-Jenkins.
  * *Valor analítico:* Demonstra a sensibilidade dos parâmetros de tendência e sazonalidade e permite comparar as duas formas de sazonalidade com os mesmos critérios.

---

## 4. Modelação Box-Jenkins: Ganho Estrutural com Covariáveis Exógenas (Capítulo 4.4 — Slides 67 a 74)
* **Amostra de estimação:** divisão cronológica 80:20 com 893 semanas de treino e 224 de teste (2022 a 2026). As 10 semanas anteriores ao primeiro *tag* de release não têm fase conhecida, pelo que todas as especificações Box-Jenkins são estimadas nas mesmas **883 semanas**. É isso que torna a comparação como-para-como: sem esta restrição, o modelo com a fase teria uma amostra diferente do modelo sem ela.
* **Confronto Univariado vs. Multivariado:**
  * **ARIMA(0, 1, 2) Univariado:** REQM = **614.76**, EPAM = **36.78%** (AIC = 13227.55).
  * **ARIMAX(0, 1, 2) + Covariáveis + Fase de Release:** REQM = **254.0**, EPAM = **10.08%** (AIC = 11423.96).
  * **Ganho Percentual:** A introdução das covariáveis exógenas reduziu o REQM em **58.7%** e o EPAM de **36.78%** para **10.08%**.
* **Fase do ciclo de release (merge window como referência):**
  * `fase_rc_inicial` (estabilização inicial, rc1-5): Coeficiente = **-83.4928** ($p$-value < 0.0001).
  * `fase_rc_final` (estabilização final, rc6+): Coeficiente = **-18.8394** ($p$-value = 0.1893).
  * **Contributo da variável:** o modelo com a fase reduz o REQM em 2.9% face à especificação sem ela (254.00 contra 261.59), na mesma amostra de treino.
  * **Como ler:** estando o modelo em primeiras diferenças, cada coeficiente compara o **ritmo semanal** dessa fase com o da merge window, e não o nível de commits. São médias de 2005-2026 e o efeito inverte de sinal em 2015 (ver `tab_02_outliers.csv`), pelo que a média esconde a heterogeneidade entre eras. A fase é a única covariável conhecida antes de a semana começar; a justificação da escolha e a comparação com as alternativas estão em `scripts/analise_covariaveis_release/README.md`.
* **Significância dos Coeficientes das Covariáveis (Slide 70):**
  * `unique_authors` (Contribuidores): Coeficiente = **+4.6262** ($p$-value < 0.0001).
  * `log_churn` (Code Churn): Coeficiente = **+102.9059** ($p$-value < 0.0001).
  * **Como ler estes coeficientes:** o modelo está estimado em **primeiras diferenças** ($d = 1$), pelo que os coeficientes medem o efeito sobre a **variação semanal** da atividade, e não sobre o nível. Um contribuidor ativo adicional está associado a cerca de **+4.6 commits de variação** de uma semana para a seguinte, e um aumento de 1% no churn a **~1.0 commits de variação**. Escrever que «cada novo contribuidor acrescenta 4.6 commits por semana ao total» seria um erro de leitura: descreveria um efeito de nível, que este modelo não estima.
* **Tabelas para o Relatório:** Inserir `tab_04_sarimax_results.csv` e `tab_05_comparacao_econometrica.csv`.
* **Tabela de Especificações Box-Jenkins (`arima_arimax.csv`):**
  * *Conteúdo:* Resumo comparativo de todas as especificações ARIMA univariadas e ARIMAX multivariadas ajustadas, com ordens $(p, d, q)$, AIC, BIC e métricas preditivas.
  * *Por que considerar:* Permite demonstrar a progressão metodológica da disciplina: desde os modelos univariados de referência até à incorporação sequencial de variáveis exógenas (code churn, autores e termos harmónicos de Fourier).
  * *Valor analítico:* Justifica formalmente porque certas combinações de covariáveis foram retidas ou descartadas com base no compromisso entre parcimónia (AIC/BIC) e precisão fora da amostra (REQM).

---

## 5. Diagnóstico Residual e Validação de Ruído Branco (Capítulo 4.5 — Slides 55 e 70)
* **Teste de Ljung-Box (Validação de Não-Autocorrelação):**
  * Lag 1: $Q = 0.20$, $p\text{-value} = 0.6527$.
  * Lag 5: $Q = 4.36$, $p\text{-value} = 0.4986$.
  * Lag 10: $Q = 13.93$, $p\text{-value} = 0.1763$.
  * Lag 15: $Q = 31.71$, $p\text{-value} = 0.0070$.
  * Lag 20: $Q = 36.05$, $p\text{-value} = 0.0152$.
  * **Conclusão:** $H_0$ não é rejeitada nos desfasamentos 1, 5 e 10, mas é rejeitada em 15 e 20. Os resíduos não podem ser declarados ruído branco: resta autocorrelação a partir do desfasamento 15, mesmo com a fase do ciclo de release no modelo. A dummy de fase capta o nível médio de cada fase, não a cadência irregular dos ciclos (entre 8 e 15 semanas), pelo que a estrutura de médio prazo fica por explicar. É uma limitação a declarar no relatório.
* **Teste de Normalidade de Jarque-Bera:**
  * $p\text{-value} < 0.0001$. A hipótese nula de normalidade estrita é rejeitada: os resíduos apresentam caudas pesadas, consistentes com os picos do ciclo de release identificados na análise exploratória.
* **Figura para o Relatório:** Inserir `fig_04_diagnostico_residuos.[png|svg]`.
* **Tabela de Validação Residual (`diagnostico_residuos.csv`):**
  * *Conteúdo:* Resultados numéricos completos do teste de normalidade de Jarque-Bera (global) e do teste de ruído branco de Ljung-Box para os desfasamentos 1, 5, 10, 15 e 20, com estatísticas, p-values e conclusões diagnósticas.
  * *Por que considerar:* Substitui os outputs textuais de consola por uma tabela limpa e auditável, adequada para inclusão direta no relatório ou em anexo metodológico.
  * *Valor analítico:* É o comprovativo estatístico do diagnóstico: confirma a ausência de autocorrelação de curto prazo, mas **não** valida ruído branco em toda a extensão testada. Deve ser citado em conjunto com a ressalva acima, e não como prova de que o modelo esgotou a dependência temporal.

---

## 6. Projeção Futura a 52 Semanas com Holt-Winters Reajustado (Slides 27, 30, 33c e class-code/2.py)
* **Procedimento:** Conforme demonstrado pelo docente em aula (`class-code/2.py`), após apurar as constantes de alisamento ótimas no conjunto de treino, o modelo Holt-Winters Multiplicativo foi reajustado à totalidade das 1.117 semanas observadas (2005–2026).
* **Horizonte Projetado:** 52 semanas, de 2026-09-14 a 2027-09-06.
* **Comportamento da Curva Futura:** A projeção preserva a amplitude cíclica anual ($s=52$), antecipando oscilações entre ~1.371 commits (semana de 2026-12-28) e ~2.271 commits (semana de 2026-09-21).
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
