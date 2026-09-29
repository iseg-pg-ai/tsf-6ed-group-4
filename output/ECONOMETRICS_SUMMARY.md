# Síntese da Modelação Econométrica Clássica
**Documento de Apoio à Redação para o Elemento A (Capítulo 4 do Relatório)**  
*Gerado automaticamente por `scripts/02_classical_econometrics.py`*

---

## 1. Testes de Raízes Unitárias e Estacionaridade (Capítulo 4.1 — Slides 46, 52 e 62)
* **Série em Níveis ($Y_t$):**
  * Estatística ADF = **-1.9514** (p-value = **0.3083**).
  * **Conclusão:** Não rejeitamos a hipótese nula de raiz unitária ($H_0$). A série em níveis é **não estacionária** e possui tendência estocástica.
* **Hierarquia de Diferenciação do Professor Jorge Caiado (Slide 62 e 78):**
  1. **Diferença Sazonal ($\\nabla_{52} Y_t = (1 - B^{52}) Y_t$):** Estatística ADF = **-4.0808** (p-value = **0.0010**). Rejeita $H_0$ a 1% de significância.
  2. **Primeira Diferença Simples ($\\nabla Y_t = (1 - B) Y_t$):** Estatística ADF = **-15.083** (p-value = **8.4021e-28**). Rejeita fortemente $H_0$.
* **Tabela para o Relatório:** Inserir `tab_03_adf.csv`.

---

## 2. Identificação através de Correlogramas (Capítulo 4.2 — Slides 35 a 45, 51)
* **Comportamento da FAC e FACP:**
  * Na série em níveis, a FAC decresce muito lentamente para zero, sintoma inequívoco de não estacionaridade (Slide 37 e 51).
  * Na série diferenciada ($\\nabla Y_t$), a FAC corta abruptamente após os primeiros desfasamentos e apresenta picos sazonais no lag 52.
  * De acordo com os **Figurinos Teóricos** (Slide 45), este corte rápido na FAC com decaimento na FACP suporta a formulação de processos de médias móveis de baixa ordem (MA(1) ou MA(2)) combinados com autoregressão.
* **Figura para o Relatório:** Inserir `fig_03_fac_facp.[png|svg]`.

---

## 3. Desempenho do Alisamento Exponencial (Capítulo 4.3 — Slides 11 a 30)
* **Holt Linear:** REQM = **618.21**, EPAM = **38.86%**.
* **Holt-Winters Multiplicativo ($s=52$):** REQM = **663.81**, EPAM = **42.84%** (AIC = 11266.0).
* **Observação Teórica:** O alisamento exponencial gera previsões que acompanham o nível médio, mas falha em capturar as oscilações pontuais extremas provocadas pelas *merge windows* do kernel.

---

## 4. Modelação Box-Jenkins: Ganho Estrutural com Covariáveis Exógenas (Capítulo 4.4 — Slides 67 a 74)
* **Confronto Univariado vs. Multivariado:**
  * **ARIMA(0, 1, 2) Univariado:** REQM = **614.75**, EPAM = **36.78%** (AIC = 13359.32).
  * **ARIMAX(0, 1, 2) + Covariáveis:** REQM = **263.24**, EPAM = **10.76%** (AIC = 11589.33).
  * **Ganho Percentual:** A introdução das covariáveis de engenharia reduziu o REQM em **57.2%** e o EPAM de **36.78%** para **10.76%**!
* **Significância dos Coeficientes das Covariáveis (Slide 70):**
  * `unique_authors` (Contribuidores): Coeficiente = **+4.6594** ($p = 0.0000e+00$). Por cada novo contribuidor ativo na semana, o kernel integra em média ~4.7 commits adicionais.
  * `log_churn` (Code Churn): Coeficiente = **+97.6727** ($p = 7.5281e-30$). Forte impacto positivo e altamente significativo.
* **Tabelas para o Relatório:** Inserir `tab_04_sarimax_results.csv` e `tab_05_comparacao_econometrica.csv`.

---

## 5. Diagnóstico Residual e Validação de Ruído Branco (Capítulo 4.5 — Slides 55 e 70)
* **Teste de Ljung-Box (Validação de Não-Autocorrelação):**
  * Lag 1: $Q = 1.65$, $p\\text{-value} = 0.1989$ ($> 0.05$).
  * Lag 10: $Q = 17.82$, $p\\text{-value} = 0.0581$ ($> 0.05$).
  * **Conclusão:** O teste confirma a ausência de autocorrelação linear sistemática nos resíduos a 5% de significância. O modelo filtrou a estrutura temporal da série, restando **ruído branco (*white noise*)**.
* **Teste de Normalidade de Jarque-Bera:**
  * $p\\text{-value} = 0.00e+00$. A hipótese nula de normalidade estrita é rejeitada devido às caudas pesadas provocadas pelos picos pontuais de *merge windows*, fenómeno clássico documentado em engenharia de sistemas.
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
