# Síntese da Modelação com Aprendizagem Profunda (Deep Learning)
**Documento de Apoio à Redação para o Elemento B (Capítulo 5 e 6 do Relatório)**  
*Gerado automaticamente por `scripts/03_deep_learning.py`*

---

## 1. Formulação por Janelas Deslizantes e Prevenção de Fuga de Informação (Slide 85)
* **Série Utilizada:** Diária (`kernel_daily_ts.csv`, $N = 7.815$ observações, frequência `D`).
* **Justificação da Escolha:** As redes neuronais treinadas por descida de gradiente (*backpropagation*) necessitam de milhares de pontos para convergir sem sobreajustar. A frequência diária contém micro-sazonalidade intra-semanal ($s = 7$, com quedas sistemáticas aos fins de semana), ideal para testar a capacidade das redes.
* **Janela Deslizante ($W = 28$ dias):** Cada observação de treino é uma matriz $X \in \mathbb{R}^{B \times 28 \times F}$, onde $B$ é o tamanho do lote, $28$ são os dias de histórico (4 semanas completas) e $F$ é o número de características.
* **Prevenção de Data Leakage:**
  * Divisão estritamente cronológica (80% treino, 20% teste).
  * `MinMaxScaler` ajustado **exclusivamente na amostra de treino** e aplicado à amostra de teste (sem contaminação).
  * Divisão interna treino/validação (90:10) para monitorização de sobreajustamento durante o treino.
* **Amostra de treino comum aos três regimes:** os 62 dias anteriores ao primeiro *tag* de release não têm fase conhecida e são removidos do treino em **todos** os regimes, ficando 6.190 dias. Se só o regime com a fase os perdesse, os outros treinariam com mais 62 janelas e a comparação entre regimes deixaria de ser justa. O conjunto de teste (2022-2026) não tem valores em falta e não é afetado.
* **Reprodutibilidade:** `torch.manual_seed(42)` no início de cada regime, para que a mesma arquitetura arranque dos mesmos pesos nos três regimes e a diferença de métricas meça o efeito das features e não o sorteio da inicialização. Sem seed, duas execuções do mesmo script davam métricas diferentes; a variabilidade entre seeds foi medida em ±0,2 de REQM, muito abaixo das diferenças discutidas.

---

## 2. Arquiteturas Implementadas (Slides 81 a 105)
* **1D-CNN (Slides 81-84, 87, 92-94):** Duas camadas convolucionais (32 e 64 filtros, kernel=3) com MaxPool, seguidas de camada densa (64 neurónios) e Dropout(0.2). Ilustra a lição do **Slide 94**: estruturas mais parcimoniosas podem superar redes profundas em séries temporais.
* **SimpleRNN (Slides 95-96, 101):** Rede recorrente elementar com estado oculto $h_t = \tanh(W_x x_t + U_h h_{t-1} + b)$. Serve de linha de base recorrente, evidenciando o desvanecimento do gradiente.
* **LSTM (Slides 97-98, 102):** Rede recorrente com célula de memória ($c_t$) e três portas: esquecimento ($f_t$), entrada ($i_t$) e saída ($o_t$).
* **GRU (Slides 99, 103):** Variante simplificada da LSTM com duas portas: atualização ($z_t$) e reposição ($r_t$). Habitualmente mais rápida a convergir e menos propensa a sobreajustamento.

---

## 3. Regimes de Features (Slides 106-108)
* **Regime Univariado ($F = 1$):** Apenas o histórico de commits diários ($Y_{d,t}$).
* **Regime Multivariado ($F = 6$):** Commits diários combinados com:
  * Contribuidores ativos diários (`unique_authors`)
  * Code churn em escala logarítmica (`log_churn`)
  * Ficheiros alterados (`files_changed`)
  * Codificação harmónica do dia da semana ($\sin$ e $\cos$ do ciclo $s=7$)
* **Regime Multivariado + Release ($F = 8$):** As seis colunas anteriores mais a **fase do ciclo de release**, codificada como duas dummies com a *merge window* como referência (`fase_rc_inicial` = semanas 2 a 5 após a release; `fase_rc_final` = semana 6 ou mais). É exatamente a codificação da especificação ARIMAX vencedora do script 02, para que as duas famílias de modelos leiam a mesma variável da mesma maneira.
  * É a **única covariável conhecida antes de a semana começar**: as restantes são medidas na mesma semana do alvo e só servem para *nowcasting*. O seu valor está menos na magnitude do ganho e mais em ser o único bloco de features que sustenta uma previsão em tempo real (Capítulo 7).
  * Justificação, alternativas testadas e variância entre seeds: `scripts/analise_covariaveis_release/README.md`.

### Comparação Univariado vs. Multivariado (REQM):
{{ comparacao_uni_multi }}

### Comparação Multivariado vs. Multivariado + Release (REQM):
{{ comparacao_release }}
* **Efeito da fase de release:** {{ efeito_release }}.
* **Melhor Modelo Global:** **{{ melhor_modelo }}** com REQM = {{ melhor_reqm }} e EPAM = {{ melhor_epam }}%.
* **Interpretação (Slide 94 e 111):** As redes recorrentes (LSTM e GRU) beneficiam significativamente da informação multivariada, capturando não-linearidades nas interações entre volume de commits, disponibilidade de contribuidores e volatilidade de código. O ganho da fase de release é muito menor do que o do bloco de autores e churn — como seria de esperar, porque descreve o calendário e não a intensidade do trabalho —, mas é o único que não exige conhecer o futuro para prever o presente.

---

## 4. Tabela de Benchmark (tab_07_dl_benchmark_metricas.csv)
* **Métricas Reportadas (Slide 13):**
  * **REQM** (Raiz do Erro Quadrático Médio / RMSE)
  * **EAM** (Erro Absoluto Médio / MAE)
  * **EPAM** (Erro Percentual Absoluto Médio / MAPE)
  * **EPAMs** (Erro Percentual Absoluto Médio Simétrico / sMAPE)
* **Previsão Estática vs. Dinâmica (Slide 108):** As métricas reportadas correspondem a previsões estáticas (passo-a-passo), onde o modelo utiliza os valores reais observados no passado imediato para prever o passo seguinte.
* **Tabela Consolidada de Deep Learning (`deep_learning.csv`):**
  * *Conteúdo:* Avaliação comparativa das 12 variantes neuronais (1D-CNN, SimpleRNN, LSTM e GRU em três regimes: univariado, multivariado e multivariado com a fase do ciclo de release) com métricas de teste (REQM, EAM, EPAM e EPAMs).
  * *Por que considerar:* Permite ao Elemento B analisar e discutir os resultados de Deep Learning de forma independente, facilitando a elaboração do Capítulo 5 antes da consolidação mestre no Capítulo 6.
  * *Valor analítico:* Demonstra empiricamente o ganho de precisão alcançado com variáveis exógenas e harmónicas em cada classe de rede neuronal (recorrentes com portas vs. convoluções temporais), e isola o contributo da fase do ciclo de release.
* **Nota obrigatória sobre o EPAM nesta frequência:** o EPAM é instável em dados diários porque muitos dias têm poucos commits (fins de semana e feriados), o que infla o erro relativo; o EPAMs, por ser simétrico, mostra um padrão mais estável. Ao acrescentar a fase do ciclo de release, o **REQM e o EAM melhoram nas quatro arquiteturas, mas o EPAM piora nas quatro** e o EPAMs é misto — a comparação linha a linha está na `tab_07`. O regime com a fase foi escolhido pelo **REQM**, que é o critério de seleção usado em todo o trabalho, e a codificação por duas dummies é a melhor das três nessa métrica, no EAM e no EPAMs (variante com uma única dummy de *merge window* testada e registada em `scripts/analise_covariaveis_release/`). Declarar esta troca é preferível a escondê-la: o relatório deve dizer que o modelo com a fase acerta melhor em erro absoluto e pior em erro relativo nos dias de baixo volume.
* **Tabela de Trajetória de Aprendizagem (`curvas_aprendizagem.csv`):**
  * *Conteúdo:* Histórico granular da perda época a época (1 a 15) de treino (`Train_Loss`) e validação (`Val_Loss`) em escala MSE para as 12 variantes (4 arquiteturas x 3 regimes).
  * *Por que considerar:* Fornece o suporte numérico exato subjacente à Figura 6 (`fig_06_curvas_aprendizagem`), permitindo ao redator citar taxas de decaimento do erro e valores finais exatos.
  * *Valor analítico:* Permite fundamentar a análise de sobreajustamento (*overfitting*) e subajustamento (*underfitting*), identificando a época em que o erro de validação estabilizou, reforçando a profundidade científica do relatório.

---

## 5. Imagens a Incluir no Documento
1. **`fig_06_curvas_aprendizagem.[png|svg]`:** Inserir no Capítulo 5 (grelha de 3 linhas por 4 colunas: as 4 arquiteturas em cada um dos 3 regimes).
2. **`fig_07_previsoes_dl_teste.[png|svg]`:** Inserir no Capítulo 5/6 (três painéis, um por regime, com as previsões das 4 arquiteturas sobrepostas no período de teste).

---

## 6. Prevenção de Sobreajustamento (Slide 94 e 111)
* **Dropout (0.2):** Aplicado na camada densa da 1D-CNN para regularização.
* **Monitorização de Validação:** As curvas de perda de validação são monitorizadas em cada época para detetar divergência (sobreajustamento).
* **Observação do Professor Jorge Caiado (Slide 94):** "Em séries temporais, arquiteturas mais simples frequentemente superam redes profundamente parametrizadas devido ao sobreajustamento." Os resultados devem ser confrontados criticamente com esta afirmação.

---

## 7. Guia de Tabelas Complementares Geradas (em `output/tables/`)
Tabelas geradas sem prefixo numérico para apoio e consulta detalhada da equipa de redação:
1. **`deep_learning.csv`:** Síntese métrica de todas as arquiteturas neuronais nos três regimes de features (Capítulo 5).
2. **`curvas_aprendizagem.csv`:** Dados brutos da evolução da perda por época (treino vs. validação) para diagnóstico de sobreajustamento (Capítulo 5).
