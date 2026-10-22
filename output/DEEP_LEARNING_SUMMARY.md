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

---

## 2. Arquiteturas Implementadas (Slides 81 a 105)
* **1D-CNN (Slides 81-84, 87, 92-94):** Duas camadas convolucionais (32 e 64 filtros, kernel=3) com MaxPool, seguidas de camada densa (64 neurónios) e Dropout(0.2). Ilustra a lição do **Slide 94**: estruturas mais parcimoniosas podem superar redes profundas em séries temporais.
* **SimpleRNN (Slides 95-96, 101):** Rede recorrente elementar com estado oculto $h_t = \tanh(W_x x_t + U_h h_{t-1} + b)$. Serve de linha de base recorrente, evidenciando o desvanecimento do gradiente.
* **LSTM (Slides 97-98, 102):** Rede recorrente com célula de memória ($c_t$) e três portas: esquecimento ($f_t$), entrada ($i_t$) e saída ($o_t$).
* **GRU (Slides 99, 103):** Variante simplificada da LSTM com duas portas: atualização ($z_t$) e reposição ($r_t$). Habitualmente mais rápida a convergir e menos propensa a sobreajustamento.

---

## 3. Regime Univariado vs. Multivariado (Slides 106-108)
* **Regime Univariado ($F = 1$):** Apenas o histórico de commits diários ($Y_{d,t}$).
* **Regime Multivariado ($F = 6$):** Commits diários combinados com:
  * Contribuidores ativos diários (`unique_authors`)
  * Code churn em escala logarítmica (`log_churn`)
  * Ficheiros alterados (`files_changed`)
  * Codificação harmónica do dia da semana ($\sin$ e $\cos$ do ciclo $s=7$)

### Comparação Univariado vs. Multivariado (REQM):
  * **1D-CNN:** Univariado REQM=144.24 (EPAM=65.04%) vs. Multivariado REQM=130.88 (EPAM=56.09%) -> Ganho de +9.3%
  * **SimpleRNN:** Univariado REQM=142.24 (EPAM=73.41%) vs. Multivariado REQM=136.14 (EPAM=67.00%) -> Ganho de +4.3%
  * **LSTM:** Univariado REQM=164.90 (EPAM=133.67%) vs. Multivariado REQM=129.94 (EPAM=61.15%) -> Ganho de +21.2%
  * **GRU:** Univariado REQM=152.87 (EPAM=99.66%) vs. Multivariado REQM=132.12 (EPAM=53.91%) -> Ganho de +13.6%

* **Melhor Modelo Global:** **LSTM Multivariado** com REQM = 129.94 e EPAM = 61.15%.
* **Interpretação (Slide 94 e 111):** As redes recorrentes (LSTM e GRU) beneficiam significativamente da informação multivariada, capturando não-linearidades nas interações entre volume de commits, disponibilidade de contribuidores e volatilidade de código.

---

## 4. Tabela de Benchmark (tab_07_dl_benchmark_metricas.csv)
* **Métricas Reportadas (Slide 13):**
  * **REQM** (Raiz do Erro Quadrático Médio / RMSE)
  * **EAM** (Erro Absoluto Médio / MAE)
  * **EPAM** (Erro Percentual Absoluto Médio / MAPE)
  * **EPAMs** (Erro Percentual Absoluto Médio Simétrico / sMAPE)
* **Previsão Estática vs. Dinâmica (Slide 108):** As métricas reportadas correspondem a previsões estáticas (passo-a-passo), onde o modelo utiliza os valores reais observados no passado imediato para prever o passo seguinte.

---

## 5. Imagens a Incluir no Documento
1. **`fig_06_curvas_aprendizagem.[png|svg]`:** Inserir no Capítulo 5 (Curvas de perda de treino e validação para as 4 arquiteturas em ambos os regimes).
2. **`fig_07_previsoes_dl_teste.[png|svg]`:** Inserir no Capítulo 5/6 (Painel duplo: Univariado vs. Multivariado com sobreposição das previsões no período de teste).

---

## 6. Prevenção de Sobreajustamento (Slide 94 e 111)
* **Dropout (0.2):** Aplicado na camada densa da 1D-CNN para regularização.
* **Monitorização de Validação:** As curvas de perda de validação são monitorizadas em cada época para detetar divergência (sobreajustamento).
* **Observação do Professor Jorge Caiado (Slide 94):** "Em séries temporais, arquiteturas mais simples frequentemente superam redes profundamente parametrizadas devido ao sobreajustamento." Os resultados devem ser confrontados criticamente com esta afirmação.
