# Análise das variáveis do ciclo de release

Registo da experiência que levou à inclusão da **fase do ciclo de release** na especificação
ARIMAX vencedora do `scripts/02_classical_econometrics.py`.

A implementação que conta para o relatório é a do script 02. Esta pasta existe para que a
decisão possa ser auditada e defendida: mostra o que foi testado, o que foi rejeitado e porquê.

---

## 1. A pergunta

O `dataset/Extractor/kernel_raw_commits.csv` é enriquecido, no fim da extração, com três
variáveis que descrevem em que ponto do ciclo de release cada semana cai — `weeks_since_release`,
`is_merge_window` e `rc_week`. Até agora nenhum modelo as usava. A pergunta era dupla:

1. Acrescentam poder preditivo à especificação ARIMAX que já usa contribuidores e code churn?
2. Em caso afirmativo, qual é a codificação defensável — contagem linear, dummy de merge window,
   ou dummies por fase?

## 2. Porque é que estas variáveis são candidatas plausíveis

A análise exploratória (script 01) já mostrava que o ciclo de release, e não o calendário,
produz as semanas extremas:

- 112 releases finais no período, com intervalo mediano de 9.9 semanas entre releases (mínimo 8.0,
  máximo 15.3, desvio-padrão 1.3) — o ciclo ronda as dez semanas, mas não é um relógio.
- A merge window (semanas 0 e 1 após o fecho de uma release) é a fase mais calma no kernel
  moderno: cerca de 1.0 mil commits por semana contra 1.8 mil nas restantes. O motivo é o
  momento em que o commit é datado: o trabalho de features é commitado no ramo do maintainer,
  não durante a janela de merge.
- Dos 40 resíduos com `|Z| > 2`, nenhum cai em rc1-rc5. E o sinal inverte-se em 2015: entre
  2005 e 2014 há 19 picos positivos na merge window contra 4 no fim do ciclo; a partir de 2015
  há 0 picos positivos e todas as 6 quedas negativas ocorrem na merge window, com 11 picos
  positivos a transitar para o fim do ciclo.

Ou seja: a variável descreve um mecanismo institucional, não uma correlação espúria, e é
conhecida **antes** de a semana começar — ao contrário de `unique_authors` e `log_churn`, que só
são observáveis depois de a semana fechar.

## 3. Método: comparação como-para-como

Para que a comparação seja honesta:

- **Mesma amostra de treino.** As variáveis de release não existem nas 10 primeiras semanas de
  2005 (anteriores ao primeiro tag). Todas as especificações Box-Jenkins do script 02 foram
  estimadas nas mesmas **883** semanas de treino (de 1 117 observações semanais, 893 formam a
  divisão 80:20 e as 10 primeiras não têm release conhecida). Comparar modelos estimados em
  amostras diferentes tornaria a escolha do vencedor arbitrária.
- **Mesma amostra de teste.** As **224** semanas finais, nunca usadas na estimação.
- **Ordem fixa `(0,1,2)`.** É a ordem escolhida pela FAC/FACP no script 02. Manter a ordem fixa
  isola o efeito de acrescentar variáveis do efeito de mudar a estrutura ARIMA.
- **Sem aleatoriedade.** Divisão cronológica, sem shuffling. O ajuste é determinístico.
- **Apenas informação passada.** As variáveis vêm de `merge_asof(direction="backward")` sobre
  releases finais; não há data da release seguinte nem tags `-rc`, portanto não há
  *look-ahead bias*.

O código é este vizinho: [`experimento_covariaveis_release.py`](experimento_covariaveis_release.py).

## 4. Resultados

SARIMAX(0, 1, 2) com tendência constante implícita, 883 semanas de treino, 224 de teste.

| Especificação | Exógenas | REQM | EPAM (%) | AIC | Ljung-Box 15 (p) | Ljung-Box 20 (p) |
|---|---:|---:|---:|---:|---:|---:|
| Covariáveis (controlo) | 2 | 261.59 | 10.74 | 11464.54 | 0.0010 | 0.0054 |
| + `is_merge_window` (dummy) | 3 | 260.81 | 10.40 | 11453.27 | 0.0014 | 0.0094 |
| + `weeks_since_release` (contagem) | 3 | 260.78 | 10.83 | 11462.92 | 0.0021 | 0.0077 |
| + `rc_week` (contagem) | 3 | 259.48 | 10.80 | 11459.70 | 0.0028 | 0.0087 |
| Só fase (sem covariáveis) | 2 | 577.37 | 34.72 | 13188.22 | 0.0000 | 0.0000 |
| **+ Fase: 2 dummies (merge = referência)** | **4** | **254.00** | **10.08** | **11423.96** | **0.0070** | **0.0152** |
| + Fase: 3 dummies (sem referência) | 5 | 254.00 | 10.08 | 11425.95 | 0.0074 | 0.0160 |
| + Fase x Era (interação 2015+) | 6 | 254.03 | 10.15 | 11427.56 | 0.0049 | 0.0121 |

Controlo: REQM 261.59. A especificação adotada reduz o REQM em **2.9%** (261.59 para 254.00) e o
AIC em **40.6 pontos**, com duas exógenas adicionais. Os números em bruto estão em
[`resultados_experimento.csv`](resultados_experimento.csv).

## 5. Leitura de cada linha

- **`is_merge_window`, `weeks_since_release` e `rc_week` isoladamente: ganho marginal.**
  Entre 259.48 e 260.81, ou seja 0.3% a 0.8% de REQM. Uma dummy binária ou uma contagem linear
  não descrevem a não-linearidade do ciclo: a atividade não decresce monotonicamente com as
  semanas desde a release, tem um perfil em U. É por isso que a codificação por fases é a única
  que se destaca.
- **Só fase, sem covariáveis: muito pior (577.37).** A fase não substitui autores e churn —
  complementa-os. Confirma que o ganho grande da Secção 4 do resumo econométrico (58.7% face ao
  ARIMA univariado) continua a vir dos contribuidores e do churn; a fase acrescenta o resto.
- **Fase com 2 dummies: vencem.** REQM 254.00, EPAM 10.08%, AIC 11423.96.
- **Fase com 3 dummies: mesma precisão, AIC pior.** O REQM é idêntico ao centésimo (254.00)
  porque o espaço de colunas é o mesmo, mas o AIC sobe 1.99 pontos — exatamente o custo do
  parâmetro redundante. Os três dummies são exaustivos e colineares com a constante que o
  SARIMAX inclui, pelo que um dos coeficientes fica por identificar. Uma tabela com coeficientes
  não identificáveis não é defensável no relatório; ficam dois dummies, com a merge window como
  referência.
- **Fase x Era: rejeitada.** A interação com a era moderna não melhora a precisão (254.03 contra
  254.00) e piora o AIC (11427.56). A inversão de sinal em 2015, documentada na análise
  exploratória, não impede uma codificação por fase na amostra completa: a interação foi testada
  e não paga o custo de dois parâmetros. Fica registada como heterogeneidade que os coeficientes
  médios escondem.

## 6. O que ficou implementado

- `scripts/02_classical_econometrics.py`, `carregar_dados_e_dividir`: construção de
  `fase_merge`, `fase_rc_inicial` (semanas 2 a 5) e `fase_rc_final` (6 ou mais). As semanas sem
  release conhecida ficam `NaN`, nunca 0 — um "desconhecido" não pode ser codificado como merge
  window.
- `scripts/02_classical_econometrics.py`, `modelar_arima_arimax`: restrição de todas as
  especificações Box-Jenkins à amostra comum de 883 semanas e nova especificação
  `ARIMAX(0, 1, 2) + Covariáveis + Fase de Release`, que passou a vencedora por REQM.
- `output/tables/tab_04_sarimax_results.csv`: coeficientes das duas dummies e respetivos
  p-values. `output/tables/tab_05_comparacao_econometrica.csv`: a nova linha na comparação
  global.
- `output/ECONOMETRICS_SUMMARY.md`, Secção 4: coeficientes da fase e contributo incremental
  (2.9% de REQM face à especificação sem a fase, na mesma amostra); Secção 5: leitura do
  Ljung-Box adaptada ao facto de a fase já estar no modelo.
- `scripts/01_eda_decomposition.py`: as três variáveis passaram a entrar na matriz de correlação
  (`fig_00_matriz_correlacao_pearson` e `fig_00_matriz_correlacao_spearman`, bloco do ciclo de
  release delimitado a preto).
- Script 03 (deep learning, série diária) usa as mesmas duas dummies como terceiro regime de features — ver a Secção 9.

## 7. Limitações a declarar

1. **O ganho global é um nowcast, a fase não.** `unique_authors` e `log_churn` são medidos na
   mesma semana do alvo; só a fase é conhecida antes de a semana começar. A fase é, por isso, a
   única covariável que serve para previsão em tempo real, e o seu contributo isolado é de 2.9%
   de REQM, não de 58.7%.
2. **Autocorrelação residual não desaparece.** O Ljung-Box passa de p = 0.0010 para 0.0070 no
   desfasamento 15 e de 0.0054 para 0.0152 no 20: melhora, mas continua a rejeitar a 5%. A dummy
   de fase capta o nível médio de cada fase, não a cadência irregular dos ciclos (8 a 15
   semanas), pelo que a estrutura de médio prazo fica por explicar. É a limitação a assumir no
   relatório, não a esconder.
3. **Coeficientes médios de 2005 a 2026.** O sinal do efeito inverte-se por volta de 2015; o
   coeficiente da fase é uma média que esconde essa heterogeneidade. A interação com a era foi
   testada e rejeitada em precisão (Secção 5).
4. **Leitura em variação, não em nível.** Com `d = 1`, cada coeficiente de fase compara o ritmo
   semanal dessa fase com o da merge window, não o número de commits. A frase "a merge window
   tem 800 commits a menos" seria um erro de leitura.
5. **Ambas as dummies são necessárias?** `fase_rc_inicial` é significativa (p < 0.0001);
   `fase_rc_final` não é (p = 0.1893). Mantêm-se as duas porque a codificação é categórica e
   retirar uma fase devolveria a não-linearidade ao termo de erro; o p-value está reportado na
   `tab_04` e o relatório deve dizê-lo.

## 8. Como reproduzir

A partir da raiz do repositório:

```bash
uv run scripts/analise_covariaveis_release/experimento_covariaveis_release.py
```

O script imprime a tabela da Secção 4 e regrava `resultados_experimento.csv`. Os números são
determinísticos e coincidem com os do script 02 para a especificação adotada (REQM 254.00, AIC
11423.96).

O experimento de deep learning é independente e mais lento (cerca de 6 minutos, 20 treinos):

```bash
uv run scripts/analise_covariaveis_release/experimento_release_dl.py
```

Para regenerar os artefactos do relatório afetados:

```bash
uv run scripts/02_classical_econometrics.py
uv run scripts/03_deep_learning.py
uv run scripts/04_benchmark_consolidado.py
```

## 9. Aplicação no ramo de deep learning (script 03)

A série usada pelas redes é a **diária** (7.815 dias), com janela de 28 dias e alvo em commits
diários. A fase é a mesma variável, com a mesma codificação (duas dummies, *merge window* como
referência), como **terceiro regime de features** ao lado do univariado (F=1) e do multivariado
(F=6). Duas decisões específicas desta frequência:

- **Amostra de treino comum.** Os 62 dias anteriores ao primeiro *tag* não têm fase conhecida e
  são removidos do treino em **todos** os regimes, ficando 6.190 dias (6.162 janelas). Se só o
  regime F=8 os perdesse, os outros treinariam com mais 62 janelas e a comparação deixaria de ser
  justa. O conjunto de teste (1.563 dias, 2022-2026) não tem valores em falta.
- **Seed.** `torch.manual_seed(42)` reposta no início de cada regime, para que a mesma
  arquitetura arranque dos mesmos pesos nos três regimes. Sem isso, a diferença entre F=6 e F=8
  confundir-se-ia com o sorteio da inicialização.

### 9.1 Escolha da codificação (LSTM, 5 seeds)

Quatro conjuntos de features, mesma semente por variante, média e desvio-padrão em 5 seeds:

| Features | F | REQM | EAM | EPAM | EPAMs |
|---|---:|---|---|---|---|
| Sem a fase | 6 | 130.29 ± 0.17 | 79.88 ± 0.47 | 61.71 ± 1.61 | 37.47 ± 0.26 |
| Só `weeks_since_release` | 7 | 130.35 ± 0.20 | 80.33 ± 0.31 | 63.74 ± 1.43 | 37.53 ± 0.04 |
| `weeks_since_release` + `is_merge_window` | 8 | 128.80 ± 0.17 | 78.72 ± 0.40 | 60.29 ± 1.05 | 37.45 ± 0.46 |
| Duas dummies (adotada) | 8 | **127.56 ± 0.18** | **78.37 ± 0.65** | 64.80 ± 2.00 | **36.91 ± 0.28** |

A variabilidade entre seeds é de ±0.2 no REQM, uma ordem de grandeza abaixo das diferenças em
causa, pelo que estas não são ruído de inicialização. Leitura:

- **A contagem sozinha não serve.** `weeks_since_release` isolada não melhora nada (130.35 contra
  130.29): a rede já vê o nível das últimas quatro semanas na janela e não consegue extrair do
  contador uma informação que não esteja lá.
- **As duas codificações com a flag/estrutura de fase funcionam**, e a diferença entre elas é
  pequena mas consistente: as dummies ganham no REQM (127.56 contra 128.80), no EAM e no sMAPE, e
  perdem no EPAM (64.80 contra 60.29).
- **Adotaram-se as dummies** por duas razões: são a codificação da especificação ARIMAX
  vencedora, o que permite dizer no relatório que as duas famílias leem a mesma variável da mesma
  forma; e o REQM é o critério de seleção usado em todo o trabalho. A perda no EPAM fica
  declarada — o EPAM é instável nesta frequência porque muitos dias têm poucos commits.

`experimento_release_dl.py` reproduz esta tabela e grava `resultados_experimento_dl.csv` com as
quatro métricas (REQM, EAM, EPAM e EPAMs) por variante.

### 9.2 Resultado no pipeline (`tab_07`)

12 redes, 15 épocas, $\mathrm{seed} = 42$, REQM fora da amostra no mesmo período de teste.

| Arquitetura | F=1 | F=6 | F=8 (com a fase) | Ganho F=6 → F=8 |
|---|---:|---:|---:|---:|
| 1D-CNN | 149.84 | 130.26 | **125.68** | +3.5% |
| SimpleRNN | 142.13 | 136.31 | 132.69 | +2.7% |
| LSTM | 164.68 | 129.94 | 126.99 | +2.3% |
| GRU | 143.21 | 132.15 | 127.49 | +3.5% |

- **A fase melhora as quatro arquiteturas**, em média 3.0% de REQM, e o melhor modelo de deep
  learning passa a ser o `1D-CNN Multivariado + Release` (125.68), à frente do melhor modelo sem
  a fase na mesma execução (`LSTM Multivariado`, 129.94).
- **O ganho é muito menor do que o do bloco de autores e churn** (13% a 21%), e isso é o esperado:
  a fase descreve o calendário, não a intensidade do trabalho.
- **Em erro absoluto melhora em todas**: o EAM cai nas quatro arquiteturas. Em erro relativo é o
  contrário: o EPAM **piora** nas quatro (por exemplo, 54.67% → 58.20% na 1D-CNN), e o sMAPE é
  misto. A escolha é feita pelo REQM, como em todo o trabalho, mas esta troca tem de ser
  declarada no relatório em vez de omitida.
- **O valor da variável não está no tamanho do ganho.** É a única feature conhecida **antes** de a
  semana começar; todas as outras são medidas na mesma semana do alvo e só servem para
  *nowcasting*. Para a discussão de cenários futuros (Capítulo 7), é a única exógena que não exige
  prever o futuro de outra variável.
