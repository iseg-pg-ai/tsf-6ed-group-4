import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pandas_datareader.data as web
import numpy as np
 
# Ler os dados do IPC dos Estados Unidos (CPI data fromFRED)
start_date = '2010-01-01'
end_date = '2024-12-31'
inflation_data = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
plt.figure(figsize=(12, 8))
plt.plot(inflation_data.index, inflation_data, color='blue')
#plt.title("Índice de preços no consumidor (CPI) nos EUA (2010–2024)", fontsize=16)
plt.ylabel("IPC", fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()
 
# Calcular a taxa de inflação mensal (CPIpercentage change)
inflation_rate = inflation_data.pct_change().dropna() * 100
plt.plot(inflation_rate.index, inflation_rate, color='blue')
#plt.title("Taxa de inflação mensal nos EUA (2010–2024)", fontsize=16)
plt.ylabel("Taxa de inflação", fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()
 
# Dividir os dados em treino e teste e testar a estacionaridade (ADFtest)
from statsmodels.tsa.stattools import adfuller, acf, pacf
h = 12
n = len(inflation_rate)
train = inflation_rate.iloc[:-h].copy()
test = inflation_rate.iloc[-h:].copy()
 
# Teste ADF em níveis
adf_train = adfuller(train)
print("\nTeste ADF (taxa de inflação em níveis):")
print(f"Estatística ADF statistic = {adf_train[0]:.4f}, valor-p = {adf_train[1]:.4f}")
if adf_train[1] < 0.05:
print("A taxa de inflação é estacionária.")
else:
print("A taxa de inflação não é estacionária! Devemos aplicar diferenças.")
 
# Aplicar uma diferenciação simples
diff_train = train.diff().dropna()# primeira diferença
adf_train2 = adfuller(diff_train)
print("\nTeste ADF (taxa de inflação em diferenças simples):")
print(f"Estatística ADF statistic = {adf_train2[0]:.4f}, valor-p = {adf_train2[1]:.4f}")
 
if adf_train2[1] < 0.05:
print("A primeira diferença da taxa de inflação é estacionária.")
else:
print("A primeira diferença da taxa de inflação não é estacionária!Devemos aplicar segundas diferenças.")
 
# ACF e PACF da taxa de inflação em diferenças
nlags = 36
train_for_acf = train.diff().dropna()
series_vals = train_for_acf.values
n_train = len(train_for_acf)
acf_vals = acf(series_vals, nlags=nlags, fft=False)[1:]  # remove lag 0
pacf_vals = pacf(series_vals, nlags=nlags, method='ywm')[1:]
lags_arr = np.arange(1, nlags+1)
conf_level = 1.96 / np.sqrt(n_train)
 
fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].stem(lags_arr, acf_vals, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].axhline(0, color='black')
axes[0].axhline(conf_level, color='red', linestyle='--')
axes[0].axhline(-conf_level, color='red', linestyle='--')
axes[0].set_title("ACF")
axes[0].set_xlabel("Lag")
axes[0].set_ylim(-0.4, 0.4)
 
axes[1].stem(lags_arr, pacf_vals, linefmt='b-',markerfmt='bo', basefmt='k-')
axes[1].axhline(0, color='black')
axes[1].axhline(conf_level, color='red', linestyle='--')
axes[1].axhline(-conf_level, color='red', linestyle='--')
axes[1].set_title("PACF")
axes[1].set_xlabel("Lag")
axes[1].set_ylim(-0.4, 0.4)
 
plt.tight_layout()
plt.show()
 
# Etapa 2:
 
# Ajustar modelos ARIMA aos dados de treino (variável dependente: inflation_rate)

d = 1

models_spec = {

    f'ARIMA(0,{d},2)': (0, d, 2),

    f'ARIMA(1,{d},1)': (1, d, 1),

    f'ARIMA(4,{d},0)': (4, d, 0)

}

fitted_models = {}

model_summaries = {}

train_series = train.squeeze()  # converte de DataFrame para Series se necessário

train_series.name = "inflation_rate"

for name, order in models_spec.items():

    try:

        print(f"\nModelo ajustado {name} para os dados de treino:")

        mod = ARIMA(train_series, order=order)

        res = mod.fit()

        fitted_models[name] = res

        model_summaries[name] = res.summary()

        print(res.summary())

    except Exception as e:

        print(f"Falha ao ajustar o modelo {name}: {e}")
 
 
 
 
