# Install the pandas_datareader
# pip install pandas_datareader
 
import warnings
warnings.filterwarnings("ignore")
 
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
# 1. Ler os dados do IPC dos Estados Unidos (CPI data from FRED)
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
 
# 2. Calcular a taxa de inflação mensal (CPI percentage change)
inflation_rate = inflation_data.pct_change().dropna() * 100  # Percentage change
 
 
plt.figure(figsize=(12, 8))
plt.plot(inflation_rate.index, inflation_rate, color='blue')
#plt.title("Taxa de inflação mensal nos EUA (2010–2024)", fontsize=16)
plt.ylabel("Taxa de inflação", fontsize=14)
plt.xlabel("Data", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()
 
# 3. Dividir os dados em treino e teste e testar a estacionaridade (ADF test)
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
diff_train = train.diff().dropna()  # primeira diferença
adf_train2 = adfuller(diff_train)
print("\nTeste ADF (taxa de inflação em diferenças simples):")
print(f"Estatística ADF statistic = {adf_train2[0]:.4f}, valor-p = {adf_train2[1]:.4f}")
 
if adf_train2[1] < 0.05:
    print("A primeira diferença da taxa de inflação é estacionária.")
else:
    print("A primeira diferença da taxa de inflação não é estacionária! Devemos aplicar segundas diferenças.")
 
# 4. ACF e PACF da taxa de inflação em diferenças
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
 
axes[1].stem(lags_arr, pacf_vals, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1].axhline(0, color='black')
axes[1].axhline(conf_level, color='red', linestyle='--')
axes[1].axhline(-conf_level, color='red', linestyle='--')
axes[1].set_title("PACF")
axes[1].set_xlabel("Lag")
axes[1].set_ylim(-0.4, 0.4)
 
plt.tight_layout()
plt.show()
 
# 5. Ajustar modelos ARIMA aos dados de treino (variável dependente: inflation_rate)
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

# 6. Previsões para o período de teste e cálculo das métricas 
metrics = {}
predictions = {}
 
for name, res in fitted_models.items():
    fc = res.get_forecast(steps=len(test))
    mean_pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    mean_pred.index = test.index
    ci.index = test.index
    predictions[name] = {'mean': mean_pred, 'ci': ci}
    y_true = test.values
    y_pred = mean_pred.values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true))) * 100
    metrics[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape}
    print(f"\nMétricas para {name}: MSE={mse:.3f}, RMSE={rmse:.3f},MAE={mae:.3f}, MAPE={mape:.2f}%")
 
metrics_df = pd.DataFrame(metrics).T
print("\nResumo das métricas de desempenho preditivo:")
print(metrics_df.round(3))
 
# 7. Gráficos comparativos das previsões vs histórico (para cada modelo)
for name, preds in predictions.items():
    mean_pred = preds['mean']
    ci = preds['ci']
    plt.figure(figsize=(12,8))
    plt.plot(train.index, train, label='Treino', color='blue', linewidth=1.2)
    plt.plot(test.index, test, label='Teste (observado)', color='green', linewidth=1.2)
    plt.plot(mean_pred.index, mean_pred, label=f'Previsão com o modelo {name}', color='red', linestyle='--', linewidth=2)
    plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2)
    #plt.title(f'Previsões da taxa de inflação versus valores reais: modelo {name}')
    plt.legend()
    plt.xlabel('Data')
    plt.ylabel('Taxa de inflação (%)')
    plt.grid(False)
    plt.show()
 
# 8. Escolher o melhor modelo à luz do MSE e prever próximos 12 (futuro) 
best_model_name = metrics_df['MSE'].idxmin()
print(f"\nMelhor modelo segundo o critério do MSE: {best_model_name} (MSE={metrics_df.loc[best_model_name,'MSE']:.4f}%)")
 
best_order = models_spec[best_model_name]
print(f"Reajustando {best_model_name} na série completa com ordem {best_order} para prever os próximos {h} passos.")
final_model = ARIMA(inflation_rate, order=best_order).fit()
 
future_fc = final_model.get_forecast(steps=h)
future_mean = future_fc.predicted_mean
future_ci = future_fc.conf_int(alpha=0.05)
 
last_index = inflation_rate.index[-1]
future_index = pd.date_range(start=last_index + pd.offsets.MonthBegin(1), periods=h, freq='MS')
 
future_mean.index = future_index
future_ci.index = future_index
 
print("\nPrevisões para os próximos 12 períodos (melhor modelo):")
print(future_mean)
 
plt.figure(figsize=(12,8))
plt.plot(inflation_rate.index, inflation_rate, label='Taxa de inflação', color='blue', linewidth=1.2)
plt.plot(future_mean.index, future_mean, label=f'Previsões a {h} meses', color='red', linestyle='--', linewidth=2)
plt.fill_between(future_ci.index, future_ci.iloc[:,0], future_ci.iloc[:,1], alpha=0.2)
#plt.title(f'Previsão para os próximos {h} períodos: Modelo {best_model_name}')
plt.legend()
plt.xlabel('Data')
plt.ylabel('Taxa de inflação (%)')
plt.grid(False)
plt.show()
