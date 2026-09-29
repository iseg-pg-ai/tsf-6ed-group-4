import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwintersimport ExponentialSmoothing
from sklearn.metricsimport mean_squared_error, mean_absolute_error
 
# Dados (para ilustração, considerámos a 1ª observação como sendo a correspondente à 1ª semana de janeiro de 2000)
data = {'Data': pd.date_range(start='2000-01', periods=161, freq='W'),
'Vendas': [45.9, 45.4, 42.8, 34.4, 31.9, 36.6, 39.2, 41.4, 40.3, 43.1, 43.2, 41.2, 38.4, 38.3, 41.9, 37.1, 34.5, 31.3, 30.2, 28.3, 25.9, 26.6, 26.2, 29.0, 34.8, 36.8, 37.2, 41.7, 41.2, 40.7, 39.5, 40.4, 38.0, 35.6, 33.9, 35.2, 41.8, 42.4, 38.9, 42.1, 41.7, 39.2, 38.5, 42.5, 47.9, 48.6, 52.0, 53.5, 53.5, 52.9, 53.4, 52.8, 51.4, 52.5, 52.4, 51.5, 51.7, 53.3, 55.4, 56.9, 60.0, 60.8, 62.3, 62.6, 63.1, 62.8, 64.7, 66.3, 63.0, 65.5, 70.6, 76.0, 80.1, 78.6, 78.3, 78.1, 73.6, 68.8, 64.4, 62.4, 61.1, 63.1, 65.3, 68.3, 72.5, 73.2, 72.9, 70.5, 69.4, 68.2, 69.3, 72.3, 73.5, 70.3, 68.3, 64.1, 62.5, 62.6, 60.4, 61.1, 64.7, 65.1, 61.5, 64.2, 67.8, 66.8, 64.1, 66.4,
68.0, 71.0, 76.9, 84.1, 85.9, 85.2, 86.2, 85.7, 81.3, 75.9, 75.0, 72.5, 69.6, 67.3, 69.8, 72.2, 75.2, 77.2, 76.8, 72.4, 69.4, 68.7, 65.1, 64.4, 64.2, 63.2, 62.1, 65.8, 73.7, 77.1, 76.0, 74.6, 70.6, 67.5, 67.9, 68.9,67.8, 65.1, 65.0, 67.6, 67.9, 66.5, 68.2, 71.7, 71.3, 68.9, 70.0, 73.1, 69.1, 67.3, 72.9, 78.6, 82.3]}
 
df = pd.DataFrame(data)
 
# Dividir dados em treino e teste
train_size = int(0.80 * len(df))
train, test = df[:train_size], df[train_size:]
 
# Método de alisamento exponencial de Holt com escolha ótima das constantes de alisamento
model_optimal = ExponentialSmoothing(train['Vendas'], trend='add', seasonal=None).fit()
predictions_optimal = model_optimal.predict(start=len(train), end=len(df) - 1)
 
alpha_optimal = model_optimal.params['smoothing_level']
beta_optimal = model_optimal.params['smoothing_trend']
print(f'Valor Ótimo de Alpha: {alpha_optimal:.4f}')
print(f'Valor Ótimo de Beta: {beta_optimal:.4f}')
 
 
print('\nResultados de Otimização:')
print(model_optimal.summary())
 
 
# Calcular os erros de previsão
rmse_optimal = np.sqrt(mean_squared_error(test['Vendas'], predictions_optimal))
mae_optimal = mean_absolute_error(test['Vendas'], predictions_optimal)
mape_optimal = np.mean(np.abs((test['Vendas'] - predictions_optimal) / test['Vendas'])) * 100
 
print(f'Raiz do Erro Quadrático Médio (RMSE) doHoltcom Alphae Beta Ótimos: {rmse_optimal:.2f}')
print(f'Erro Absoluto Médio (MAE) do Holtcom Alphae Beta Ótimos: {mae_optimal:.2f}')
print(f'Erro Percentual Absoluto Médio (MAPE) do Holtcom Alphae Beta Ótimos: {mape_optimal:.2f}%')
 
 
# Representação gráfica das vendas e previsões
plt.plot(df['Data'], df['Vendas'],label='Vendas')
plt.plot(test['Data'], predictions_optimal, label='Previsões (Alphae Beta Ótimos)', linestyle='dashed')
plt.title('Previsões de Vendas de DVD com o Alis. Exponencial de Holt')
plt.xlabel('Data')
plt.xticks(rotation='vertical')
plt.ylabel('Vendas')
plt.legend()
plt.show()
 
# Previsões para as próximas 17 semanas com a escolha ótima de alpha e beta

horizon = 17

model_df_optimal = ExponentialSmoothing(df['Vendas'], trend='add', seasonal=None, seasonal_periods=12).fit(smoothing_level=alpha_optimal, smoothing_trend=beta_optimal)

print(model_df_optimal.summary())


predictions10_optimal = model_df_optimal.predict(start=len(df['Vendas']), end=len(df['Vendas']) + horizon - 1)

extended_index = pd.date_range(start=df['Data'].iloc[-1] + pd.DateOffset(1), periods=horizon, freq='W')

extended_df = pd.DataFrame({'Data': extended_index, 'Vendas': predictions10_optimal})

df_new = pd.concat([df, extended_df], ignore_index=True)

plt.plot(df['Data'], df['Vendas'], label='Observado')

plt.plot(extended_df['Data'], predictions10_optimal, label='Previsões (Alpha e Beta Ótimos)')

plt.title('Previsões de Vendas de DVD com Alis. Exponencial de Holt')

plt.xlabel('Data')

plt.xticks(rotation='vertical')

plt.ylabel('Vendas')

plt.legend()

plt.show()
 
