

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dados importados de link da internet

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(url)

df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month', inplace=True)

# Dividir dados em treino e teste

test_size = 24

train, test = df[:-test_size], df[-test_size:]

# Método de Holt-Winters com Sazonalidade Multiplicativa (com escolha ótima das constantes de alisamento)

model_optimal = ExponentialSmoothing(train['Passengers'], 

                                     trend='add', 

                                     seasonal='multiplicative', 

                                     seasonal_periods=12).fit()

predictions_optimal = model_optimal.predict(start=len(train), end=len(df) - 1)

alpha_optimal = model_optimal.params['smoothing_level']

beta_optimal = model_optimal.params['smoothing_trend']

gamma_optimal = model_optimal.params['smoothing_seasonal']

print(f'Valor Ótimo de Alpha: {alpha_optimal:.4f}')

print(f'Valor Ótimo de Beta: {beta_optimal:.4f}')

print(f'Valor Ótimo de Gamma: {gamma_optimal:.4f}')


print('\nResultados de Otimização:')

print(model_optimal.summary())
 
# Calcular os erros de previsão

rmse_optimal = np.sqrt(mean_squared_error(test['Passengers'], predictions_optimal))

mae_optimal = mean_absolute_error(test['Passengers'], predictions_optimal)

mape_optimal = np.mean(np.abs((test['Passengers'] - predictions_optimal) / test['Passengers'])) * 100

print(f'Raiz do Erro Quadrático Médio (RMSE) com Alpha, Beta e Gamma Ótimos: {rmse_optimal:.2f}')

print(f'Erro Absoluto Médio (MAE) do Holt-Winters com Alpha, Beta e Gamma Ótimos: {mae_optimal:.2f}')

print(f'Erro Percentual Absoluto Médio (MAPE) com Alpha, Beta e Gamma Ótimos: {mape_optimal:.2f}%')

 
# Representação gráfica do número de passageiros e previsões
plt.plot(df.index, df['Passengers'], label='Passageiros')
plt.plot(test.index, predictions_optimal, label='Previsões (Alpha, Beta e GammaÓtimos)', linestyle='dashed')
plt.title('Previsões de Passageiros Aéreos com o Holt-Winters Multiplicativo')
plt.xlabel('Data')
plt.ylabel('Passageiros')
plt.legend()
plt.show()
 
horizon = 24
model_df_optimal = ExponentialSmoothing(df['Passengers'], trend='add', seasonal='multiplicative', seasonal_periods=12).fit(
smoothing_level=alpha_optimal, smoothing_trend=beta_optimal, smoothing_seasonal=gamma_optimal)
 
print(model_df_optimal.summary())
 
predictions_optimal = model_df_optimal.predict(start=len(df['Passengers']), end=len(df['Passengers']) + horizon - 1)
extended_index = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=horizon, freq='M')
extended_df = pd.DataFrame({'Passengers':predictions_optimal}, index=extended_index)
df_new = pd.concat([df, extended_df])
 
plt.plot(df.index, df['Passengers'], label='Observado')
plt.plot(extended_df.index, predictions_optimal, label='Previsões(Alpha, Beta e Gamma Ótimos)')
plt.title('Previsões de Passageiros Aéreos com o Holt-Winters Multiplicativo')
plt.xlabel('Data')
plt.ylabel('Passageiros')
plt.legend()
plt.show()
 
 
