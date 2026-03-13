import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("winners_f1.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=['time','laps','year'])

def time_to_sec(t):
    h, m, s = t.split(':')
    return int(h)*3600 + int(m)*60 + int(s)

df['time_sec'] = df['time'].apply(time_to_sec)

print("Tempo (s) - Média:", round(df['time_sec'].mean(), 2))
print("Tempo (s) - Mediana:", round(df['time_sec'].median(), 2))
print("Tempo (s) - Moda:", round(df['time_sec'].mode()[0], 2))

print("Voltas - Média:", round(df['laps'].mean(), 2))
print("Voltas - Mediana:", round(df['laps'].median(), 2))
print("Voltas - Moda:", round(df['laps'].mode()[0], 2))

print("Tempo (s) - Variância:", round(df['time_sec'].var(), 2))
print("Tempo (s) - Desvio padrão:", round(df['time_sec'].std(), 2))

print("Voltas - Variância:", round(df['laps'].var(), 2))
print("Voltas - Desvio padrão:", round(df['laps'].std(), 2))

plt.hist(df['time_sec'], bins='auto', color="#9FFFF5", edgecolor='black')
plt.title('Tempo de Corrida (s) 🏁')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência')
plt.show()

plt.hist(df['laps'], bins='auto', color='lightgreen', edgecolor='black')
plt.title('Número de Voltas 🏎️')
plt.xlabel('Voltas')
plt.ylabel('Frequência')
plt.show()

corr1 = df['time_sec'].corr(df['laps'])
corr2 = df['time_sec'].corr(df['year'])

print("Correlação tempo x voltas:", round(corr1, 2))
print("Correlação tempo x ano:", round(corr2, 2))

sns.scatterplot(x='laps', y='time_sec', data=df)
plt.title('Tempo de Corrida x Voltas')
plt.xlabel('Voltas')
plt.ylabel('Tempo (s)')
plt.show()

sns.scatterplot(x='year', y='time_sec', data=df)
plt.title('Tempo de Corrida x Ano')
plt.xlabel('Ano')
plt.ylabel('Tempo (s)')
plt.show()

def iqr_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    return mask, lower, upper

mask, lower, upper = iqr_outliers(df['time_sec'])
print(f"Tempo - limites IQR: [{round(lower,2)}, {round(upper,2)}]")
print(f"Outliers encontrados: {mask.sum()} de {len(df)}")

plt.boxplot(df['time_sec'])
plt.title("Tempo de Corrida - Antes de remover outliers")
plt.ylabel("Tempo (s)")
plt.ylim(df['time_sec'].min() - 1000, df['time_sec'].max() + 1000)  
plt.show()

df_no_outliers = df.loc[~mask]
plt.boxplot(df_no_outliers['time_sec'])
plt.title("Tempo de Corrida - Depois de remover outliers")
plt.ylabel("Tempo (s)")
plt.ylim(df['time_sec'].min() - 1000, df['time_sec'].max() + 1000)  
plt.show()