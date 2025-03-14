import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
df['Número Días de Estancia'] = pd.to_numeric(df['Número Días de Estancia'], errors='coerce')
df['Precio Pagado'] = pd.to_numeric(df['Precio Pagado'], errors='coerce')
df['Puntuación satisfacción'] = pd.to_numeric(df['Puntuación satisfacción'], errors='coerce')

def generar_tabla_frecuencia(df, columna):
    df_freq = df.groupby(columna).agg(Freq=(columna, 'count')).reset_index()
    df_freq['Freq_Rel'] = 100 * df_freq['Freq'] / df_freq['Freq'].sum()
    df_freq['Freq_Acum'] = df_freq['Freq'].cumsum()
    df_freq['Freq_Rel_Acum'] = df_freq['Freq_Rel'].cumsum()
    return df_freq

for var in ['Sexo', 'Nacionalidad', 'Tipo Habitación']:
    df_freq = generar_tabla_frecuencia(df, var)
    plt.bar(df_freq[var], df_freq['Freq'], color='forestgreen', edgecolor='black')
    plt.title(f'Frecuencia según {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.show()

bins_dict = {
    'Edad': [10, 20, 30, 40, 50, 60, 70, 80],
    'Número Días de Estancia': [0, 7, 14, 21, 28],
    'Precio Pagado': [0, 300, 600, 900, 1200, 1500],
    'Puntuación satisfacción': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

def agrupar_datos(df, columna, bins):
    df[columna] = pd.to_numeric(df[columna], errors='coerce')

    if df[columna].isna().all():
        print(f"⚠️ La columna '{columna}' está vacía después de la conversión. No se puede agrupar.")
        return None

    labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]

    df[f"{columna}_grupo"] = pd.cut(df[columna], bins=bins, labels=labels, right=True)

    return generar_tabla_frecuencia(df, f"{columna}_grupo")

for var, bins in bins_dict.items():
    df_freq = agrupar_datos(df, var, bins)
    
    if df_freq is not None:  
        columna_agrupada = f"{var}_grupo"
        if columna_agrupada in df_freq.columns:  
            plt.bar(df_freq[columna_agrupada], df_freq['Freq'], color='forestgreen', edgecolor='black')
            plt.title(f'Frecuencia según {var}')
            plt.xlabel(var)
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f" La columna agrupada '{columna_agrupada}' no se encuentra en df_freq.")
    else:
        print(f"No se pudo generar la tabla de frecuencias para '{var}'.")


variables_numericas = ['Edad', 'Número Días de Estancia', 'Precio Pagado', 'Puntuación satisfacción']

plt.figure(figsize=(12, 6))
for i, col in enumerate(variables_numericas, 1):
    plt.subplot(1, len(variables_numericas), i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot de {col}")

plt.tight_layout()
plt.show()


sns.pairplot(df, hue='Tipo Habitación', palette='tab10')
plt.show()
sns.pairplot(df, hue='Nacionalidad', palette='mako')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df[['Precio Pagado', 'Número Días de Estancia', 'Puntuación satisfacción', 'Edad']].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Mapa de Correlación")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

def entrenar_modelo(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return model, y_pred


x = df[["Edad"]]
y = df[["Número Días de Estancia"]]
modelo_edad_estancia, y_pred = entrenar_modelo(x, y)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlabel("Edad")
plt.ylabel("Número de días de estancia")
plt.title("Relación entre Edad y Días de Estancia")
plt.show()

print("Intercepto:", modelo_edad_estancia.intercept_)
print("Coeficiente:", modelo_edad_estancia.coef_)
print("R²:", r2_score(y, y_pred))

x_pred = np.array([[45]])
y_pred = modelo_edad_estancia.predict(x_pred)
print(f"Una persona de 45 años se quedaría aproximadamente {y_pred[0][0]:.2f} días en el hotel.")

x = df[['Edad', 'Puntuación satisfacción', 'Número Días de Estancia']]
y = df[['Precio Pagado']]
modelo_multiple, y_pred = entrenar_modelo(x, y)

plt.scatter(y, y_pred)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Comparación de Precio Real vs Predicho")
plt.show()

print("R²:", r2_score(y, y_pred))


x_pred = np.array([[30, 8, 10]])
y_pred = modelo_multiple.predict(x_pred)
print(f"Una persona de 30 años con 8 de satisfacción y 10 días de estancia pagaría aproximadamente {y_pred[0][0]:.2f} euros.")


x = df[['Edad','Número Días de Estancia','Puntuación satisfacción']]
y = df[['Precio Pagado']]

model = Lasso()
model.fit(x, y)

y_pred = model.predict(x)
print('MAE:', mean_absolute_error(y, y_pred))
print('MAPE:', mean_absolute_percentage_error(y, y_pred))
print('MSE:', mean_squared_error(y, y_pred))
print("R^2: ", r2_score(y, y_pred))


dummies_sexo = pd.get_dummies(df['Sexo'], drop_first=True)
dummies_nacionalidad = pd.get_dummies(df['Nacionalidad'], drop_first=True)
dummies_hab = pd.get_dummies(df['Tipo Habitación'], drop_first=True)


df = pd.concat([df,dummies_sexo,dummies_nacionalidad,dummies_hab], axis=1)
df.head()

X = df[['Edad', 'mujer']]
y = df['Precio Pagado']

modelo = LinearRegression()

modelo.fit(X, y)

print('COEFICIENTES:',modelo.coef_)
print('INTERCEPTO:',modelo.intercept_)

y_pred=modelo.predict(X)

print('MAE:',mean_absolute_error(y, y_pred))
print('MAPE:',mean_absolute_percentage_error(y, y_pred))
print('MSE:',mean_squared_error(y, y_pred))
print("R^2: ",r2_score(y,y_pred))


x = df[['Edad','Número Días de Estancia','Puntuación satisfacción','mujer','No Europea','individual','suite','triple']]
y = df[['Precio Pagado']]


model = Lasso() 
model.fit(x, y)

print('COEFICIENTES:',model.coef_)
print('INTERCEPTO:',model.intercept_)

y_pred=model.predict(x)

print('MAE:',mean_absolute_error(y, y_pred))
print('MAPE:',mean_absolute_percentage_error(y, y_pred))
print('MSE:',mean_squared_error(y, y_pred))
print("R^2: ",r2_score(y,y_pred))