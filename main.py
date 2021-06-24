import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#Importar tabela para analise
tabela = pd.read_csv("advertising.csv")

#sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
#sns.pairplot(tabela)
#plt.show()

y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

#Criar as AI
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

count= 0
while count < 5:
#treina a AI
  modelo_regressaolinear.fit(x_treino, y_treino)
  modelo_arvoredecisao.fit(x_treino, y_treino)
  count+=1

#criar as previsores
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#comparar modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))


#plt.figure(figsize=(15,5))
#sns.lineplot(data=tabela_auxiliar)



sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()