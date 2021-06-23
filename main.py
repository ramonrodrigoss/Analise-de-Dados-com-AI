import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt



#Importar tabela para analise
tabela = pd.read_csv("advertising.csv")

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()


sns.pairplot(tabela)
plt.show()