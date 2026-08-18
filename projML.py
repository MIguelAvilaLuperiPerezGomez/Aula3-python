import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#dados = pd.read_csv("03 - Cap. 3 - data small.csv")
#print (dados)
dados = pd.DataFrame({
    "media" : [5.5, 8.0, 6.2],
    "frequencia" : [70, 90, 76],
    "trabalhos_entregues" : [2, 5, 3],
    "historico_reprovacao" : [1, 0, 1],
    "risco" : ["alto", "baixo", "moderado"]
})
x = dados[["media", "frequencia",  "trabalhos_entregues", "historico_reprovacao"]]
y = dados["risco"]
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(x, y)

novo_aluno = [[5.4, 72, 2, 1], [8.2, 92, 5, 0], [6.0, 75, 3, 1]]
predicao = modelo.predict(novo_aluno)
print(predicao)