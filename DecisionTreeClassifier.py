import sklearn
import matplotlib.pyplot as plt
import graphviz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz


def modelo_DecisionTreeClassifier(x,y):
    SEED = 3000
    np.random.seed(SEED)
    Rtreino_x, Rteste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

    modelo = DecisionTreeClassifier()
    modelo.fit(Rtreino_x, treino_y)
    previsoes = modelo.predict(Rteste_x)
    acuracia = accuracy_score(teste_y, previsoes) * 100
    print("Acuracia do modelo_SVM: %.2f%%" % acuracia)

    features = x.columns
    dot_data = export_graphviz(modelo, out_file=None,
                               filled=True, rounded=True,
                               feature_names=features,
                               class_names=["não", "sim"])
    grafico = graphviz.Source(dot_data)
    #print(grafico)

