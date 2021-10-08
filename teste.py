import sklearn
import graphviz
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz


def modelo_teste(x1,y1):
    SEED = 3000
    np.random.seed(SEED)
    treino_x1, teste_x1, treino_y1, teste_y1 = train_test_split(x1, y1, test_size=0.25, stratify=y1)


    scaler = StandardScaler()
    scaler.fit(treino_x1)
    treino_x1 = scaler.transform(treino_x1)
    teste_x1 = scaler.transform(teste_x1)

    modelo = LinearSVC()
    modelo.fit(treino_x1, treino_y1)
    previsoes = modelo.predict(teste_x1)

    misterioso1 = [75.0, 0, 0, 1, 0, 1, 100]
    misterioso2 = [55.0, 0, 0, 0, 0, 1, 4]
    testeM = [misterioso1, misterioso2]
    resultado_teste = modelo.predict(testeM)
    print(resultado_teste)


    acuracia = accuracy_score(teste_y1, previsoes) * 100
    print("treino com %d elementos e teste com %d elementos" % (len(treino_x1), len(teste_x1)))
    print("Acuracia de: %.2f%%" % acuracia)

    features1 = x1.columns
    dot_data1 = export_graphviz(modelo, out_file=None,
                                filled=True, rounded=True,
                                feature_names=features1,
                                class_names=["sim", "não"])
    grafico1 = graphviz.Source(dot_data1)
    #grafico1