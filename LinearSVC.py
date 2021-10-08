import sklearn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def modelo_LinearSVC(x,y):
    SEED = 3000
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

    scaler = StandardScaler()
    scaler.fit(treino_x)
    treino_x = scaler.transform(treino_x)
    teste_x = scaler.transform(teste_x)

    modelo = LinearSVC()
    modelo.fit(treino_x, treino_y)
    previsoes = modelo.predict(teste_x)
    acuracia = accuracy_score(teste_y, previsoes) * 100

    print("Acuracia do modelo_LinearSVC: %.2f%%" % acuracia)
    print("treino com %d elementos e teste com %d elementos" % (len(treino_x), len(teste_x)))