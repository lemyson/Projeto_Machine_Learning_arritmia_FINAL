import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def modelo_SVM(x,y):
    SEED = 3000
    np.random.seed(SEED)
    Rtreino_x, Rteste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

    scaler = StandardScaler()
    scaler.fit(Rtreino_x)
    treino_x = scaler.transform(Rtreino_x)
    teste_x = scaler.transform(Rteste_x)

    modelo = SVC()
    modelo.fit(treino_x, treino_y)
    previsoes = modelo.predict(teste_x)

    acuracia = accuracy_score(teste_y, previsoes) * 100
    print("Acuracia do modelo_SVM: %.2f%%" % acuracia)
    print("treino com %d elementos e teste com %d elementos" % (len(treino_x), len(teste_x)))