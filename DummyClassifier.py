import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def modelo_DummyClassifier(x,y):
    SEED = 3000
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

    dummy_stratified = DummyClassifier()
    dummy_stratified.fit(treino_x, treino_y)
    acuracia = dummy_stratified.score(teste_x, teste_y)*100

    print("Acuracia do modelo_DummyClassifier: %.2f%%" % acuracia)
    print("treino com %d elementos e teste com %d elementos" % (len(treino_x), len(teste_x)))