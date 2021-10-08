import sklearn
import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.dummy import DummyClassifier
from sklearn.tree import export_graphviz
from LinearSVC import modelo_LinearSVC
from DummyClassifier import modelo_DummyClassifier
from SVM import modelo_SVM
from DecisionTreeClassifier import modelo_DecisionTreeClassifier

coracao = pd.read_csv("coracao")
mapa = {
    "age":"idade","anaemia": "anemia","creatinine_phosphokinase":"creatinina fosfoquinase","diabetes":"diabetes","ejection_fraction":"fração de ejeção",
    "high_blood_pressure":"pressão alta","platelets":"plaquetas","serum_creatinine":"creatinina sérica","serum_sodium":"soro sódico","sex":"sexo",
    "smoking":"fuma","time":"tempo", "DEATH_EVENT":"Evento de morte"
}
coracao = coracao.rename(columns=mapa)
#print(coracao)
x = coracao[["idade","anemia","creatinina fosfoquinase","diabetes","fração de ejeção","pressão alta","plaquetas","creatinina sérica","soro sódico","sexo","fuma","tempo"]]
y = coracao["Evento de morte"]
xt = coracao[["idade","anemia","diabetes","pressão alta","sexo","fuma","tempo"]]

modelo_LinearSVC(x,y)
modelo_DummyClassifier(x,y)
modelo_SVM(x,y)
modelo_DecisionTreeClassifier(x,y)




