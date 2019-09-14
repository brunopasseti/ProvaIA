import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from util import *

from pandas import DataFrame, read_csv
f = './Data/train.csv'
df = read_csv(f)
df = df.drop('Name', axis=1)


# Tratando os dados, removendo colunas esparsas e dados que parecem irrelevantes:
flagsdf = df.isna()
a = []
for columns in flagsdf:
    sum = 0
    for row in flagsdf[columns]:
        if row: sum+=1
    a.append({"Column": columns, "Sum": sum})

# Removing Cabin
df = df.drop("Cabin", axis=1)
df = df.drop("Ticket", axis=1)

# Removing Rows that is missing age data
b = []
for idx,row in enumerate(flagsdf["Age"]):
    if row: b.append(idx)
df = df.drop(b)

# Removing Rows that is missing Embarked data
b = []
for idx,row in enumerate(flagsdf["Embarked"]):
    if row: b.append(idx)
df = df.drop(b)

# Histograma para ver balanceamento
df.hist()
# Boxplot confirmando presença de outliers
df.plot.box()

# Tratando dados que não são númericos
df["Sex"] = df["Sex"].apply(isMale)
df["Embarked"] = df["Embarked"].apply(embarkCodeToInt)


#Aleatorizando Datasete
df = df.sample(frac=1).reset_index(drop=True) # Randomizing full dataset

#Separando em treino e Teste
trainSample = df.iloc[:570] # Getting 80% train sample
testSample = df.iloc[570:712] # Getting 20% testSample

trainSample_x = trainSample.filter(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])
trainSample_y = trainSample.filter(['Survived'])
testSample_x = testSample.filter(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])
testSample_y = testSample.filter(['Survived'])

print("=============== Starting SVM ===============")
clfSVM = svm.SVC(gamma="auto", kernel="poly", degree=2, coef0=6).fit(trainSample_x.values, trainSample_y.values.ravel())
print(testSample_x)
print(trainSample_x)
testSampleResultSVM = clfSVM.predict(testSample_x)
print(metrics.confusion_matrix(testSample_y.values.ravel(), testSampleResultSVM))
print(metrics.f1_score(testSample_y.values.ravel(), testSampleResultSVM))
print(metrics.accuracy_score(testSample_y.values.ravel(), testSampleResultSVM))
print(metrics.classification_report(testSample_y.values.ravel(), testSampleResultSVM))

print("=============== Starting KNN ===============")
clfKNN = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree").fit(trainSample_x.values, trainSample_y.values.ravel())
correlation_matrix(trainSample_x)
testSampleResultKNN = clfKNN.predict(testSample_x)
print(metrics.confusion_matrix(testSample_y.values.ravel(), testSampleResultKNN))
print(metrics.f1_score(testSample_y.values.ravel(), testSampleResultKNN))
print(metrics.accuracy_score(testSample_y.values.ravel(), testSampleResultKNN))
print(metrics.classification_report(testSample_y.values.ravel(), testSampleResultKNN))


