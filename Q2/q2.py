import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from util import *
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import metrics

f = "./Data/train.csv"
df = read_csv(f)
# index = df.index
eliminate = []
candidates = df.isna().sum().sort_values(ascending=False)
for row,value in candidates.items():
   if value > 8: eliminate.append(row)
   else: df.fillna(inplace=True, value=df[row].mode()[0])

df = df.drop(columns=eliminate)
df.drop(columns="Id")
eliminate = []
example = df.sample(n=1)
for idx, columns in example.items():
    for row in columns:
        if(type(row) == type("foi")): eliminate.append(idx)
df = df.drop(columns=eliminate)
corr = df.corr()['SalePrice']
goodCorr = corr.drop(index="SalePrice").abs().sort_values(ascending=False).head(n=10)
index = list(goodCorr.index) 
index.append("SalePrice")
df = read_csv(f, usecols=index)
eliminate = []
candidates = df.isna().sum().sort_values(ascending=False)
for row,value in candidates.items():
   if value > 8: eliminate.append(row)
   else: df.fillna(inplace=True, value=df[row].mode()[0])

df = df.drop(columns=eliminate)
eliminate = []
example = df.sample(n=1)
for idx, columns in example.items():
    for row in columns:
        if(type(row) == type("foi")): eliminate.append(idx)
df = df.drop(columns=eliminate)
df = df.sample(frac=1).reset_index(drop=True) # Randomizing full dataset
trainSample = df.iloc[:1168] # Getting 80% train sample
testSample = df.iloc[1168:1460] # Getting 20% testSample
BR = linear_model.BayesianRidge()
PC = linear_model.Perceptron()
SVR = svm.SVR()
RF = ensemble.RandomForestRegressor()

trainY = DataFrame(trainSample["SalePrice"])
trainX = trainSample.drop(columns="SalePrice")
testY = DataFrame(testSample["SalePrice"])
testX = testSample.drop(columns="SalePrice")

BR.fit(trainX,trainY.values.ravel())
PC.fit(trainX,trainY.values.ravel())
RF.fit(trainX,trainY.values.ravel())
SVR.fit(trainX,trainY.values.ravel())

resultBR = BR.predict(testX)
resultPC = PC.predict(testX)
resultSVR = SVR.predict(testX)
resultRF = RF.predict(testX)

resultBRdf = DataFrame(resultBR, columns=["SalePrice"])
differenceBR = resultBRdf.sub(testY.reset_index(drop=True))
resultPCdf = DataFrame(resultPC, columns=["SalePrice"])
differencePC = resultPCdf.sub(testY.reset_index(drop=True))
resultRFdf = DataFrame(resultRF, columns=["SalePrice"])
differenceRF = resultRFdf.sub(testY.reset_index(drop=True))
resultSVRdf = DataFrame(resultSVR, columns=["SalePrice"])
differenceSVR = resultSVRdf.sub(testY.reset_index(drop=True))

print(metrics.mean_absolute_error(resultBR, testY.reset_index(drop=True)))
print(metrics.mean_absolute_error(resultPC, testY.reset_index(drop=True)))
print(metrics.mean_absolute_error(resultRF, testY.reset_index(drop=True)))
print(metrics.mean_absolute_error(resultSVR, testY.reset_index(drop=True)))
