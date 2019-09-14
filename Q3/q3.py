import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from util import *
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn import cluster, metrics, preprocessing

f = "./Data/epl_1819.csv"
df = read_csv(f)
print(df.columns)
# df.replace(["Champions League", "Champions League Qualification", "Europa League", "Europa League Qualification", "No UEFA Competitions","Relegated"], [0, 1, 2, 3, 4, 6])
a = []
for i in [0, 2, 4, 6, 7, 17]:
    a.append(df["category"][i])
df = df.replace(a, [5, 4, 3, 2, 1, 0])
df = df.drop(columns="Team")
KM_2 = cluster.KMeans(n_clusters=2).fit(df)
KM_5 = cluster.KMeans(n_clusters=5).fit(df)
KM_10 = cluster.KMeans(n_clusters=10).fit(df)
Ward = cluster.AgglomerativeClustering(linkage="ward", n_clusters=5).fit(df)
Complete = cluster.AgglomerativeClustering(linkage="complete", n_clusters=5).fit(df)
Average = cluster.AgglomerativeClustering(linkage="average", n_clusters=5).fit(df)
Single = cluster.AgglomerativeClustering(linkage="single", n_clusters=5).fit(df)
print("Clustering:")
for i in [KM_2, KM_5, KM_10, Ward, Complete, Average, Single]:
    print(i.labels_)
print("Iter:")
KM_10_1 = cluster.KMeans(n_clusters=10, max_iter=1).fit(df)
KM_10_10 = cluster.KMeans(n_clusters=10, max_iter=10).fit(df)
KM_10_100 = cluster.KMeans(n_clusters=10, max_iter=100).fit(df)
for i in [KM_10_1, KM_10_10, KM_10_100, KM_10]:
    print(i.labels_)
print("Silhueta:")
for i in [KM_2, KM_5, KM_10, Ward, Complete, Average, Single, KM_10_1, KM_10_10, KM_10_100]:
    print(metrics.silhouette_score(df, i.labels_) )
print("Normalizing")
df = DataFrame(preprocessing.Normalizer().fit_transform(df), columns=df.columns)
print(df)
KM_2 = cluster.KMeans(n_clusters=2).fit(df)
KM_5 = cluster.KMeans(n_clusters=5).fit(df)
KM_10 = cluster.KMeans(n_clusters=10).fit(df)
Ward = cluster.AgglomerativeClustering(linkage="ward", n_clusters=5).fit(df)
Complete = cluster.AgglomerativeClustering(linkage="complete", n_clusters=5).fit(df)
Average = cluster.AgglomerativeClustering(linkage="average", n_clusters=5).fit(df)
Single = cluster.AgglomerativeClustering(linkage="single", n_clusters=5).fit(df)
print("Clustering:")
for i in [KM_2, KM_5, KM_10, Ward, Complete, Average, Single]:
    print(i.labels_, metrics.silhouette_score(df, i.labels_))