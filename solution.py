#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import warnings
from sklearn import metrics

colnames = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","outcome"]

df = pd.read_csv('dataset/adult.csv', names=colnames, header=None)
df_test = pd.read_csv('dataset/adult_test.csv', names=colnames, header=None)

df.append(df_test, ignore_index=True)

df['occupation'].replace(to_replace=' ?', value="Unknown", inplace=True)
df['workclass'].replace(to_replace=' ?', value="Unknown", inplace=True)
df['native-country'].replace(to_replace=' ?', value="Unknown", inplace=True)


laben = pp.LabelEncoder()

laben.fit(df['race'])
df['race'] = laben.transform(df['race'])

laben.fit(df['sex'])
df['sex'] = laben.transform(df['sex'])

laben.fit(df['relationship'])
df['relationship'] = laben.transform(df['relationship'])

laben.fit(df['occupation'])
df['occupation'] = laben.transform(df['occupation'])

laben.fit(df['workclass'])
df['workclass'] = laben.transform(df['workclass'])

laben.fit(df['education'])
df['education'] = laben.transform(df['education'])

laben.fit(df['marital-status'])
df['marital-status'] = laben.transform(df['marital-status'])

laben.fit(df['native-country'])
df['native-country'] = laben.transform(df['native-country'])

laben.fit(df['outcome'])
df['outcome'] = laben.transform(df['outcome'])

vector = [x for x in range(42)]
vector.remove(39)
df['native-country'].replace(to_replace=vector, value=0, inplace=True)
df['native-country'].replace(to_replace=39, value=1, inplace=True)

X = df.drop(columns=['outcome'])
y = df['outcome']

X['squared-education-num']= X['education-num']**2
X['squared-relationship']= X['relationship']**2
X['gain-loss'] = X['capital-gain'] - X['capital-loss']

columns = ['capital-gain','capital-loss','hours-per-week', 'squared-education-num', 'squared-relationship', 'age', 'gain-loss']

scaler = pp.MinMaxScaler()
X[columns] = scaler.fit_transform(X[columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=145)

over_sampler = RandomOverSampler(random_state=100)
X_train, y_train = over_sampler.fit_resample(X_train, y_train)

clf = RandomForestClassifier(max_depth=20, min_samples_leaf=3, min_samples_split=6, n_estimators=250)
clf.fit(X_train, y_train)

f1_score = metrics.f1_score(y_test, clf.predict(X_test))
accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
precision = metrics.precision_score(y_test, clf.predict(X_test))
recall = metrics.recall_score(y_test, clf.predict(X_test))

print("F1 score: ", f1_score)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

