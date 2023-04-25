import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

dataframe=pd.read_csv("./data.csv")

dataframe['diagnosis']=dataframe['diagnosis'].map({'M':1,'B':0})

dataframe = dataframe.drop(["id"],axis=1)

X = dataframe.loc[:,dataframe.columns[1:]]
y = dataframe['diagnosis']

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X, y)

joblib.dump(dt, 'model.pkl')
