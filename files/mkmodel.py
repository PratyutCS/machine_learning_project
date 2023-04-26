import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

dataframe=pd.read_csv("./data.csv")

dataframe['diagnosis']=dataframe['diagnosis'].map({'M':1,'B':0})

dataframe = dataframe.drop(["id"],axis=1)

X = dataframe.loc[:,dataframe.columns[1:]]
y = dataframe['diagnosis']

# Decision Tree
rf = RandomForestClassifier()
rf.fit(X, y)

joblib.dump(rf, 'model.pkl')
