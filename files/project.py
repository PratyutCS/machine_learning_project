import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

dataframe=pd.read_csv("./data.csv")

print(dataframe.head())
print(dataframe.tail())
print(dataframe.shape)
print(dataframe.info())
print(dataframe.describe())
print(dataframe.duplicated().sum())
print(dataframe.isnull().sum())

dataframe['diagnosis']=dataframe['diagnosis'].map({'M':1,'B':0})
print(dataframe.head())

df = dataframe.drop(["id"],axis=1)
dataframe = dataframe.drop(["id"],axis=1)

print(df)

plt.figure(figsize=(7,5))

plt.subplot(231)
sns.histplot(data=dataframe, x="fractal_dimension_se", hue="diagnosis", multiple="stack",binwidth=2)
plt.title('fractal_dimension_se VS diagnosis')

plt.subplot(234)
sns.histplot(data=dataframe, x="perimeter_worst", hue="diagnosis", multiple="stack",binwidth=3)
plt.title('perimeter_worst VS diagnosis')

df_corr = df.drop(['diagnosis'], axis=1)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
cax = ax.matshow(df_corr.corr(), vmin=-1, vmax=1, interpolation='none')
ax.grid(False)
fig.colorbar(cax)
ticks = np.arange(0, len(df_corr.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df_corr.columns, rotation=90)
ax.set_yticklabels(df_corr.columns)
plt.show()

X = dataframe.loc[:,dataframe.columns[1:]]
y = dataframe['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.iloc[0])

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 
y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
aur_dt = roc_auc_score(y_test, y_pred_dt)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
aur_rf = roc_auc_score(y_test, y_pred_rf)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train) 
y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
aur_lr = roc_auc_score(y_test, y_pred_lr)


# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train) 
y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
aur_knn = roc_auc_score(y_test, y_pred_knn)

print("Decision Tree Accuracy: {:.2f}".format(acc_dt))
print("Decision Tree AUROC: {:.2f}".format(aur_dt))
print("Random Forest Accuracy: {:.2f}".format(acc_rf))
print("Random Forest AUROC: {:.2f}".format(aur_rf))
print("Logistic Regression Accuracy: {:.2f}".format(acc_lr))
print("Logistic Regression AUROC: {:.2f}".format(aur_lr))
print("KNN Accuracy: {:.2f}".format(acc_knn))
print("KNN AUROC: {:.2f}".format(aur_knn))

data = {'dt':acc_dt, 'rf':acc_rf, 'lr':acc_lr,'knn':acc_knn}

mname = list(data.keys())
values = list(data.values())
plt.bar(mname, values, color ='blue',width = 0.4)
 
plt.xlabel("models")
plt.ylabel("accuracy")
plt.title("model Comparision")
plt.show()

fpr_dt, tpr_dt, thresholds = roc_curve(y_test, y_pred_dt)
roc_auc = auc(fpr_dt, tpr_dt)
print("AUROC: {:.2f}".format(roc_auc))

plt.figure(figsize=(7,5))
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve descision tree')
plt.legend(loc="lower right")
plt.show()


fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr_rf, tpr_rf)
print("AUROC: {:.2f}".format(roc_auc))

plt.figure(figsize=(7,5))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve random forest')
plt.legend(loc="lower right")
plt.show()

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
roc_auc = auc(fpr_lr, tpr_lr)
print("AUROC: {:.2f}".format(roc_auc))

plt.figure(figsize=(7,5))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve Logistic Regression')
plt.legend(loc="lower right")
plt.show()

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)
roc_auc = auc(fpr_knn, tpr_knn)
print("AUROC: {:.2f}".format(roc_auc))

plt.figure(figsize=(7,5))
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve KNN')
plt.legend(loc="lower right")
plt.show()
