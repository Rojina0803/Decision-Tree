import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('kyphosis.csv')
# print(df.head())

# print(df.info())
# print(df.describe())
sns.pairplot(df,hue='Kyphosis')
# plt.show()

from sklearn.model_selection import train_test_split
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction= dtree.predict(X_test)
# print(prediction)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))