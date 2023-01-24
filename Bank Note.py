import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
train_data=pd.read_csv('BankNote_Authentication.csv')
x=train_data.iloc[:,:-1]
y=train_data.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)
classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,y_pred)
score
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
classifier.predict([[-2.2,-4.7428,6.3489,0.11162]])

