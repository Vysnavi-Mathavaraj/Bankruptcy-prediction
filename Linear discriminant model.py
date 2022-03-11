"""
Linear Discriminant Analysis with Training & Test Sampling, Stratified KFold Sampling
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (train_test_split)
from sklearn.model_selection import RepeatedStratifiedKFold

df=pd.read_csv('h2Bank.csv', header=None) # use it when file does not have headers

# Rename column titles
df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'decision']
print (df.head()) # see first six rows to check everything

# Define independent variables and class variables
X = df[['v1', 'v2', 'v3', 'v4', 'v5']]
y = df['decision']

# split dataset into training and testing 70-30 ratio
X_train, X_test, y_train, y_test=train_test_split (X,y, test_size=0.3)# add fourth parameter random_state=10 for seeded random number generation
print('size of test dataset:',len(X_test), ' size of training dataset: ', len(X_train))

#Fit the LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train) #learn Discriminant Function

from sklearn.metrics import confusion_matrix
#Create Training and Test Dataset
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1)
y_pred=model.predict(X_test) # predict test dataset
confusion=confusion_matrix(y_test,y_pred)#,labels=[0,1])
print(confusion) # Column is Actual and Row Title is Predicted

# Print Training and Test Accuracies
result1 = model.score(X_train, y_train)
print(("LDA Training Accuracy: %.2f%%") % (result1*100))

result = model.score(X_test, y_test)
print(("LDA Test Accuracy: %.2f%%") % (result*100.0))

# Cross Validation using 10 fold sampling
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

model1 = LinearDiscriminantAnalysis()
model1.fit(X, y)

#evaluate model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(("Stratified KFold Accuracy: %.2f%%") % (np.mean(scores)*100)) 


