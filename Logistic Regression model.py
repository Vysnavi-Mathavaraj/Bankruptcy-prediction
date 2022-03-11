"""
Logistic Regression with Training & Test Sampling, Stratified KFold Sampling
"""
import pandas as pd
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

# Logistic Regression with Area Under the Curve-may not work with Iris due to dataset loading
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X_train, y_train)
from sklearn.metrics import roc_auc_score
print("ROC-AUC score: %.3f" % roc_auc_score(y, clf.predict_proba(X)[:, 1]))

# Print Training and Test Accuracies
result1 = clf.score(X_train, y_train)
print(("LR Training Accuracy: %.2f%%") % (result1*100))

result = clf.score(X_test, y_test)
print(("LR Test Accuracy: %.2f%%") % (result*100))

# Cross Validation using 10 fold sampling
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X, y)

#evaluate model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(("Stratified KFold Accuracy: %.2f%%") % (np.mean(scores)*100)) 
