
# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from mlxtend.plotting import plot_confusion_matrix
from sklearn import linear_model

# open and read excel file to dataframe


file_csv =("C:/Users/eve10/Desktop/Alloys Composition weight predict.csv")
dataframe = pd.read_csv(file_csv, encoding= 'unicode_escape',header=0)

# head of the data
print(dataframe.head())

# details of data
print(dataframe.shape)
dataframe.describe()

# check Multi-Collinearity
data=dataframe.drop(['Printability'], axis=1)
kwargs = {'alpha':.9, 'linecolor':'k','rasterized':False, 'edgecolor':'w', 'capstyle':'projecting',}
fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(data.corr(),vmin=-1,center=0, cmap="BrBG",linewidth=.2, annot=True, **kwargs)
ax.set_title("Multi-Collinearity of Alloy Attributes")

# prepare data for normalization
# Normalise training data
target_column = ['Printability']
predictors = list(set(list(dataframe.columns))-set(target_column))
dataframe[predictors] = dataframe[predictors]/dataframe[predictors].max()
dataframe.describe()

# get features and target from dataframe
X = dataframe[predictors].values
y = dataframe[target_column].values

  
# using the train test split function
X_train, X_test,y_train, y_test = train_test_split(X,y ,random_state=42, test_size=0.3, shuffle=True)

# logistic regression model 
# instantiation of the Model
logisticRegr = linear_model.LogisticRegression()

# Training the model on the data, storing the information learned from the data
logisticRegr.fit(X_train, y_train)

# Predict labels for new data
# Use information the model learned during the model training process
y_predictions = logisticRegr.predict(X_test)

# Measuring model's accuracy 
score = accuracy_score(y_predictions, y_test)
print(score)

# confusion matrix describe the performance of a classification model

def model_matrix(y_test, y_predictions):
  con_matrix= metrics.confusion_matrix(y_test, y_predictions)
  print(con_matrix)
  plt.figure(figsize=(9,9))
  sns.heatmap(con_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  all_sample_title = 'Accuracy Score: {0}'.format(score)
  plt.title(all_sample_title, size = 15)
  return plt.show()


# Support vector machine classification model
# instantiating the model
param = {'C':(0,0.01,0.5,0.1,1,2,5,10,50,100,500,1000)}
kernel=["linear","rbf","poly"]
for i in kernel:
 svm1 = svm.SVC(kernel="i",gamma=0.5)
 svm.grid = GridSearchCV(svm1,param,n_jobs=1,cv=10,verbose=1,scoring='accuracy')

# Training the model on the data, storing the information learned from the data
 svm.grid.fit(X_train,y_train)

 svm.grid.best_params_

 linsvm_clf = svm.grid.best_estimator_
# Predict labels for new data
 y_predictions=  linsvm_clf.predict(X_test)

# Measuring model's accuracy 
 score=accuracy_score(y_test,y_predictions)

# confusion matrix describe the performance of a classification model
 model_matrix(y_test, y_predictions)


 # Decision tree classification model
# instantiating the model
deci_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

# training of model
deci_tree.fit(X_train, y_train)


from mlxtend.plotting import plot_decision_regions
 
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
 
fig, ax = plt.subplots(figsize=(7, 7))
plot_decision_regions(X_combined, y_combined, clf=deci_tree)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
 

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()



