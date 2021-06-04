#import libraries

import pandas as pd
import numpy as np
from datetime import datetime
from statistics import mean
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import os
import warnings
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,ExtraTreesClassifier ,GradientBoostingRegressor
from sklearn.feature_selection import RFECV,RFE,VarianceThreshold
from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split,learning_curve,GridSearchCV,StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import Counter
import time
#import scikitplot as skplt
import itertools
warnings.filterwarnings('ignore')
# Import label encoder 
from sklearn import preprocessing 
import json
from sklearn.preprocessing import StandardScaler

# load the dataset
df = pd.read_csv("stock_quotes.csv")    
X = df.iloc[:,2:].values
y = df.iloc[:,1].values
    
#Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
check_test = y_test

### Feature scaling StandardScaler based transform
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    
# Baseline - comparing model accuracy using all features across classifiers 
classifiers = [
    LinearRegression(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    SVR(),
    DecisionTreeRegressor()
    ]



 # Naive Train Accuracy
r2score = []
MAE = []
MSE = []

for clf in classifiers:
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf)
    r2score.append(clf.score(X_test,y_test))
    MAE.append(mean_absolute_error(y_test,y_pred))
    MSE.append(mean_squared_error(y_test,y_pred))

    print(r2score)
    print(MAE)
    print(MSE)
    
    r2score.clear()
    MAE.clear()
    MSE.clear()

# Naive Test Accuracy
r2score = []
MAE = []
MSE = []

for clf in classifiers:
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf)
    r2score.append(r2_score(y_test,y_pred))
    MAE.append(mean_absolute_error(y_test,y_pred))
    MSE.append(mean_squared_error(y_test,y_pred))

    print(r2score)
    print(MAE)
    print(MSE)

    r2score.clear()
    MAE.clear()
    MSE.clear()


# predict the results
rf = LinearRegression()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
df1 = pd.DataFrame()
df1['Check_test'] = check_test
df1['y_pred'] = y_pred
print(r2_score(y_test,y_pred))