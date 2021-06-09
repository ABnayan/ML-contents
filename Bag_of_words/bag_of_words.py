#import the dataset
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVCimport numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,precision_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#read the dataset
data = pd.read_excel('train_data_nlp.xlsx')
X, y = data.word, data.polarity


#Lemmatizer
documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

#text to vector
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()

#shufflesplit
sss = StratifiedShuffleSplit(n_splits=4, test_size=0.2,
                             random_state=0)
sss.get_n_splits(X, y)


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# model to classify the predict the values
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 


# classifier = SVC(C= 1,gamma= 1,kernel= 'rbf')
# classifier.fit(X_train, y_train)

#predict the values based on testing dataset
y_pred = classifier.predict(X_test)

#metrics values based on the predicted outcomes

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))   #0.5168539325842697

# saving the model for future use
with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

# for check the prediction of user input values
a = "err-free"
data = [a]
X = vectorizer.transform(data).toarray()
my_prediction = classifier.predict(X)
