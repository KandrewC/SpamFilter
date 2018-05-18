# Initialization
import time
start_time = time.time()


import pandas as pd
import numpy as np
import nltk
import string
import sklearn

np.set_printoptions(threshold=np.nan)

emails_df = pd.read_csv("./data/spam.csv", encoding= "ISO-8859-1")
emails_df = emails_df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
emails_df.columns = ['type', 'body']

# Separating the dataset between the message bodies and the email types.
X = emails_df.iloc[:, 1].values
y = emails_df.iloc[:, 0].values

# separating the data into training and testing data, with an 80-20 split of training to testing, respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Training the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# predicting our test data
y_pred = classifier.predict(X_test) 

# creating a confusion matrix and printing out the report of the classifications
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

# printing the accuracy of predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

print("Execution takes %s seconds" % (time.time() - start_time))