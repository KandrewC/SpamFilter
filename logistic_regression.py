# Importing packages 
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
lemmatizer = nltk.WordNetLemmatizer()


import time


start_time = time.time()

# Reading in test data and changing column names.
emails_df = pd.read_csv('./data/spam.csv', encoding= "ISO-8859-1")
emails_df = emails_df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
emails_df.columns = ['type', 'body']

for i, email in emails_df.iterrows():
    if email['type'] == "ham":
        email['type'] = 0
    else: 
        email['type'] = 1


emails_smalldf = emails_df

# Split data and convert into integer in order to have binary values
X_train_raw, X_test_raw, y_train, y_test = train_test_split(emails_smalldf['body'],emails_smalldf['type'])
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Create vectorizer to have tfidf featuers
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)

# Create logistic regression classifier to fit data.
log_classifier = LogisticRegression()
log_classifier.fit(X_train, y_train)
 
X_test = vectorizer.transform( X_test_raw)
predictions = log_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
print("Execution takes %s seconds"% (time.time() - start_time))