# Initialization
import pandas as pd
import numpy as np
import nltk
import string
lemmatizer = nltk.WordNetLemmatizer()
np.set_printoptions(threshold=np.nan)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
translator = str.maketrans('', '', string.punctuation)

# Read in data, drop unimportant columns and rename column names to be more descriptive

emails_df = pd.read_csv("./data/spam.csv", encoding= "ISO-8859-1")
emails_df = emails_df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
emails_df.columns = ['type', 'body']

#changing classification to binary values
for i, email in emails_df.iterrows():
    if email['type'] == "ham":
        email['type'] = 0
    else: 
        email['type'] = 1

emails_smalldf = emails_df

#creating array of stop words
stop_words = ["a", "and", "are", "as", "at", "be", "by", "for", "from",  "the", "or", "us", "this"]

#clean function that gets rid of punctuation and lemmatizes a word (e.g. words -> word)
def clean(word) :
    cleaned_word = word.translate(translator)
    cleaned_word = lemmatizer.lemmatize(cleaned_word)
    return cleaned_word


#create dictionary with occurrences of each word in every email
word_dict = {}
for i, email in emails_smalldf.iterrows():
    for word in email['body'].split():
        # Remove punctuation and lemmatize
        cleaned_word = clean(word)
    
        if word in word_dict.keys(): 
            word_dict[cleaned_word] += 1
        else: 
            word_dict[cleaned_word] = 1
            
#remove stopwords       
for key in list(word_dict):
    if key in stop_words:
        del word_dict[key]

# make feature vectors
dictionary_keys = sorted(word_dict.keys(), key=str.lower)
emails_smalldf.shape[0]
feature_vector = np.zeros((emails_smalldf.shape[0], len(dictionary_keys)))
for i, email in emails_smalldf.iterrows():
    for word in email['body'].split():
        cleaned_word = clean(word)
        if cleaned_word in dictionary_keys and cleaned_word not in stop_words:
            feature_vector[i, list(dictionary_keys).index(cleaned_word)] += 1

# Split training data and change to int types
X_train, X_test, y_train, y_test = train_test_split(feature_vector, emails_smalldf['type'], test_size=0.33, random_state=42)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Gaussian Naive Bayes
gnb = GaussianNB()

# Fit and Predict
model = gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)

# Check Accuracy
print(accuracy_score(y_test, preds))