# Import libraries

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
import re
import time

# nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import WordNetLemmatizer

# initialize dataframe
df = pd.read_csv('data/employee_reviews.csv', index_col=0)
df = df[['pros', 'cons']]
#df.drop(columns=['location', 'dates', 'advice-to-mgmt', 'summary', 'helpful-count', 'job-title', 'link', 'overall-rating', 'work-balance-stars', 'culture-values-stars', 'carrer-opportunities-stars'], inplace=True)

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# tokenization with stop words removed
pros_batch = list()
cons_batch = list()
for pros, cons in zip(df['pros'], df['cons']):
    pros = replace_contractions(pros)
    words = tokenizer.tokenize(str(pros))
    pros_batch.append(words)
    cons = replace_contractions(cons)
    words = tokenizer.tokenize(str(cons))
    cons_batch.append(words)
    
# lemmatization
pros_lem = list()
cons_lem = list()
for p, c in zip(pros_batch, cons_batch):
    pros_no_stop = [word for word in p if word.lower() not in stop_words]
    lemm = WordNetLemmatizer()
    pros_lem_temp = [lemm.lemmatize(word) for word in pros_no_stop]
    pros_lem.append(" ".join(pros_lem_temp))
    cons_no_stop = [word for word in c if word.lower() not in stop_words]
    lemm = WordNetLemmatizer()
    cons_lem_temp = [lemm.lemmatize(word) for word in cons_no_stop]
    cons_lem.append(" ".join(cons_lem_temp))

# combine all reviews into one long list
batch = pros_lem + cons_lem
# pros are labeled as '1' and cons are labeled as '0'
labels = [1] * len(pros_batch) + [0] * len(cons_batch)

# specify path for saving models
path = 'models'
if not os.path.exists(path):
    os.mkdir(path)
    
# building the classifier
count_vec = CountVectorizer()
X_counts = count_vec.fit_transform(batch)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

pickle.dump(count_vec, open(path + "/" + "count_vec.pk1", "wb"))

# have 70% of the data as training data, 30% as testing data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.3)

# Naive Bayes Classifier: ~91% accuracy
nb = MultinomialNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)
    
acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open(path + "/" + "model_nb.pk1", "wb"))
print("Model NB created")

# Linear Support Vector Classifier: ~92% accuracy
l_svc = LinearSVC().fit(X_train, y_train)
y_pred = l_svc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Linear SVC Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open(path + "/" + "model_lsvc.pk1", "wb"))
print("Model LinearSVC created")

# Logistic Regression: ~92% accuracy
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_pred = lr.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open(path + "/" + "model_lr.pk1", "wb"))
print("Model Log Reg created")

# Random Forest: ~90% accuracy with the following params values (3 mins)
start = time.time()
rf = RandomForestClassifier(max_depth=100, n_estimators=100).fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open(path + "/" + "model_rf.pk1", "wb"))
print("Model Random Forest created")

end = time.time()
print("Time elapsed: {} seconds".format(end - start))




