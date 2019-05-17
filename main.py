# Import libraries

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np 
import pickle
import os

# nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import WordNetLemmatizer

# initialize dataframe
df = pd.read_csv('data/reviews/employee_reviews.csv', index_col=0)
df = df[['pros', 'cons']]
#df.drop(columns=['location', 'dates', 'advice-to-mgmt', 'summary', 'helpful-count', 'job-title', 'link', 'overall-rating', 'work-balance-stars', 'culture-values-stars', 'carrer-opportunities-stars'], inplace=True)

pros_batch = list()
cons_batch = list()
for pros, cons in zip(df['pros'], df['cons']):
    words = tokenizer.tokenize(str(pros))
    pros_batch.append(words)
    words = tokenizer.tokenize(str(cons))
    cons_batch.append(words)
    
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

# building the classifier
count_vec = CountVectorizer()
X_counts = count_vec.fit_transform(batch)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# have 70% of the data as training data, 30% as testing data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.3, random_state=10)

# Naive Bayes Classifier: ~91% accuracy
nb = MultinomialNB().fit(X_train, y_train)
y_pred = nb.predict(X_test)


os.mkdir('models')
acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open("models/model_nb.pk1", "wb"))
print("Model NB created")

# Linear Support Vector Classifier: ~92% accuracy
l_svc = LinearSVC().fit(X_train, y_train)
y_pred = l_svc.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Linear SVC Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open("models/model_lsvc.pk1", "wb"))
print("Model LinearSVC created")

# Logistic Regression: ~92% accuracy
lr = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
y_pred = lr.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open("models/model_lr.pk1", "wb"))
print("Model Log Reg created")

# Random Forest: ~86% accuracy with the following params values
rf = RandomForestClassifier(max_depth=5, n_estimators=100).fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy on the companies dataset: {:.2f}%".format(acc*100))
pickle.dump(nb, open("models/model_rf.pk1", "wb"))
print("Model Random Forest created")

# Try with user input
print("Please type a review: ", end='')
sentence = str(input())
sentence = [word for word in sentence.split() if word.lower() not in stop_words]
sentence = " ".join(sentence)
sentence = count_vec.transform([sentence])

model = pickle.load(open("models/model_nb.pk1", "rb"))
pred = model.predict(sentence)[0]
res = "This is a positive sentence!" if pred == 1 else "This is a negative sentence!"
print(res)






