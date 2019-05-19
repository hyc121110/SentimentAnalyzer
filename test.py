import pickle
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

model1 = pickle.load(open("models/model_nb.pk1", "rb"))
model2 = pickle.load(open("models/model_lsvc.pk1", "rb"))
model3 = pickle.load(open("models/model_lr.pk1", "rb"))
model4 = pickle.load(open("models/model_rf.pk1", "rb"))
count_vec = pickle.load(open("models/count_vec.pk1", "rb"))

# Try with user input
print("Please enter a review of a company: ", end='')
sentence = str(input())
sentence = [word for word in sentence.split() if word.lower() not in stop_words]
sentence = " ".join(sentence)
sent = count_vec.transform([sentence])

# pick one model (lr has the highest acc)
pred1 = model1.predict(sent)[0]
pred2 = model2.predict(sent)[0]
pred3 = model3.predict(sent)[0]
pred4 = model4.predict(sent)[0]
pred = pred1 + pred2 + pred3 + pred4
res = "\nThis is a positive review!" if pred > 2 else "This is a negative review!"
print(res)