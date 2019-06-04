import pickle
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

model1 = pickle.load(open("models/model_nb.pk1", "rb"))
model2 = pickle.load(open("models/model_lsvc.pk1", "rb"))
model3 = pickle.load(open("models/model_lr.pk1", "rb"))
count_vec = pickle.load(open("models/count_vec.pk1", "rb"))

# Try with user input
print("Please enter a review of a company: ", end='')
sentence = str(input())
sentence = [word for word in sentence.split() if word.lower() not in stop_words]
sentence = " ".join(sentence)
sent = count_vec.transform([sentence])

pred1 = model1.predict_proba(sent)
print("\nNaive Bayes model is " + '{0:.2f}'.format(pred1[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred1[0][1]*100) + "% sure that this sentence is positive.")
pred2 = model2.predict_proba(sent)
print("Linear SVC model is " + '{0:.2f}'.format(pred2[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred2[0][1]*100) + "% sure that this sentence is positive.")
pred3 = model3.predict_proba(sent)
print("Logistic Regression model is " + '{0:.2f}'.format(pred3[0][0]*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format(pred3[0][1]*100) + "% sure that this sentence is positive.")

# average the score
score = (pred1[0][0] + pred2[0][0] + pred3[0][0]) / 3
print("\nAveraging the three models: the new model is " + '{0:.2f}'.format(score*100) + "% sure that this sentence is negative, " + '{0:.2f}'.format((1-score)*100) + "% sure that this sentence is positive.")

if score > 0.8:
    print("\nThis is a negative review!")
elif score < 0.2:
    print("\nThis is a positive review!")
else:
    print("\nThis is a neutral review!")