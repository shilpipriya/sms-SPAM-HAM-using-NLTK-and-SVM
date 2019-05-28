# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('spam_msg.csv', error_bad_lines=False,encoding='latin-1')
dataset=dataset.iloc[:,0:2]

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

for i in range(0,5572):
    if(y[i]=='ham'):
        y[i]=0
    else:
        y[i]=1
y=y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#calculation
true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
for i in range(0,1115):
    if(y_test[i] == 1 and y_pred[i] == 1):
        true_pos=true_pos+1
    if(y_test[i] == 0 and y_pred[i] == 0):
        true_neg=true_neg+1
    if(y_test[i] == 0 and y_pred[i] == 1):
        false_pos=false_pos+1
    if(y_test[i] == 1 and y_pred[i] == 0):
        false_neg=false_neg+1
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
Fscore = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", Fscore)
print("Accuracy: ", accuracy)