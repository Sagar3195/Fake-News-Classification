### Fake News Classification based on the News Title

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Loading fake news dataset
df = pd.read_csv("C:/PythonCSV/CSV_file/fake_news_train.csv")

print(df.head(10))

print(df.shape)

#Now we check missing values in dataset
print(df.isnull().sum())

#Now we drop missing values from dataset
df = df.dropna()

#There is no missing values in dataset
print(df.isnull().sum())

message = df.copy()

print(message.head(10)) #we can see that after drop nan values we get 5 then 7 id,so we reset index.

message.reset_index(inplace= True)

print(message.head(10))

#Now we get id 6 value
print(message['title'][6])


import nltk
import re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#Create object for stemming
stemmer = PorterStemmer()

corpus = []
for i in range(len(message)):
    text = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)

#print(corpus)

print(len(corpus))

#creating model for Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features= 5000 , ngram_range= (1,3))


#now independent variable 
X = vectorizer.fit_transform(corpus)

X = X.toarray()

print(X.size)

print(vectorizer.get_feature_names())

print(len(vectorizer.get_feature_names()))

y = message['label']

#y.head()

print(y.shape)

#now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size = 0.3, random_state = 0)

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

#here we can see that combiations of 2 words and 3 words bcoz of ngram_range parameter.
print(vectorizer.get_feature_names()[:10])

import itertools

#now create model for multinomialNB classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

#now fit the model and predict the model
clf.fit(x_train, y_train)

predi = clf.predict(x_test)


#now checking the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, predi)
print("Accuracy of the model: ",accuracy)

cm = confusion_matrix(y_test, predi)
print("Confusion matrix of the model:\n",cm)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#visualize the confusion matrix
plot_confusion_matrix(cm, classes = ['FAKE', "REAL"])

##Classification report of the matrix
print("Classification report of the model: \n", classification_report(y_test, predi))

import joblib
#joblib.dump(clf, 'fake_news_model.pkl')
#joblib.dump(vectorizer, 'transform.pkl')


