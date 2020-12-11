from flask import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
import joblib

classifier = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('transform.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['title']
        data = [message]
        vectorizer = tfidf.transform(data).toarray()
        prediction = classifier.predict(vectorizer)
    return render_template('result.html', prediction_title = prediction)


if __name__ == '__main__':
    app.run(debug= True)

