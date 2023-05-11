from flask import Flask, request,render_template,url_for,redirect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle
import re
import string
import joblib
nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

from keras.models import load_model
model = joblib.load('model1.pkl')


#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()
@app.route('/')
def home():
    return render_template('index.html')
# Bind predict function to URL
@app.route('/predict',methods =['POST','GET'])
def predict():
    message = request.form['text']
    data=transform_text(message)
    vector_sent=tfidf3.fit_transform([data])
    prediction3=model.predict(vector_sent)[0]
    output = prediction3
    # Check the output values and retrive the result with html tag based on the value
    if output>=0:
        print('inside')
        return render_template('index2.html')
    else:
        print("outside")
        return render_template('stress.html')

@app.route('/otherpage', methods=['GET', 'POST'])
def otherpage():
    if request.method == 'GET':
        # Handle form submission or perform necessary actions
        # ...
        return render_template('record.html')  # Redirect to a success page

    return render_template('record.html')
if __name__ == '__main__':
    app.run(debug=True)
