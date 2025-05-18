from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3
from googletrans import Translator
import warnings

app = Flask(__name__)

translator = Translator()

#Reading data

df = []

data_files = ['schizophrenia.csv','fitness.csv','jokes.csv','meditation.csv','parenting.csv','relationships.csv','teaching.csv']
for file in data_files:
    data = pd.read_csv(file,encoding="latin-1")
    df.append(data)
df = pd.concat(df)

df = df.dropna()
df['text'] = df['Post Text']

Tweet = []
Labels = []

for row in df["text"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)

# Encoded Sentiment columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Label'] = encoder.fit_transform(df['Label'])

df['message']=df['text']

#df = df[0:2000]
X = df['message']
y = df['Label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
classifier.fit(X_train, y_train) 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

@app.route('/')
def home():
	return render_template('home.html')

@app.route("/signup")
def signup():
    
    
    name = request.args.get('username','')
    number = request.args.get('number','')
    email = request.args.get('email','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `detail` (`name`,`number`,`email`, `password`) VALUES (?, ?, ?, ?)",(name,number,email,password))
    con.commit()
    con.close()

    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('name','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from detail where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()
    print(data)

    if data == None:
        return render_template("signup.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")





@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        message = request.form['message']
        translations = translator.translate(message, dest='en')
        message =  translations.text
        data = [message]
        #cv = CountVectorizer()
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        
        print(my_prediction[0])
        
        return render_template('result.html',prediction = my_prediction[0],message=message)


@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/reg')
def reg():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/reddit')
def reddit():
	return render_template('reddit.html')

@app.route('/notebook')
def notebook():
	return render_template('notebook.html')

if __name__ == '__main__':
	app.run(debug=False)
